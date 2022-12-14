#!/usr/bin/env python3
#
# Copyright (c) Facebook, Inc. and its affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.
#

import functools
import json
import os
from collections import Counter

import torch
import torchvision.transforms as transforms
from data.dataset import JsonlDataset
from data.vocab import Vocab
from pytorch_pretrained_bert import BertTokenizer
from torch.utils.data import DataLoader
from transformers import ViltProcessor, FlavaProcessor


def get_transforms(args):
    if args.model in ["vilt", "flava"]:
        return transforms.Compose(
            [
                transforms.Resize((256, 256)),
                transforms.ToTensor()
            ]
        )
    else:
        return transforms.Compose(
            [
                transforms.Resize(256),
                transforms.CenterCrop(224),
                transforms.ToTensor(),
                transforms.Normalize(
                    mean=[0.46777044, 0.44531429, 0.40661017],
                    std=[0.12221994, 0.12145835, 0.14380469],
                ),
            ]
        )

def salt_and_pepper(frequency=0.05):
    def _salt_and_pepper(image):
        # mask =  torch.rand(image.shape, device=torch.device("cuda")) < frequency
        mask =  torch.rand(image.shape) < frequency
        # new_vals = (torch.rand(image.shape, device=torch.device("cuda")) < 0.5).float()
        new_vals = (torch.rand(image.shape) < 0.5).float()
        image[mask] = new_vals[mask]
        return image
    return _salt_and_pepper


def get_noisy_transforms(args):
    noisy_transforms = []

    # Color jitter
    if args.model in ["vilt", "flava"]:
        noisy_transforms.append(transforms.Compose(
            [
                transforms.Resize((256, 256)),
                transforms.ToTensor(),
                transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.2),
            ]
        )
        )
    else:
        noisy_transforms.append(transforms.Compose(
            [
                transforms.Resize(256),
                transforms.CenterCrop(224),
                transforms.ToTensor(),
                transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.2),
                transforms.Normalize(
                    mean=[0.46777044, 0.44531429, 0.40661017],
                    std=[0.12221994, 0.12145835, 0.14380469],
                ),
            ]
        ))

    # Gaussian blur
    if args.model in ["vilt", "flava"]:
        noisy_transforms.append(transforms.Compose(
            [
                transforms.Resize((256, 256)),
                transforms.ToTensor(),
                transforms.GaussianBlur(kernel_size=15, sigma=(1., 2.)),
            ]
        )
        )
    else:
        noisy_transforms.append(transforms.Compose(
            [
                transforms.Resize(256),
                transforms.CenterCrop(224),
                transforms.ToTensor(),
                transforms.GaussianBlur(kernel_size=15, sigma=(1.,2.)),
                transforms.Normalize(
                    mean=[0.46777044, 0.44531429, 0.40661017],
                    std=[0.12221994, 0.12145835, 0.14380469],
                ),
            ]
        ))

    # Salt and pepper
    if args.model in ["vilt", "flava"]:
        noisy_transforms.append(transforms.Compose(
            [
                transforms.Resize((256, 256)),
                transforms.ToTensor(),
                salt_and_pepper(frequency=0.05),
            ]
        )
        )
    else:
        noisy_transforms.append(transforms.Compose(
            [
                transforms.Resize(256),
                transforms.CenterCrop(224),
                transforms.ToTensor(),
                salt_and_pepper(frequency=.05),
                transforms.Normalize(
                    mean=[0.46777044, 0.44531429, 0.40661017],
                    std=[0.12221994, 0.12145835, 0.14380469],
                ),
            ]
        ))
    return noisy_transforms

def get_labels_and_frequencies(path):
    label_freqs = Counter()
    path = path.split('.jsonl')[0] + '_filtered' + '.jsonl'
    data_labels = [json.loads(line)["label"] for line in open(path)]
    if type(data_labels[0]) == list:
        for label_row in data_labels:
            label_freqs.update(label_row)
    else:
        label_freqs.update(data_labels)

    return list(label_freqs.keys()), label_freqs


def get_glove_words(path):
    word_list = []
    for line in open(path):
        w, _ = line.split(" ", 1)
        word_list.append(w)
    return word_list


def get_vocab(args):
    vocab = Vocab()
    if args.model in ["bert", "mmbt", "concatbert"]:
        bert_tokenizer = BertTokenizer.from_pretrained(
            args.bert_model, do_lower_case=True
        )
        vocab.stoi = bert_tokenizer.vocab
        vocab.itos = bert_tokenizer.ids_to_tokens
        vocab.vocab_sz = len(vocab.itos)

    else:
        # word_list = get_glove_words(args.glove_path)
        word_list = []
        vocab.add(word_list)

    return vocab


def collate_fn(batch, args):
    lens = [len(row[0]) for row in batch]
    bsz, max_seq_len = len(batch), max(lens)

    mask_tensor = torch.zeros(bsz, max_seq_len).long()
    text_tensor = torch.zeros(bsz, max_seq_len).long()
    segment_tensor = torch.zeros(bsz, max_seq_len).long()

    img_tensor = None
    if args.model in ["img", "concatbow", "concatbert", "mmbt"]:
        img_tensor = torch.stack([row[2] for row in batch])

    if args.task_type == "multilabel":
        # Multilabel case
        tgt_tensor = torch.stack([row[3] for row in batch])
    else:
        # Single Label case
        tgt_tensor = torch.cat([row[3] for row in batch]).long()

    for i_batch, (input_row, length) in enumerate(zip(batch, lens)):
        tokens, segment = input_row[:2]
        text_tensor[i_batch, :length] = tokens
        segment_tensor[i_batch, :length] = segment
        mask_tensor[i_batch, :length] = 1

    return text_tensor, segment_tensor, mask_tensor, img_tensor, tgt_tensor


def get_data_loaders(args):
    if args.model in ["bert", "mmbt", "concatbert"]:
        tokenizer = BertTokenizer.from_pretrained(args.bert_model, do_lower_case=True).tokenize
    elif args.model == "vilt":
        tokenizer = ViltProcessor.from_pretrained("dandelin/vilt-b32-mlm")
    elif args.model == "flava":
        tokenizer = FlavaProcessor.from_pretrained("facebook/flava-full")

    transforms = get_transforms(args)
    if args.image_noise_probability>0:
        noisy_transforms = get_noisy_transforms(args)
    else:
        noisy_transforms = None
    args.labels, args.label_freqs = get_labels_and_frequencies(os.path.join(args.data_path, args.task, "train.jsonl"))
    vocab = get_vocab(args)
    args.vocab = vocab
    args.vocab_sz = vocab.vocab_sz
    args.n_classes = len(args.labels)

    train = JsonlDataset(
        os.path.join(args.data_path, args.task, "train.jsonl"),
        tokenizer,
        transforms,
        vocab,
        args,
        noisy_transforms=noisy_transforms
    )

    args.train_data_len = len(train)

    dev = JsonlDataset(
        os.path.join(args.data_path, args.task, "dev.jsonl"),
        tokenizer,
        transforms,
        vocab,
        args,
    )

    collate = functools.partial(collate_fn, args=args)

    if args.model == "vilt" or args.model == "flava":
        train_loader = DataLoader(
            train,
            batch_size=args.batch_sz,
            shuffle=True,
            num_workers=args.n_workers
        )
    else:
        train_loader = DataLoader(
            train,
            batch_size=args.batch_sz,
            shuffle=True,
            num_workers=args.n_workers,
            collate_fn=collate,
        )

    if args.model == "vilt" or args.model == "flava":
        val_loader = DataLoader(
            dev,
            batch_size=args.batch_sz,
            shuffle=False,
            num_workers=args.n_workers
        )
    else:
        val_loader = DataLoader(
            dev,
            batch_size=args.batch_sz,
            shuffle=False,
            num_workers=args.n_workers, 
            collate_fn = collate
        )

    test_set = JsonlDataset(
        os.path.join(args.data_path, args.task, "test.jsonl"),
        tokenizer,
        transforms,
        vocab,
        args,
    )
    if args.model == "vilt" or args.model == "flava":
        test_loader = DataLoader(
            test_set,
            batch_size=args.batch_sz,
            shuffle=False,
            num_workers=args.n_workers
        )
    else:
        test_loader = DataLoader(
            test_set,
            batch_size=args.batch_sz,
            shuffle=False,
            num_workers=args.n_workers,
            collate_fn=collate,
        )
    return train_loader, val_loader, test_loader
