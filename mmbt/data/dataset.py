#!/usr/bin/env python3
#
# Copyright (c) Facebook, Inc. and its affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.
#

import json
from xxlimited import new
import numpy as np
import os
from PIL import Image

import torch
from torch.utils.data import Dataset
from utils.utils import truncate_seq_pair, numpy_seed
# import augly.image as imaugs
from transformers import ViltProcessor
import torchvision.transforms as torch_transforms
import re
from pytorch_pretrained_bert import BertTokenizer
from data.vocab import Vocab
from nltk.tokenize import word_tokenize
from nltk.corpus import wordnet
from nltk.corpus import stopwords
import random


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


stop_words = set(stopwords.words('english'))


def get_syn_flipped(s, p=0.3):
    word_tokens = word_tokenize(s)
    x = []
    for word in word_tokens:
        if not word.isalpha():
            x.append(word)
        elif word.lower() in stop_words:
            x.append(word)
        else:
            u = random.random()
            if u <= p:
                syns = list(set([i.name() for syn in wordnet.synsets(word) for i in syn.lemmas() ]))
                if syns:
                    x.append(random.choice(syns))
                else:
                    x.append(word)
            else:
                x.append(word)
    return " ".join(x)

class JsonlDataset(Dataset):
    def __init__(self, data_path, tokenizer, transforms, vocab, args, noisy_transforms=None):
        path = data_path.split('.jsonl')[0]+'_filtered'+'.jsonl'
        data_path = '.'.join([data_path.split('.jsonl')[0]+'_filtered','jsonl'])
        self.data = [json.loads(l) for l in open(data_path)]

        self.train_mode = False
        self.train_mode_improvement = None
        if 'train' in data_path:
            self.train_mode = True
            if args.training_improvement in ["augment", "contrast"]:
                self.train_mode_improvement = args.training_improvement
        self.syn_flip_prob = args.text_syn_probability

        #   if 'test' in data_path:
        #       data_path = '../food101/test_filtered_del.jsonl'
        #       self.data = [json.loads(l) for l in open(data_path)][:500]

        print("Loading ", data_path, len(self.data))
        self.data_dir = os.path.dirname(data_path)
        self.tokenizer = tokenizer
        self.args = args
        if self.args.model == "vilt":
            self.args.max_seq_len = 40
        self.vocab = vocab
        self.n_classes = len(args.labels)
        self.text_start_token = ["[CLS]"] if args.model != "mmbt" else ["[SEP]"]

        with numpy_seed(0):
            for row in self.data:
                if np.random.random() < args.drop_img_percent:
                    row["img"] = None

        self.max_seq_len = args.max_seq_len
        if args.model == "mmbt":
            self.max_seq_len -= args.num_image_embeds

        self.transforms = transforms
        self.noisy_transforms = noisy_transforms
        if self.args.model == "vilt" or self.args.model == "flava":
            self.transforms = torch_transforms.Compose([torch_transforms.Resize((256,256)),torch_transforms.ToTensor()])
    
    def __len__(self):
        return len(self.data)

    def __getitem__(self, index):
        if self.args.model in ["vilt","flava"]:
            new_path = 'images/' + ('/').join(self.data[index]["img"].split('/')[1:])
            orig_image = Image.open(os.path.join(self.data_dir, new_path)).convert("RGB")
            text = self.data[index]["text"]
            
            if self.train_mode:
                # if == contrast {1. get aug 2. transform both}
                # if == aug {1. flip coin and change image 2. change text 3. transform both}
                # else {1. just transform both}
                if self.train_mode_improvement == "contrast":
                    image = self.transforms(orig_image)
                    inputs = self.tokenizer(image, text, return_tensors="pt", padding = "max_length", truncation = True, max_length = self.args.max_seq_len)
                    image_aug = random.choice(self.noisy_transforms)(orig_image)
                    text_aug = get_syn_flipped(text, self.syn_flip_prob)
                    inputs_aug = self.tokenizer(image_aug, text_aug, return_tensors="pt", padding = "max_length", truncation = True, max_length = self.args.max_seq_len)
            
                elif self.train_mode_improvement == "augment":
                    if torch.rand(1) < self.args.image_noise_probability:
                        image = random.choice(self.noisy_transforms)(orig_image)
                    else:
                        image = self.transforms(orig_image)
                    text = get_syn_flipped(text, self.syn_flip_prob)
                    inputs = self.tokenizer(image, text, return_tensors="pt", padding = "max_length", truncation = True, max_length = self.args.max_seq_len)
                    
                else:
                    image = self.transforms(orig_image)
                    inputs = self.tokenizer(image, text, return_tensors="pt", padding = "max_length", truncation = True, max_length = self.args.max_seq_len)
                    
            else:
                image = self.transforms(orig_image)
                inputs = self.tokenizer(image, text, return_tensors="pt", padding = "max_length", truncation = True, max_length = self.args.max_seq_len)
            
#             image = self.transforms(orig_image)
            
#             inputs = self.tokenizer(image, text, return_tensors="pt", padding = "max_length", truncation = True, max_length = self.args.max_seq_len)
            
#             if self.train_mode:
#                 image_aug = random.choice(self.noisy_transforms)(orig_image)
#                 text_aug = get_syn_flipped(text, self.syn_flip_prob)
#                 inputs_aug = self.tokenizer(image_aug, text_aug, return_tensors="pt", padding = "max_length", truncation = True, max_length = self.args.max_seq_len)
            
            if self.args.task_type == "multilabel":
                label = torch.zeros(self.n_classes)
                label[
                    [self.args.labels.index(tgt) for tgt in self.data[index]["label"]]
                ] = 1
            else:
                label = torch.LongTensor(
                    [self.args.labels.index(self.data[index]["label"])]
                )
            
            if self.train_mode and self.train_mode_improvement == "contrast":
                return inputs, inputs_aug, label
            else:
                return inputs, label

        if self.args.task == "vsnli":
            sent1 = self.data[index]["sentence1"]
            if self.train_mode:
                sent1 = get_syn_flipped(sent1, self.syn_flip_prob)
            sent1 = self.tokenizer(sent1)
            sent2 = self.data[index]["sentence2"]
            if self.train_mode:
                sent2 = get_syn_flipped(sent2, self.syn_flip_prob)
            sent2 = self.tokenizer(sent2)
            truncate_seq_pair(sent1, sent2, self.args.max_seq_len - 3)
            sentence = self.text_start_token + sent1 + ["[SEP]"] + sent2 + ["[SEP]"]
            segment = torch.cat(
                [torch.zeros(2 + len(sent1)), torch.ones(len(sent2) + 1)]
            )
        else:
            temp = self.data[index]["text"]
            if self.train_mode:
                temp = get_syn_flipped(temp, self.syn_flip_prob)
            sentence = (
                self.text_start_token
                + self.tokenizer(temp)[
                    : (self.args.max_seq_len - 1)
                ]
            )
            segment = torch.zeros(len(sentence))

        sentence = torch.LongTensor(
            [
                self.vocab.stoi[w] if w in self.vocab.stoi else self.vocab.stoi["[UNK]"]
                for w in sentence
            ]
        )

        if self.args.task_type == "multilabel":
            label = torch.zeros(self.n_classes)
            label[
                [self.args.labels.index(tgt) for tgt in self.data[index]["label"]]
            ] = 1
        else:
            label = torch.LongTensor(
                [self.args.labels.index(self.data[index]["label"])]
            )

        image = None
        if self.args.model in ["img", "concatbow", "concatbert", "mmbt"]:
            if self.data[index]["img"]:
                new_path = 'images/' + ('/').join(self.data[index]["img"].split('/')[1:])
                image = Image.open(
                    os.path.join(self.data_dir, new_path)
                ).convert("RGB")
            else:
                image = Image.fromarray(128 * np.ones((256, 256, 3), dtype=np.uint8))

            if torch.rand(1) < self.args.image_noise_probability and self.train_mode:
                image = random.choice(self.noisy_transforms)(image)
            else:
                image = self.transforms(image)

        if self.args.model == "mmbt":
            # The first SEP is part of Image Token.
            segment = segment[1:]
            sentence = sentence[1:]
            # The first segment (0) is of images.
            segment += 1
        return sentence, segment, image, label



class TextAttackDataset(Dataset):
    def __init__(self, data_path, tokenizer, args):
        path = data_path.split('.jsonl')[0]+'_filtered'+'.jsonl'
        data_path = '.'.join([data_path.split('.jsonl')[0]+'_filtered','jsonl'])
        self.data = [json.loads(l) for l in open(data_path)][:args.attack_size]
        self.data_dir = os.path.dirname(data_path)
        self.tokenizer = tokenizer
        self.args = args
        if self.args.model == "vilt":
            self.max_seq_len = 40
        else:
            self.max_seq_len = args.max_seq_len
    def __len__(self):
        return len(self.data)

    def __getitem__(self, index):
        sentence = (' ').join(self.data[index]["text"].split(' ')[: (self.max_seq_len - 1)])
        new_path = 'images/' + ('/').join(self.data[index]["img"].split('/')[1:])
        label = torch.LongTensor([self.args.labels.index(self.data[index]["label"])])
        return (new_path, sentence) , label.item()



class ImageAttackDataset(Dataset):
    def __init__(self, data,  tokenizer, transforms, args):
        self.data = data[:args.attack_size]
        self.args = args
        if self.args.model == "vilt":
            self.max_seq_len = 40
        else:
            self.max_seq_len = args.max_seq_len
        self.transforms = transforms
        self.tokenizer = tokenizer
        self.data_dir = args.data_model_path
        self.text_start_token = ["[CLS]"]
        self.vocab = get_vocab(args)
        print(self.data.columns)
    def __len__(self):
        return len(self.data)

    def __getitem__(self, index):
        text = self.data.iloc[index]["perturbed_text"].split('<SPLIT>')[-1].split('[[[[Hypothesis]]]]:')[-1]
        text = re.sub(r"[\([{})\]]", "", text)
        img_path = self.data.iloc[index]["perturbed_text"].split('<SPLIT>')[0].split('[[[[Premise]]]]:')[-1][1:]
        # print(img_path)
        image = Image.open(os.path.join(self.data_dir, img_path)).convert("RGB")
        image = self.transforms(image)
        label = int(self.data.iloc[index]['ground_truth_output'])
        label = torch.LongTensor([label])
        if self.args.model in ["vilt","flava"]:
            inputs = self.tokenizer(image, text, return_tensors="pt", padding = "max_length", truncation = True, max_length = self.args.max_seq_len)
            return inputs, label
        else:
            sentence = (self.text_start_token+ self.tokenizer(text)[: (self.args.max_seq_len - 1)])
            segment = torch.zeros(len(sentence))
            sentence = torch.LongTensor([self.vocab.stoi[w] if w in self.vocab.stoi else self.vocab.stoi["[UNK]"] for w in sentence])
            return sentence, segment, image, label
                    
        