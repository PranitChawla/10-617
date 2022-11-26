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

class JsonlDataset(Dataset):
    def __init__(self, data_path, tokenizer, transforms, vocab, args):
      path = data_path.split('.jsonl')[0]+'_filtered'+'.jsonl'
      data_path = '.'.join([data_path.split('.jsonl')[0]+'_filtered','jsonl'])
      self.data = [json.loads(l) for l in open(data_path)][:50]
      
    #   if 'test' in data_path:
    #       data_path = '../food101/test_filtered_del.jsonl'
    #       self.data = [json.loads(l) for l in open(data_path)][:500]

      print(len(self.data))
      self.data_dir = os.path.dirname(data_path)
      self.tokenizer = tokenizer
      self.args = args
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
    #   COLOR_JITTER_PARAMS = {
    #       "brightness_factor": 1.2,
    #       "contrast_factor": 1.2,
    #       "saturation_factor": 1.4,
    #   }
    #   params = {"level":0.7}


      # self.AUGMENTATIONS = imaugs.ColorJitter(**COLOR_JITTER_PARAMS)
    #   self.AUGMENTATIONS = imaugs.Opacity(**params)

    def __len__(self):
        return len(self.data)

    def __getitem__(self, index):
        if self.args.task == "vsnli":
            sent1 = self.tokenizer(self.data[index]["sentence1"])
            sent2 = self.tokenizer(self.data[index]["sentence2"])
            truncate_seq_pair(sent1, sent2, self.args.max_seq_len - 3)
            sentence = self.text_start_token + sent1 + ["[SEP]"] + sent2 + ["[SEP]"]
            segment = torch.cat(
                [torch.zeros(2 + len(sent1)), torch.ones(len(sent2) + 1)]
            )
        else:
            sentence = (
                self.text_start_token
                + self.tokenizer(self.data[index]["text"])[
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
                new_path = 'image_filtered/' + ('/').join(self.data[index]["img"].split('/')[1:])
                # print(new_path)
                image = Image.open(
                    os.path.join(self.data_dir, new_path)
                ).convert("RGB")
                # print("reading image")
                # image = self.AUGMENTATIONS(image)
            else:
                image = Image.fromarray(128 * np.ones((256, 256, 3), dtype=np.uint8))
            image = self.transforms(image)

        if self.args.model == "mmbt":
            # The first SEP is part of Image Token.
            segment = segment[1:]
            sentence = sentence[1:]
            # The first segment (0) is of images.
            segment += 1

        return sentence, segment, image, label



class text_attackDataset(Dataset):
    def __init__(self, data_path, tokenizer, transforms, vocab, args):
      path = data_path.split('.jsonl')[0]+'_filtered'+'.jsonl'
      data_path = '.'.join([data_path.split('.jsonl')[0]+'_filtered','jsonl'])
      self.data = [json.loads(l) for l in open(data_path)][:50]
      
    #   if 'test' in data_path:
    #       data_path = '../food101/test_filtered_del.jsonl'
    #       self.data = [json.loads(l) for l in open(data_path)][:500]

      print(len(self.data))
      self.data_dir = os.path.dirname(data_path)
      self.tokenizer = tokenizer
      self.args = args
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
    #   COLOR_JITTER_PARAMS = {
    #       "brightness_factor": 1.2,
    #       "contrast_factor": 1.2,
    #       "saturation_factor": 1.4,
    #   }
    #   params = {"level":0.7}


      # self.AUGMENTATIONS = imaugs.ColorJitter(**COLOR_JITTER_PARAMS)
    #   self.AUGMENTATIONS = imaugs.Opacity(**params)

    def __len__(self):
        return len(self.data)

    def __getitem__(self, index):
        if self.args.task == "vsnli":
            sent1 = self.tokenizer(self.data[index]["sentence1"])
            sent2 = self.tokenizer(self.data[index]["sentence2"])
            truncate_seq_pair(sent1, sent2, self.args.max_seq_len - 3)
            sentence = self.text_start_token + sent1 + ["[SEP]"] + sent2 + ["[SEP]"]
            segment = torch.cat(
                [torch.zeros(2 + len(sent1)), torch.ones(len(sent2) + 1)]
            )
        else:
            sentence = (
                self.text_start_token
                + self.tokenizer(self.data[index]["text"])[
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

        sentence2 = (' ').join(self.tokenizer(self.data[index]["text"])[: (self.args.max_seq_len - 1)])
        # sentence2 = (' ').join(self.tokenizer(self.data[index]["text"])[: 256])

        # sentence2 = 
        # print(sentence2)
                    
                

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
                # print(new_path)
                image = Image.open(
                    os.path.join(self.data_dir, new_path)
                ).convert("RGB")
                # print("reading image")
                # image = self.AUGMENTATIONS(image)
            else:
                image = Image.fromarray(128 * np.ones((256, 256, 3), dtype=np.uint8))
            image = self.transforms(image)

        if self.args.model == "mmbt":
            # The first SEP is part of Image Token.
            segment = segment[1:]
            sentence = sentence[1:]
            # The first segment (0) is of images.
            segment += 1
        # print(sentence2)
        return (new_path, sentence2) , label.item()