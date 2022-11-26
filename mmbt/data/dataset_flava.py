#!/usr/bin/env python3
#
# Copyright (c) Facebook, Inc. and its affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.
#

import json
import numpy as np
import os
from PIL import Image

import torch
from torch.utils.data import Dataset

from utils.utils import truncate_seq_pair, numpy_seed
# import augly.image as imaugs
from transformers import ViltProcessor
import torchvision.transforms as transforms
from transformers import FlavaProcessor, FlavaModel
import ipdb

class JsonlDataset(Dataset):
    def __init__(self, data_path, vocab, args):
      path = data_path.split('.jsonl')[0]+'_filtered'+'.jsonl'
      data_path = '.'.join([data_path.split('.jsonl')[0]+'_filtered','jsonl'])
      self.data = [json.loads(l) for l in open(data_path)]
      
      # if 'test' in data_path:
      #     data_path = '../food101/test_filtered_del.jsonl'
      #     self.data = [json.loads(l) for l in open(data_path)][:500]

      # print(len(self.data))
      self.data_dir = os.path.dirname(data_path)
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
      self.processor = FlavaProcessor.from_pretrained("facebook/flava-full")
      self.transforms = transforms.Compose(
        [
            transforms.Resize((256,256)),
            # transforms.CenterCrop(224),
            transforms.ToTensor()
            # transforms.Normalize(
            #     mean=[0.46777044, 0.44531429, 0.40661017],
            #     std=[0.12221994, 0.12145835, 0.14380469],
            # ),
        ]
    )

    def __len__(self):
        return len(self.data)

    def __getitem__(self, index):
      if self.data[index]["img"]:
        new_path = 'images/' + ('/').join(self.data[index]["img"].split('/')[1:])
        # image = Image.fromarray(128 * np.ones((256, 256, 3), dtype=np.uint8))
        image = Image.open(os.path.join(self.data_dir, new_path)).convert("RGB")
        # ipdb.set_trace()
        image = self.transforms(image)
        # print(image.shape)

      else:
        image = Image.fromarray(128 * np.ones((256, 256, 3), dtype=np.uint8))
      text = self.data[index]["text"]
      inputs = self.processor(image, text, return_tensors="pt", padding = "max_length", truncation = True, max_length = self.args.max_seq_len)
    #   print(inputs['input_ids'].shape)
      # ipdb.set_trace()
      label = torch.LongTensor([self.args.labels.index(self.data[index]["label"])])
      #inputs["label"] = label
      return inputs, label