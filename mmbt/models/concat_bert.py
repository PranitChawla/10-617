#!/usr/bin/env python3
#
# Copyright (c) Facebook, Inc. and its affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.
#

import torch
import torch.nn as nn
import os
from PIL import Image

from models.bert import BertEncoder
from models.image import ImageEncoder
from pytorch_pretrained_bert import BertTokenizer
from data.vocab import Vocab
from data.helpers import collate_fn

from data.helpers import get_transforms
device = 'cuda' if torch.cuda.is_available() else 'cpu'
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
class MultimodalConcatBertClf(nn.Module):
    def __init__(self, args):
        super(MultimodalConcatBertClf, self).__init__()
        self.args = args
        self.model = BertEncoder(args)
        # self.txtenc = BertEncoder(args)
        # self.model = self.txtenc
        self.imgenc = ImageEncoder(args)
        self.tokenizer = BertTokenizer.from_pretrained(args.bert_model, do_lower_case=True).tokenize
        last_size = args.hidden_sz + (args.img_hidden_sz * args.num_image_embeds)
        self.clf = nn.ModuleList()
        for hidden in args.hidden:
            self.clf.append(nn.Linear(last_size, hidden))
            if args.include_bn:
                self.clf.append(nn.BatchNorm1d(hidden))
            self.clf.append(nn.ReLU())
            self.clf.append(nn.Dropout(args.dropout))
            last_size = hidden

        self.clf.append(nn.Linear(last_size, args.n_classes))
        self.vocab = get_vocab(args)
        self.text_start_token = ["[CLS]"] if args.model != "mmbt" else ["[SEP]"]
        self.transforms = get_transforms(args)
        self.data_dir = "/Users/pranitchawla/Desktop/mmbt/food101"

    def custom_collate_fn(self,batch):
        lens = [row.shape[0] for row in batch]
        bsz, max_seq_len = len(batch), max(lens)
        max_seq_len = 512
        mask_tensor = torch.zeros(bsz, max_seq_len).long()
        text_tensor = torch.zeros(bsz, max_seq_len).long()
        segment_tensor = torch.zeros(bsz, max_seq_len).long()

        # img_tensor = None
        # if args.model in ["img", "concatbow", "concatbert", "mmbt"]:
        #     img_tensor = torch.stack([row[2] for row in batch])

        # if args.task_type == "multilabel":
        #     # Multilabel case
        #     tgt_tensor = torch.stack([row[3] for row in batch])
        # else:
        #     # Single Label case
        #     tgt_tensor = torch.cat([row[3] for row in batch]).long()

        for i_batch, (input_row, length) in enumerate(zip(batch, lens)):
            tokens = input_row
            text_tensor[i_batch, :length] = tokens
            # segment_tensor[i_batch, :length] = segment
            mask_tensor[i_batch, :length] = 1

        return text_tensor, mask_tensor, segment_tensor



    def convert_text(self, text):
        sentence = (self.text_start_token+ self.tokenizer(text)[: (self.args.max_seq_len - 1)])
        # segment = torch.zeros(len(sentence))

        sentence = torch.LongTensor(
            [
                self.vocab.stoi[w] if w in self.vocab.stoi else self.vocab.stoi["[UNK]"]
                for w in sentence
            ]
        )
        return sentence
    
    def forward(self, txt, mask = None, segment = None, img = None):
        import ipdb
        print(txt)
        # ipdb.set_trace()
        if type(txt) != type(torch.ones(1)):
            txt_list = []
            img = []
            # segment = []
            for input in txt:
                txt_list.append(self.convert_text(input[1]))
                # img.append(input[1])
                new_path = input[0]
                # print(new_path)
                # print()
                image = Image.open(
                    os.path.join(self.data_dir, new_path)
                ).convert("RGB")
                image = self.transforms(image)
                img.append(image)
                # segment.append(input[1])
            #ipdb.set_trace()
            #txt_list.append(torch.LongTensor([1,2,3,4,5,6]))

            #txt = torch.Tensor(txt).long()
            txt, mask, segment = self.custom_collate_fn(txt_list)
            #txt, mask = self.text_to_ids(txt).to(device)
            # ipdb.set_trace()
            txt = txt.to(device).long()
            mask = mask.to(device).long()
            segment = segment.to(device).long()
            img = torch.stack(tuple(img),dim=0).to(device)
            # print(img.shape)
            
            # segment = torch.stack(segment,dim=0).to(device).long()
        # txt = self.txtenc(txt, mask, segment)
        txt = self.model(txt, mask, segment) 

        img = self.imgenc(img)
        img = torch.flatten(img, start_dim=1)
        out = torch.cat([txt, img], -1)
        for layer in self.clf:
            out = layer(out)
        return out.detach()
        # if self.regime == "attack"
        #     return out
