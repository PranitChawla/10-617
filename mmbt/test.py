#!/usr/bin/env python3
#
# Copyright (c) Facebook, Inc. and its affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.
#


import argparse
from sklearn.metrics import f1_score, accuracy_score
from tqdm import tqdm

import torch
import torch.nn as nn
import torch.optim as optim
from pytorch_pretrained_bert import BertAdam

from data.helpers import get_data_loaders
from models import get_model
from utils.logger import create_logger
from utils.utils import *

device = 'cuda' if torch.cuda.is_available() else 'cpu'
def get_args(parser):
    parser.add_argument("--batch_sz", type=int, default=4)
    parser.add_argument("--bert_model", type=str, default="bert-base-uncased", choices=["bert-base-uncased", "bert-large-uncased"])
    parser.add_argument("--data_path", type=str, default="/home/scratch/rsaxena2/")
    parser.add_argument("--data_model_path", type=str, default="/home/scratch/rsaxena2/food101/")
    parser.add_argument("--drop_img_percent", type=float, default=0.0)
    parser.add_argument("--dropout", type=float, default=0.1)
    parser.add_argument("--embed_sz", type=int, default=300)
    parser.add_argument("--freeze_img", type=int, default=0)
    parser.add_argument("--freeze_txt", type=int, default=0)
    parser.add_argument("--glove_path", type=str, default="/path/to/glove_embeds/glove.840B.300d.txt")
    parser.add_argument("--gradient_accumulation_steps", type=int, default=4)
    parser.add_argument("--hidden", nargs="*", type=int, default=[])
    parser.add_argument("--hidden_sz", type=int, default=768)
    parser.add_argument("--img_embed_pool_type", type=str, default="avg", choices=["max", "avg"])
    parser.add_argument("--img_hidden_sz", type=int, default=2048)
    parser.add_argument("--include_bn", type=int, default=True)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--lr_factor", type=float, default=0.5)
    parser.add_argument("--lr_patience", type=int, default=2)
    parser.add_argument("--max_epochs", type=int, default=100)
    parser.add_argument("--max_seq_len", type=int, default=512)
    parser.add_argument("--model", type=str, default="concatbert", choices=["bow", "img", "bert", "concatbow", "concatbert", "mmbt", "vilt", "flava"])
    parser.add_argument("--n_workers", type=int, default=8)
    parser.add_argument("--name", type=str, default="concat_bert_model")
    parser.add_argument("--num_image_embeds", type=int, default=1)
    parser.add_argument("--patience", type=int, default=10)
    parser.add_argument("--savedir", type=str, default="/home/scratch/rsaxena2/saved_models/")
    parser.add_argument("--seed", type=int, default=123)
    parser.add_argument("--task", type=str, default="food101", choices=["mmimdb", "vsnli", "food101"])
    parser.add_argument("--task_type", type=str, default="classification", choices=["multilabel", "classification"])
    parser.add_argument("--warmup", type=float, default=0.1)
    parser.add_argument("--weight_classes", type=int, default=1)
    parser.add_argument("--regime", type=str, default="train", choices = ["attack", "train", "test"])
    parser.add_argument("--training_improvement", type=str, default="none", choices=["none", "augment", "contrast"])
    parser.add_argument("--text_syn_probability", type=float, default=0.3)
    parser.add_argument("--image_noise_probability", type=float, default=0.2)


def get_criterion(args):
    if args.task_type == "multilabel":
        if args.weight_classes:
            freqs = [args.label_freqs[l] for l in args.labels]
            label_weights = (torch.FloatTensor(freqs) / args.train_data_len) ** -1
            criterion = nn.BCEWithLogitsLoss(pos_weight=label_weights.cuda())
        else:
            criterion = nn.BCEWithLogitsLoss()
    else:
        criterion = nn.CrossEntropyLoss()

    return criterion


def get_optimizer(model, args):
    if args.model in ["bert", "concatbert", "mmbt"]:
        total_steps = (
            args.train_data_len
            / args.batch_sz
            / args.gradient_accumulation_steps
            * args.max_epochs
        )
        param_optimizer = list(model.named_parameters())
        no_decay = ["bias", "LayerNorm.bias", "LayerNorm.weight"]
        optimizer_grouped_parameters = [
            {"params": [p for n, p in param_optimizer if not any(nd in n for nd in no_decay)], "weight_decay": 0.01},
            {"params": [p for n, p in param_optimizer if any(nd in n for nd in no_decay)], "weight_decay": 0.0,},
        ]
        optimizer = BertAdam(
            optimizer_grouped_parameters,
            lr=args.lr,
            warmup=args.warmup,
            t_total=total_steps,
        )
    else:
        optimizer = optim.Adam(model.parameters(), lr=args.lr)

    return optimizer


def get_scheduler(optimizer, args):
    return optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, "max", patience=args.lr_patience, verbose=True, factor=args.lr_factor
    )


def model_eval(i_epoch, data, model, args, criterion, store_preds=False):
    with torch.no_grad():
        losses, preds, tgts = [], [], []
        cnt = 0
        for batch in tqdm(data):
                # print(cnt,len(data))
            cnt += 1
            loss, out, tgt = model_forward(i_epoch, model, args, criterion, batch)
            losses.append(loss.item())

            if args.task_type == "multilabel":
                pred = torch.sigmoid(out).cpu().detach().numpy() > 0.5
            else:
                pred = torch.nn.functional.softmax(out, dim=1).argmax(dim=1).cpu().detach().numpy()

            preds.append(pred)
            tgt = tgt.cpu().detach().numpy()

            tgts.append(tgt)

    metrics = {"loss": np.mean(losses)}
    if args.task_type == "multilabel":
        tgts = np.vstack(tgts)
        preds = np.vstack(preds)
        metrics["macro_f1"] = f1_score(tgts, preds, average="macro")
        metrics["micro_f1"] = f1_score(tgts, preds, average="micro")
    else:
        tgts = [l for sl in tgts for l in sl]
        preds = [l for sl in preds for l in sl]
        print(preds)
        metrics["acc"] = accuracy_score(tgts, preds)

    if store_preds:
        store_preds_to_disk(tgts, preds, args)

    return metrics


def model_forward(i_epoch, model, args, criterion, batch):
    with torch.no_grad():
        if args.model in ["vilt","flava"]:
            inputs, tgt = batch
            for key in list(inputs.keys()):
                inputs[key] = inputs[key].squeeze(dim=1).cuda()
            tgt = tgt.squeeze(dim=1)
        else:
            txt, segment, mask, img, tgt = batch

        freeze_img = i_epoch < args.freeze_img
        freeze_txt = i_epoch < args.freeze_txt

        if args.model == "bow":
            txt = txt.cuda()
            out = model(txt)
        elif args.model == "img":
            img = img.cuda()
            out = model(img)
        elif args.model == "concatbow":
            txt, img = txt.cuda(), img.cuda()
            out = model(txt, img)
        elif args.model == "bert":
            txt, mask, segment = txt.cuda(), mask.cuda(), segment.cuda()
            out = model(txt, mask, segment)
        elif args.model == "concatbert":
            txt, img = txt.cuda(), img.cuda()
            mask, segment = mask.cuda(), segment.cuda()
            out = model(txt, mask, segment, img)
        elif args.model in ["vilt","flava"]:
            out = model(inputs)
        else:
            assert args.model == "mmbt"
            for param in model.enc.img_encoder.parameters():
                param.requires_grad = not freeze_img
            for param in model.enc.encoder.parameters():
                param.requires_grad = not freeze_txt

            txt, img = txt.cuda(), img.cuda()
            mask, segment = mask.cuda(), segment.cuda()
            out = model(txt, mask, segment, img)

        tgt = tgt.cuda()
        loss = criterion(out, tgt)
        return loss, out, tgt


def test(args):

    # set_seed(args.seed)
    # args.savedir = os.path.join(args.savedir, args.name)
    # os.makedirs(args.savedir, exist_ok=True)

    train_loader, val_loader, test_loaders = get_data_loaders(args)

    model = get_model(args)
    criterion = get_criterion(args)

    logger = create_logger("%s/logfile.log" % args.savedir, args)
    logger.info(model)
    model.to(device)
    args.savedir = os.path.join(args.savedir,args.name)
    load_checkpoint(model, os.path.join(args.savedir, "model_best.pt"))
    model.eval()
    test_metrics = model_eval(
        np.inf, test_loaders, model, args, criterion, store_preds=True
    )
    log_metrics("Test", test_metrics, args, logger)
    print(test_metrics)


def cli_main():
    parser = argparse.ArgumentParser(description="Test Models")
    get_args(parser)
    args, remaining_args = parser.parse_known_args()
    assert remaining_args == [], remaining_args
    test(args)


if __name__ == "__main__":
    import warnings

    warnings.filterwarnings("ignore")

    cli_main()
