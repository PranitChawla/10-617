#!/usr/bin/env python3
#
# Copyright (c) Facebook, Inc. and its affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.
#


import argparse
import sklearn
from sklearn.metrics import f1_score, accuracy_score
from tqdm import tqdm

import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from pytorch_pretrained_bert import BertAdam

from data.helpers import get_data_loaders
from models import get_model
from utils.logger import create_logger
from utils.utils import *


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
    parser.add_argument("--name", type=str, default="concatbert_test_syn_2")
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
    

def device_as(t1, t2):
    """
    Moves t1 to the device of t2
    """
    return t1.to(t2.device)

class ContrastiveLoss(nn.Module):
    """
    Vanilla Contrastive loss, also called InfoNceLoss as in SimCLR paper
    """
    def __init__(self, batch_size, temperature=0.5):
        super().__init__()
#         self.batch_size = batch_size
        self.temperature = temperature
#         self.mask = (~torch.eye(batch_size * 2, batch_size * 2, dtype=bool)).float()

    def calc_similarity_batch(self, a, b):
        representations = torch.cat([a, b], dim=0)
        return F.cosine_similarity(representations.unsqueeze(1), representations.unsqueeze(0), dim=2)

    def forward(self, proj_1, proj_2):
        """
        proj_1 and proj_2 are batched embeddings [batch, embedding_dim]
        where corresponding indices are pairs
        z_i, z_j in the SimCLR paper
        """
        batch_size = proj_1.shape[0]
        z_i = F.normalize(proj_1, p=2, dim=1)
        z_j = F.normalize(proj_2, p=2, dim=1)

        similarity_matrix = self.calc_similarity_batch(z_i, z_j)

        sim_ij = torch.diag(similarity_matrix, batch_size)
        sim_ji = torch.diag(similarity_matrix, -batch_size)

        positives = torch.cat([sim_ij, sim_ji], dim=0)

        nominator = torch.exp(positives / self.temperature)

        mask = (~torch.eye(batch_size * 2, batch_size * 2, dtype=bool)).float()
        denominator = device_as(mask, similarity_matrix) * torch.exp(similarity_matrix / self.temperature)

        all_losses = -torch.log(nominator / torch.sum(denominator, dim=1))
        loss = torch.sum(all_losses) / (2 * batch_size)
        return loss    

    
# class ContrastiveLoss(nn.Module):
#     def __init__(self, batch_size, temperature, verbose):
#         super().__init__()
#         self.batch_size = batch_size
#         self.register_buffer("temperature", torch.tensor(temperature))
#         self.verbose = verbose
            
#     def forward(self, emb_i, emb_j):
#         """
#         emb_i and emb_j are batches of embeddings, where corresponding indices are pairs
#         z_i, z_j as per SimCLR paper
#         """
#         z_i = F.normalize(emb_i, dim=1)
#         z_j = F.normalize(emb_j, dim=1)

#         representations = torch.cat([z_i, z_j], dim=0)
#         similarity_matrix = F.cosine_similarity(representations.unsqueeze(1), representations.unsqueeze(0), dim=2)
# #         if self.verbose: print("Similarity matrix\n", similarity_matrix, "\n")
            
#         def l_ij(i, j):
#             z_i_, z_j_ = representations[i], representations[j]
#             sim_i_j = similarity_matrix[i, j]
# #             if self.verbose: print(f"sim({i}, {j})={sim_i_j}")
                
#             numerator = torch.exp(sim_i_j / self.temperature)
#             one_for_not_i = torch.ones((2 * self.batch_size, )).to(emb_i.device).scatter_(0, torch.tensor([i]).to(emb_i.device), 0.0)
# #             if self.verbose: print(f"1{{k!={i}}}",one_for_not_i)
            
#             denominator = torch.sum(
#                 one_for_not_i * torch.exp(similarity_matrix[i, :] / self.temperature)
#             )    
# #             if self.verbose: print("Denominator", denominator)
                
#             loss_ij = -torch.log(numerator / denominator)
# #             if self.verbose: print(f"loss({i},{j})={loss_ij}\n")
                
#             return loss_ij.squeeze(0)

#         N = self.batch_size
#         loss = 0.0
#         for k in range(0, N):
#             loss += l_ij(k, k + N) + l_ij(k + N, k)
#         return 1.0 / (2*N) * loss
    
    
class TotalLoss(nn.Module):
    def __init__(self, batch_size, temperature=0.5):
        super().__init__()
        self.batch_size = batch_size
        self.CL = ContrastiveLoss(batch_size, temperature=temperature)
        self.CE = nn.CrossEntropyLoss()
    
    def forward(self, emb, emb_aug, tgt):
        return self.CL(emb, emb_aug) + self.CE(emb, tgt) + self.CE(emb_aug, tgt)
    
def get_criterion(args, train_mode=False):
    if args.task_type == "multilabel":
        if args.weight_classes:
            freqs = [args.label_freqs[l] for l in args.labels]
            label_weights = (torch.FloatTensor(freqs) / args.train_data_len) ** -1
            criterion = nn.BCEWithLogitsLoss(pos_weight=label_weights.cuda())
        else:
            criterion = nn.BCEWithLogitsLoss()
    else:
        if train_mode and args.training_improvement == "contrast":
            criterion = TotalLoss(args.batch_sz)
        else:
            criterion = nn.CrossEntropyLoss()

    return criterion


def get_optimizer(model, args):
    optimizer = optim.Adam(model.parameters(), lr=args.lr)
    return optimizer


def get_scheduler(optimizer, args):
    return optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, "max", patience=args.lr_patience, verbose=True, factor=args.lr_factor
    )


def model_eval(i_epoch, data, model, args, criterion, store_preds=False):
    with torch.no_grad():
        losses, preds, tgts = [], [], []
        for batch in data:
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
        metrics["acc"] = accuracy_score(tgts, preds)

    if store_preds:
        store_preds_to_disk(tgts, preds, args)

    return metrics


def model_forward(i_epoch, model, args, criterion, batch, train_mode=False):
    if args.model in ["vilt","flava"]:
        if train_mode and args.training_improvement == "contrast":
            inputs, inputs_aug, tgt = batch
        else:
            inputs, tgt = batch
        for key in list(inputs.keys()):
            inputs[key] = inputs[key].squeeze().cuda()
            if train_mode and args.training_improvement == "contrast":
                inputs_aug[key] = inputs_aug[key].squeeze().cuda()
        tgt = tgt.squeeze()
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
        if train_mode and args.training_improvement == "contrast":
            out_aug = model(inputs_aug)
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
    if train_mode and args.training_improvement == "contrast":
        loss = criterion(out, out_aug, tgt)
    else:
        loss = criterion(out, tgt)
    return loss, out, tgt


def train(args):

    set_seed(args.seed)
    args.savedir = os.path.join(args.savedir, args.name)
    os.makedirs(args.savedir, exist_ok=True)

    train_loader, val_loader, test_loaders = get_data_loaders(args)

    model = get_model(args)
    criterion = get_criterion(args, True)
    criterion_eval = get_criterion(args, False)
    optimizer = get_optimizer(model, args)
    scheduler = get_scheduler(optimizer, args)

    logger = create_logger("%s/logfile.log" % args.savedir, args)
    logger.info(model)
#     model = nn.DataParallel(model)
    model.cuda()

    torch.save(args, os.path.join(args.savedir, "args.pt"))

    start_epoch, global_step, n_no_improve, best_metric = 0, 0, 0, -np.inf

    if os.path.exists(os.path.join(args.savedir, "checkpoint.pt")):
        checkpoint = torch.load(os.path.join(args.savedir, "checkpoint.pt"))
        start_epoch = checkpoint["epoch"]
        n_no_improve = checkpoint["n_no_improve"]
        best_metric = checkpoint["best_metric"]
        model.load_state_dict(checkpoint["state_dict"])
        optimizer.load_state_dict(checkpoint["optimizer"])
        scheduler.load_state_dict(checkpoint["scheduler"])

    logger.info("Training..")
    for i_epoch in range(start_epoch, args.max_epochs):
        train_losses = []
        model.train()
        optimizer.zero_grad()

        for batch in tqdm(train_loader, total=len(train_loader)):
            loss, _, _ = model_forward(i_epoch, model, args, criterion, batch, True)
            if args.gradient_accumulation_steps > 1:
                loss = loss / args.gradient_accumulation_steps

            train_losses.append(loss.item())
            loss.backward()
            global_step += 1
            if global_step % args.gradient_accumulation_steps == 0:
                optimizer.step()
                optimizer.zero_grad()

        model.eval()
        metrics = model_eval(i_epoch, val_loader, model, args, criterion_eval)
        logger.info("Train Loss: {:.4f}".format(np.mean(train_losses)))
        log_metrics("Val", metrics, args, logger)

        tuning_metric = (
            metrics["micro_f1"] if args.task_type == "multilabel" else metrics["acc"]
        )
        scheduler.step(tuning_metric)
        is_improvement = tuning_metric > best_metric
        if is_improvement:
            best_metric = tuning_metric
            n_no_improve = 0
            save_checkpoint(
            {
                "epoch": i_epoch + 1,
                "state_dict": model.state_dict(),
                "optimizer": optimizer.state_dict(),
                "scheduler": scheduler.state_dict(),
                "n_no_improve": n_no_improve,
                "best_metric": best_metric,
            },
            is_improvement,
            args.savedir,
        )
        else:
            n_no_improve += 1



        if n_no_improve >= args.patience:
            logger.info("No improvement. Breaking out of loop.")
            break

    load_checkpoint(model, os.path.join(args.savedir, "model_best.pt"))
    model.eval()
    for test_name, test_loader in test_loaders.items():
        test_metrics = model_eval(
            np.inf, test_loader, model, args, criterion_eval, store_preds=True
        )
        log_metrics(f"Test - {test_name}", test_metrics, args, logger)


def cli_main():
    parser = argparse.ArgumentParser(description="Train Models")
    get_args(parser)
    args, remaining_args = parser.parse_known_args()
    assert remaining_args == [], remaining_args
    train(args)


if __name__ == "__main__":
    import warnings

    warnings.filterwarnings("ignore")

    cli_main()
