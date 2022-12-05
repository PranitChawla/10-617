from pytorch_pretrained_bert import BertTokenizer

import sys, os

import ipdb
import argparse
import torchvision.transforms as transforms
from utils.utils import *
from models import get_model

from data.helpers import *
from data.dataset import ImageAttackDataset
from data.vocab import Vocab
from torch.utils.data import DataLoader
import pandas as pd
import re
import torch.nn.functional as F
from transformers import ViltProcessor, FlavaProcessor
import copy
from tqdm import tqdm

def get_args(parser):
    parser.add_argument("--batch_sz", type=int, default=1)
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
    parser.add_argument("--regime", type=str, default="test", choices = ["attack", "train", "test"])
    parser.add_argument("--attack_size", type=int, default=500)
    parser.add_argument('--norm', type=str, default='Linf')
    parser.add_argument('--epsilon', type=float, default=8./255.)
    parser.add_argument('--version', type=str, default="custom",choices = ['standard', 'plus', 'rand', 'custom'])
    parser.add_argument('--qbudget', type=int, default=300)
    parser.add_argument('--text_attack_path', type=str, default="attacked_texts")



def get_probs(model, inputs, y):
    inputs_cuda = copy.deepcopy(inputs)
    txt, segment, mask, img = inputs_cuda
    txt, segment, mask, img = txt.cuda(), segment.cuda(), mask.cuda(), img.cuda()
    with torch.no_grad():
        output = model(txt, mask, segment, img).cpu()
    
    probs = torch.index_select(F.softmax(output, dim=-1).data, 1, y)
    best_idx = torch.argmax(output)
    top2 = torch.topk(F.softmax(output, dim=-1).data, 2)
    return torch.diag(probs), best_idx.item(), top2.values





def simba_single(model, x, y, num_iters=1500, epsilon=0.2, targeted=False):
    n_dims = x[3].view(1, -1).size(1)
    perm = torch.randperm(n_dims)
    last_prob, best_idx, top2 = get_probs(model, x, y)
    start_first = top2[0][0]
    start_second = top2[0][1]
    total_iters = 10*num_iters
    pbar = tqdm(range(num_iters))
    for i in pbar:
        diff = torch.zeros(n_dims)
        diff[perm[i]] = epsilon
        x_copy_left = copy.deepcopy(x)
        x_copy_right = copy.deepcopy(x)
        x_copy_left = list(x_copy_left)
        x_copy_right = list(x_copy_right)
        x_copy_left[3] = x_copy_left[3] - diff.view(x_copy_left[3].size()).clamp(0,1)
        x_copy_right[3] = x_copy_right[3] + diff.view(x_copy_right[3].size()).clamp(0,1)
        x_copy_left = tuple(x_copy_left)
        x_copy_right = tuple(x_copy_right)
        left_prob, best_idx, top2 = get_probs(model, x_copy_left, y)
        if targeted != (left_prob < last_prob):
            x = list(x)
            x_copy_left = list(x_copy_left)
            x[3] = x_copy_left[3]
            x = tuple(x)
            x_copy_left = tuple(x_copy_left)
            last_prob = left_prob
        else:
            right_prob, best_idx, top2 = get_probs(model, x_copy_right, y)
            if targeted != (right_prob < last_prob):
                x = list(x)
                x_copy_right = list(x_copy_right)
                x[3] = x_copy_right[3]
                x = tuple(x)
                x_copy_right = tuple(x_copy_right)
                last_prob = right_prob
        
        if best_idx != y.item():
            print("Broken exiting")
            return 0
        
        pbar.set_postfix({'last_prob': last_prob.item()})
    return 1



def attack_image(args):
    text_file_path = os.path.join(args.text_attack_path,args.name+'_'+str(args.qbudget)+'.csv')
    args.savedir = os.path.join(args.savedir,args.name)
    transforms = get_transforms(args)
    data = pd.read_csv(text_file_path)
    data = data[data["result_type"]=="Failed"]
    args.labels, args.label_freqs = get_labels_and_frequencies(os.path.join(args.data_path, args.task, "train.jsonl"))
    args.n_classes = len(args.labels)
    if args.model in ["img", "bert", "mmbt", "concatbert"]:
        tokenizer = BertTokenizer.from_pretrained(args.bert_model, do_lower_case=True).tokenize
    elif args.model == "vilt":
        tokenizer = ViltProcessor.from_pretrained("dandelin/vilt-b32-mlm")
    elif args.model == "flava":
        tokenizer = FlavaProcessor.from_pretrained("facebook/flava-full")
    test_set = ImageAttackDataset(data, tokenizer, transforms, args)  
    collate = functools.partial(collate_fn, args=args)
    test_loader = DataLoader(
        test_set,
        batch_size=args.batch_sz,
        shuffle=False,
        num_workers=args.n_workers,
        collate_fn=collate,
    )
    model = get_model(args).cuda()
    model.eval()
    load_checkpoint(model, os.path.join(args.savedir, "model_best.pt"))
    num_left = 0
    for batch in test_loader:
        fts = batch[:4]
        label = batch[-1]
        num_left += simba_single(model,fts,label)
    print("Final roboust accuracy: ",num_left/args.attack_size)






def cli_main():
    parser = argparse.ArgumentParser(description="Attack Models")
    get_args(parser)
    args, remaining_args = parser.parse_known_args()
    assert remaining_args == [], remaining_args
    attack_image(args)


if __name__ == "__main__":
    import warnings
    warnings.filterwarnings("ignore")
    cli_main()