from __future__ import absolute_import
import sys, os
# sys.path.append("textattack_lib/textattack/.")
# sys.path.append("customattacks/.")
from textattack.attacker import Attacker
import ipdb
import argparse
import torchvision.transforms as transforms
from textattack.attack_recipes import TextFoolerJin2019
import textattack

from textattack.constraints.pre_transformation import (
    InputColumnModification,
    RepeatModification,
    StopwordModification,
)
from textattack import AttackArgs
from textattack.datasets import HuggingFaceDataset, Dataset

# import tensorflow as tf
# os.environ["WANDB_DISABLED"] = "true"
from pytorch_pretrained_bert import BertTokenizer
from textattack.augmentation import EmbeddingAugmenter

from utils.utils import *
from models import get_model

from data.helpers import *
from data.dataset import TextAttackDataset
from data.vocab import Vocab

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
    parser.add_argument("--n_workers", type=int, default=0)
    parser.add_argument("--name", type=str, default="concat_bert_model")
    parser.add_argument("--num_image_embeds", type=int, default=1)
    parser.add_argument("--patience", type=int, default=10)
    parser.add_argument("--savedir", type=str, default="/home/scratch/rsaxena2/saved_models/")
    parser.add_argument("--seed", type=int, default=123)
    parser.add_argument("--task", type=str, default="food101", choices=["mmimdb", "vsnli", "food101"])
    parser.add_argument("--task_type", type=str, default="classification", choices=["multilabel", "classification"])
    parser.add_argument("--warmup", type=float, default=0.1)
    parser.add_argument("--weight_classes", type=int, default=1)
    parser.add_argument("--regime", type=str, default="attack", choices = ["attack", "train", "test"])
    parser.add_argument("--attack_size", type=int, default=500)



def attack (args):
    model = get_model(args)
    args.savedir = os.path.join(args.savedir,args.name)
    args.labels, args.label_freqs = get_labels_and_frequencies(os.path.join(args.data_path, args.task, "train.jsonl"))
    load_checkpoint(model, os.path.join(args.savedir, "model_best.pt"))
    tokenizer = BertTokenizer.from_pretrained(args.bert_model, do_lower_case=True).tokenize
    test_set = TextAttackDataset(
        os.path.join(args.data_path, args.task, "test.jsonl"),
        tokenizer,
        args)
    my_dataset = Dataset(test_set, input_columns = ["premise",  "hypothesis"])           
    attack_class = TextFoolerJin2019
    attack = attack_class.build(model)
    model.eval()
    attack_args = AttackArgs(num_examples=args.attack_size, 
                            log_to_csv="log_to_csv.csv", \
                            checkpoint_interval=100, 
                            checkpoint_dir=args.savedir, 
                            disable_stdout=True,
                            query_budget = 300,parallel=False)

    attacker = Attacker(attack, my_dataset, attack_args)
    attacker.attack.goal_function.batch_size = 2
    max_percent_words = 0.3
    for i,constraint in enumerate(attacker.attack.constraints):
        if type(constraint) == textattack.constraints.overlap.max_words_perturbed.MaxWordsPerturbed:
            attacker.attack.constraints[i].max_percent = max_percent_words
    print(attacker)
    attacker.attack_dataset()



def cli_main():
    parser = argparse.ArgumentParser(description="Attack Models")
    get_args(parser)
    args, remaining_args = parser.parse_known_args()
    assert remaining_args == [], remaining_args
    attack(args)


if __name__ == "__main__":
    import warnings
    warnings.filterwarnings("ignore")
    cli_main()