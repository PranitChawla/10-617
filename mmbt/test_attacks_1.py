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
# from TextBuggerCustom import TextBuggerCustom
# from Pruthi2019Custom import Pruthi2019Custom
# from HomoglyphCustom import HomoglyphAttack
# from HotFlipCustom import HotFlipCustom
# from FasterGeneticAlgorithmCustom import FasterGeneticAlgorithmCustom
# from ChecklistCustom import CheckList2020Custom
# from PWWSCustom import PWWSCustom
# from BERTAttackCustom import BERTAttackCustom
# from CLARECustom import CLARECustom
# from A2TCustom import A2TCustom
# from funcs import *
# from custom_models import get_model
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

from data.helpers import get_data_loaders
from data.dataset import JsonlDataset, text_attackDataset

def get_args(parser):
    parser.add_argument("--batch_sz", type=int, default=2)
    parser.add_argument("--bert_model", type=str, default="bert-base-uncased", choices=["bert-base-uncased", "bert-large-uncased"])
    parser.add_argument("--data_path", type=str, default="../")
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
    parser.add_argument("--model", type=str, default="concatbert", choices=["bow", "img", "bert", "concatbow", "concatbert", "mmbt"])
    parser.add_argument("--n_workers", type=int, default=12)
    parser.add_argument("--name", type=str, default="concat_bert_model")
    parser.add_argument("--num_image_embeds", type=int, default=1)
    parser.add_argument("--patience", type=int, default=10)
    parser.add_argument("--savedir", type=str, default="saved_models/")
    parser.add_argument("--seed", type=int, default=123)
    parser.add_argument("--task", type=str, default="food101", choices=["mmimdb", "vsnli", "food101"])
    parser.add_argument("--task_type", type=str, default="classification", choices=["multilabel", "classification"])
    parser.add_argument("--warmup", type=float, default=0.1)
    parser.add_argument("--weight_classes", type=int, default=1)

# def attacker(args):
    #ipdb.set_trace()
class Vocab(object):
    def __init__(self, emptyInit=False):
        if emptyInit:
            self.stoi, self.itos, self.vocab_sz = {}, [], 0
        else:
            self.stoi = {
                w: i
                for i, w in enumerate(["[PAD]", "[UNK]", "[CLS]", "[SEP]", "[MASK]"])
            }
            self.itos = [w for w in self.stoi]
            self.vocab_sz = len(self.itos)

    def add(self, words):
        cnt = len(self.itos)
        for w in words:
            if w in self.stoi:
                continue
            self.stoi[w] = cnt
            self.itos.append(w)
            cnt += 1
        self.vocab_sz = len(self.itos)
parser = argparse.ArgumentParser(description="Train Models")
get_args(parser)
args, remaining_args = parser.parse_known_args()
assert remaining_args == [], remaining_args
# if not os.path.exists(args.savedir): os.makedirs(args.savedir)
file = open(f"{args.savedir}/eval.txt", "a")  
def myprint(a): print(a); file.write(a); file.write("\n"); file.flush()
# chkpt_name = os.path.basename(args.path)
#train dataset is needed to get the right vocabulary for the problem
# my_dataset, tokenizer, data_collator = prepare_huggingface_dataset(args)
# verbalizer, templates = get_prompts(args) 
train_loader, val_loader, test_loaders = get_data_loaders(args)
model = get_model(args)


def load_checkpoint(model, path):
    if device == 'cpu':
        print(path)
        best_checkpoint = torch.load(path,map_location='cpu')
    else:
        best_checkpoint = torch.load(path)
    model.load_state_dict(best_checkpoint["state_dict"],strict=False)



transforms = transforms.Compose(
        [
            transforms.Resize((256,256)),
            transforms.CenterCrop(224),
            transforms.ToTensor(),
            transforms.Normalize(
                mean=[0.46777044, 0.44531429, 0.40661017],
                std=[0.12221994, 0.12145835, 0.14380469],
            ),
        ]
    )
tokenizer = BertTokenizer.from_pretrained(args.bert_model, do_lower_case=True).tokenize
vocab = Vocab()
test_set = text_attackDataset(
    os.path.join(args.data_path, args.task, "test.jsonl"),
    tokenizer,
    transforms,
    vocab,
    args,
)
my_dataset = test_set

# split = args.split 
# print(my_dataset[split].num_rows)
# args.num_examples = min(my_dataset[split].num_rows, args.num_examples )
my_dataset = Dataset(my_dataset, input_columns = ["premise",  "hypothesis"])
# else:
    # dataset = HuggingFaceDataset(args.dataset, split="test") if args.dataset!="sst2" else HuggingFaceDataset("glue", args.dataset, split = "validation")

# attack_name = args.attack_name


# attack_name_mapper = {"textfooler":TextFoolerCustom, "pruthi":Pruthi2019Custom, 
#                     "hotflip":HotFlipCustom, "genetic":FasterGeneticAlgorithmCustom,
#                     "homoglyph":HomoglyphAttack, "checklist":CheckList2020Custom,
#                     "pwws":PWWSCustom,
#                     "clare":CLARECustom,
#                     "bertattack":BERTAttackCustom,
#                     "a2t":A2TCustom,
#                     "textbugger":TextBuggerCustom,
#                     }
                    
attack_class = TextFoolerJin2019
# log_to_csv=f"{args.model_dir}/{attack_name}_log_{args.swap_algo}_{args.token_swap}_{args.pool_label_words}_{args.pool_templates}.csv"
# if args.query_budget > 0:
#     log_to_csv=f"{args.model_dir}/{attack_name}_log_{args.swap_algo}_{args.token_swap}_{args.query_budget}_{args.pool_label_words}_{args.pool_templates}.csv"
# if os.path.exists(log_to_csv):
#     log_to_csv=f"{args.model_dir}/{attack_name}_log_{args.swap_algo}_{args.token_swap}_{args.query_budget}_{args.pool_label_words}_{args.pool_templates}_1.csv"
# print(templates)
# print(verbalizer)
attack = attack_class.build(model)
args.savedir = os.path.join(args.savedir,args.name)
load_checkpoint(model, os.path.join(args.savedir, "model_best.pt"))
model.eval()
# if args.query_budget < 0: args.query_budget = None 
attack_args = AttackArgs(num_examples=50, 
                        log_to_csv="log_to_csv.csv", \
                        checkpoint_interval=100, 
                        checkpoint_dir=args.savedir, 
                        disable_stdout=True,
                        query_budget = 500,parallel=False)

attacker = Attacker(attack, my_dataset, attack_args)

#set batch size of goal function
attacker.attack.goal_function.batch_size = 2
#set max words pertubed constraint
max_percent_words = 0.3
#flag = 0

for i,constraint in enumerate(attacker.attack.constraints):
    if type(constraint) == textattack.constraints.overlap.max_words_perturbed.MaxWordsPerturbed:
        attacker.attack.constraints[i].max_percent = max_percent_words
    # if type(constraint) == textattack.constraints.pre_transformation.input_column_modification.InputColumnModification:
    #     attacker.attack.constraints[i].matching_column_labels = ["text", "image_path"]
    #     attacker.attack.constraints[i].columns_to_ignore = {"image_path"}

# multimodal_constraint = InputColumnModification(
#             ["text", "image_path"], {"text"}
#         )
# attacker.attack.constraints.append(multimodal_constraint)    
# if(args.attack_name == "textbugger"):
#     attacker.attack.transformation = textattack.transformations.WordSwapEmbedding(max_candidates=5)
#     attacker.attack.constraints.append(textattack.constraints.overlap.max_words_perturbed.MaxWordsPerturbed(max_percent=max_percent_words))
print(attacker)
attacker.attack_dataset()
