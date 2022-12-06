# Adv Robustness Project (10-617)

## Introduction

Install all dependencies using the requirements.txt file after creating a conda environment


### Model Training

train.py provides the common training pipeline for all datasets. 
- **task**: mmimdb, food101, vsnli
- **model**: bow, img, concatbow, bert, concatbert, mmbt

The following paths need to be set to start training.

- **data_path**: Assumes a subfolder for each dataset. 
- **savedir**: Location to save model checkpoints.

Example command to train:

```
python mmbt/train.py --batch_sz 4 --gradient_accumulation_steps 40 \
 --savedir /path/to/savedir/ --name mmbt_model_run \
 --data_path /path/to/datasets/ \
 --task food101 --task_type classification \
 --model vilt --num_image_embeds 3 \
 --patience 5 --dropout 0.1 --lr 5e-05 --warmup 0.1 --max_epochs 100 --seed 1
```  

Example command to test:

```
python mmbt/train.py --batch_sz 4 \
 --savedir /path/to/savedir/ --model vilt --name vilt_model \
```  

Example command for text attacks:

```
python mmbt/attack_text.py --batch_sz 4 \
 --savedir /path/to/savedir/ --model vilt --name vilt_model --qbudget 300\
```  

Example command for image attacks (Ensure the corresponding text attack file is saved in text_attacks folder):

```
python mmbt/attack_image_batch.py \
 --savedir /path/to/savedir/ --model vilt --name vilt_model --qbudget 300\
```  





