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

Example command:

```
python mmbt/train.py --batch_sz 4 --gradient_accumulation_steps 40 \
 --savedir /path/to/savedir/ --name mmbt_model_run \
 --data_path /path/to/datasets/ \
 --task food101 --task_type classification \
 --model mmbt --num_image_embeds 3 --freeze_txt 5 --freeze_img 3  \
 --patience 5 --dropout 0.1 --lr 5e-05 --warmup 0.1 --max_epochs 100 --seed 1
```  

