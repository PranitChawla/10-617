from transformers import ViltProcessor, ViltModel
import torch.nn as nn
from data.helpers import *
from transformers import default_data_collator
import os
from PIL import Image
class ViltClf(nn.Module):
    def __init__(self, args):
        super(ViltClf, self).__init__()
        self.args = args
        self.backbone = ViltModel.from_pretrained("dandelin/vilt-b32-mlm")
        self.processor = ViltProcessor.from_pretrained("dandelin/vilt-b32-mlm")
        last_size = args.hidden_sz
        args.n_classes = len(args.labels)
        self.regime = args.regime
        if self.regime == "attack":
            self.model = self.backbone
        self.clf = nn.ModuleList()
        for hidden in args.hidden:
            self.clf.append(nn.Linear(last_size, hidden))
            if args.include_bn:
                self.clf.append(nn.BatchNorm1d(hidden))
            self.clf.append(nn.ReLU())
            self.clf.append(nn.Dropout(args.dropout))
            last_size = hidden

        self.clf.append(nn.Linear(last_size, args.n_classes))

        self.transforms = get_transforms(args)
        self.data_dir = args.data_model_path
        self.args.max_seq_len = 40

    def convert_to_attack_output(self, inputs):
        list_objs = []
        for input_vals in inputs:
            image_name = input_vals[0]
            text = input_vals[1]
            image = Image.open(os.path.join(self.data_dir, image_name)).convert("RGB")
            image = self.transforms(image)
            merged_ins = self.processor(image, text, return_tensors="pt", padding = "max_length", truncation = True, max_length = self.args.max_seq_len)
            list_objs.append(merged_ins)
        inputs = default_data_collator(list_objs)
        for key in list(inputs.keys()):
            inputs[key] = inputs[key].squeeze(dim=1).cuda()
        return inputs


    def forward(self, inputs):
        if self.regime == "attack":
            inputs = self.convert_to_attack_output(inputs)
        outputs= self.backbone(**inputs)
        out = outputs.pooler_output
        for layer in self.clf:
            out = layer(out)
        if self.regime == "attack":
            return out.detach()
        else:
            return out