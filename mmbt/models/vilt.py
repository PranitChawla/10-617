from transformers import ViltProcessor, ViltModel
import torch.nn as nn


class ViltClf(nn.Module):
    def __init__(self, args):
        super(ViltClf, self).__init__()
        self.args = args
        self.backbone = ViltModel.from_pretrained("dandelin/vilt-b32-mlm")
        # self.processor = ViltProcessor
        last_size = 768
        self.clf = nn.ModuleList()
        for hidden in args.hidden:
            self.clf.append(nn.Linear(last_size, hidden))
            if args.include_bn:
                self.clf.append(nn.BatchNorm1d(hidden))
            self.clf.append(nn.ReLU())
            self.clf.append(nn.Dropout(args.dropout))
            last_size = hidden

        self.clf.append(nn.Linear(last_size, args.n_classes))

    def forward(self, inputs):
        # inputs = self.processor(img, txt, return_tensors="pt")
        outputs = self.backbone(**inputs)
        out = outputs.pooler_output
        # print(out.shape)
        for layer in self.clf:
            out = layer(out)
        return out