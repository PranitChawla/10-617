from transformers import FlavaProcessor, FlavaModel
import torch.nn as nn
import torch
class FlavaClf(nn.Module):
    def __init__(self, args):
        super(FlavaClf, self).__init__()
        self.args = args
        self.backbone = FlavaModel.from_pretrained("facebook/flava-full")
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
        outputs = self.backbone(**inputs)
        out = outputs.multimodal_embeddings
        # print(out.shape)
        # out = torch.mean(out,axis=1)
        # print(out.shape)
        out = out[:,0,:]
        # print(out.shape)
        for layer in self.clf:
            out = layer(out)
        return out