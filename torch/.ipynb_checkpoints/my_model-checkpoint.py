import torch
import torch.nn as nn
import torch.nn.functional as F

class Model(nn.Module):
    def __init__(self,):
        super(Model,self).__init__()
        self.base_model = torch.hub.load('pytorch/vision:v0.6.0', 'mobilenet_v2', pretrained=False)
        self.top_layer = nn.Sequential(
             nn.Linear(1000,10),
         )
    def forward(self, inp):
        out = self.base_model(inp)
        logit = self.top_layer(out)
        return F.softmax(logit, dim=1),logit