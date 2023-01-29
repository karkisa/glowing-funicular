import torchvision
import torch
from torchsummary import summary

class get_model(torch.nn.Module):
    def __init__(self,num_classes):
        super(get_model,self).__init__()
        self.num_classes = num_classes
        self.model =torchvision.models.efficientnet_b0(pretrained = True)
        self.model.classifier[1]=torch.nn.Linear(1280,self.num_classes)

    def forward(self,images):
        return self.model(images)
  



