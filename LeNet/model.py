import torch
import torch.nn as nn
# LeNet Architecture ---->>

# 1*32*32 input -> (5,5) kernel , s = 2 , p = 0 -> avg pool s = 2 , p = 0 -> (5,5) kernel , s = 1 , p = 0 -> avg pool s = 2 , p = 0

# Cnv 5 X 5 to 120 channels X Linear 84 X Linear 10
class LeNet(nn.Module):
  def __init__ (self):
    super(LeNet , self).__init__()
    self.features = nn.Sequential(
        nn.Conv2d(1 , 6 , kernel_size = 5),
        nn.ReLU(),
        nn.MaxPool2d(kernel_size=2),
        nn.Conv2d(6 , 16 , kernel_size = 5),
        nn.ReLU(),
        nn.MaxPool2d(kernel_size=2)
    )

    self.classifier = nn.Sequential(
        nn.Linear(16 * 4 * 4 , 120),
        nn.ReLU(),
        nn.Linear(120 , 84),
        nn.ReLU(),
        nn.Linear(84 , 10)
    )




  def forward(self , x):
    x = self.features(x)
    x = torch.flatten(x , 1)
    x = self.classifier(x)
    return x
