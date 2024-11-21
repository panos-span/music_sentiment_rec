import torch
import torch.nn as nn

class CNNBackbone(nn.Module):
    def __init__(self, input_dims, in_channels, filters, feature_size):
        super(CNNBackbone, self).__init__()
        self.input_dims = input_dims
        self.in_channels = in_channels
        self.filters = filters
        self.feature_size = feature_size
        
        self.conv1 = nn.Sequential(
            nn.Conv2d(self.in_channels, filters[0], kernel_size=(5,5), stride=1, padding=2),
            nn.BatchNorm2d((self.in_channels**1) * filters[0]),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2))

        self.conv2 = nn.Sequential(
                nn.Conv2d(filters[0], filters[1], kernel_size=(5,5), stride=1, padding=2),
                nn.BatchNorm2d((self.in_channels**2) * filters[1]),
                nn.ReLU(),
                nn.MaxPool2d(kernel_size=2, stride=2))

        self.conv3 = nn.Sequential(
                nn.Conv2d(filters[1], filters[2], kernel_size=(3,3), stride=1, padding=1),
                nn.BatchNorm2d((self.in_channels**3) * filters[2]),
                nn.ReLU(),
                nn.MaxPool2d(kernel_size=2, stride=2))

        self.conv4 = nn.Sequential(
                nn.Conv2d(filters[2], filters[3], kernel_size=(3,3), stride=1, padding=1),
                nn.BatchNorm2d((self.in_channels**4) * filters[3]),
                nn.ReLU(),
                nn.MaxPool2d(kernel_size=2, stride=2))
        
        shape_after_convs = [input_dims[0]//2**(len(filters)), input_dims[1]//2**(len(filters))]
        self.fc1 = nn.Linear(filters[3] * shape_after_convs[0] * shape_after_convs[1], self.feature_size)
        
    def forward(self, x):
        x = x.view(x.shape[0], self.in_channels, x.shape[1], x.shape[2])
        out = self.conv1(x)
        out = self.conv2(out)
        out = self.conv3(out)
        out = self.conv4(out)
        out = out.reshape(out.size(0), -1)
        out = self.fc1(out)
        return out