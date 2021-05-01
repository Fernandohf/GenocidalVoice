import torch
import torch.nn as nn
import torchvision.models as models


class GenocidalClassifier(nn.Module):
    def __init__(self, pretrained_path=None):

        super().__init__()
        # Load pretrained image model
        self.model = models.resnet18(pretrained=True)

        # Change first layer for data input format
        self.model.conv1 = nn.Conv2d(1, self.model.conv1.out_channels,
                                     kernel_size=self.model.conv1.kernel_size[0],
                                     stride=self.model.conv1.stride[0],
                                     padding=self.model.conv1.padding[0])
        num_features = self.model.fc.in_features

        # Change last layer for classification task
        self.model.fc = nn.Linear(num_features, 1)

        # Load pretrained
        if pretrained_path:
            self.load_model(pretrained_path)

    def load_model(self, checkpoint):
        self.model.load_state_dict(torch.load(checkpoint))
        self.model.cpu()

    def forward(self, x):
        output = self.model(x)
        return output

    def save_model(self, path="model.pt"):
        torch.save(self.model.state_dict(), path)

    def is_genocidal(self, batch, thresh=.5):
        with torch.no_grad():
            out = self.model(batch)
            pred = (torch.sigmoid(out)).cpu().numpy()
            return (pred > thresh, pred)
