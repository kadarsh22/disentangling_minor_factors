from torch import nn
from torchvision.models import resnet18


def save_hook(module, input, output):
    setattr(module, 'output', output)


class Classifier(nn.Module):
    def __init__(self, num_dirs=10):
        super(Classifier, self).__init__()

        self.classifier_layer_1 = nn.Linear(512, 512)
        self.classifier_layer_2 = nn.Linear(512, 10)

    def forward(self, x):
        out = self.classifier_layer_1(x)
        out = self.classifier_layer_2(out)
        return out