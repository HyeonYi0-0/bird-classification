from torch import nn
import torchvision

class BaseModel(nn.Module):
    def __init__(self, num_classes, args):
        super(BaseModel, self).__init__()
        self.backbone = torchvision.models.get_model(args.model, weights=args.weights)
        self.classifier = nn.Linear(1000, num_classes)

    def forward(self, x):
        x = self.backbone(x)
        x = self.classifier(x)
        return x