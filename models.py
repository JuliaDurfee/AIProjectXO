import torch.nn as nn
 
from preprocess import SIZE
 
 
class MLP(nn.Module):
    """Classifier 2: flatten the image, fully connected layers only."""
    def __init__(self, size=SIZE, hidden=(256, 64), dropout=0.3):
        super().__init__()
        layers, d = [nn.Flatten()], size * size
        for h in hidden:
            layers += [nn.Linear(d, h), nn.ReLU(), nn.Dropout(dropout)]
            d = h
        layers.append(nn.Linear(d, 2))              # 2 logits: O, X
        self.net = nn.Sequential(*layers)
 
    def forward(self, x):
        return self.net(x)
 
 
class CNN(nn.Module):
    """Classifier 3: small convolutional net. 28x28 -> 14 -> 7 -> global average pool."""
    def __init__(self, dropout=0.3):
        super().__init__()
        def block(cin, cout):
            return [nn.Conv2d(cin, cout, 3, padding=1), nn.BatchNorm2d(cout), nn.ReLU()]
        self.features = nn.Sequential(
            *block(1, 16), nn.MaxPool2d(2),
            *block(16, 32), nn.MaxPool2d(2),
            *block(32, 64),
            *block(64, 64),
            nn.AdaptiveAvgPool2d(1),
        )
        self.head = nn.Sequential(nn.Flatten(), nn.Dropout(dropout), nn.Linear(64, 2))
 
    def forward(self, x):
        return self.head(self.features(x))
 
 
def build(name):
    return {"mlp": MLP, "cnn": CNN}[name]()
