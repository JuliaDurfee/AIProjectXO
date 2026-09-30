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
    """Classifier 3: small convolutional neural network."""

    def __init__(self, size=SIZE, dropout=0.3):
        super().__init__()

        self.features = nn.Sequential(
            nn.Conv2d(1, 16, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2),

            nn.Conv2d(16, 32, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2),

            nn.Conv2d(32, 64, kernel_size=3, padding=1),
            nn.ReLU()
        )

        reduced_size = size // 4

        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Linear(64 * reduced_size * reduced_size, 128),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(128, 2)
        )

    def forward(self, x):
        x = self.features(x)
        return self.classifier(x)
 
def build(name):
    return {"mlp": MLP, "cnn": CNN}[name]()
