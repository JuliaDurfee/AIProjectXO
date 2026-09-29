from torchvision import datasets, transforms
from torch.utils.data import DataLoader

transform = transforms.Compose([
    transforms.Grayscale(num_output_channels=1),
    transforms.Resize((32, 32)),
    transforms.ToTensor()
])

dataset = datasets.ImageFolder(
    root="data",
    transform=transform
)

loader = DataLoader(
    dataset,
    batch_size=8,
    shuffle=True
)

print("Classes:", dataset.classes)
print("Number of images:", len(dataset))

images, labels = next(iter(loader))

print("Image batch shape:", images.shape)
print("Labels:", labels)