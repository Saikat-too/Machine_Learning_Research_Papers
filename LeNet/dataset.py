import torchvision
from torch.utils.data import DataLoader, Dataset, ConcatDataset, random_split
from torchvision import datasets, transforms

def get_mnist_dataloader():
    # Define a transform to normalize the data
    transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.1307,), (0.3081,))
    ])

    # Download and load the training data
    train_dataset = torchvision.datasets.MNIST(root='./data', train=True,  download=True, transform=transform)

    # Download and load the test data
    test_dataset = torchvision.datasets.MNIST(root='./data', train=False,  download=True, transform=transform)

    # Create data loaders
    train_loader = DataLoader(train_dataset, batch_size=64, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=1000, shuffle=False)
    
    return train_loader , test_loader 