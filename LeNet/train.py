import torch 
import torch.nn as nn
import torchvision
import torch.optim as optim
from torch.optim.lr_scheduler import StepLR

from model import LeNet
from dataset import get_mnist_dataloader

# Initialize the model with Xavier/Glorot initialization
def init_weights(m):
    if isinstance(m, nn.Linear) or isinstance(m, nn.Conv2d):
        nn.init.xavier_uniform_(m.weight)
        if m.bias is not None:
            nn.init.zeros_(m.bias)

model = LeNet()
model.apply(init_weights)

# Loading Dataset
train_loader , test_loader = get_mnist_dataloader()

# Define loss function and optimizer
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=0.001)
# scheduler = StepLR(optimizer, step_size=1, gamma=0.7)

# Training loop
def train(model, train_loader, criterion, optimizer ,  epochs=5):
    model.train()
    for epoch in range(epochs):
        running_loss = 0.0
        for i, (inputs, labels) in enumerate(train_loader):


            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            running_loss += loss.item()
            if i % 100 == 99:
                print(f'[{epoch + 1}, {i + 1:5d}] loss: {running_loss / 100:.3f}')
                running_loss = 0.0

        # scheduler.step()

train(model , train_loader , criterion , optimizer )
print('Finished Training')

# save the trained model
torch.save(model.state_dict() , 'mnist_cnn.pth')

# Test the model
def test(model, test_loader):
    model.eval()
    correct = 0
    total = 0
    with torch.no_grad():
        for images, labels in test_loader:
            outputs = model(images)
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()

    accuracy = 100 * correct / total
    print(f'Accuracy on test images: {accuracy:.2f}%')
    return accuracy


accuracy = test(model, test_loader)

