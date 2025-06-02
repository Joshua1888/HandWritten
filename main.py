import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import datasets, transforms
from torch.utils.data import DataLoader

# Set up
learning_rate = 0.001
batch_size = 64
num_epochs = 10
num_classes = 26

if torch.cuda.is_available() :
    device = "cuda"
else: 
    device = "cpu"

# data pipe line
transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.5,), (0.5,))
])


# train test split 
train_dataset = datasets.EMNIST(root='./data', split='letters', train=True, transform=transform, download=True)
test_dataset = datasets.EMNIST(root='./data', split='letters', train=False, transform=transform, download=True)
train_loader = DataLoader(dataset=train_dataset, batch_size=batch_size, shuffle=True)
test_loader = DataLoader(dataset=test_dataset, batch_size=batch_size, shuffle=False)


class ConvoNN(nn.Module):
    def __init__(self) :
        super(ConvoNN, self).__init__()
        self.conv1 = nn.Conv2d(in_channels=1, out_channels=16, kernel_size=3, padding=1)
        self.conv2 = nn.Conv2d(16, 48, kernel_size=5, padding=2)
        self.relu = nn.ReLU()
        self.pool = nn.MaxPool2d(kernel_size=2, stride=2)
        self.l1 = nn.Linear(48 * 7 * 7, 1176+300)
        self.l2 = nn.Linear(1176+300, 520)
        self.l3 = nn.Linear(520,26)
    
    def forward(self, x) :
        # first convo
        out = self.conv1(x)
        out = self.relu(out)
        out = self.pool(out)
        # second convo
        out = self.conv2(out)
        out = self.relu(out)
        out = self.pool(out)

        out = out.view(out.size(0), -1) # flatten 

        out = self.l1(out)
        out = self.relu(out)

        out = self.l2(out)
        out = self.relu(out)

        out = self.l3(out)
        


criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.SGD(model.parameters(), lr=learning_rate)

'''
class LinearRegression(nn.Module):
    def __init__(self, input_dim, output_dim), input_dim, output_dim):
        super(LinearRegression, self).__init__()
        # define diferent layers
        self.lin = nn.Linear(input_dim, output_dim)

    def forward(self, x):
        return self.lin(x)

model = LinearRegression(input_size, output_size)
'''
'''
# 3) Training loop
for epoch in range(n_iters):
    # predict = forward pass with our model
    y_predicted = model(X)

    # loss
    l = loss(Y, y_predicted)

    # calculate gradients = backward pass
    l.backward()

    # update weights
    optimizer.step()

    # zero the gradients after updating
    optimizer.zero_grad()
'''

# chat version

model = ConvoNN().to(device)


criterion = nn.CrossEntropyLoss()


optimizer = torch.optim.SGD(model.parameters(), lr=learning_rate)




for epoch in range(num_epochs):
    model.train()  
    running_loss = 0.0
    for images, labels in train_loader:
        labels = labels - 1

        images = images.to(device)
        labels = labels.to(device)

        outputs = model(images)
        loss = criterion(outputs, labels)

        loss.backward()

        optimizer.step()

        optimizer.zero_grad()

        running_loss += loss.item()

    print(f"Epoch [{epoch+1}/{num_epochs}], Loss: {running_loss/len(train_loader):.4f}")

model.eval()  
correct = 0
total = 0

with torch.no_grad():
    for images, labels in test_loader:
        labels = labels - 1
        images = images.to(device)
        labels = labels.to(device)

        outputs = model(images)
        _, predicted = torch.max(outputs, 1)
        total += labels.size(0)
        correct += (predicted == labels).sum().item()

print(f"Test Accuracy: {100 * correct / total:.2f}%")

