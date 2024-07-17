import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import models, datasets, transforms

def ok_compression(image, depth=10):
    height, width = image.shape
    x, y = np.meshgrid(np.arange(width), np.arange(height))
    c = image

    x_norm = x / width
    y_norm = y / height
    c_norm = c / 255.0

    x_norm_flat = x_norm.flatten()
    y_norm_flat = y_norm.flatten()
    c_norm_flat = c_norm.flatten()

    def to_ternary(value, depth):
        ternary = []
        for _ in range(depth):
            value *= 3
            digit = int(value)
            ternary.append(digit)
            value -= digit
        return ternary

    def cantor_rewrite(ternary):
        binary = []
        for digit in ternary:
            if digit == 2:
                binary.append(1)
            else:
                binary.append(0)
        binary_fraction = sum(b * (2 ** -(i + 1)) for i, b in enumerate(binary))
        return binary_fraction

    x_ternary = [to_ternary(val, depth) for val in x_norm_flat]
    y_ternary = [to_ternary(val, depth) for val in y_norm_flat]
    c_ternary = [to_ternary(val, depth) for val in c_norm_flat]

    x_binary = np.array([cantor_rewrite(t) * width for t in x_ternary])
    y_binary = np.array([cantor_rewrite(t) * height for t in y_ternary])
    c_binary = np.array([cantor_rewrite(t) * 255 for t in c_ternary])

    x_binary_reshaped = x_binary.reshape(height, width)
    y_binary_reshaped = y_binary.reshape(height, width)
    c_binary_reshaped = c_binary.reshape(height, width)

    compressed_image = c_binary_reshaped  # Use c_binary_reshaped as the OKCompressed image
    return compressed_image


class CatCNN(nn.Module):
    def __init__(self):
        super(CatCNN, self).__init__()
        self.model = models.resnet18(weights='IMAGENET1K_V1')
        self.fc = nn.Linear(512, 3 * 64 * 64)  # Example output shape for cat images

    def forward(self, x):
        x = self.model(x)
        x = self.fc(x)
        x = x.view(-1, 3, 64, 64)
        return x

def train_catcnn(depth):
    transform = transforms.Compose([
        transforms.Grayscale(num_output_channels=1),
        transforms.Resize((64, 64)),
        transforms.ToTensor()
    ])

    train_dataset = datasets.ImageFolder('/path/to/train_data', transform=transform)
    val_dataset = datasets.ImageFolder('/path/to/val_data', transform=transform)

    train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=32, shuffle=True)
    val_loader = torch.utils.data.DataLoader(val_dataset, batch_size=32, shuffle=False)

    model = CatCNN()
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=0.001)

    num_epochs = 10
    for epoch in range(num_epochs):
        model.train()
        for inputs, labels in train_loader:
            inputs = ok_compression(inputs, depth)
            inputs = torch.tensor(inputs).float().unsqueeze(1)
            outputs = model(inputs)
            loss = criterion(outputs, labels)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

        model.eval()
        val_loss = 0
        with torch.no_grad():
            for inputs, labels in val_loader:
                inputs = ok_compression(inputs, depth)
                inputs = torch.tensor(inputs).float().unsqueeze(1)
                outputs = model(inputs)
                loss = criterion(outputs, labels)
                val_loss += loss.item()

        print(f'Epoch {epoch + 1}/{num_epochs}, Validation Loss: {val_loss / len(val_loader)}')

    torch.save(model.state_dict(), f'catcnn_depth_{depth}.pth')
    return model

class MetaCatNN(nn.Module):
    def __init__(self, depths):
        super(MetaCatNN, self).__init__()
        self.catcnns = nn.ModuleList([CatCNN() for _ in range(depths)])
        self.depths = depths

    def forward(self, x):
        for i in range(self.depths):
            x = ok_compression(x, i + 1)
            x = torch.tensor(x).float().unsqueeze(1)
            x = self.catcnns[i](x)
        return x

# Initialize and load pre-trained weights
depths = 10
meta_catcnn = MetaCatNN(depths)

for i in range(depths):
    meta_catcnn.catcnns[i].load_state_dict(torch.load(f'catcnn_depth_{i + 1}.pth'))

# Test MetaCatNN with an example input
example_input = np.random.rand(64, 64)  # Replace with actual input data
example_output = meta_catcnn(example_input)

# Visualize the output
example_output_np = example_output.detach().numpy().squeeze().transpose(1, 2, 0)
plt.imshow(example_output_np)
plt.title("MetaCatNN Output")
plt.show()
