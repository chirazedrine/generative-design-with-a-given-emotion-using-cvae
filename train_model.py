import torch
from torch import optim
from torchvision import datasets, transforms
from torch.utils.data import DataLoader
from models.cvae import CVAE as VAE
from dataset.dataset import EmoSetDataset
import torch.nn.functional as F

# Define the training parameters
log_interval = 10  # Log training status every 10 batches
epochs = 10  # Number of epochs to train the model


def train(epoch, model, train_loader, optimizer):
    model.train()
    train_loss = 0
    for batch_idx, batch_data in enumerate(train_loader):
        images = batch_data['image'].to(device)
        emotions = batch_data['emotion_label_idx'].to(device)
        optimizer.zero_grad()
        recon_batch, mu, logvar = model(images, emotions)
        loss = loss_function(recon_batch, images, mu, logvar)
        loss.backward()
        train_loss += loss.item()
        optimizer.step()
        
        if batch_idx % log_interval == 0:
            print(f'Train Epoch: {epoch} [{batch_idx * len(images)}/{len(train_loader.dataset)}'
                  f' ({100. * batch_idx / len(train_loader):.0f}%)]\tLoss: {loss.item() / len(images):.6f}')

    print(f'====> Epoch: {epoch} Average loss: {train_loss / len(train_loader.dataset):.4f}')

def loss_function(recon_x, x, mu, logvar):
    BCE = F.binary_cross_entropy(recon_x, x.view(-1, 224*224*3), reduction='sum')
    KLD = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp())
    return BCE + KLD

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = VAE(image_channels=3, latent_dim=10, num_emotions=8).to(device)
optimizer = optim.Adam(model.parameters(), lr=1e-3)

transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
])
train_dataset = EmoSetDataset(data_root='./dataset/EmoSet-118K', phase='train', transform=transform)
train_loader = DataLoader(train_dataset, batch_size=16, shuffle=True)

for epoch in range(1, epochs + 1):
    train(epoch, model, train_loader, optimizer)
