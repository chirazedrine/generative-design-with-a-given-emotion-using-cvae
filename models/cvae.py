import torch
from torch import nn
from torch.nn import functional as F

class CVAE(nn.Module):
    def __init__(self, image_channels, latent_dim, num_emotions, height=224, width=224):
        super(CVAE, self).__init__()
        self.latent_dim = latent_dim
        self.num_emotions = num_emotions

        # Define size for the embedding to match the input spatial dimensions
        self.emotion_embedding = nn.Embedding(num_emotions, height * width)
        self.embedding_to_channels = nn.Linear(height * width, image_channels * height * width)

        # Adjust the first Conv2d layer to accept only image_channels, since emotion embedding will be added later
        self.encoder = nn.Sequential(
            nn.Conv2d(image_channels, 32, kernel_size=3, stride=2, padding=1),
            nn.ReLU(),
            nn.Conv2d(32, 64, kernel_size=3, stride=2, padding=1),
            nn.ReLU(),
            nn.Flatten(),
            nn.Linear(64 * (height // 4) * (width // 4), 128),
            nn.ReLU(),
        )

        self.fc_mu = nn.Linear(128, latent_dim)
        self.fc_log_var = nn.Linear(128, latent_dim)

        self.decoder_input = nn.Linear(latent_dim, 128)

        self.decoder = nn.Sequential(
            nn.Linear(128, 64 * (height // 4) * (width // 4)),
            nn.ReLU(),
            nn.Unflatten(1, (64, height // 4, width // 4)),
            nn.ConvTranspose2d(64, 32, kernel_size=3, stride=2, padding=1, output_padding=1),
            nn.ReLU(),
            nn.ConvTranspose2d(32, image_channels, kernel_size=3, stride=2, padding=1, output_padding=1),
            nn.Sigmoid(),
        )

    def encode(self, x, emotions):
        # Embed emotions and reshape to match the spatial dimensions of x
        emotion_embedded = self.emotion_embedding(emotions)
        emotion_embedded = self.embedding_to_channels(emotion_embedded).view(-1, 3, 224, 224)
        # Combine embedded emotions with x before passing to the encoder
        x_cond = x + emotion_embedded
        h = self.encoder(x_cond)
        return self.fc_mu(h), self.fc_log_var(h)

    def forward(self, x, emotions):
        mu, log_var = self.encode(x, emotions)
        z = self.reparameterize(mu, log_var)
        return self.decode(z), mu, log_var

    def reparameterize(self, mu, logvar):
        std = torch.exp(0.5*logvar)
        eps = torch.randn_like(std)
        return mu + eps*std

    def decode(self, z):
        h = self.decoder_input(z)
        return self.decoder(h)

def loss_function(recon_x, x, mu, logvar):
    BCE = F.binary_cross_entropy(recon_x, x.view(-1, 224*224*3), reduction='sum')
    KLD = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp())
    return BCE + KLD
