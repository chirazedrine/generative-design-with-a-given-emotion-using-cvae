import torch
from models.cvae import CVAE as VAE
import matplotlib.pyplot as plt

def generate_image(model, emotion_label):
    model.eval()
    with torch.no_grad():
        sample = torch.randn(64, 20).to(device)  # Randomly generated sample
        sample = model.decode(sample).cpu()
        plt.imshow(sample.view(224, 224, 3))
        plt.title(f'Generated Image for Emotion: {emotion_label}')
        plt.show()

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = VAE().to(device)
model.load_state_dict(torch.load('path/to/trained_model.pth'))

emotion_label = "happy"  # Example emotion label
generate_image(model, emotion_label)
