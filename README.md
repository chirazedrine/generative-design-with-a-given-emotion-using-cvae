# Emotion-Conditioned Shape Generation with CVAE

This project implements a Conditional Variational Autoencoder (CVAE) to generate shapes conditioned on emotions, using the EmoSet dataset. It explores the intersection of emotion recognition and generative models to create visual representations based on emotional input.

## Getting Started

### Prerequisites

- Python 3.6+
- PyTorch
- torchvision
- PIL
- matplotlib

See `requirements.txt` for detailed dependencies.

### Installation

1. Clone the repository:

```shell
git clone https://github.com/chirazedrine/generative-design-with-a-given-emotion-using-cvae.git
```

2. Install the required packages:

```shell
pip install -r requirements.txt
```

## Usage

### Dataset Preparation

Download and prepare the [EmoSet dataset](https://github.com/JingyuanYY/EmoSet). Unzip it and place it in ./dataset/EmoSet-118K.

### Training the Model

Run the training script:

```shell
python train_model.py
```

### Generating Images

Generate shapes based on emotions:

```shell
python generate_images.py
```

## Project Structure

- `dataset/`: Dataset preparation scripts.
- `models/`: Contains the CVAE model definition.
<!-- - `training/`: Training script for the CVAE model. -->
- `generation/`: Script for generating images based on emotions.
- `requirements.txt`: Project dependencies.

## License

This project is licensed under the MIT License - see the LICENSE file for details.

## Acknowledgments

- Thanks to the creators of the EmoSet dataset for their valuable resource. This project is inspired by research on emotion recognition and generative models.
- Reference: Jingyuan Yang, Qirui Huang, Tingting Ding, Dani Lischinski, Danny Cohen-Or, and Hui Huang, "EmoSet: A Large-scale Visual Emotion Dataset with Rich Attributes," in Proceedings of the IEEE/CVF International Conference on Computer Vision, pp. 20383–20394, 2023.
