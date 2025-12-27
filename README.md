# Image Generation Using Variational Autoencoders (VAEs)

This repository implements a Variational Autoencoder (VAE) for image generation, specifically trained on face images. VAEs are generative models that learn compact latent representations of data, enabling the synthesis of novel samples that capture the underlying distribution of the training set.

## Introduction

VAEs extend traditional autoencoders by enforcing an approximate normal distribution in the latent space, allowing for effective sampling and generation of new data. The key innovation is the reparameterization trick, which enables backpropagation through the stochastic sampling process. This implementation uses a β-VAE variant with a tunable β parameter to balance reconstruction quality and latent space regularization, mitigating issues like posterior collapse.

![VAE Architecture](./.github/assets/VAE_Architecture.png)

Variational Autoencoders (VAEs) can be viewed as a non-linear extension of Probabilistic Principal Component Analysis (p-PCA). Unlike traditional autoencoders, VAEs enforce an approximate normal distribution in the latent space, improving the model's ability to capture underlying data structures and enabling effective sampling for generation.

The core challenge in generative modeling is that the latent space distribution is unknown, preventing realistic image generation from a trained decoder alone. VAEs address this by assuming the latent space follows a normal distribution, allowing sampling to produce diverse outputs. The decoder output is treated as the mean of a distribution, with added noise for final image generation.

To enforce the normal distribution, the encoder outputs mean (μ) and log-variance (σ²) parameters. However, sampling from this distribution is non-differentiable, blocking backpropagation. The **reparameterization trick** resolves this by expressing the latent variable as z = μ + ε \* σ, where ε ~ N(0,1), making the process differentiable.

VAEs can suffer from **posterior collapse**, where the learned distribution matches the prior too closely, reducing latent informativeness. This is mitigated using β-VAE, where a hyperparameter β scales the KL divergence term in the loss function. A lower β (e.g., 0.5) preserves latent structure, while higher values enforce stricter normality but risk collapse.

## Prerequisites

- Python 3.10 or higher

## Installation

1. **Clone the Repository**:

   ```bash
   git clone https://github.com/arghyaxcodes/Image_Generation_VAES.git
   cd Image_Generation_VAES
   ```

2. **Install Dependencies**:

   ```bash
   pip install torch torchvision numpy matplotlib
   ```

3. **Install Jupyter (Optional)**:
   If Jupyter is not already installed:
   ```bash
   pip install jupyter
   ```

## Usage

### Running the Jupyter Notebook

1. Start the Jupyter server:

   ```bash
   jupyter notebook
   ```

2. Navigate to and open `notebooks/image-generation-using-vaes.ipynb`.

3. Execute the cells sequentially to:
   - Load and preprocess the face dataset
   - Train the VAE model (or load pre-trained weights)
   - Generate new face images by sampling from the latent space

### Generating Images Programmatically

After loading the trained model, generate new images as follows:

```python
import torch
from model import Decoder  # Assuming decoder is defined in model.py

# Load pre-trained decoder
decoder = Decoder(latent_dim=10)
decoder.load_state_dict(torch.load('models/img_generation_model.pth'))
decoder.eval()

# Generate images
with torch.no_grad():
    latent_sample = torch.randn(batch_size, latent_dim)
    generated_images = decoder(latent_sample)
```

## Architecture

The VAE consists of:

- **Encoder**:

  - Two convolutional layers (32 and 64 filters, kernel size 4, stride 2, padding 1) for feature extraction.
  - Fully-connected layers output mean (μ) and log-variance (log σ²) for the latent distribution.
  - Input: 24x24 grayscale images; Output: Latent vectors of dimension 10 (default).

- **Decoder**:

  - Fully-connected layer maps latent vectors back to feature maps (64x6x6).
  - Two transposed convolutional layers for upsampling to original image size.
  - Sigmoid activation constrains output to [0,1] range.
  - Output: Reconstructed 24x24 images.

- **Reparameterization**: Implemented as z = μ + ε _ exp(0.5 _ log σ²), where ε ~ N(0,1).

- **Loss Function**: Combines Binary Cross-Entropy (BCE) for reconstruction and β-scaled KL divergence for regularization.

## Training Details

### Training

- **Hyperparameters**: Latent dimension = 20, batch size = 64, epochs = 5, learning rate = 1e-3, β = 0.5.
- **Optimizer**: Adam optimizer.
- **Loss**: BCE + β \* KL divergence.
- **Device**: CUDA if available, else CPU.

The training loop iterates through the dataset, computes reconstruction and KL losses, and updates model parameters via backpropagation.

### Generation

- **Sampling**: Random latent vectors from N(0,1) are passed through the decoder to generate new images.
- **Grid Generation**: Produces 10x10 grids of 100 generated faces.
- **Latent Space Interpolation**: Demonstrates smooth transitions between random latent points, showcasing the continuity of the learned manifold.

## Results

The trained VAE successfully generates diverse face images by sampling from the latent space. Key experiments include:

- **Image Grid Generation**: A 10x10 grid of 100 randomly generated face images, demonstrating the model's ability to produce varied outputs.
- **Latent Space Interpolation**: Smooth transitions between two random latent points, visualized in a 4x5 grid of 20 interpolated images, highlighting the continuous and structured nature of the learned latent manifold.

These results illustrate the VAE's effectiveness in capturing facial features and enabling controlled generation through latent space manipulation.

## Project Structure

```
Image_Generation_VAES/
├── README.md
├── data/
│   └── faces_vae.npy              # Face image dataset
├── images/                        # Generated image outputs
├── models/
│   └── img_generation_model.pth   # Pre-trained VAE weights
└── notebooks/
    └── image-generation-using-vaes.ipynb  # Main implementation notebook
```

## Dependencies

- PyTorch >= 1.9.0
- Torchvision >= 0.10.0
- NumPy >= 1.19.0
- Matplotlib >= 3.3.0

## License

This project is licensed under the MIT License. See the [LICENSE](LICENSE) file for details.

## References

- [Kingma, D. P., & Welling, M. (2013). Auto-Encoding Variational Bayes. arXiv preprint arXiv:1312.6114](https://arxiv.org/abs/1312.6114)
- PyTorch Documentation: https://pytorch.org/docs/

**Note: This implementation is for educational purposes. For production use, consider more robust architectures and extensive hyperparameter tuning.**
