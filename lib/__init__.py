import torch

# Set device
if torch.cuda.is_available():
    device = "cuda:0"  # CUDA GPU
elif torch.mps.is_available():
    device = "mps"  # Apple silicon GPU
else:
    device = "cpu"
