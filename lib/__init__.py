import torch

# Set device
if torch.cuda.is_available():
    device = torch.device("cuda:0")  # CUDA GPU
elif torch.mps.is_available():
    device = torch.device("mps")  # Apple silicon GPU
else:
    device = torch.device("cpu")  # CPU
