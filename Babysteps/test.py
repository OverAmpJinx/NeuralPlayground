import sys
import torch

print("Interpreter:", sys.executable)
print("PyTorch:", torch.__version__)
print("MPS available:", torch.backends.mps.is_available())