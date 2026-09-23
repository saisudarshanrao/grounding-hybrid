"""Print the compute environment. Run first on any new machine: python scripts/env_check.py"""
import platform

import torch
import transformers

print(f"Python        {platform.python_version()} ({platform.machine()})")
print(f"torch         {torch.__version__}")
print(f"transformers  {transformers.__version__}")

if torch.cuda.is_available():
    for i in range(torch.cuda.device_count()):
        p = torch.cuda.get_device_properties(i)
        print(f"CUDA GPU {i}    {p.name}, {p.total_memory / 1e9:.1f} GB")
    print("Device to use: cuda  (Kaggle: this is where full runs happen)")
elif torch.backends.mps.is_available():
    print("Apple MPS     available")
    print("Device to use: mps   (Mac: small tests only; final numbers come from Kaggle)")
else:
    print("No GPU found. Device to use: cpu")
