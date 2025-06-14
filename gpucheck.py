import torch

# Check if CUDA is available
print(f"PyTorch version: {torch.__version__}")
print(f"CUDA available: {torch.cuda.is_available()}")

# If CUDA is available, print additional information
if torch.cuda.is_available():
    print(f"CUDA version: {torch.version.cuda}")
    print(f"Num GPUs Available: {torch.cuda.device_count()}")
    
    # Print details for each GPU
    for i in range(torch.cuda.device_count()):
        print(f"GPU {i}: {torch.cuda.get_device_name(i)}")
        print(f"GPU {i} memory: {torch.cuda.get_device_properties(i).total_memory / 1e9:.2f} GB")
else:
    print("No GPU available, using CPU")