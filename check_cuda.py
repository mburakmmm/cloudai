import torch

print("=== CUDA DURUMU KONTROLÜ ===")
print(f"PyTorch Version: {torch.__version__}")
print(f"CUDA Available: {torch.cuda.is_available()}")
print(f"CUDA Version: {torch.version.cuda}")

if torch.cuda.is_available():
    print(f"CUDA Device Count: {torch.cuda.device_count()}")
    print(f"Current CUDA Device: {torch.cuda.current_device()}")
    print(f"Device Name: {torch.cuda.get_device_name(0)}")
    
    # Test tensor oluştur
    device = torch.device("cuda:0")
    x = torch.randn(3, 3).to(device)
    print(f"Test tensor on CUDA: {x.device}")
    print(f"Tensor shape: {x.shape}")
else:
    print("❌ CUDA kullanılamıyor!")
    print("💡 CPU kullanılıyor")
