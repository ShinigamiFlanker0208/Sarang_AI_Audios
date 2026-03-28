import torch


def check_demucs_hardware():
    print("=== Sarang AI Hardware Check ===")
    print(f"PyTorch Version: {torch.__version__}")

    # Check if CUDA (NVIDIA GPU) is accessible
    has_gpu = torch.cuda.is_available()
    print(f"Is CUDA (GPU) accessible? {has_gpu}")

    if has_gpu:
        device_count = torch.cuda.device_count()
        current_device = torch.cuda.current_device()
        device_name = torch.cuda.get_device_name(current_device)

        print(f"GPUs detected: {device_count}")
        print(f"Active GPU: {device_name}")
        print("\n✅ SUCCESS: Demucs will automatically use your GPU for lightning-fast separation!")
    else:
        print("\n❌ WARNING: PyTorch cannot see a GPU.")
        print("Demucs will fall back to the CPU. Processing a 3-minute song may take several minutes.")
        print("Fix: You may need to install the CUDA-enabled version of PyTorch.")


if __name__ == "__main__":
    check_demucs_hardware()