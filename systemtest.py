import torch

print("CUDA is available" if torch.cuda.is_available() else "CUDA is not available")
print(torch.zeros(1).cuda())
available_gpus = [torch.cuda.device(i) for i in range(torch.cuda.device_count())]

print(available_gpus)