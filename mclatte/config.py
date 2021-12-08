import torch

DEVICE = torch.device(f'cuda:{torch.cuda.device_count() - 1}' if torch.cuda.is_available() else 'cpu')
