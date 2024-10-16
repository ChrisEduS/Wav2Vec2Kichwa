import torch

print(f"Memoria asignada: {torch.cuda.memory_allocated()} bytes")
print(f"Memoria reservada: {torch.cuda.memory_reserved()} bytes")
