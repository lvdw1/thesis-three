import torch

model = torch.load("transformer_v3.pt")

# If it's a model, you might want to print its state_dict
if isinstance(model, torch.nn.Module):
    print(model.state_dict())
else:
    # If it's a tensor or other data, just print it
    print(model)
