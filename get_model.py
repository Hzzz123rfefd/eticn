import torch

model = torch.load("model.pth")
torch.save(model["state_dict"],"model2.pth")
