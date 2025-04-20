import torch

model = torch.load("saved_model/0.0001/model.pth")
torch.save(model["state_dict"],"model2.pth")
