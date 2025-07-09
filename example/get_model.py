import torch

model = torch.load("saved_model/fbr/camvid/stf/0.18/model.pth")
torch.save(model["state_dict"],"model2.pth")
