import torch
from compressai import models
from compressai.utils import load_config

config = load_config("config/imagenetc/stf.yml")
model = models[config["model_type"]](**config["model"])
checkpoint = torch.load('saved_model/vbr/imagenet/stfcqvr/model.pth', map_location="cuda")
model.load_state_dict(checkpoint, strict=False)
model.save_pretrained("saved_model/")