import torch
from compressai import models
from compressai.utils import load_config

config = load_config("config/eticncqvr.yml")
model = models[config["model_type"]](**config["model"])
checkpoint = torch.load('saved_model/eticncqvr/model.pth', map_location="cuda")
model.load_state_dict(checkpoint, strict=False)
model.save_pretrained("saved_model/")


# if __name__ == "__main__":
#     parser = argparse.ArgumentParser()
#     parser.add_argument("--model_config_path",type=str,default = "config/stfqvrf.yml")
    
#     args = parser.parse_args()
#     main(args)
