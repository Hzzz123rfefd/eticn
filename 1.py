
import torch


a = torch.load("saved_model/fbr/camvid/vic/0.0932/model.pth")


new_state_dict = {}
for k, v in a.items():
    if k.startswith("image_transform_encoder"):
         new_key = k.replace("image_transform_encoder", "g_a")
    elif k.startswith("image_transform_decoder"):
        new_key = k.replace("image_transform_decoder", "g_s")
    elif k.startswith("hyperpriori_encoder.h_a"):
        new_key = k.replace("hyperpriori_encoder.h_a", "h_a")
    elif k.startswith("hyperpriori_decoder.h_s"):
        new_key = k.replace("hyperpriori_decoder.h_s", "h_s")
    else:
        new_key = k
    new_state_dict[new_key] = v
torch.save(new_state_dict, "saved_model/fbr/camvid/vic/0.0932/model2.pth")