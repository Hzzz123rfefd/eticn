import torch

save_path = "saved_model/fbr/soda/eticn/0.0018/model.pth"
model = torch.load(save_path)
a = model["state_dict"]
# new_state_dict = {}
# for k, v in a.items():
#     if k.startswith("image_transform_encoder"):
#          new_key = k.replace("image_transform_encoder", "g_a")
#     elif k.startswith("image_transform_decoder"):
#         new_key = k.replace("image_transform_decoder", "g_s")
#     elif k.startswith("hyperpriori_encoder.h_a"):
#         new_key = k.replace("hyperpriori_encoder.h_a", "h_a")
#     elif k.startswith("hyperpriori_decoder.h_s"):
#         new_key = k.replace("hyperpriori_decoder.h_s", "h_s")
#     else:
#         new_key = k
#     new_state_dict[new_key] = v
# torch.save(new_state_dict, "saved_model/fbr/camvid/vic/0.0018/model.pth")
torch.save(a , save_path)

