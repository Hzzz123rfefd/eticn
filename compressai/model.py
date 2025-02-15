from torch.utils.data import DataLoader
import torch
import torch.nn as nn
import math
import numpy as np

from compressai.entropy_models import *
from compressai.utils import AverageMeter,clip_gradient
from compressai.modules import *
from compressai.layers import *
from compressai.base import *

# class CompressionModel(nn.Module):
#     def __init__(self,image_channel,image_height,image_weight, out_channel_m, out_channel_n, 
#         lamda = 0.0002, 
#         use_finetune_model = False, 
#         finetune_model_lamda = None, 
#         device = "cuda"
#     ):
#         super().__init__()
#         self.use_finetune_model = use_finetune_model
#         self.finetune_model_lamda = finetune_model_lamda
#         self.device = device if torch.cuda.is_available() else "cpu"
#         self.image_channel = image_channel
#         self.image_height = image_height
#         self.image_weight = image_weight
#         self.image_shape = [image_channel,image_height,image_weight]
#         self.lamda = lamda
#         self.out_channel_m = out_channel_m
#         self.out_channel_n = out_channel_n
#         self.device =  device if torch.cuda.is_available() else "cpu"
        
#         self.entropy_bottleneck = EntropyBottleneck(out_channel_n).to(self.device)
#         self.gaussian_conditional = GaussianConditional(None).to(self.device)

#     def forward(self,inputs):
#         raise NotImplementedError("Subclasses should implement this method")

#     def load_pretrained(self, save_model_dir, lamda = None):
#         lamda = self.lamda if lamda == None else lamda
#         self.load_state_dict(torch.load(save_model_dir + "/" + str(lamda) + "/model.pth"))
    
#     def save_pretrained(self,  save_model_dir, lamda = None):
#         lamda = self.lamda if lamda == None else lamda
#         torch.save(self.state_dict(), save_model_dir + "/" + str(lamda) + "/model.pth")

#     def trainning(
#             self,
#             train_dataloader:DataLoader = None,
#             test_dataloader:DataLoader = None,
#             optimizer_name:str = "Adam",
#             weight_decay:float = 1e-4,
#             clip_max_norm:float = 0.5,
#             factor:float = 0.3,
#             patience:int = 15,
#             lr:float = 1e-4,
#             total_epoch:int = 1000,
#             save_checkpoint_step:str = 10,
#             save_model_dir:str = "models"
#         ):
#             ## 1 trainning log path 
#             first_trainning = True
#             save_model_dir_ = save_model_dir  + "/" + str(self.lamda)
#             check_point_path = save_model_dir_   + "/checkpoint.pth"
#             log_path = save_model_dir_  + "/train.log"

#             ## 2 get net pretrain parameters if need 
#             """
#                 If there is  training history record, load pretrain parameters
#             """
#             if  os.path.isdir(save_model_dir_) and os.path.exists(check_point_path) and os.path.exists(log_path):
#                 self.load_pretrained(save_model_dir, self.finetune_model_lamda)  
#                 first_trainning = False

#             else:
#                 if not os.path.isdir(save_model_dir_):
#                     os.makedirs(save_model_dir_)
#                 with open(log_path, "w") as file:
#                     pass

#                 if self.use_finetune_model:
#                     self.load_pretrained(save_model_dir,)


#             ##  3 get optimizer
#             if optimizer_name == "Adam":
#                 optimizer = optim.Adam(self.parameters(),lr,weight_decay = weight_decay)
#             elif optimizer_name == "AdamW":
#                 optimizer = optim.AdamW(self.parameters(),lr,weight_decay = weight_decay)
#             else:
#                 optimizer = optim.Adam(self.parameters(),lr,weight_decay = weight_decay)
#             lr_scheduler = optim.lr_scheduler.ReduceLROnPlateau(
#                 optimizer = optimizer, 
#                 mode = "min", 
#                 factor = factor, 
#                 patience = patience
#             )

#             ## init trainng log
#             if first_trainning:
#                 best_loss = float("inf")
#                 last_epoch = 0
#             else:
#                 checkpoint = torch.load(check_point_path, map_location=self.device)
#                 optimizer.load_state_dict(checkpoint["optimizer"])
#                 lr_scheduler.load_state_dict(checkpoint["lr_scheduler"])
#                 best_loss = checkpoint["loss"]
#                 last_epoch = checkpoint["epoch"] + 1

#             try:
#                 for epoch in range(last_epoch,total_epoch):
#                     print(f"Learning rate: {optimizer.param_groups[0]['lr']}")
#                     train_loss = self.train_one_epoch(epoch,train_dataloader, optimizer,clip_max_norm,log_path)
#                     test_loss = self.test_epoch(epoch,test_dataloader,log_path)
#                     loss = train_loss + test_loss
#                     lr_scheduler.step(loss)
#                     is_best = loss < best_loss
#                     best_loss = min(loss, best_loss)
#                     torch.save(                
#                         {
#                             "epoch": epoch,
#                             "loss": loss,
#                             "optimizer": None,
#                             "lr_scheduler": None
#                         },
#                         check_point_path
#                     )

#                     if epoch % save_checkpoint_step == 0:
#                         os.makedirs(save_model_dir_ + "/" + "chaeckpoint-"+str(epoch))
#                         torch.save(
#                             {
#                                 "epoch": epoch,
#                                 "loss": loss,
#                                 "optimizer": optimizer.state_dict(),
#                                 "lr_scheduler": lr_scheduler.state_dict()
#                             },
#                             save_model_dir_ + "/" + "chaeckpoint-"+str(epoch)+"/checkpoint.pth"
#                         )
#                     if is_best:
#                         self.save_pretrained(save_model_dir)

#             # interrupt trianning
#             except KeyboardInterrupt:
#                     torch.save(                
#                         {
#                             "epoch": epoch,
#                             "loss": loss,
#                             "optimizer": optimizer.state_dict(),
#                             "lr_scheduler": lr_scheduler.state_dict()
#                         },
#                         check_point_path
#                     )

#     def compute_loss(self, input):
#         out = {}
#         mse_loss = nn.MSELoss()
#         N, _, H, W = input["image"].size()
#         num_pixels = N * H * W

#         out["bpp_loss"] = sum(
#             (torch.log(likelihoods).sum() / (-math.log(2) * num_pixels))
#             for likelihoods in input["likelihoods"].values()
#         )

#         """ reconstruction loss """
#         out["reconstruction_loss"] = mse_loss(input["reconstruction_image"], input["image"])

#         """ all loss """
#         out["total_loss"] = out["reconstruction_loss"]  + self.lamda * out["bpp_loss"]
#         return out

#     def train_one_epoch(self, epoch, train_dataloader, optimizer, clip_max_norm, log_path = None):
#         self.train()
#         self.to(self.device)
#         pbar = tqdm(train_dataloader,desc="Processing epoch "+str(epoch), unit="batch")
#         total_loss = AverageMeter()
#         reconstruction_loss = AverageMeter()
#         bpp_loss = AverageMeter()
        
#         for batch_id, inputs in enumerate(train_dataloader):
#             """ grad zeroing """
#             optimizer.zero_grad()

#             """ forward """
#             used_memory = 0 if self.device != "cuda" else torch.cuda.memory_allocated(torch.cuda.current_device()) / (1024 ** 3)  
#             output = self.forward(inputs)

#             """ calculate loss """
#             out_criterion = self.compute_loss(output)
#             out_criterion["total_loss"].backward()
#             total_loss.update(out_criterion["total_loss"].item())
#             bpp_loss.update(out_criterion["bpp_loss"].item())
#             reconstruction_loss.update(out_criterion["reconstruction_loss"].item())

#             """ grad clip """
#             if clip_max_norm > 0:
#                 clip_gradient(optimizer,clip_max_norm)

#             """ modify parameters """
#             optimizer.step()
#             after_used_memory = 0 if self.device != "cuda" else torch.cuda.memory_allocated(torch.cuda.current_device()) / (1024 ** 3) 
#             postfix_str = "total_loss: {:.4f},reconstruction_loss:{:.4f},bpp_loss:{:.4f},use_memory: {:.1f}G".format(
#                 total_loss.avg, 
#                 reconstruction_loss.avg,
#                 bpp_loss.avg,
#                 after_used_memory - used_memory
#             )
#             pbar.set_postfix_str(postfix_str)
#             pbar.update()
#         with open(log_path, "a") as file:
#             file.write(postfix_str+"\n")
#         return total_loss.avg

#     def test_epoch(self,epoch, test_dataloader,trainning_log_path = None):
#         total_loss = AverageMeter()
#         reconstruction_loss = AverageMeter()
#         bpp_loss = AverageMeter()
#         self.eval()
#         self.to(self.device)
#         with torch.no_grad():
#             for batch_id, inputs in enumerate(test_dataloader):
#                 """ forward """
#                 output = self.forward(inputs)

#                 """ calculate loss """
#                 out_criterion = self.compute_loss(output)
#                 total_loss.update(out_criterion["total_loss"].item())
#                 bpp_loss.update(out_criterion["bpp_loss"].item())
#                 reconstruction_loss.update(out_criterion["reconstruction_loss"].item())

#             str = "Test Epoch: {:d}, total_loss: {:.4f},reconstruction_loss:{:.4f},bpp_loss:{:.4f}".format(
#                 epoch,
#                 total_loss.avg, 
#                 bpp_loss.avg,
#                 reconstruction_loss.avg
#             )
#         print(str)
#         with open(trainning_log_path, "a") as file:
#             file.write(str+"\n")
#         return total_loss.avg

#     def get_reconstrction_image(self, image):
#         image = image.unsqueeze(0)
#         image = image.to(self.device)
#         N ,C, H ,W = image.shape
#         self = self.to(self.device)
#         image = image.contiguous() / 255.0
#         self.eval()
#         inputs = {"images":image}
#         with torch.no_grad():
#             output = self.forward(inputs)

#         num_pixels = N * H * W
#         bpp = sum(
#             (torch.log(likelihoods).sum() / (-math.log(2) * num_pixels))
#             for likelihoods in output["likelihoods"].values()
#         )

#         return output["reconstruction_image"].cpu().squeeze(0) * 255, bpp.cpu()

# class ETICN(CompressionModel):
#     def __init__(self,image_channel,image_height,image_weight,patch_size,embedding_dim,
#         window_size,head_num,shift_size,out_channel_m,out_channel_n,group_num,
#         codebook_size,transfomer_head,transfomer_blocks,
#         drop_prob = 0.1,
#         stage = 2,
#         lamda = 0.0002,
#         sigma = 0.0001,
#         beta = 0.0001,
#         use_finetune_model = False, 
#         finetune_model_lamda = None, 
#         use_university_pretrain = False,
#         university_pretrain_path = None,
#         device = "cuda"
#     ):
#         super().__init__(image_channel, image_height, image_weight, out_channel_m,out_channel_n, lamda, use_finetune_model, finetune_model_lamda, device)
#         self.use_university_pretrain = use_university_pretrain
#         self.university_pretrain_path = university_pretrain_path
#         self.sigma = sigma
#         self.beta = beta
#         self.image_shape = [image_channel,image_height,image_weight]
#         self.patch_size = patch_size
#         self.embed_dim = embedding_dim
#         self.group_num = group_num
#         self.codebook_size = codebook_size
#         self.stage = stage
#         self.out_channel_m = out_channel_m
#         self.feather_shape = [
#             embedding_dim*8,
#             (int)(self.image_shape[1]/patch_size/8),
#             (int)(self.image_shape[2]/patch_size/8)
#         ]
        
#         self.image_transform_encoder = Encoder(
#             image_shape = self.image_shape,
#             patch_size = self.patch_size,
#             embed_dim = self.embed_dim,
#             window_size = window_size,
#             head_num = head_num,
#             shift_size = shift_size,
#             out_channel_m = self.out_channel_m
#         ).to(self.device)

#         self.image_transform_decoder = Decoder(
#             image_shape = self.image_shape,
#             patch_size = self.patch_size,
#             embed_dim = embedding_dim,
#             window_size = window_size,
#             head_num=head_num,
#             shift_size=shift_size,                                    
#             out_channel_m= out_channel_m
#         ).to(self.device)

#         self.tedm = TEDM(
#             in_c = self.image_shape[0], 
#             embed_dim = embedding_dim
#         ).to(self.device)

#         self.hyperpriori_encoder = HyperprioriEncoder(
#             feather_shape = [out_channel_m,(int)(self.image_shape[1]/16),(int)(self.image_shape[2]/16)],
#             out_channel_m = out_channel_m,
#             out_channel_n = out_channel_n
#         ).to(self.device)

#         self.side_context = nn.Sequential(
#             deconv(out_channel_n, out_channel_m, kernel_size = 5,stride = 2),
#             nn.LeakyReLU(inplace=True),
#             deconv(out_channel_m, out_channel_m * 3 // 2,kernel_size = 5,stride = 2),
#             nn.LeakyReLU(inplace=True),
#             conv(out_channel_m * 3 // 2, out_channel_m * 2, kernel_size=3,stride = 1)
#         ).to(self.device)


#         self.universal_context = UniversalContext(
#             out_channel_m = self.out_channel_m,
#             codebook_size = self.codebook_size,
#             group_num = self.group_num
#         ).to(self.device)

#         self.local_context = MaskedConv2d(
#             in_channels = out_channel_m , 
#             out_channels = 2 * out_channel_m, 
#             kernel_size = 5, 
#             padding = 2, 
#             stride = 1
#         ).to(self.device)

#         self.global_context = GlobalContext(
#             head = transfomer_head ,
#             layers= transfomer_blocks,
#             d_model_1 = out_channel_m,
#             d_model_2 = out_channel_m * 2,
#             drop_prob = drop_prob
#         ).to(self.device)
        
#         self.parm1 = nn.Sequential(
#             nn.Conv2d(out_channel_m * 15 // 3,out_channel_m * 10 // 3, 1),
#             nn.LeakyReLU(inplace=True),
#             nn.Conv2d(out_channel_m * 10 // 3, out_channel_m * 8 // 3, 1),
#             nn.LeakyReLU(inplace=True),
#             nn.Conv2d(out_channel_m * 8 // 3, out_channel_m * 6 // 3, 1),
#         ).to(self.device)

#         self.parm2 = nn.Sequential(
#             nn.Conv2d(out_channel_m * 15 // 3,out_channel_m * 10 // 3, 1),
#             nn.LeakyReLU(inplace=True),
#             nn.Conv2d(out_channel_m * 10 // 3, out_channel_m * 8 // 3, 1),
#             nn.LeakyReLU(inplace=True),
#             nn.Conv2d(out_channel_m * 8 // 3, out_channel_m * 6 // 3, 1),
#         ).to(self.device)

#     def forward(self,inputs):
#             image = inputs["image"].to(self.device)
#             """ get latent vector """
#             y,mid_feather = self.image_transform_encoder(image)

#             """ detect traffic element"""
#             logits,mask = self.tedm(image,mid_feather)

#             """ get side message """
#             z = self.hyperpriori_encoder(y)
#             z_hat, z_likelihoods = self.entropy_bottleneck(z)
#             side_ctx = self.side_context(z_hat)

#             y_hat = self.gaussian_conditional.quantize(
#                 y, "noise" if self.training else "dequantize"
#             )

#             """ get local message """
#             local_ctx = self.local_context(y_hat)

#             """ get global message """
#             global_ctx = self.global_context(y_hat,local_ctx)

#             """ get university message"""
#             y_ = y * (1 - mask.unsqueeze(1).repeat(1, self.out_channel_m, 1, 1))
#             universal_ctx,y_ba,code_index = self.universal_context(y_)
#             y_ba = y_ba * (1 - mask.unsqueeze(1).repeat(1, self.out_channel_m, 1, 1))
#             universal_ctx = universal_ctx * (1 - mask.unsqueeze(1).repeat(1, self.out_channel_m, 1, 1))

#             """ parameters estimation"""
#             gaussian_params1 = self.parm1(
#                 torch.concat((local_ctx,global_ctx,side_ctx),dim=1)
#             )
#             gaussian_params2 = self.parm2(
#                 torch.concat((local_ctx,universal_ctx,side_ctx),dim=1)
#             )
#             scales_hat1, means_hat1 = gaussian_params1.chunk(2, 1)
#             scales_hat2, means_hat2 = gaussian_params2.chunk(2, 1)
#             scales_hat = scales_hat2*(1 - mask.unsqueeze(1).repeat(1, self.out_channel_m, 1, 1)) + scales_hat1*mask.unsqueeze(1).repeat(1, self.out_channel_m, 1, 1)
#             means_hat = means_hat2*(1 - mask.unsqueeze(1).repeat(1, self.out_channel_m, 1, 1)) + means_hat1*mask.unsqueeze(1).repeat(1, self.out_channel_m, 1, 1)


#             _,y_likelihoods = self.gaussian_conditional(y,scales_hat,means_hat)
#             """ inverse transformation"""
#             x_hat = self.image_transform_decoder(y_hat)
#             x_hat = torch.clamp(x_hat,0,1)

#             """ output """
#             """
#                 reconstruction_image: reconstruction image
#                 feather: transport feather to CGVQ
#                 zq: re-feather
#                 likelihoods: likelihoods of lather（y） and super prior latent（z）
#                 logits: logits score of mask
#                 mask: predict transport mask
#             """
#             output = {
#                 "image":inputs["image"].to(self.device),
#                 "reconstruction_image":x_hat,
#                 "feather":y_,
#                 "zq":y_ba,
#                 "likelihoods": {"y": y_likelihoods, "z": z_likelihoods},
#                 "logits":logits,
#                 "mask":mask,
#                 "latent":y,
#                 "labels":inputs["label"].long().to(self.device) if "label" in inputs else None
#             }
#             return output
    
#     def compute_loss(self, input):
#         mse_loss = nn.MSELoss()
#         cross = nn.CrossEntropyLoss()
#         N, _, H, W = input["image"].size()
#         out = {}
#         num_pixels = N * H * W

#         out["bpp_loss"] = sum(
#             (torch.log(likelihoods).sum() / (-math.log(2) * num_pixels))
#             for likelihoods in input["likelihoods"].values()
#         )

#         """ reconstruction loss """
#         out["reconstruction_loss"] = mse_loss(input["reconstruction_image"], input["image"])

#         """ codebook loss """
#         out["codebook_loss"] =  mse_loss(input["zq"].detach(), input["feather"]) + 0.25 * mse_loss(input["zq"], input["feather"].detach())

#         """ mask loss """
#         true_labels = input["labels"]
#         pre_labels = F.interpolate(input["logits"], scale_factor=16, mode='bicubic', align_corners=True)
#         out["mask_loss"] = cross(pre_labels, true_labels)

#         """ recall """
#         pre = torch.argmax(pre_labels, dim=1)
#         TP = torch.sum((pre == 0) & (true_labels == 0))
#         FN = torch.sum((pre == 1) & (true_labels == 0))
#         if TP + FN == 0:
#             out["recall"] = 0.5
#         else:
#             out["recall"] = TP.float() / (TP + FN).float()

#         """ all loss """
#         out["loss"] = out["reconstruction_loss"]  + self.sigma * out["codebook_loss"] + self.lamda * out["bpp_loss"] + self.beta * out["mask_loss"]
#         return out

#     def trainning(self, train_dataloader: DataLoader = None, test_dataloader: DataLoader = None, optimizer_name: str = "Adam", weight_decay: float = 0.0001, clip_max_norm: float = 0.5, factor: float = 0.3, patience: int = 15, lr: float = 0.0001, total_epoch: int = 1000, save_checkpoint_step: str = 10, save_model_dir: str = "models"):
#             ## 1 trainning log path 
#             first_trainning = True
#             save_model_dir_ = save_model_dir  + "/" + str(self.lamda)
#             check_point_path = save_model_dir_   + "/checkpoint.pth"
#             log_path = save_model_dir_  + "/train.log"

#             ## 2 get net pretrain parameters if need 
#             """
#                 If there is  training history record, load pretrain parameters
#             """
#             if  os.path.isdir(save_model_dir_) and os.path.exists(check_point_path) and os.path.exists(log_path):
#                 self.load_pretrained(save_model_dir,self.finetune_model_lamda)  
#                 first_trainning = False

#             else:
#                 if not os.path.isdir(save_model_dir_):
#                     os.makedirs(save_model_dir_)
#                 with open(log_path, "w") as file:
#                     pass

#                 if self.use_finetune_model:
#                     self.load_pretrained(save_model_dir,)

#             if self.use_university_pretrain:
#                 print("use university pretrain...")
#                 assert self.university_pretrain_path != None, "use university pretrain,but there is no university pretrain path"
#                 parameters = np.load(self.university_pretrain_path) # (g_n,c_s,g_s)
#                 parameters = torch.from_numpy(parameters)
#                 self.universal_context.from_pretrain(parameters,requires_grad = False)


#             ##  3 get optimizer
#             if optimizer_name == "Adam":
#                 optimizer = optim.Adam(self.parameters(),lr,weight_decay = weight_decay)
#             elif optimizer_name == "AdamW":
#                 optimizer = optim.AdamW(self.parameters(),lr,weight_decay = weight_decay)
#             else:
#                 optimizer = optim.Adam(self.parameters(),lr,weight_decay = weight_decay)
#             lr_scheduler = optim.lr_scheduler.ReduceLROnPlateau(
#                 optimizer = optimizer, 
#                 mode = "min", 
#                 factor = factor, 
#                 patience = patience
#             )

#             ## init trainng log
#             if first_trainning:
#                 best_loss = float("inf")
#                 last_epoch = 0
#             else:
#                 checkpoint = torch.load(check_point_path, map_location=self.device)
#                 optimizer.load_state_dict(checkpoint["optimizer"])
#                 lr_scheduler.load_state_dict(checkpoint["lr_scheduler"])
#                 best_loss = checkpoint["loss"]
#                 last_epoch = checkpoint["epoch"] + 1

#             try:
#                 for epoch in range(last_epoch,total_epoch):
#                     print(f"Learning rate: {optimizer.param_groups[0]['lr']}")
#                     train_loss = self.train_one_epoch(epoch,train_dataloader, optimizer,clip_max_norm,log_path)
#                     test_loss = self.test_epoch(epoch,test_dataloader,log_path)
#                     loss = train_loss + test_loss
#                     lr_scheduler.step(loss)
#                     is_best = loss < best_loss
#                     best_loss = min(loss, best_loss)
#                     torch.save(                
#                         {
#                             "epoch": epoch,
#                             "loss": loss,
#                             "optimizer": None,
#                             "lr_scheduler": None
#                         },
#                         check_point_path
#                     )

#                     if epoch % save_checkpoint_step == 0:
#                         os.makedirs(save_model_dir_ + "/" + "chaeckpoint-"+str(epoch))
#                         torch.save(
#                             {
#                                 "epoch": epoch,
#                                 "loss": loss,
#                                 "optimizer": optimizer.state_dict(),
#                                 "lr_scheduler": lr_scheduler.state_dict()
#                             },
#                             save_model_dir_ + "/" + "chaeckpoint-"+str(epoch)+"/checkpoint.pth"
#                         )
#                     if is_best:
#                         self.save_pretrained(save_model_dir)

#             # interrupt trianning
#             except KeyboardInterrupt:
#                     torch.save(                
#                         {
#                             "epoch": epoch,
#                             "loss": loss,
#                             "optimizer": optimizer.state_dict(),
#                             "lr_scheduler": lr_scheduler.state_dict()
#                         },
#                         check_point_path
#                     )

#     def train_one_epoch(
#         self, epoch, train_dataloader, optimizer, clip_max_norm, log_path = None
#     ):
#         total_loss = AverageMeter()
#         reconstruction_loss = AverageMeter()
#         bpp_loss = AverageMeter()
#         codebook_loss = AverageMeter()
#         mask_loss = AverageMeter()
#         average_hit_rate = AverageMeter()
#         recall = AverageMeter()
#         self.train()
#         pbar = tqdm(train_dataloader,desc="Processing epoch "+str(epoch), unit="batch")
#         for batch_id, inputs in enumerate(pbar):
#             """ grad zeroing """
#             optimizer.zero_grad()

#             """ forward """
#             used_memory = 0 if self.device != "cuda" else torch.cuda.memory_allocated(torch.cuda.current_device()) / (1024 ** 3)  
#             output = self.forward(inputs)

#             """ calculate loss """
#             out_criterion = self.compute_loss(output)
#             after_used_memory = 0 if self.device != "cuda" else torch.cuda.memory_allocated(torch.cuda.current_device()) / (1024 ** 3) 
#             out_criterion["loss"].backward()

#             """ grad clip """
#             if clip_max_norm > 0:
#                 clip_gradient(optimizer,clip_max_norm)
            
#             """ modify parameters """
#             optimizer.step()

#             total_loss.update(out_criterion["loss"].item())
#             reconstruction_loss.update(out_criterion["reconstruction_loss"].item())
#             codebook_loss.update(out_criterion["codebook_loss"].item())
#             bpp_loss.update(out_criterion["bpp_loss"].item())
#             mask_loss.update(out_criterion["mask_loss"])
#             average_hit_rate.update(math.exp(-mask_loss.avg))
#             recall.update(out_criterion["recall"])

#             postfix_str = "total_loss: {:.4f}, reconstruction_loss: {:.4f}, codebook_loss:{:.4f}, bpp_loss:{:.4f}, average_hit_rate:{:.2f},recall:{:.2f},use_memory: {:.1f}G".format(
#                 math.sqrt(total_loss.avg), 
#                 math.sqrt(reconstruction_loss.avg),
#                 math.sqrt(codebook_loss.avg),
#                 bpp_loss.avg,
#                 average_hit_rate.avg,
#                 recall.avg,
#                 after_used_memory - used_memory
#             )
#             pbar.set_postfix_str(postfix_str)
#             pbar.update()
#         with open(log_path, "a") as file:
#             file.write(postfix_str+"\n")
#         return total_loss.avg

#     def test_epoch(
#         self,epoch, test_dataloader,trainning_log_path = None
#     ):
#         self.eval()
#         total_loss = AverageMeter()
#         reconstruction_loss = AverageMeter()
#         codebook_loss = AverageMeter()
#         bpp_loss = AverageMeter()
#         mask_loss = AverageMeter()
#         average_hit_rate = AverageMeter()
#         recall = AverageMeter()

#         with torch.no_grad():
#             for batch_id,inputs in enumerate(test_dataloader):
#                 output = self.forward(inputs)
#                 out_criterion = self.compute_loss(output)
#                 total_loss.update(out_criterion["loss"])
#                 reconstruction_loss.update(out_criterion["reconstruction_loss"])
#                 codebook_loss.update(out_criterion["codebook_loss"])
#                 bpp_loss.update(out_criterion["bpp_loss"])
#                 mask_loss.update(out_criterion["mask_loss"])
#                 average_hit_rate.update(math.exp(-mask_loss.avg))
#                 recall.update(out_criterion["recall"])
#         str = (        
#             f"Test epoch {epoch}:"
#             f" total_loss: {math.sqrt(total_loss.avg):.4f} |"
#             f" reconstruction_loss: {math.sqrt(reconstruction_loss.avg):.4f}|"
#             f" codebook_loss: {math.sqrt(codebook_loss.avg):.4f}|"
#             f" bpp_loss: {bpp_loss.avg:.4f} |"
#             f" average_hit_rate: {average_hit_rate.avg:.2f}|"
#             f" recall: {recall.avg:.2f} \n"
#             )
#         print(str)
#         with open(trainning_log_path, "a") as file:
#             file.write(str+"\n")
#         return total_loss.avg
    
#     def eval_model(
#         self,
#         epoch = None,
#         val_dataloader = None, 
#         log_path = None
#     ):
#         psnr = AverageMeter()
#         bpp = AverageMeter()
#         with torch.no_grad():
#             for batch_id,inputs in enumerate(val_dataloader):
#                 b, c, h, w = inputs["image"].shape
#                 output = self.forward(inputs)
#                 bpp.update(
#                     sum(
#                         (torch.log(likelihoods).sum() / (-math.log(2) * b * h * w))
#                         for likelihoods in output["likelihoods"].values()
#                     )
#                 )
#                 for i in range(b):
#                     psnr.update(calculate_psnr(output["reconstruction_image"][i].cpu() * 255, inputs["image"][i].cpu() * 255))
        
#         log_message = "Eval Epoch: {:d}, PSNR = {:.4f}, BPP = {:.2f}\n".format(epoch, psnr.avg, bpp.avg)
#         print(log_message)
#         if log_path != None:
#             with open(log_path, "a") as file:
#                 file.write(log_message+"\n")
#         return log_message

class ETICN(ModelCompressionBase):
    def __init__(
        self,
        image_channel,
        image_height,
        image_weight,
        patch_size,
        embedding_dim,
        window_size,
        head_num,
        shift_size,
        out_channel_m,
        out_channel_n,
        group_num,
        codebook_size,
        transfomer_head,
        transfomer_blocks,
        drop_prob = 0.1,
        stage = 2,
        lamda = None,
        sigma = 0.0001,
        beta = 0.0001,
        finetune_model_dir = None, 
        university_pretrain_path = None,
        device = "cuda"
    ):
        super().__init__(image_channel, image_height, image_weight, out_channel_m, out_channel_n, lamda, finetune_model_dir, device)
        self.sigma = sigma
        self.beta = beta
        self.university_pretrain_path = university_pretrain_path
        self.stage = stage
        self.patch_size = patch_size
        self.embed_dim = embedding_dim
        self.window_size = window_size
        self.head_num = head_num
        self.shift_size = shift_size
        self.group_num = group_num
        self.codebook_size = codebook_size
        self.transfomer_head = transfomer_head
        self.transfomer_blocks = transfomer_blocks
        self.drop_prob = drop_prob
        self.feather_shape = [
            embedding_dim*8,
            (int)(self.image_shape[1]/patch_size/8),
            (int)(self.image_shape[2]/patch_size/8)
        ]
        self.image_transform_encoder = Encoder(
            image_shape = self.image_shape,
            patch_size = self.patch_size,
            embed_dim = self.embed_dim,
            window_size = window_size,
            head_num = head_num,
            shift_size = shift_size,
            out_channel_m = self.out_channel_m
        ).to(self.device)

        self.image_transform_decoder = Decoder(
            image_shape = self.image_shape,
            patch_size = self.patch_size,
            embed_dim = embedding_dim,
            window_size = window_size,
            head_num=head_num,
            shift_size=shift_size,                                    
            out_channel_m= out_channel_m
        ).to(self.device)

        self.tedm = TEDM(
            in_c = self.image_shape[0], 
            embed_dim = embedding_dim
        ).to(self.device)

        self.hyperpriori_encoder = HyperprioriEncoder(
            feather_shape = [out_channel_m,(int)(self.image_shape[1]/16),(int)(self.image_shape[2]/16)],
            out_channel_m = out_channel_m,
            out_channel_n = out_channel_n
        ).to(self.device)

        self.side_context = nn.Sequential(
            deconv(out_channel_n, out_channel_m, kernel_size = 5,stride = 2),
            nn.LeakyReLU(inplace=True),
            deconv(out_channel_m, out_channel_m * 3 // 2,kernel_size = 5,stride = 2),
            nn.LeakyReLU(inplace=True),
            conv(out_channel_m * 3 // 2, out_channel_m * 2, kernel_size=3,stride = 1)
        ).to(self.device)


        self.universal_context = UniversalContext(
            out_channel_m = self.out_channel_m,
            codebook_size = self.codebook_size,
            group_num = self.group_num
        ).to(self.device)

        self.local_context = MaskedConv2d(
            in_channels = out_channel_m , 
            out_channels = 2 * out_channel_m, 
            kernel_size = 5, 
            padding = 2, 
            stride = 1
        ).to(self.device)

        self.global_context = GlobalContext(
            head = transfomer_head ,
            layers= transfomer_blocks,
            d_model_1 = out_channel_m,
            d_model_2 = out_channel_m * 2,
            drop_prob = drop_prob
        ).to(self.device)
        
        self.parm1 = nn.Sequential(
            nn.Conv2d(out_channel_m * 15 // 3,out_channel_m * 10 // 3, 1),
            nn.LeakyReLU(inplace=True),
            nn.Conv2d(out_channel_m * 10 // 3, out_channel_m * 8 // 3, 1),
            nn.LeakyReLU(inplace=True),
            nn.Conv2d(out_channel_m * 8 // 3, out_channel_m * 6 // 3, 1),
        ).to(self.device)

        self.parm2 = nn.Sequential(
            nn.Conv2d(out_channel_m * 15 // 3,out_channel_m * 10 // 3, 1),
            nn.LeakyReLU(inplace=True),
            nn.Conv2d(out_channel_m * 10 // 3, out_channel_m * 8 // 3, 1),
            nn.LeakyReLU(inplace=True),
            nn.Conv2d(out_channel_m * 8 // 3, out_channel_m * 6 // 3, 1),
        ).to(self.device)

    def forward(self,inputs):
            image = inputs["image"].to(self.device)
            """ get latent vector """
            y,mid_feather = self.image_transform_encoder(image)

            """ detect traffic element"""
            logits,mask = self.tedm(image,mid_feather)

            """ get side message """
            z = self.hyperpriori_encoder(y)
            z_hat, z_likelihoods = self.entropy_bottleneck(z)
            side_ctx = self.side_context(z_hat)

            y_hat = self.gaussian_conditional.quantize(
                y, "noise" if self.training else "dequantize"
            )

            """ get local message """
            local_ctx = self.local_context(y_hat)

            """ get global message """
            global_ctx = self.global_context(y_hat,local_ctx)

            """ get university message"""
            y_ = y * (1 - mask.unsqueeze(1).repeat(1, self.out_channel_m, 1, 1))
            universal_ctx,y_ba,code_index = self.universal_context(y_)
            y_ba = y_ba * (1 - mask.unsqueeze(1).repeat(1, self.out_channel_m, 1, 1))
            universal_ctx = universal_ctx * (1 - mask.unsqueeze(1).repeat(1, self.out_channel_m, 1, 1))

            """ parameters estimation"""
            gaussian_params1 = self.parm1(
                torch.concat((local_ctx,global_ctx,side_ctx),dim=1)
            )
            gaussian_params2 = self.parm2(
                torch.concat((local_ctx,universal_ctx,side_ctx),dim=1)
            )
            scales_hat1, means_hat1 = gaussian_params1.chunk(2, 1)
            scales_hat2, means_hat2 = gaussian_params2.chunk(2, 1)
            scales_hat = scales_hat2*(1 - mask.unsqueeze(1).repeat(1, self.out_channel_m, 1, 1)) + scales_hat1*mask.unsqueeze(1).repeat(1, self.out_channel_m, 1, 1)
            means_hat = means_hat2*(1 - mask.unsqueeze(1).repeat(1, self.out_channel_m, 1, 1)) + means_hat1*mask.unsqueeze(1).repeat(1, self.out_channel_m, 1, 1)


            _,y_likelihoods = self.gaussian_conditional(y,scales_hat,means_hat)
            """ inverse transformation"""
            x_hat = self.image_transform_decoder(y_hat)
            x_hat = torch.clamp(x_hat,0,1)

            """ output """
            """
                reconstruction_image: reconstruction image
                feather: transport feather to CGVQ
                zq: re-feather
                likelihoods: likelihoods of lather（y） and super prior latent（z）
                logits: logits score of mask
                mask: predict transport mask
            """
            output = {
                "image":inputs["image"].to(self.device),
                "reconstruction_image":x_hat,
                "feather":y_,
                "zq":y_ba,
                "likelihoods": {"y": y_likelihoods, "z": z_likelihoods},
                "logits":logits,
                "mask":mask,
                "latent":y,
                "labels":inputs["label"].long().to(self.device) if "label" in inputs else None
            }
            return output

    def trainning(            
            self,
            train_dataloader: DataLoader = None,
            test_dataloader: DataLoader = None,
            val_dataloader: DataLoader = None,
            optimizer_name:str = "Adam",
            weight_decay:float = 0,
            clip_max_norm:float = 0.5,
            factor:float = 0.3,
            patience:int = 15,
            lr:float = 1e-4,
            total_epoch:int = 1000,
            eval_interval:int = 10,
            save_model_dir:str = None
        ):
            if self.university_pretrain_path:
                print("use university pretrain...")
                assert self.university_pretrain_path != None, "use university pretrain,but there is no university pretrain path"
                parameters = np.load(self.university_pretrain_path) # (g_n,c_s,g_s)
                parameters = torch.from_numpy(parameters)
                self.universal_context.from_pretrain(parameters,requires_grad = False)
                
            super().trainning(train_dataloader, test_dataloader, val_dataloader, optimizer_name, weight_decay, clip_max_norm, factor, patience, lr, total_epoch, eval_interval, save_model_dir)

    def compute_loss(self, input):
        output = super().compute_loss(input)
        """ codebook loss """
        output["codebook_loss"] =  F.mse_loss(input["zq"].detach(), input["feather"]) + 0.25 * F.mse_loss(input["zq"], input["feather"].detach())

        """ mask loss """
        true_labels = input["labels"]
        pre_labels = F.interpolate(input["logits"], scale_factor=16, mode='bicubic', align_corners=True)
        output["mask_loss"] = F.cross_entropy(pre_labels, true_labels)
        
        output["total_loss"] =  output["total_loss"] + self.sigma * output["codebook_loss"] + self.beta * output["mask_loss"]
        return output
        
    def eval_model(
        self,
        val_dataloader = None, 
        log_path = None
    ):
        psnr = AverageMeter()
        bpp = AverageMeter()
        with torch.no_grad():
            for batch_id,inputs in enumerate(val_dataloader):
                b, c, h, w = inputs["image"].shape
                output = self.forward(inputs)
                bpp.update(
                    sum(
                        (torch.log(likelihoods).sum() / (-math.log(2) * b * h * w))
                        for likelihoods in output["likelihoods"].values()
                    )
                )
                for i in range(b):
                    psnr.update(calculate_psnr(output["reconstruction_image"][i].cpu() * 255, inputs["image"][i].cpu() * 255))
    
        log_message = "PSNR = {:.4f}, BPP = {:.2f}\n".format(psnr.avg, bpp.avg)
        print(log_message)
        if log_path != None:
            with open(log_path, "a") as file:
                file.write(log_message+"\n")
                
        output = {
            "log_message":log_message,
            "PSNR": psnr.avg,
            "bpp": bpp.avg
        }
        
        return output

class ETICNVBR(ModelCompressionBase):
    def __init__(
        self,
        image_channel,
        image_height,
        image_weight,
        patch_size,
        embedding_dim,
        window_size,
        head_num,
        shift_size,
        out_channel_m,
        out_channel_n,
        group_num,
        codebook_size,
        transfomer_head,
        transfomer_blocks,
        drop_prob = 0.1,
        stage = 2,
        lamda = None,
        sigma = 0.0001,
        beta = 0.0001,
        finetune_model_dir = None, 
        university_pretrain_path = None,
        device = "cuda"
    ):
        super().__init__(image_channel, image_height, image_weight, out_channel_m, out_channel_n, lamda, finetune_model_dir, device)
        self.sigma = sigma
        self.beta = beta
        self.university_pretrain_path = university_pretrain_path
        self.stage = stage
        self.patch_size = patch_size
        self.embed_dim = embedding_dim
        self.window_size = window_size
        self.head_num = head_num
        self.shift_size = shift_size
        self.group_num = group_num
        self.codebook_size = codebook_size
        self.transfomer_head = transfomer_head
        self.transfomer_blocks = transfomer_blocks
        self.drop_prob = drop_prob
        self.feather_shape = [
            embedding_dim*8,
            (int)(self.image_shape[1]/patch_size/8),
            (int)(self.image_shape[2]/patch_size/8)
        ]
        self.image_transform_encoder = Encoder(
            image_shape = self.image_shape,
            patch_size = self.patch_size,
            embed_dim = self.embed_dim,
            window_size = window_size,
            head_num = head_num,
            shift_size = shift_size,
            out_channel_m = self.out_channel_m
        ).to(self.device)
        
        self.param_pre = ParameterEstimation(
            latent_channel = self.out_channel_m, 
            latent_width = (int)(self.image_weight / 16), 
            latent_heigh = (int)(self.image_height / 16), 
        ).to(self.device)

        self.image_transform_decoder = Decoder(
            image_shape = self.image_shape,
            patch_size = self.patch_size,
            embed_dim = embedding_dim,
            window_size = window_size,
            head_num=head_num,
            shift_size=shift_size,                                    
            out_channel_m= out_channel_m
        ).to(self.device)

        self.tedm = TEDM(
            in_c = self.image_shape[0], 
            embed_dim = embedding_dim
        ).to(self.device)

        self.hyperpriori_encoder = HyperprioriEncoder(
            feather_shape = [out_channel_m,(int)(self.image_shape[1]/16),(int)(self.image_shape[2]/16)],
            out_channel_m = out_channel_m,
            out_channel_n = out_channel_n
        ).to(self.device)

        self.side_context = nn.Sequential(
            deconv(out_channel_n, out_channel_m, kernel_size = 5,stride = 2),
            nn.LeakyReLU(inplace=True),
            deconv(out_channel_m, out_channel_m * 3 // 2,kernel_size = 5,stride = 2),
            nn.LeakyReLU(inplace=True),
            conv(out_channel_m * 3 // 2, out_channel_m * 2, kernel_size=3,stride = 1)
        ).to(self.device)


        self.universal_context = UniversalContext(
            out_channel_m = self.out_channel_m,
            codebook_size = self.codebook_size,
            group_num = self.group_num
        ).to(self.device)

        self.local_context = MaskedConv2d(
            in_channels = out_channel_m , 
            out_channels = 2 * out_channel_m, 
            kernel_size = 5, 
            padding = 2, 
            stride = 1
        ).to(self.device)

        self.global_context = GlobalContext(
            head = transfomer_head ,
            layers= transfomer_blocks,
            d_model_1 = out_channel_m,
            d_model_2 = out_channel_m * 2,
            drop_prob = drop_prob
        ).to(self.device)
        
        self.parm1 = nn.Sequential(
            nn.Conv2d(out_channel_m * 15 // 3,out_channel_m * 10 // 3, 1),
            nn.LeakyReLU(inplace=True),
            nn.Conv2d(out_channel_m * 10 // 3, out_channel_m * 8 // 3, 1),
            nn.LeakyReLU(inplace=True),
            nn.Conv2d(out_channel_m * 8 // 3, out_channel_m * 6 // 3, 1),
        ).to(self.device)

        self.parm2 = nn.Sequential(
            nn.Conv2d(out_channel_m * 15 // 3,out_channel_m * 10 // 3, 1),
            nn.LeakyReLU(inplace=True),
            nn.Conv2d(out_channel_m * 10 // 3, out_channel_m * 8 // 3, 1),
            nn.LeakyReLU(inplace=True),
            nn.Conv2d(out_channel_m * 8 // 3, out_channel_m * 6 // 3, 1),
        ).to(self.device)
    
    def forward(self,inputs):
            image = inputs["image"].to(self.device)
            """ get latent vector """
            y, mid_feather = self.image_transform_encoder(image)
            
            """ detect traffic element"""
            logits, mask = self.tedm(image, mid_feather)
            
            """ get university message"""
            y_ = y * (1 - mask.unsqueeze(1).repeat(1, self.out_channel_m, 1, 1))
            universal_ctx, y_ba, code_index = self.universal_context(y_)
            y_ba = y_ba * (1 - mask.unsqueeze(1).repeat(1, self.out_channel_m, 1, 1))
            universal_ctx = universal_ctx * (1 - mask.unsqueeze(1).repeat(1, self.out_channel_m, 1, 1))
            
            lamdas = torch.tensor([0.0002, 0.0004, 0.0009, 0.0016, 0.0036, 0.0081]).to(self.device)
            args = []
            for lamda in lamdas:
                args.append(self.param_pre(y, lamda))

            y_trnas = []
            y_hats = []
            for arg in args:
                k, b = arg.chunk(2, 1)
                y_tran = y * k + b
                y_trnas.append(y_tran)
                y_hats.append(self.gaussian_conditional.quantize(
                    y_tran, "noise" if self.training else "dequantize"
                ))
                
            x_hats = [] 
            all_likelihoods = []
            
            for y_tran, y_hat in zip(y_trnas, y_hats):
                """ get side message """
                z = self.hyperpriori_encoder(y_tran)
                z_hat, z_likelihoods = self.entropy_bottleneck(z)
                side_ctx = self.side_context(z_hat)

                """ get local message """
                local_ctx = self.local_context(y_hat)

                """ get global message """
                global_ctx = self.global_context(y_hat, local_ctx)

                """ parameters estimation"""
                gaussian_params1 = self.parm1(
                    torch.concat((local_ctx, global_ctx, side_ctx),dim=1)
                )
                
                gaussian_params2 = self.parm2(
                    torch.concat((local_ctx, universal_ctx, side_ctx),dim=1)
                )
                
                scales_hat1, means_hat1 = gaussian_params1.chunk(2, 1)
                scales_hat2, means_hat2 = gaussian_params2.chunk(2, 1)
                scales_hat = scales_hat2*(1 - mask.unsqueeze(1).repeat(1, self.out_channel_m, 1, 1)) + scales_hat1*mask.unsqueeze(1).repeat(1, self.out_channel_m, 1, 1)
                means_hat = means_hat2*(1 - mask.unsqueeze(1).repeat(1, self.out_channel_m, 1, 1)) + means_hat1*mask.unsqueeze(1).repeat(1, self.out_channel_m, 1, 1)
                _,y_likelihoods = self.gaussian_conditional(y_tran, scales_hat, means_hat)
                all_likelihoods.append({
                    "y": y_likelihoods, "z": z_likelihoods
                })
                
                """ inverse transformation"""
                x_hat = self.image_transform_decoder(y_hat)
                x_hat = torch.clamp(x_hat, 0, 1)
                x_hats.append(x_hat)

            """ output """
            """
                reconstruction_image: reconstruction image
                feather: transport feather to CGVQ
                zq: re-feather
                likelihoods: likelihoods of lather（y） and super prior latent（z）
                logits: logits score of mask
                mask: predict transport mask
            """
            output = {
                "image":inputs["image"].to(self.device),
                "reconstruction_image":x_hats,
                "feather":y_,
                "zq":y_ba,
                "all_likelihoods": all_likelihoods,
                "logits":logits,
                "mask":mask,
                "latent":y,
                "labels":inputs["label"].long().to(self.device) if "label" in inputs else None
            }
            return output

    def compute_loss(self, input):
        N, _, H, W = input["image"].size()
        output = {}
        num_pixels = N * H * W

        lamdas = torch.tensor([0.0002, 0.0004, 0.0009, 0.0016, 0.0036, 0.0081]).to(self.device)
        bpp_losses = []
        reconstruction_losses = []
        for likelihoods in input["all_likelihoods"]:
            bpp_losses.append(
                sum(
                    (torch.log(likelihood).sum() / (-math.log(2) * num_pixels))
                    for likelihood in likelihoods.values()
                )
            )

        for x_hat in input["reconstruction_image"]:
            """ reconstruction loss """
            reconstruction_losses.append(F.mse_loss(x_hat, input["image"]))

        """ codebook loss """
        output["codebook_loss"] =  F.mse_loss(input["zq"].detach(), input["feather"]) + 0.25 * F.mse_loss(input["zq"], input["feather"].detach())

        """ mask loss """
        true_labels = input["labels"]
        pre_labels = F.interpolate(input["logits"], scale_factor=16, mode='bicubic', align_corners=True)
        output["mask_loss"] = F.cross_entropy(pre_labels, true_labels)

        # """ recall """
        # pre = torch.argmax(pre_labels, dim=1)
        # TP = torch.sum((pre == 0) & (true_labels == 0))
        # FN = torch.sum((pre == 1) & (true_labels == 0))
        # if TP + FN == 0:
        #     out["recall"] = 0.5
        # else:
        #     out["recall"] = TP.float() / (TP + FN).float()

        """ all loss """
        output["total_loss"] = torch.Tensor([0]).to(self.device)
        for index, lamda in enumerate(lamdas):
            output["total_loss"]  = output["total_loss"] + lamda * bpp_losses[index] + reconstruction_losses[index]
        output["total_loss"] =  output["total_loss"] + self.sigma * output["codebook_loss"] + self.beta * output["mask_loss"]
        return output

    def trainning(            
            self,
            train_dataloader: DataLoader = None,
            test_dataloader: DataLoader = None,
            val_dataloader: DataLoader = None,
            optimizer_name:str = "Adam",
            weight_decay:float = 0,
            clip_max_norm:float = 0.5,
            factor:float = 0.3,
            patience:int = 15,
            lr:float = 1e-4,
            total_epoch:int = 1000,
            eval_interval:int = 10,
            save_model_dir:str = None
        ):
            if self.university_pretrain_path:
                print("use university pretrain...")
                assert self.university_pretrain_path != None, "use university pretrain,but there is no university pretrain path"
                parameters = np.load(self.university_pretrain_path) # (g_n,c_s,g_s)
                parameters = torch.from_numpy(parameters)
                self.universal_context.from_pretrain(parameters,requires_grad = False)
                
            super().trainning(train_dataloader, test_dataloader, val_dataloader, optimizer_name, weight_decay, clip_max_norm, factor, patience, lr, total_epoch, eval_interval, save_model_dir)

    def eval_model(
        self,
        val_dataloader = None, 
        log_path = None
    ):
        # psnr = AverageMeter()
        # bpp = AverageMeter()
        # with torch.no_grad():
        #     for batch_id,inputs in enumerate(val_dataloader):
        #         b, c, h, w = inputs["image"].shape
        #         output = self.forward(inputs)
        #         bpp.update(
        #             sum(
        #                 (torch.log(likelihoods).sum() / (-math.log(2) * b * h * w))
        #                 for likelihoods in output["likelihoods"].values()
        #             )
        #         )
        #         for i in range(b):
        #             psnr.update(calculate_psnr(output["reconstruction_image"][i].cpu() * 255, inputs["image"][i].cpu() * 255))
    
        # log_message = "PSNR = {:.4f}, BPP = {:.2f}\n".format(psnr.avg, bpp.avg)
        # print(log_message)
        # if log_path != None:
        #     with open(log_path, "a") as file:
        #         file.write(log_message+"\n")
                
        # output = {
        #     "log_message":log_message,
        #     "PSNR": psnr.avg,
        #     "bpp": bpp.avg
        # }
        pass