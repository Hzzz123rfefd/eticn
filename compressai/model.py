import random
from torch.utils.data import DataLoader
import torch
import torch.nn as nn
import math
import numpy as np

from compressai.ops import ste_round
from compressai.entropy_models import *
from compressai.utils import AverageMeter,clip_gradient
from compressai.modules import *
from compressai.layers import *
from compressai.base import *

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


            _,y_likelihoods = self.gaussian_conditional(y, scales_hat, means_hat)
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
                "labels":inputs["label"].long().to(self.device) if "label" in inputs else None,
                "lamda":self.lamda
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
        # print(f"mask_loss = {output['mask_loss']}, mask_loss = {output['mask_loss']}")
        return output
        
    # def eval_model(
    #     self,
    #     val_dataloader = None, 
    #     log_path = None
    # ):
    #     psnr = AverageMeter()
    #     bpp = AverageMeter()
    #     with torch.no_grad():
    #         for batch_id,inputs in enumerate(val_dataloader):
    #             b, c, h, w = inputs["image"].shape
    #             output = self.forward(inputs)
    #             bpp.update(
    #                 sum(
    #                     (torch.log(likelihoods).sum() / (-math.log(2) * b * h * w))
    #                     for likelihoods in output["likelihoods"].values()
    #                 )
    #             )
    #             for i in range(b):
    #                 psnr.update(calculate_psnr(output["reconstruction_image"][i].cpu() * 255, inputs["image"][i].cpu() * 255))
    
    #     log_message = "PSNR = {:.4f}, BPP = {:.2f}\n".format(psnr.avg, bpp.avg)
    #     print(log_message)
    #     if log_path != None:
    #         with open(log_path, "a") as file:
    #             file.write(log_message+"\n")
                
    #     output = {
    #         "log_message":log_message,
    #         "PSNR": psnr.avg,
    #         "bpp": bpp.avg
    #     }
        
    #     return output

class ETICNQVRF(ModelVBRCompressionBase):
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
        stage = 1,
        sigma = 0.0001,
        beta = 0.0001,
        university_pretrain_path = None,
        device = "cuda"
    ):
        super().__init__(image_channel, image_height, image_weight, out_channel_m, out_channel_n, stage, device)
        self.sigma = sigma
        self.beta = beta
        self.university_pretrain_path = university_pretrain_path
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

        self.Gain = torch.nn.Parameter(torch.tensor(
            [1.0000, 1.3944, 1.9293, 2.6874, 3.7268, 5.1801, 7.1957, 10.0000]), requires_grad=True)
        
    def forward(self, inputs, s = 1, is_train = True):
        image = inputs["image"].to(self.device)
        if is_train == True:
            if self.stage > 1:
                s = random.randint(0, self.levels - 1)  # choose random level from [0, levels-1]
                if s != 0:
                    scale = torch.max(self.Gain[s], torch.tensor(1e-4)) + 1e-9
                else:
                    s = 0
                    scale = self.Gain[s].detach().clone()
            else:
                s = self.levels - 1
                scale = self.Gain[0].detach()
        else:
            scale = self.Gain[s]
        rescale = 1.0 / scale.clone().detach()

        if self.stage <= 2:
            """ get latent vector """
            y, mid_feather = self.image_transform_encoder(image)
        
            """ detect traffic element"""
            logits, mask = self.tedm(image, mid_feather)
        
            """ get university message"""
            y_ = y * (1 - mask.unsqueeze(1).repeat(1, self.out_channel_m, 1, 1))
            universal_ctx, y_ba, code_index = self.universal_context(y_)
            y_ba = y_ba * (1 - mask.unsqueeze(1).repeat(1, self.out_channel_m, 1, 1))
            universal_ctx = universal_ctx * (1 - mask.unsqueeze(1).repeat(1, self.out_channel_m, 1, 1))
            
            """ get side message """
            z = self.hyperpriori_encoder(y)
            z_hat, z_likelihoods = self.entropy_bottleneck(z)
            side_ctx = self.side_context(z_hat)
        
            y_hat = self.gaussian_conditional.quantize(y * scale, "noise" if self.training else "dequantize") * rescale

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
            _,y_likelihoods = self.gaussian_conditional(y * scale - means_hat * scale, scales_hat * scale)
        
            """ inverse transformation"""
            x_hat = self.image_transform_decoder(y_hat)
            x_hat = torch.clamp(x_hat, 0, 1)

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
            "labels":inputs["label"].long().to(self.device) if "label" in inputs else None,
            "lamda": self.lmbda[s]
        }
        return output

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
        
class VICQVRF(ModelVBRCompressionBase):
    def __init__(
        self, 
        image_channel = 3,
        image_height = 512,
        image_weight = 512,
        out_channel_m= 192, 
        out_channel_n = 192,
        stage = 1,
        device = "cuda"
    ):
        super().__init__(image_channel, image_height, image_weight, out_channel_m, out_channel_n, stage, device)
        self.N = out_channel_n
        self.M = out_channel_m
        self.stage = stage

        self.g_a = nn.Sequential(
            conv(3, self.N, kernel_size=5, stride=2),
            GDN(self.N),
            conv(self.N, self.N, kernel_size=5, stride=2),
            GDN(self.N),
            conv(self.N, self.N, kernel_size=5, stride=2),
            GDN(self.N),
            conv(self.N, self.M, kernel_size=5, stride=2),
        )

        self.g_s = nn.Sequential(
            deconv(self.M, self.N, kernel_size=5, stride=2),
            GDN(self.N, inverse=True),
            deconv(self.N, self.N, kernel_size=5, stride=2),
            GDN(self.N, inverse=True),
            deconv(self.N, self.N, kernel_size=5, stride=2),
            GDN(self.N, inverse=True),
            deconv(self.N, 3, kernel_size=5, stride=2),
        )

        self.h_a = nn.Sequential(
            conv(self.M, self.N, stride=1, kernel_size=3),
            nn.ReLU(inplace=True),
            conv(self.N, self.N, stride=2, kernel_size=5),
            nn.ReLU(inplace=True),
            conv(self.N, self.N, stride=2, kernel_size=5),
        )

        self.h_s = nn.Sequential(
            deconv(self.N, self.M, stride=2, kernel_size=5),
            nn.ReLU(inplace=True),
            deconv(self.M, self.M * 3 // 2, stride=2, kernel_size=5),
            nn.ReLU(inplace=True),
            conv(self.M * 3 // 2, self.M * 2, stride=1, kernel_size=3),
        )


        self.entropy_parameters = nn.Sequential(
            nn.Conv2d(self.M * 12 // 3, self.M * 10 // 3, 1),
            nn.ReLU(inplace=True),
            nn.Conv2d(self.M * 10 // 3, self.M * 8 // 3, 1),
            nn.ReLU(inplace=True),
            nn.Conv2d(self.M * 8 // 3, self.M * 6 // 3, 1),
        )

        self.context_prediction = MaskedConv2d(
            self.M, 2 * self.M, kernel_size=5, padding=2, stride=1
        )

        self.quantizer = Quantizer()
        
        self.Gain = torch.nn.Parameter(torch.tensor(
            [1.0000, 1.3944, 1.9293, 2.6874, 3.7268, 5.1801, 7.1957, 10.0000]), requires_grad = True)

    @property
    def downsampling_factor(self) -> int:
        return 2 ** (4 + 2)

    def forward(self, inputs, s = 1, is_train = True):
        x = inputs["image"].to(self.device)
        if is_train == True:
            if self.stage > 1:
                s = random.randint(0, self.levels - 1)  # choose random level from [0, levels-1]
                if s != 0:
                    scale = torch.max(self.Gain[s], torch.tensor(1e-4)) + 1e-9
                else:
                    s = 0
                    scale = self.Gain[s].detach().clone()
            else:
                s = self.levels - 1
                scale = self.Gain[0].detach()
        else:
            scale = self.Gain[s]
        rescale = 1.0 / scale.clone().detach()

        if self.stage <= 2:
            # print("noise quant: True, ste quant:False, stage:{}, s = {}, scale = {}".format(self.stage, s, self.Gain[s]))
            y = self.g_a(x)
            z = self.h_a(y)
            z_hat, z_likelihoods = self.entropy_bottleneck(z)
            params = self.h_s(z_hat)

            y_hat = self.gaussian_conditional.quantize(y * scale, "noise" if self.training else "dequantize") * rescale
            ctx_params = self.context_prediction(y_hat)
            gaussian_params = self.entropy_parameters(
                torch.cat((params, ctx_params), dim=1)
            )
            scales_hat, means_hat = gaussian_params.chunk(2, 1)
            _, y_likelihoods = self.gaussian_conditional(y * scale - means_hat * scale, scales_hat * scale)
            x_hat = self.g_s(y_hat)
        else:
            print("noise quant: False, ste quant: True, stage:{}".format(self.stage))
            y = self.g_a(x)
            z = self.h_a(y)
            _, z_likelihoods = self.entropy_bottleneck(z)

            z_offset = self.entropy_bottleneck._get_medians()
            z_tmp = z - z_offset
            z_hat = ste_round(z_tmp) + z_offset

            params = self.h_s(z_hat)
            kernel_size = 5  # context prediction kernel size
            padding = (kernel_size - 1) // 2
            y_hat = F.pad(y, (padding, padding, padding, padding))
            y_hat, y_likelihoods = self._stequantization(y_hat, params, y.size(2), y.size(3), kernel_size, padding, scale, rescale)

            x_hat = self.g_s(y_hat)
        # x_hat = torch.clamp(x_hat, 0, 1)
        output = {
                "image":inputs["image"].to(self.device),
                "reconstruction_image":x_hat,
                "likelihoods": {"y": y_likelihoods, "z": z_likelihoods},
                "lamda": self.lmbda[s]
            }
        return output

    def eval_epoch(
        self,
        val_dataloader = None, 
        log_path = None
    ):
        psnrs = [AverageMeter() for i in self.lmbda]
        bpps = [AverageMeter() for i in self.lmbda]
        with torch.no_grad():
            for batch_id, inputs in enumerate(val_dataloader):
                b, c, h, w = inputs["image"].shape
                for s in range(self.levels):
                    output = self.forward(inputs = inputs, s = s, is_train = False)
                    
                    bpps[s].update(
                        sum(
                            (torch.log(likelihoods).sum() / (-math.log(2) * b * h * w))
                            for likelihoods in output["likelihoods"].values()
                        )
                    )
                    
                    for i in range(b):
                        psnrs[s].update(calculate_psnr(output["reconstruction_image"].cpu() * 255, inputs["image"].cpu() * 255))

        log_message = ""
        for index, lamda in enumerate(self.lmbda):
            log_message = log_message + f"lamda = {lamda}, s = {index}, scale: {self.Gain.data[index].cpu().numpy():0.4f},   stage {self.stage}, PSNR = {psnrs[index].avg},  BPP = {bpps[index].avg}\n"
            
        print(log_message)
        if log_path != None:
            with open(log_path, "a") as file:
                file.write(log_message+"\n")
                
        output = {
            "log_message":log_message,
            "PSNR": psnrs,
            "bpp": bpps
        }

    def _stequantization(self, y_hat, params, height, width, kernel_size, padding, scale, rescale):
        y_likelihoods = torch.zeros([y_hat.size(0), y_hat.size(1), height, width]).to(scale.device)
        # TODO: profile the calls to the bindings...
        masked_weight = self.context_prediction.weight * self.context_prediction.mask
        for h in range(height):
            for w in range(width):
                y_crop = y_hat[:, :, h: h + kernel_size, w: w + kernel_size].clone()
                ctx_p = F.conv2d(
                    y_crop,
                    masked_weight,
                    bias=self.context_prediction.bias,
                )

                # 1x1 conv for the entropy parameters prediction network, so
                # we only keep the elements in the "center"
                p = params[:, :, h: h + 1, w: w + 1]
                gaussian_params = self.entropy_parameters(torch.cat((p, ctx_p), dim=1))
                gaussian_params = gaussian_params.squeeze(3).squeeze(2)
                scales_hat, means_hat = gaussian_params.chunk(2, 1)
                y_crop = y_crop[:, :, padding, padding]
                _, y_likelihoods[:, :, h: h + 1, w: w + 1] = self.gaussian_conditional(
                    ((y_crop - means_hat) * scale).unsqueeze(2).unsqueeze(3),
                    (scales_hat * scale).unsqueeze(2).unsqueeze(3))
                y_q = self.quantizer.quantize((y_crop - means_hat.detach()) * scale,
                                        "ste") * rescale + means_hat.detach()
                y_hat[:, :, h + padding, w + padding] = y_q
        y_hat = F.pad(y_hat, (-padding, -padding, -padding, -padding))
        return y_hat, y_likelihoods
    
class VICVBR2(ModelVBRCompressionBase):

    def __init__(
        self, 
        image_channel = 3,
        image_height = 512,
        image_weight = 512,
        out_channel_m= 192, 
        out_channel_n = 192,
        stage = 1,
        device = "cuda"
    ):
        super().__init__(image_channel, image_height, image_weight, out_channel_m, out_channel_n, stage, device)
        self.N = out_channel_n
        self.M = out_channel_m

        self.g_a = nn.Sequential(
            conv(3, self.N, kernel_size=5, stride=2),
            GDN(self.N),
            conv(self.N, self.N, kernel_size=5, stride=2),
            GDN(self.N),
            conv(self.N, self.N, kernel_size=5, stride=2),
            GDN(self.N),
            conv(self.N, self.M, kernel_size=5, stride=2),
        )

        self.g_s = nn.Sequential(
            deconv(self.M, self.N, kernel_size=5, stride=2),
            GDN(self.N, inverse=True),
            deconv(self.N, self.N, kernel_size=5, stride=2),
            GDN(self.N, inverse=True),
            deconv(self.N, self.N, kernel_size=5, stride=2),
            GDN(self.N, inverse=True),
            deconv(self.N, 3, kernel_size=5, stride=2),
        )

        self.h_a = nn.Sequential(
            conv(self.M, self.N, stride=1, kernel_size=3),
            nn.LeakyReLU(inplace=True),
            conv(self.N, self.N, stride=2, kernel_size=5),
            nn.LeakyReLU(inplace=True),
            conv(self.N, self.N, stride=2, kernel_size=5),
        )

        self.h_s = nn.Sequential(
            deconv(self.N, self.M, stride=2, kernel_size=5),
            nn.LeakyReLU(inplace=True),
            deconv(self.M, self.M * 3 // 2, stride=2, kernel_size=5),
            nn.LeakyReLU(inplace=True),
            conv(self.M * 3 // 2, self.M * 2, stride=1, kernel_size=3),
        )


        self.entropy_parameters = nn.Sequential(
            nn.Conv2d(self.M * 12 // 3, self.M * 10 // 3, 1),
            nn.LeakyReLU(inplace=True),
            nn.Conv2d(self.M * 10 // 3, self.M * 8 // 3, 1),
            nn.LeakyReLU(inplace=True),
            nn.Conv2d(self.M * 8 // 3, self.M * 6 // 3, 1),
        )

        self.context_prediction = MaskedConv2d(
            self.M, 2 * self.M, kernel_size=5, padding=2, stride=1
        )

        self.gaussian_conditional = GaussianConditional(None)
        
        self.Gain = torch.nn.Parameter(torch.tensor(
             [[1.0000, 1.3944, 1.9293, 2.6874, 3.7268, 5.1801, 7.1957, 10.0000]] * self.M, dtype=torch.float32), requires_grad=True)

        self.param_pre = ParameterEstimation(
            latent_channel = self.out_channel_m, 
            latent_width = (int)(self.image_weight / 16), 
            latent_heigh = (int)(self.image_height / 16), 
        ).to(self.device)

    def forward(self, inputs, s = 1, is_train = True):
        x = inputs["image"].to(self.device)
        if is_train == True:
            if self.stage > 1:
                s = random.randint(0, self.levels - 1)  # choose random level from [0, levels-1]
                if s != 0:
                    scale = torch.max(self.Gain[:, s], torch.tensor(1e-4)) + 1e-9
                else:
                    s = 0
                    scale = self.Gain[:, s].detach().clone()
            else:
                s = self.levels - 1
                scale = self.Gain[:, 0].detach()
        else:
            scale = self.Gain[:, s]
        scale = scale.unsqueeze(0).unsqueeze(2).unsqueeze(3)
        rescale = 1.0 / scale.clone().detach()

        if self.stage <= 2:
            # print("noise quant: True, ste quant:False, stage:{}, s = {}, scale = {}".format(self.stage, s, self.Gain[s]))
            y = self.g_a(x)
            z = self.h_a(y)
            z_hat, z_likelihoods = self.entropy_bottleneck(z)
            params = self.h_s(z_hat)

            y_hat = self.gaussian_conditional.quantize(y * scale, "noise" if self.training else "dequantize") * rescale
            ctx_params = self.context_prediction(y_hat)
            gaussian_params = self.entropy_parameters(
                torch.cat((params, ctx_params), dim=1)
            )
            scales_hat, means_hat = gaussian_params.chunk(2, 1)
            _, y_likelihoods = self.gaussian_conditional(y * scale - means_hat * scale, scales_hat * scale)
            x_hat = self.g_s(y_hat)
        else:
            print("noise quant: False, ste quant: True, stage:{}".format(self.stage))
            y = self.g_a(x)
            z = self.h_a(y)
            _, z_likelihoods = self.entropy_bottleneck(z)

            z_offset = self.entropy_bottleneck._get_medians()
            z_tmp = z - z_offset
            z_hat = ste_round(z_tmp) + z_offset

            params = self.h_s(z_hat)
            kernel_size = 5  # context prediction kernel size
            padding = (kernel_size - 1) // 2
            y_hat = F.pad(y, (padding, padding, padding, padding))
            y_hat, y_likelihoods = self._stequantization(y_hat, params, y.size(2), y.size(3), kernel_size, padding, scale, rescale)

            x_hat = self.g_s(y_hat)
        # x_hat = torch.clamp(x_hat, 0, 1)
        output = {
                "image":inputs["image"].to(self.device),
                "reconstruction_image":x_hat,
                "likelihoods": {"y": y_likelihoods, "z": z_likelihoods},
                "lamda": self.lmbda[s]
            }
        return output

class VICVBR3(ModelVBRCompressionBase):
    
    def __init__(
        self, 
        image_channel = 3,
        image_height = 512,
        image_weight = 512,
        out_channel_m= 192, 
        out_channel_n = 192,
        stage = 1,
        device = "cuda"
    ):
        super().__init__(image_channel, image_height, image_weight, out_channel_m, out_channel_n, stage, device)
        self.N = out_channel_n
        self.M = out_channel_m

        self.g_a = nn.Sequential(
            conv(3, self.N, kernel_size=5, stride=2),
            GDN(self.N),
            conv(self.N, self.N, kernel_size=5, stride=2),
            GDN(self.N),
            conv(self.N, self.N, kernel_size=5, stride=2),
            GDN(self.N),
            conv(self.N, self.M, kernel_size=5, stride=2),
        )

        self.g_s = nn.Sequential(
            deconv(self.M, self.N, kernel_size=5, stride=2),
            GDN(self.N, inverse=True),
            deconv(self.N, self.N, kernel_size=5, stride=2),
            GDN(self.N, inverse=True),
            deconv(self.N, self.N, kernel_size=5, stride=2),
            GDN(self.N, inverse=True),
            deconv(self.N, 3, kernel_size=5, stride=2),
        )

        self.h_a = nn.Sequential(
            conv(self.M, self.N, stride=1, kernel_size=3),
            nn.LeakyReLU(inplace=True),
            conv(self.N, self.N, stride=2, kernel_size=5),
            nn.LeakyReLU(inplace=True),
            conv(self.N, self.N, stride=2, kernel_size=5),
        )

        self.h_s = nn.Sequential(
            deconv(self.N, self.M, stride=2, kernel_size=5),
            nn.LeakyReLU(inplace=True),
            deconv(self.M, self.M * 3 // 2, stride=2, kernel_size=5),
            nn.LeakyReLU(inplace=True),
            conv(self.M * 3 // 2, self.M * 2, stride=1, kernel_size=3),
        )

        self.ddpm = ModelDDPM(
            width = image_weight,
            height = image_height,
            channel = image_channel
        )

        self.entropy_parameters = nn.Sequential(
            nn.Conv2d(self.M * 12 // 3, self.M * 10 // 3, 1),
            nn.LeakyReLU(inplace=True),
            nn.Conv2d(self.M * 10 // 3, self.M * 8 // 3, 1),
            nn.LeakyReLU(inplace=True),
            nn.Conv2d(self.M * 8 // 3, self.M * 6 // 3, 1),
        )

        self.context_prediction = MaskedConv2d(
            self.M, 2 * self.M, kernel_size=5, padding=2, stride=1
        )

        self.gaussian_conditional = GaussianConditional(None)
        
        self.Gain = torch.nn.Parameter(torch.tensor(
             [[1.0000, 1.3944, 1.9293, 2.6874, 3.7268, 5.1801, 7.1957, 10.0000]] * self.M, dtype=torch.float32), requires_grad=True)

    def forward(self, inputs, s = 1, is_train = True):
        x = inputs["image"].to(self.device)
        if is_train == True:
            if self.stage > 1:
                s = random.randint(0, self.levels - 1)  # choose random level from [0, levels-1]
                if s != 0:
                    scale = torch.max(self.Gain[:, s], torch.tensor(1e-4)) + 1e-9
                else:
                    s = 0
                    scale = self.Gain[:, s].detach().clone()
            else:
                s = self.levels - 1
                scale = self.Gain[:, 0].detach()
        else:
            scale = self.Gain[:, s].detach()
        scale = scale.unsqueeze(0).unsqueeze(2).unsqueeze(3)
        rescale = 1.0 / scale.clone().detach()

        if self.stage <= 2:
            # print("noise quant: True, ste quant:False, stage:{}, s = {}, scale = {}".format(self.stage, s, self.Gain[s]))
            y = self.g_a(x)
            z = self.h_a(y)
            z_hat, z_likelihoods = self.entropy_bottleneck(z)
            params = self.h_s(z_hat)

            y_hat = self.gaussian_conditional.quantize(y * scale, "noise" if self.training else "dequantize") * rescale
            ctx_params = self.context_prediction(y_hat)
            gaussian_params = self.entropy_parameters(
                torch.cat((params, ctx_params), dim=1)
            )
            scales_hat, means_hat = gaussian_params.chunk(2, 1)
            _, y_likelihoods = self.gaussian_conditional(y * scale - means_hat * scale, scales_hat * scale)
            if is_train:
                x_hat, alpha_t, z_t, noise = self.ddpm(x, y_hat)
            else:
                b, _, _, _ = y_hat.shape
                x_hat = self.ddpm.sample(
                    sample_num = b,
                    context = y_hat
                )
                alpha_t = None
                z_t = None
                noise = None
            
        output = {
                "image":inputs["image"].to(self.device),
                "reconstruction_image":x_hat,
                "likelihoods": {"y": y_likelihoods, "z": z_likelihoods},
                "lamda": self.lmbda[s],
                "alpha_t":alpha_t,
                "z_t": z_t,
                "noise": noise
            }
        return output
    
    def compute_loss(self, input):
        lamda = input["lamda"]
        alpha_t = input["alpha_t"]
        z_t = input["z_t"]
        noise = input["noise"]
        N, _, H, W = input["image"].size()
        output = {}
        num_pixels = N * H * W

        output["bpp_loss"] = sum(
            (torch.log(likelihoods).sum() / (-math.log(2) * num_pixels))
            for likelihoods in input["likelihoods"].values()
        )

        if alpha_t != None:
            output["true_reconstruction_loss"] = F.mse_loss(input["reconstruction_image"], input["image"])* 255**2
            # output["reconstruction_loss"] = torch.Tensor([0]).to(self.device)
            # for i in range(N):
            #     output["reconstruction_loss"] = output["reconstruction_loss"] + (alpha_t[i] / (1 - alpha_t[i])) * F.mse_loss(input["reconstruction_image"][i, :, :, :], input["image"][i, :, :, :]) * 255**2
            # output["reconstruction_loss"] = output["reconstruction_loss"] / N
            output["reconstruction_loss"] = F.mse_loss(z_t, noise)* 255**2
        else:
            output["reconstruction_loss"] = F.mse_loss(input["reconstruction_image"], input["image"])* 255**2
        output["total_loss"] = output["bpp_loss"] + lamda * output["reconstruction_loss"]
        return output
    
    def test_epoch(self, epoch, test_dataloader, log_path = None):
        total_loss = AverageMeter()
        self.eval()
        self.to(self.device)
        with torch.no_grad():
            for s in range(self.levels):
                for batch_id, inputs in enumerate(test_dataloader):
                    """ forward """
                    output = self.forward(inputs, s = s, is_train = True)

                    """ calculate loss """
                    out_criterion = self.compute_loss(output)
                    total_loss.update(out_criterion["total_loss"].item())

            str = "Test Epoch: {:d}, total_loss: {:.4f}".format(
                epoch,
                total_loss.avg, 
            )
        print(str)
        with open(log_path, "a") as file:
            file.write(str+"\n")
        return total_loss.avg
    
    
    # def test_epoch(self, epoch, test_dataloader, log_path = None):
    #     total_loss = AverageMeter()
    #     self.eval()
    #     self.to(self.device)
        
    #     with torch.no_grad():
    #         for s in range(self.levels):
    #             for batch_id, inputs in enumerate(test_dataloader):
    #                 """ forward """
    #                 output = self.forward(inputs, s = s, is_train = False)

    #                 """ calculate loss """
    #                 out_criterion = self.compute_loss(output)
    #                 total_loss.update(out_criterion["total_loss"].item())

    #         str = "Test Epoch: {:d}, total_loss: {:.4f}".format(
    #             epoch,
    #             total_loss.avg, 
    #         )
    #     print(str)
    #     with open(log_path, "a") as file:
    #         file.write(str+"\n")
    #     return total_loss.avg


    # def eval_epoch(
    #     self,
    #     val_dataloader = None, 
    #     log_path = None
    # ):
    #     psnrs = [AverageMeter() for i in self.lmbda]
    #     bpps = [AverageMeter() for i in self.lmbda]
    #     with torch.no_grad():
    #         for batch_id, inputs in enumerate(val_dataloader):
    #             b, c, h, w = inputs["image"].shape
    #             for s in range(self.levels):
    #                 output = self.forward(inputs = inputs, s = s, is_train = False)
                    
    #                 bpps[s].update(
    #                     sum(
    #                         (torch.log(likelihoods).sum() / (-math.log(2) * b * h * w))
    #                         for likelihoods in output["likelihoods"].values()
    #                     )
    #                 )
                    
    #                 for i in range(b):
    #                     psnrs[s].update(calculate_psnr(output["reconstruction_image"].cpu() * 255, inputs["image"].cpu() * 255))

    #     log_message = ""
    #     for index, lamda in enumerate(self.lmbda):
    #         log_message = log_message + f"lamda = {lamda}, s = {index}, scale: {self.Gain.data[0, index].cpu().numpy():0.4f},   stage {self.stage}, PSNR = {psnrs[index].avg},  BPP = {bpps[index].avg}\n"
            
    #     print(log_message)
    #     if log_path != None:
    #         with open(log_path, "a") as file:
    #             file.write(log_message+"\n")
                
    #     output = {
    #         "log_message":log_message,
    #         "PSNR": psnrs,
    #         "bpp": bpps
    #     }

