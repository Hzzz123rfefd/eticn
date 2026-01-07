import torch
from compressai.entropy_models import *
from compressai.modules import *
from compressai.layers import *
from compressai.base import *

class VIC_CQVR(ModelCQVRBase):
    def __init__(self, image_channel, image_height, image_weight, time_dim, out_channel_m, out_channel_n, stage, device):
        super().__init__(image_channel, image_height, image_weight, time_dim, out_channel_m, out_channel_n, stage, device)
        self.N = self.out_channel_n
        self.M = self.out_channel_m
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
            conv(self.M * 3 // 2, self.M, stride=1, kernel_size=3),
        )

    def forward(self, inputs, s = 1, is_train = True):
        image = inputs["image"].to(self.device)
        b, _, _, _ = image.shape
        scale, rescale, s = self.get_scale(s, is_train)
        
        if self.stage <= 3:
            y = self.g_a(image)
            z = self.h_a(y)
            z_hat, z_likelihoods = self.entropy_bottleneck(z)
            scales_hat = self.h_s(z_hat)
            y_hat, noisy, predict_noisy = self.y_hat_enhance(y, scale, rescale, s, b)
            _, y_likelihoods = self.gaussian_conditional(y * scale, scales_hat * scale)
            x_hat = self.g_s(y_hat)
        x_hat = torch.clamp(x_hat, 0, 1)
        output = {
            "image":inputs["image"].to(self.device),
            "reconstruction_image":x_hat,
            "likelihoods": {"y": y_likelihoods, "z": z_likelihoods},
            "noisy": noisy,
            "predict_noisy": predict_noisy,
            "lamda": self.lmbda[s]
        }
        return output

class VAIC_CQVR(ModelCQVRBase):
    def __init__(self, image_channel, image_height, image_weight, time_dim, out_channel_m, out_channel_n, stage, device):
        super().__init__(image_channel, image_height, image_weight, time_dim, out_channel_m, out_channel_n, stage, device)
        self.N = self.out_channel_n
        self.M = self.out_channel_m
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
            nn.LeakyReLU(inplace=True),
            nn.Conv2d(self.M * 10 // 3, self.M * 8 // 3, 1),
            nn.LeakyReLU(inplace=True),
            nn.Conv2d(self.M * 8 // 3, self.M * 6 // 3, 1),
        )
        self.context_prediction = MaskedConv2d(
            self.M, 2 * self.M, kernel_size=5, padding=2, stride=1
        )

    def forward(self, inputs, s = 1, is_train = True):
        image = inputs["image"].to(self.device)
        b, _, _, _ = image.shape
        scale, rescale, s = self.get_scale(s, is_train)
        
        if self.stage <= 3:
            y = self.g_a(image)
            z = self.h_a(y)
            z_hat, z_likelihoods = self.entropy_bottleneck(z)
            side_ctx = self.h_s(z_hat)
            y_hat, noisy, predict_noisy = self.y_hat_enhance(y, scale, rescale, s, b)
            local_ctx = self.context_prediction(y_hat)
            gaussian_params = self.entropy_parameters(
                torch.cat((side_ctx, local_ctx), dim=1)
            )
            scales_hat, means_hat = gaussian_params.chunk(2, 1)
            _, y_likelihoods = self.gaussian_conditional(y * scale - means_hat * scale, scales_hat * scale)
            x_hat = self.g_s(y_hat)
        x_hat = torch.clamp(x_hat, 0, 1)
        output = {
            "image":inputs["image"].to(self.device),
            "reconstruction_image":x_hat,
            "likelihoods": {"y": y_likelihoods, "z": z_likelihoods},
            "noisy": noisy,
            "predict_noisy": predict_noisy,
            "lamda": self.lmbda[s]
        }
        return output

class STF_CQVR(ModelCQVRBase):
    def __init__(self, image_channel, image_height, image_weight, patch_size, embedding_dim, time_dim, out_channel_m, out_channel_n, stage, device):
        super().__init__(image_channel, image_height, image_weight, time_dim, out_channel_m, out_channel_n, stage, device)
        self.patch_size = patch_size
        self.embed_dim = embedding_dim
        self.feather_shape = [embedding_dim*8,
                                            (int)(self.image_shape[1]/patch_size/8),
                                            (int)(self.image_shape[2]/patch_size/8)]
        self.image_transform_encoder = Encoder(image_shape = self.image_shape,
                                                                            patch_size = patch_size,
                                                                            embed_dim = embedding_dim,
                                                                            window_size = 4,
                                                                            head_num = 1,
                                                                            shift_size = 0,
                                                                            out_channel_m= out_channel_m
                                                            )
        self.image_transform_decoder = Decoder(image_shape = self.image_shape,
                                                                            patch_size = patch_size,
                                                                            embed_dim = embedding_dim,
                                                                            window_size = 4,
                                                                            head_num = 1,
                                                                            shift_size = 0,
                                                                            out_channel_m= out_channel_m
                                                            )
        self.hyperpriori_encoder = HyperprioriEncoder(feather_shape = [out_channel_m,(int)(self.image_shape[1]/16),(int)(self.image_shape[2]/16)],
                                                                                    out_channel_m = out_channel_m,
                                                                                    out_channel_n = out_channel_n)
        self.side_context = nn.Sequential(
            deconv(out_channel_n, out_channel_m, kernel_size = 5,stride = 2),
            nn.LeakyReLU(inplace=True),
            deconv(out_channel_m, out_channel_m * 3 // 2,kernel_size = 5,stride = 2),
            nn.LeakyReLU(inplace=True),
            conv(out_channel_m * 3 // 2, out_channel_m * 2, kernel_size=3,stride = 1)
        ).to(self.device)
        
        self.local_context = MaskedConv2d(
            in_channels = out_channel_m , 
            out_channels = 2 * out_channel_m, 
            kernel_size = 5, 
            padding = 2, 
            stride = 1
        ).to(self.device)

        self.parm = nn.Sequential(
            nn.Conv2d(out_channel_m * 12 // 3,out_channel_m * 10 // 3, 1),
            nn.LeakyReLU(inplace=True),
            nn.Conv2d(out_channel_m * 10 // 3, out_channel_m * 8 // 3, 1),
            nn.LeakyReLU(inplace=True),
            nn.Conv2d(out_channel_m * 8 // 3, out_channel_m * 6 // 3, 1),
        ).to(self.device)

    def forward(self, inputs, s = 1, is_train = True):
        image = inputs["image"].to(self.device)
        b, _, _, _ = image.shape
        scale, rescale, s = self.get_scale(s, is_train)
        
        if self.stage <= 3:
            """ forward transformation """
            y, mid_feather = self.image_transform_encoder(image)
            """ super prior forward transformation """
            z = self.hyperpriori_encoder(y)
            """ quantization and likelihood estimation of z"""
            z_hat, z_likelihoods = self.entropy_bottleneck(z)
            side_ctx = self.side_context(z_hat)
            y_hat, noisy, predict_noisy = self.y_hat_enhance(y, scale, rescale, s, b)
            local_ctx = self.local_context(y_hat)
            gaussian_params = self.parm(
                torch.concat((local_ctx, side_ctx),dim=1)
            )
            scales_hat, means_hat = gaussian_params.chunk(2, 1)
            _, y_likelihoods = self.gaussian_conditional(y * scale - means_hat * scale, scales_hat * scale)
            x_hat = self.image_transform_decoder(y_hat)
        x_hat = torch.clamp(x_hat, 0, 1)
        output = {
            "image":inputs["image"].to(self.device),
            "reconstruction_image":x_hat,
            "likelihoods": {"y": y_likelihoods, "z": z_likelihoods},
            "noisy": noisy,
            "predict_noisy": predict_noisy,
            "lamda": self.lmbda[s]
        }
        return output
 
class GRIC_CQVR(ModelCQVRBase):
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
        time_dim, 
        out_channel_m,
        out_channel_n,
        transfomer_head,
        transfomer_blocks,
        drop_prob = 0.1,
        stage = 1,
        device = "cuda"
    ):
        super().__init__(image_channel, image_height, image_weight, time_dim, out_channel_m, out_channel_n, stage, device)
        self.patch_size = patch_size
        self.embed_dim = embedding_dim
        self.window_size = window_size
        self.head_num = head_num
        self.shift_size = shift_size
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
    
    def forward(self, inputs, s = 1, is_train = True):
        image = inputs["image"].to(self.device)
        b, _, _, _ = image.shape
        scale, rescale, s = self.get_scale(s, is_train)
        
        if self.stage <= 3:
            image = inputs["image"].to(self.device)
            """ get latent vector """
            y, mid_feather = self.image_transform_encoder(image)

            """ get side message """
            z = self.hyperpriori_encoder(y)
            z_hat, z_likelihoods = self.entropy_bottleneck(z)
            side_ctx = self.side_context(z_hat)

            """ get local message """
            y_hat, noisy, predict_noisy = self.y_hat_enhance(y, scale, rescale, s, b)
            local_ctx = self.local_context(y_hat)
            
            """ get global message """
            global_ctx = self.global_context(y_hat,local_ctx)

            """ parameters estimation"""
            gaussian_params1 = self.parm1(
                torch.concat((local_ctx,global_ctx,side_ctx),dim=1)
            )
            scales_hat, means_hat = gaussian_params1.chunk(2, 1)
            _, y_likelihoods = self.gaussian_conditional(y * scale - means_hat * scale, scales_hat * scale)
            """ inverse transformation"""
            x_hat = self.image_transform_decoder(y_hat)
        x_hat = torch.clamp(x_hat,0,1)
        output = {
            "image":inputs["image"].to(self.device),
            "reconstruction_image":x_hat,
            "likelihoods": {"y": y_likelihoods, "z": z_likelihoods},
            "noisy": noisy,
            "predict_noisy": predict_noisy,
            "lamda": self.lmbda[s]
        }
        return output

class ETICN_CQVR(ModelCQVRBase):
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
        time_dim,
        drop_prob = 0.1,
        stage = None,
        sigma = 0.0001,
        beta = 0.0001,
        finetune_model_dir = None, 
        university_pretrain_path = None,
        device = "cuda"
    ):
        super().__init__(image_channel, image_height, image_weight, time_dim, out_channel_m, out_channel_n, stage, finetune_model_dir, device)
        self.sigma = sigma
        self.beta = beta
        self.university_pretrain_path = university_pretrain_path
        self.group_num = group_num
        self.codebook_size = codebook_size
        self.patch_size = patch_size
        self.embed_dim = embedding_dim
        self.window_size = window_size
        self.head_num = head_num
        self.shift_size = shift_size
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
    
    def forward(self, inputs, s = 1, is_train = True):
        image = inputs["image"].to(self.device)
        b, _, _, _ = image.shape
        scale, rescale, s = self.get_scale(s, is_train)
        
        if self.stage <= 3:
            image = inputs["image"].to(self.device)
            """ get latent vector """
            y, mid_feather = self.image_transform_encoder(image)

            """ detect traffic element"""
            logits,mask = self.tedm(image,mid_feather)

            """ get side message """
            z = self.hyperpriori_encoder(y)
            z_hat, z_likelihoods = self.entropy_bottleneck(z)
            side_ctx = self.side_context(z_hat)

            """ get local message """
            y_hat, noisy, predict_noisy = self.y_hat_enhance(y, scale, rescale, s, b)
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
            "noisy": noisy,
            "predict_noisy": predict_noisy,
            "lamda": self.lmbda[s]
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
    
class VIC_QVRF(ModelQVRFBase):
    def __init__(self, image_channel, image_height, image_weight, out_channel_m, out_channel_n, stage, device):
        super().__init__(image_channel, image_height, image_weight, out_channel_m, out_channel_n, stage, device)
        self.N = self.out_channel_n
        self.M = self.out_channel_m
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
            conv(self.M * 3 // 2, self.M, stride=1, kernel_size=3),
        )

    def forward(self, inputs, s = 1, is_train = True):
        image = inputs["image"].to(self.device)
        scale, rescale, s = self.get_scale(s, is_train)
        
        if self.stage <= 2:
            y = self.g_a(image)
            z = self.h_a(y)
            z_hat, z_likelihoods = self.entropy_bottleneck(z)
            scales_hat = self.h_s(z_hat)
            y_hat, y_likelihoods = self.gaussian_conditional(y * scale, scales_hat * scale)
            y_hat = y_hat * rescale
            x_hat = self.g_s(y_hat)
        x_hat = torch.clamp(x_hat, 0, 1)
        output = {
            "image":inputs["image"].to(self.device),
            "reconstruction_image":x_hat,
            "likelihoods": {"y": y_likelihoods, "z": z_likelihoods},
            "lamda": self.lmbda[s]
        }
        return output
    
class VAIC_QVRF(ModelQVRFBase):
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

    def forward(self, inputs, s = 1, is_train = True):
        x = inputs["image"].to(self.device)
        scale, rescale, s = self.get_scale(s, is_train)

        if self.stage <= 2:
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
        x_hat = torch.clamp(x_hat, 0, 1)
        output = {
                "image":inputs["image"].to(self.device),
                "reconstruction_image":x_hat,
                "likelihoods": {"y": y_likelihoods, "z": z_likelihoods},
                "lamda": self.lmbda[s]
            }
        return output

class STF_QVRF(ModelQVRFBase):
    def __init__(self, image_channel, image_height, image_weight, patch_size, embedding_dim, out_channel_m, out_channel_n, stage, device):
        super().__init__(image_channel, image_height, image_weight, out_channel_m, out_channel_n, stage, device)
        self.patch_size = patch_size
        self.embed_dim = embedding_dim
        self.feather_shape = [embedding_dim*8,
                                            (int)(self.image_shape[1]/patch_size/8),
                                            (int)(self.image_shape[2]/patch_size/8)]
        self.image_transform_encoder = Encoder(image_shape = self.image_shape,
                                                                            patch_size = patch_size,
                                                                            embed_dim = embedding_dim,
                                                                            window_size = 4,
                                                                            head_num = 1,
                                                                            shift_size = 0,
                                                                            out_channel_m= out_channel_m
                                                            )
        self.image_transform_decoder = Decoder(image_shape = self.image_shape,
                                                                            patch_size = patch_size,
                                                                            embed_dim = embedding_dim,
                                                                            window_size = 4,
                                                                            head_num = 1,
                                                                            shift_size = 0,
                                                                            out_channel_m= out_channel_m
                                                            )
        self.hyperpriori_encoder = HyperprioriEncoder(feather_shape = [out_channel_m,(int)(self.image_shape[1]/16),(int)(self.image_shape[2]/16)],
                                                                                    out_channel_m = out_channel_m,
                                                                                    out_channel_n = out_channel_n)
        self.hyperpriori_decoder = HyperprioriDecoder(feather_shape = [out_channel_m,(int)(self.image_shape[1]/16),(int)(self.image_shape[2]/16)],
                                                                                    out_channel_m = out_channel_m,
                                                                                    out_channel_n = out_channel_n)

    def forward(self, inputs, s = 1, is_train = True):
        image = inputs["image"].to(self.device)
        scale, rescale, s = self.get_scale(s, is_train)
        
        if self.stage <= 2:
            """ forward transformation """
            y, mid_feather = self.image_transform_encoder(image)
            """ super prior forward transformation """
            z = self.hyperpriori_encoder(y)
            """ quantization and likelihood estimation of z"""
            z_hat, z_likelihoods = self.entropy_bottleneck(z)
            """ lather feature variance"""
            scales_hat = self.hyperpriori_decoder(z_hat)

            y_hat = self.gaussian_conditional.quantize(y * scale, "noise" if self.training else "dequantize") * rescale
            """ quantization and likelihood estimation of y"""
            _, y_likelihoods = self.gaussian_conditional(y * scale , scales_hat * scale)
            """ reverse transformation """
            x_hat = self.image_transform_decoder(y_hat)
        x_hat = torch.clamp(x_hat, 0, 1)
        output = {
            "image":inputs["image"].to(self.device),
            "reconstruction_image":x_hat,
            "likelihoods": {"y": y_likelihoods, "z": z_likelihoods},
            "lamda":self.lmbda[s]
        }
        return output
 
class GRIC_QVRF(ModelQVRFBase):
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
        transfomer_head,
        transfomer_blocks,
        drop_prob = 0.1,
        stage = 1,
        device = "cuda"
    ):
        super().__init__(image_channel, image_height, image_weight, out_channel_m, out_channel_n, stage, device)
        self.patch_size = patch_size
        self.embed_dim = embedding_dim
        self.window_size = window_size
        self.head_num = head_num
        self.shift_size = shift_size
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

    def forward(self, inputs, s = 1, is_train = True):
        image = inputs["image"].to(self.device)
        scale, rescale, s = self.get_scale(s, is_train)
        
        if self.stage <= 3:
            image = inputs["image"].to(self.device)
            """ get latent vector """
            y, mid_feather = self.image_transform_encoder(image)

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
                torch.concat((local_ctx, global_ctx, side_ctx), dim=1)
            )
            scales_hat, means_hat = gaussian_params1.chunk(2, 1)
            _, y_likelihoods = self.gaussian_conditional(y * scale - means_hat * scale, scales_hat * scale)
            """ inverse transformation"""
            x_hat = self.image_transform_decoder(y_hat)
        x_hat = torch.clamp(x_hat,0,1)
        output = {
            "image":inputs["image"].to(self.device),
            "reconstruction_image":x_hat,
            "likelihoods": {"y": y_likelihoods, "z": z_likelihoods},
            "lamda": self.lmbda[s]
        }
        return output
    
class ETICN_QVRF(ModelQVRFBase):
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
        stage = None,
        sigma = 0.0001,
        beta = 0.0001,
        finetune_model_dir = None, 
        university_pretrain_path = None,
        device = "cuda"
    ):
        super().__init__(image_channel, image_height, image_weight, out_channel_m, out_channel_n, stage, finetune_model_dir, device)
        self.sigma = sigma
        self.beta = beta
        self.university_pretrain_path = university_pretrain_path
        self.group_num = group_num
        self.codebook_size = codebook_size
        self.patch_size = patch_size
        self.embed_dim = embedding_dim
        self.window_size = window_size
        self.head_num = head_num
        self.shift_size = shift_size
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

        self.local_context = MaskedConv2d(
            in_channels = out_channel_m , 
            out_channels = 2 * out_channel_m, 
            kernel_size = 5, 
            padding = 2, 
            stride = 1
        ).to(self.device)

        self.universal_context = UniversalContext(
            out_channel_m = self.out_channel_m,
            codebook_size = self.codebook_size,
            group_num = self.group_num
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

    def forward(self, inputs, s = 1, is_train = True):
        image = inputs["image"].to(self.device)
        scale, rescale, s = self.get_scale(s, is_train)
        
        if self.stage <= 3:
            image = inputs["image"].to(self.device)
            """ get latent vector """
            y, mid_feather = self.image_transform_encoder(image)
            
            """ detect traffic element"""
            logits,mask = self.tedm(image,mid_feather)

            """ get side message """
            z = self.hyperpriori_encoder(y)
            z_hat, z_likelihoods = self.entropy_bottleneck(z)
            side_ctx = self.side_context(z_hat)

            y_hat = self.gaussian_conditional.quantize(y * scale, "noise" if self.training else "dequantize") * rescale

            """ get local message """
            local_ctx = self.local_context(y_hat)
            
            """ get global message """
            global_ctx = self.global_context(y_hat, local_ctx)

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
    
class ETICN_VGVRF(ModelVGVRFBase):
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
        stage = None,
        sigma = 0.0001,
        beta = 0.0001,
        finetune_model_dir = None, 
        university_pretrain_path = None,
        device = "cuda"
    ):
        super().__init__(image_channel, image_height, image_weight, out_channel_m, out_channel_n, finetune_model_dir, device)
        self.sigma = sigma
        self.beta = beta
        self.university_pretrain_path = university_pretrain_path
        self.group_num = group_num
        self.codebook_size = codebook_size
        self.patch_size = patch_size
        self.embed_dim = embedding_dim
        self.window_size = window_size
        self.head_num = head_num
        self.shift_size = shift_size
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

        self.local_context = MaskedConv2d(
            in_channels = out_channel_m , 
            out_channels = 2 * out_channel_m, 
            kernel_size = 5, 
            padding = 2, 
            stride = 1
        ).to(self.device)

        self.universal_context = UniversalContext(
            out_channel_m = self.out_channel_m,
            codebook_size = self.codebook_size,
            group_num = self.group_num
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

    def forward(self, inputs, s = 1, is_train = True):
        image = inputs["image"].to(self.device)
        if is_train == True:
            s = random.randint(0, self.levels - 1)  # choose random level from [0, levels-1]
            if s != 0:
                scale = torch.max(self.Gain[:, s], torch.tensor(1e-4)) + 1e-9
            else:
                s = 0
                scale = self.Gain[:, s].detach().clone()
        else:
            scale = self.Gain[:, s]
        scale = scale.unsqueeze(0).unsqueeze(2).unsqueeze(3)
        rescale = 1.0 / scale.clone().detach()
        """ get latent vector """
        y, mid_feather = self.image_transform_encoder(image)
        
        """ detect traffic element"""
        logits,mask = self.tedm(image,mid_feather)
        
        y_bar = y * scale

        """ get side message """
        z = self.hyperpriori_encoder(y_bar)
        z_hat, z_likelihoods = self.entropy_bottleneck(z)
        side_ctx = self.side_context(z_hat)

        y_hat = self.gaussian_conditional.quantize(y_bar, "noise" if self.training else "dequantize")

        """ get local message """
        local_ctx = self.local_context(y_hat)
        
        """ get global message """
        global_ctx = self.global_context(y_hat, local_ctx)

        """ get university message"""
        y_ = y * (1 - mask.unsqueeze(1).repeat(1, self.out_channel_m, 1, 1))
        universal_ctx,y_ba,code_index = self.universal_context(y_)
        y_ba = y_ba * (1 - mask.unsqueeze(1).repeat(1, self.out_channel_m, 1, 1))
        universal_ctx = universal_ctx * (1 - mask.unsqueeze(1).repeat(1, self.out_channel_m, 1, 1))

        """ parameters estimation"""
        gaussian_params1 = self.parm1(
            torch.concat((local_ctx, global_ctx,side_ctx),dim=1)
        )
        gaussian_params2 = self.parm2(
            torch.concat((local_ctx, universal_ctx,side_ctx),dim=1)
        )
        scales_hat1, means_hat1 = gaussian_params1.chunk(2, 1)
        scales_hat2, means_hat2 = gaussian_params2.chunk(2, 1)
        scales_hat = scales_hat2*(1 - mask.unsqueeze(1).repeat(1, self.out_channel_m, 1, 1)) + scales_hat1*mask.unsqueeze(1).repeat(1, self.out_channel_m, 1, 1)
        means_hat = means_hat2*(1 - mask.unsqueeze(1).repeat(1, self.out_channel_m, 1, 1)) + means_hat1*mask.unsqueeze(1).repeat(1, self.out_channel_m, 1, 1)


        _,y_likelihoods = self.gaussian_conditional(y_bar, scales_hat, means_hat)
        """ inverse transformation"""
        x_hat = self.image_transform_decoder(y_hat * rescale)
        x_hat = torch.clamp(x_hat, 0, 1)
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