import random
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
        b, _, _, _ = image.shape
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
 
