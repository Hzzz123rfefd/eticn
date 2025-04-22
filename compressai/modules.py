import torch.nn as nn
from einops import rearrange 

from compressai.layers import *
from compressai.utils import *


class Quantizer():
    def quantize(self, inputs, quantize_type="noise"):
        if quantize_type == "noise":
            half = float(0.5)
            noise = torch.empty_like(inputs).uniform_(-half, half)
            inputs = inputs + noise
            return inputs
        elif quantize_type == "ste":
            return torch.round(inputs) - inputs.detach() + inputs
        else:
            return torch.round(inputs)

class Encoder(nn.Module):
    def __init__(self, image_shape, patch_size, embed_dim, window_size, head_num, shift_size, out_channel_m):
        super(Encoder, self).__init__()
        self.image_shape = image_shape
        self.patch_size = patch_size
        self.embed_dim = embed_dim
        

        self.patchEmbed = PatchEmbed(
            patch_size=patch_size, 
            in_chans=image_shape[0], 
            embed_dim=embed_dim, 
            norm_layer=None
        )

        # 循环构建3个attention层  
        self.swinTransfomerBlockLayers = nn.ModuleList()
        self.patchMergerLayers = nn.ModuleList()
        channel = embed_dim
        resolution = [(int)(image_shape[1]/patch_size),(int)(image_shape[2]/patch_size)]
        for i in range(3):  
            # 创建注意力层并添加到列表中
            if(i == 2):
                swinTransfomerBlock2Layers = nn.ModuleList()
                for j in range(3):
                    swinTransfomerBlock = SwinTransformerBlock2(dim=channel, input_resolution=resolution, num_heads=head_num, window_size=window_size, shift_size=shift_size)
                    swinTransfomerBlock2Layers.append(swinTransfomerBlock)
                self.swinTransfomerBlockLayers.append(swinTransfomerBlock2Layers)
            else:
                swinTransfomerBlock = SwinTransformerBlock2(dim=channel, input_resolution=resolution, num_heads=head_num, window_size=window_size , shift_size=shift_size)
                self.swinTransfomerBlockLayers.append(swinTransfomerBlock)
            patchMerge = PatchMerging(input_resolution = resolution, dim = channel, norm_layer=nn.LayerNorm)
            self.patchMergerLayers.append(patchMerge)
            channel = channel*2
            resolution = [(int)((x+1) / 2) for x in resolution]
        # 最后一层attention
        self.lastSwinTransfomerBlockLayer = SwinTransformerBlock2(dim=channel, input_resolution=resolution, num_heads=head_num, window_size=window_size,shift_size=shift_size)
        self.last_out = nn.Conv2d(embed_dim*8,out_channel_m,1,1)

    def forward(self,x):  
        b,c,h,w = x.shape
        # 1 词嵌入
        mid_feather = []
        x = self.patchEmbed(x)    # (b,c,h,w)
        mid_feather.append(x)
        # 2 窗口注意力
        x = x.permute(0,2,3,1).contiguous().view(b,-1,self.embed_dim)   #x:[b,w*h,c]
        for i in range(3):  
            if(i == 2):
                for j in range(3):
                    x = self.swinTransfomerBlockLayers[i][j](x) #x:[b,w*h,c]
            else:
                x = self.swinTransfomerBlockLayers[i](x) #x:[b,w*h,c]
            x = self.patchMergerLayers[i](x)
            b,t,c = x.shape
            mid_feather.append(x.contiguous().view(b,(int)(self.image_shape[1]/self.patch_size/(2**(i+1))),
                                                     (int)(self.image_shape[2]/self.patch_size/(2**(i+1))),
                                                      c).permute(0,3,1,2))
        # 3 最后一层窗口注意力
        x = self.lastSwinTransfomerBlockLayer(x)
        x = x.contiguous().view(b,(int)(self.image_shape[1]/self.patch_size/8),
                     (int)(self.image_shape[2]/self.patch_size/8),self.embed_dim*8).permute(0,3,1,2) #[b,c,h,w]
        x = self.last_out(x)
        return x,mid_feather

"""
    init:
        image_shape(3,H,W): shape of image
        patch_size:linear_embedding arg
        embed_dim:linear_embedding arg


    input:
        feather(B,embed_dim*4,H/patch_size/8,W/patch_size/8): encoder feather of image

    output:
        x(B,3,H,W): reconstruction image
"""
class Decoder(nn.Module):
    def __init__(self,image_shape,patch_size,embed_dim,window_size,head_num,shift_size,out_channel_m):
        super(Decoder, self).__init__()
        self.image_shape = image_shape
        self.patch_size = patch_size
        self.embed_dim = embed_dim
        self.end_conv = nn.Sequential(nn.Conv2d(embed_dim, embed_dim * patch_size ** 2, kernel_size=5, stride=1, padding=2),
                                      nn.PixelShuffle(patch_size),
                                      nn.Conv2d(embed_dim, 3, kernel_size=3, stride=1, padding=1),
                                      )
        channel = embed_dim*8
        resolution = [(int)(image_shape[1]/patch_size/8),(int)(image_shape[2]/patch_size/8)]
        #第一个attention
        self.first_in = nn.Conv2d(out_channel_m,embed_dim*8,1,1)
        self.firstSwinTransfomerBlockLayer = SwinTransformerBlock2(dim=channel, input_resolution=resolution, num_heads=head_num, window_size=window_size,shift_size=shift_size)
        #循环构建SwinTransformerBlock
        self.swinTransfomerBlockLayers = nn.ModuleList()
        self.patchMergerLayers = nn.ModuleList()
        for i in range(3):  
            patchMerge = PatchExpanding(input_resolution = resolution, dim = channel, norm_layer=nn.LayerNorm)
            self.patchMergerLayers.append(patchMerge)
            channel = (int)(channel/2)
            resolution = [(int)(2*x) for x in resolution]
            if(i == 0):
                swinTransfomerBlock2Layers = nn.ModuleList()
                for j in range(3):
                    swinTransfomerBlock = SwinTransformerBlock2(dim=channel, input_resolution=resolution, num_heads=head_num, window_size=window_size,shift_size=shift_size)
                    swinTransfomerBlock2Layers.append(swinTransfomerBlock)
                self.swinTransfomerBlockLayers.append(swinTransfomerBlock2Layers)
            else:
                swinTransfomerBlock = SwinTransformerBlock2(dim=channel, input_resolution=resolution, num_heads=head_num, window_size=window_size,shift_size=shift_size)
                self.swinTransfomerBlockLayers.append(swinTransfomerBlock)

    def forward(self,x):  # x:[B,C,H,W]
        # 1 第一层窗口注意力
        x = self.first_in(x)
        b,c,h,w = x.shape
        x = x.permute(0,2,3,1).view(b,-1,c)   #x:[b,w*h,c]
        x = self.firstSwinTransfomerBlockLayer(x)
        # 2 窗口注意力
        for i in range(3):  
            x = self.patchMergerLayers[i](x)
            if(i == 0):
                for j in range(3):
                    x = self.swinTransfomerBlockLayers[i][j](x) #x:[b,w*h,c]
            else:
                x = self.swinTransfomerBlockLayers[i](x) #x:[b,w*h,c]
        # 3 反向词嵌入
        x = x.view(b,(int)(self.image_shape[1]/self.patch_size),(int)(self.image_shape[2]/self.patch_size),self.embed_dim).permute(0,3,1,2) #[b,c,h,w]
        x = self.end_conv(x)
        return x
    
class RoundSTE(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x):
        return torch.round(x)
    
    @staticmethod
    def backward(ctx, grad_output):
        return grad_output

class TEDM(nn.Module):
    def __init__(self, in_c , embed_dim):
        super(TEDM, self).__init__()

        self.first_layer = nn.Sequential(
            DoubleConv(in_c, embed_dim),
            DownSample(embed_dim, embed_dim)
        )

        self.mid_layers = nn.ModuleList()
        self.res_layers = nn.ModuleList()
        for i in range(3):
            in_channels = embed_dim * (2 ** (i + 1)) 
            out_channels = embed_dim * (2 ** (i + 1))
            self.res_layers.append(nn.Conv2d(in_channels, out_channels, (3, 3), (1, 1), 1))
            self.mid_layers.append(nn.Sequential(
                DoubleConv(in_channels, out_channels),
                DownSample(out_channels, out_channels)))


        self.round_stf = RoundSTE.apply
        self.output = nn.Sequential(
            nn.Conv2d(embed_dim * 8, 2, 3, 1, 1),
        )
        
        self.round_stf = RoundSTE.apply
    def forward(self, x, mid_feather):
        x = self.first_layer(x)
        for i in range(3):
            x = torch.cat((x,mid_feather[i]),dim = 1) + self.res_layers[i](torch.cat((x,mid_feather[i]),dim = 1))
            x = self.mid_layers[i](x)
        logits = self.output(x)
        mask = torch.argmax(logits, dim=1)
        return logits,mask

class HyperprioriEncoder(nn.Module):
    def __init__(self,feather_shape,out_channel_m,out_channel_n):
        super(HyperprioriEncoder, self).__init__()
        self.feather_shape = feather_shape
        self.h_a = nn.Sequential(
            conv(out_channel_m, out_channel_n, stride=1, kernel_size=3),
            nn.LeakyReLU(inplace=True),
            conv(out_channel_n, out_channel_n),
            nn.LeakyReLU(inplace=True),
            conv(out_channel_n, out_channel_n),
        )

    def forward(self,x):
        return self.h_a(torch.abs(x))

class HyperprioriDecoder(nn.Module):
    def __init__(self,feather_shape,out_channel_m,out_channel_n):
        super(HyperprioriDecoder, self).__init__()
        self.feather_shape = feather_shape
        self.h_s = nn.Sequential(
            deconv(out_channel_n, out_channel_n),
            nn.ReLU(inplace=True),
            deconv(out_channel_n, out_channel_n),
            nn.ReLU(inplace=True),
            conv(out_channel_n, out_channel_m, stride=1, kernel_size=3),
            nn.ReLU(inplace=True),
        )

    def forward(self,x):
        return F.interpolate(self.h_s(x), size=(self.feather_shape[1], self.feather_shape[2]), mode='bilinear', align_corners=False)
    
class GlobalContext(nn.Module):
    def __init__(self,head,layers,drop_prob,d_model_1,d_model_2):
        super(GlobalContext, self).__init__()
        self.transformer_encoder = TransfomerEncoder(d_model = d_model_2, 
                                                    ffn_hidden = 2 * d_model_2, 
                                                    n_head = head, 
                                                    n_layers = layers, 
                                                    drop_prob = drop_prob)
        self.transformer_decoder = TransfomerDecoder(d_model_1 = d_model_1, 
                                                     d_model_2=d_model_2,
                                                    ffn_hidden = 2 * d_model_1, 
                                                    n_head = head, 
                                                    n_layers = layers, 
                                                    drop_prob = drop_prob)

    def create_decoder_init_status_and_mask(self,y):
        # add begin to y
        b,t,f = y.shape
        begin = torch.zeros(b, 1, f).to(y.device)
        y_hat = torch.cat((begin, y), dim=1)
        # create mask to y
        b,t,f = y_hat.shape
        mask = torch.ones(b,t,t).to(y.device)
        for i in range(t):
            mask[:,i,i+1:] = 0
        return y_hat,mask

    def forward(self,y2,local_context):
        b,c,h,w = local_context.shape
        local_context = local_context.reshape(b,c,h*w).permute(0,2,1)
        pos_h = positionalencoding1d(b,h,c).repeat(1,w,1)
        pos_w = positionalencoding1d(b,w,c).repeat(1,h,1)
        pos = (pos_h + pos_w).to(local_context.device)
        pos.requires_grad = False
        hidden_feather = self.transformer_encoder(local_context + pos)

        b,c,h,w = y2.shape
        y2 = y2.reshape(b,c,h*w).permute(0,2,1)
        y2_hat,mask = self.create_decoder_init_status_and_mask(y2)
        pos_h = positionalencoding1d(b,h,c).repeat(1,w,1)
        pos_w = positionalencoding1d(b,w,c).repeat(1,h,1)
        pos = (pos_h + pos_w)
        begin_pos = torch.zeros(b,1,c).float()
        begin_pos.requires_grad = False
        pos = torch.cat((begin_pos,pos),dim=1).to(y2_hat.device)
        pos.requires_grad = False

        global_context = self.transformer_decoder(y2_hat + pos, hidden_feather, trg_mask = mask)
        global_context = global_context[:,0:h*w,:].permute(0,2,1).reshape(b,c,h,w)
        return global_context
    
    def get_hidden_feather(self,local_context):
        b,c,h,w = local_context.shape
        local_context = local_context.reshape(b,c,h*w).permute(0,2,1)
        hidden_feather = self.transformer_encoder(local_context)
        return hidden_feather
    
class UniversalContext(nn.Module):
    def __init__(self,out_channel_m,codebook_size,group_num):
        super(UniversalContext, self).__init__()
        self.group_vector_quantizer = ChannelGroupVectorQuantizer(
            embedding_dim = out_channel_m,
            group_num = group_num,
            codebook_size = codebook_size
        )
    def forward(self,y):
        # mapping y to discrete space
        code_index = self.group_vector_quantizer.to_index(y)

        # extract from discrete space
        y_ba = self.group_vector_quantizer.to_feather(code_index)

        # back propagation
        universal_ctx = y + (y_ba - y).detach()   
        
        return universal_ctx,y_ba,code_index.unsqueeze(1)
    
    def from_pretrain(self,parameters,requires_grad = False):
        device = next(self.parameters()).device
        index = 0
        for vq in self.group_vector_quantizer.vector_quantization_groups:
            vq.embedding.weight.data.copy_(parameters[index,:,:].to(device))
            vq.embedding.weight.requires_grad = requires_grad
            index = index + 1


class ParameterEstimation(nn.Module):
    def __init__(
        self, 
        latent_channel = 192, 
        latent_width = 32, 
        latent_heigh = 32, 
    ):
        super().__init__()
        self.latent_channel = latent_channel
        self.latent_width = latent_width
        self.latent_heigh = latent_heigh
        self.quality_emb1 = EmbedFC(input_dim = 1, emb_dim = self.latent_channel)
        self.quality_emb2 = EmbedFC(input_dim = 1, emb_dim = self.latent_channel * 2)
        
        self.conv1 = nn.Conv2d(in_channels = self.latent_channel, out_channels = 2 * self.latent_channel, kernel_size = 3, stride = 1, padding = 1)
        self.conv2 = nn.Conv2d(in_channels = self.latent_channel * 2, out_channels = 4 * self.latent_channel, kernel_size = 3, stride = 1, padding = 1)
        self.conv3 = nn.Conv2d(in_channels = self.latent_channel * 4, out_channels = 2 * self.latent_channel, kernel_size = 3, stride = 1, padding = 1)
        self.silu = nn.SiLU()
        
    def forward(self, latent, lamda):
        lamda_emb1 = self.quality_emb1(lamda).view(-1, self.latent_channel, 1, 1)
        lamda_emb2 = self.quality_emb2(lamda).view(-1, self.latent_channel * 2, 1, 1)
        latent = self.silu(self.conv1(latent * lamda_emb1))
        latent = self.silu(self.conv2(latent * lamda_emb2))
        args = self.conv3(latent)
        return args


class EmbedFC(nn.Module):
    def __init__(self, input_dim, emb_dim):
        super(EmbedFC, self).__init__()
        '''
        generic one layer FC NN for embedding things  
        '''
        self.input_dim = input_dim
        layers = [
            nn.Linear(input_dim, emb_dim),
            nn.GELU(),
            nn.Linear(emb_dim, emb_dim),
        ]
        self.model = nn.Sequential(*layers)

    def forward(self, x):
        x = x.view(-1, self.input_dim)
        return self.model(x)
    
class Gain(nn.Module):
    def __init__(self, channel_m = 192, levels = 8):
        self.channel_m = channel_m
        self.levels = levels
        super(Gain, self).__init__()
        self.transform_matrix = nn.Parameter(torch.randn(self.channel_m, self.levels))  

    def forward(self, x):
        weight = self.transform_matrix.view(1, self.channel_m, self.levels, 1, 1)  
        x_transformed = x.unsqueeze(2) * weight  
        x_transformed = x_transformed.permute(0, 2, 1, 3, 4)  
        return x_transformed  
    
class LowerBoundFunction(torch.autograd.Function):
    """Autograd function for the `LowerBound` operator."""

    @staticmethod
    def forward(ctx, x, bound):
        ctx.save_for_backward(x, bound)
        return lower_bound_fwd(x, bound)

    @staticmethod
    def backward(ctx, grad_output):
        x, bound = ctx.saved_tensors
        return lower_bound_bwd(x, bound, grad_output)
    
class LowerBound(nn.Module):
    """Lower bound operator, computes `torch.max(x, bound)` with a custom
    gradient.

    The derivative is replaced by the identity function when `x` is moved
    towards the `bound`, otherwise the gradient is kept to zero.
    """

    bound: Tensor

    def __init__(self, bound: float):
        super().__init__()
        self.register_buffer("bound", torch.Tensor([float(bound)]))

    @torch.jit.unused
    def lower_bound(self, x):
        return LowerBoundFunction.apply(x, self.bound)

    def forward(self, x):
        if torch.jit.is_scripting():
            return torch.max(x, self.bound)
        return self.lower_bound(x)
    
    
class NonNegativeParametrizer(nn.Module):
    """
    Non negative reparametrization.

    Used for stability during training.
    """

    pedestal: Tensor

    def __init__(self, minimum: float = 0, reparam_offset: float = 2**-18):
        super().__init__()

        self.minimum = float(minimum)
        self.reparam_offset = float(reparam_offset)

        pedestal = self.reparam_offset**2
        self.register_buffer("pedestal", torch.Tensor([pedestal]))
        bound = (self.minimum + self.reparam_offset**2) ** 0.5
        self.lower_bound = LowerBound(bound)

    def init(self, x: Tensor) -> Tensor:
        return torch.sqrt(torch.max(x + self.pedestal, self.pedestal))

    def forward(self, x: Tensor) -> Tensor:
        out = self.lower_bound(x)
        out = out**2 - self.pedestal
        return out

class GDN(nn.Module):
    r"""Generalized Divisive Normalization layer.

    Introduced in `"Density Modeling of Images Using a Generalized Normalization
    Transformation" <https://arxiv.org/abs/1511.06281>`_,
    by Balle Johannes, Valero Laparra, and Eero P. Simoncelli, (2016).

    .. math::

       y[i] = \frac{x[i]}{\sqrt{\beta[i] + \sum_j(\gamma[j, i] * x[j]^2)}}

    """

    def __init__(
        self,
        in_channels: int,
        inverse: bool = False,
        beta_min: float = 1e-6,
        gamma_init: float = 0.1,
    ):
        super().__init__()

        beta_min = float(beta_min)
        gamma_init = float(gamma_init)
        self.inverse = bool(inverse)

        self.beta_reparam = NonNegativeParametrizer(minimum=beta_min)
        beta = torch.ones(in_channels)
        beta = self.beta_reparam.init(beta)
        self.beta = nn.Parameter(beta)

        self.gamma_reparam = NonNegativeParametrizer()
        gamma = gamma_init * torch.eye(in_channels)
        gamma = self.gamma_reparam.init(gamma)
        self.gamma = nn.Parameter(gamma)

    def forward(self, x: Tensor) -> Tensor:
        _, C, _, _ = x.size()

        beta = self.beta_reparam(self.beta)
        gamma = self.gamma_reparam(self.gamma)
        gamma = gamma.reshape(C, C, 1, 1)
        norm = F.conv2d(x**2, gamma, beta)

        if self.inverse:
            norm = torch.sqrt(norm)
        else:
            norm = torch.rsqrt(norm)

        out = x * norm

        return out
    
class DoubleConv(nn.Module):
    def __init__(self, in_c, out_c):
        super(DoubleConv, self).__init__()
        self.conv = nn.Sequential(
            torch.nn.Conv2d(in_channels=in_c, out_channels=out_c, kernel_size=(3, 3), stride=(1, 1), padding=1),
            torch.nn.BatchNorm2d(out_c),
            torch.nn.ReLU(inplace=True),
            torch.nn.Conv2d(out_c, out_c, (3, 3), (1, 1), 1),
            torch.nn.BatchNorm2d(out_c),
            torch.nn.ReLU(inplace=True)
        )

    def forward(self, x):
        return self.conv(x)

class DownSample(nn.Module):
    def __init__(self, in_c, out_c):
        super(DownSample, self).__init__()
        self.down_sample = nn.Sequential(
            nn.Conv2d(in_c, out_c, kernel_size=(1, 1), stride=(2, 2)),
            nn.BatchNorm2d(out_c),
            nn.ReLU(inplace=True)
        )

    def forward(self, x):
        return self.down_sample(x)

class TimeEmbed(nn.Module):
    def __init__(self, time_dim, out_dim):
        super(TimeEmbed, self).__init__()
        self.time_embed_layer = nn.Sequential(
            nn.SiLU(),
            nn.Linear(time_dim,out_dim)
        )
    def forward(self,x):
        x = self.time_embed_layer(x)
        return rearrange(x,"b c -> b c 1 1")

class ContextEmbed(nn.Module):
    def __init__(self, up_scale, in_dim, out_dim):
        super(ContextEmbed, self).__init__()
        self.up_scale = up_scale
        self.in_dim = in_dim
        self.out_dim = out_dim
        if self.up_scale == 1:
            self.net = nn.Conv2d(in_dim, out_dim, kernel_size=1)
        else:
            self.net = nn.ConvTranspose2d(
                in_channels = in_dim,  
                out_channels = out_dim, 
                kernel_size = self.up_scale * 2,  
                stride = self.up_scale, 
                padding =  self.up_scale // 2,  
                output_padding = self.up_scale % 2  
            )

    def forward(self, x):
        x = self.net(x)
        return x

class SelfAttation3D(nn.Module):
    def __init__(self, channels, h,w):
        super(SelfAttation3D, self).__init__()
        self.channels = channels
        self.h = h
        self.w = w
        self.mha = nn.MultiheadAttention(channels, 4, batch_first=True)
        self.ln = nn.LayerNorm([channels])
        self.ff_self = nn.Sequential(
            nn.LayerNorm([channels]),
            nn.Linear(channels, channels),
            nn.GELU(),
            nn.Linear(channels, channels),
        )

    def forward(self, x):
        x = x.view(-1, self.channels, self.h * self.w).swapaxes(1, 2)
        x_ln = self.ln(x)
        attention_value, _ = self.mha(x_ln, x_ln, x_ln)
        attention_value = attention_value + x
        attention_value = self.ff_self(attention_value) + attention_value
        return attention_value.swapaxes(2, 1).view(-1, self.channels, self.h, self.w)
    
    
class Unet(nn.Module):
    def __init__(self, width,  height,  in_c = 3, out_c = 3, time_dim=32):
        super(Unet, self).__init__()
        self.h = height
        self.w = width
        self.time_dim = time_dim
        
        self.conv1 = DoubleConv(in_c, 64)
        self.att1 = SelfAttation3D(64, self.h, self.w)
        self.e_emd_1 = TimeEmbed(time_dim, 64)
        self.context_emd_1 = ContextEmbed(16, 192, 64)
        self.pool1 = DownSample(64, 64)  
        
        self.conv2 = DoubleConv(64, 128)
        self.att2 = SelfAttation3D(128,(int)(self.h/2),(int)(self.w/2))
        self.e_emd_2 = TimeEmbed(time_dim,128)
        self.context_emd_2 = ContextEmbed(8, 192, 128)
        self.pool2 = DownSample(128, 128)  
        
        self.conv3 = DoubleConv(128, 256)
        self.att3 = SelfAttation3D(256,(int)(self.h/4),(int)(self.w/4))
        self.e_emd_3 = TimeEmbed(time_dim,256)
        self.context_emd_3 = ContextEmbed(4, 192, 256)
        self.pool3 = DownSample(256, 256)  
        
        self.conv4 = DoubleConv(256, 512)
        self.att4 = SelfAttation3D(512,(int)(self.h/8),(int)(self.w/8))
        self.e_emd_4 = TimeEmbed(time_dim,512)
        self.context_emd_4 = ContextEmbed(2, 192, 512)
        self.pool4 = DownSample(512, 512)  

        self.conv5 = DoubleConv(512, 1024)

        self.up6 = torch.nn.ConvTranspose2d(1024, 512, kernel_size=(2, 2), stride=(2, 2))  
        self.conv6 = DoubleConv(1024, 512)
        self.att6 = SelfAttation3D(512,(int)(self.h/8),(int)(self.w/8))
        self.d_emd_1 = TimeEmbed(time_dim,512)
        self.d_context_emd_1 = ContextEmbed(2, 192, 512)

        self.up7 = torch.nn.ConvTranspose2d(512, 256, (2, 2), (2, 2))
        self.conv7 = DoubleConv(512, 256)
        self.att7 = SelfAttation3D(256,(int)(self.h/4),(int)(self.w/4))
        self.d_emd_2 = TimeEmbed(time_dim,256)
        self.d_context_emd_2 = ContextEmbed(4, 192, 256)

        self.up8 = torch.nn.ConvTranspose2d(256, 128, (2, 2), (2, 2))
        self.conv8 = DoubleConv(256, 128)
        self.att8 = SelfAttation3D(128,(int)(self.h/2),(int)(self.w/2))
        self.d_emd_3 = TimeEmbed(time_dim,128)
        self.d_context_emd_3 = ContextEmbed(8, 192, 128)

        self.up9 = torch.nn.ConvTranspose2d(128, 64, (2, 2), (2, 2))
        self.conv9 = DoubleConv(128, 64)
        self.att9 = SelfAttation3D(64,(int)(self.h),(int)(self.w))
        self.d_emd_4 = TimeEmbed(time_dim,64)
        self.d_context_emd_4 = ContextEmbed(16, 192, 64)

        self.conv10 = torch.nn.Conv2d(64, out_c, kernel_size=(1, 1))  
        
    def pos_encoding(self, t, channels):
        inv_freq = 1.0 / (
            10000
            ** (torch.arange(0, channels, 2).float() / channels)
        ).to(t.device)
        pos_enc_a = torch.sin(t.repeat(1, channels // 2) * inv_freq)
        pos_enc_b = torch.cos(t.repeat(1, channels // 2) * inv_freq)
        pos_enc = torch.cat([pos_enc_a, pos_enc_b], dim=-1)
        return pos_enc        
 

    def forward(self, x, t, context):
        t = t.unsqueeze(-1).type(torch.float)             #
        t = self.pos_encoding(t, self.time_dim)        
        c1 = self.conv1(x) + self.e_emd_1(t) + self.context_emd_1(context)      
        # c1 = self.att1(c1)                                            
        p1 = self.pool1(c1)                                        
        c2 = self.conv2(p1) + self.e_emd_2(t) + self.context_emd_2(context)    
        # c2 = self.att2(c2)
        p2 = self.pool2(c2)
        c3 = self.conv3(p2) + self.e_emd_3(t) + self.context_emd_3(context)    
        # c3 = self.att3(c3)
        p3 = self.pool3(c3)
        c4 = self.conv4(p3) + self.e_emd_4(t) + self.context_emd_4(context)    
        # c4 = self.att4(c4)
        p4 = self.pool4(c4) 

        c5 = self.conv5(p4)

        up6 = self.up6(c5)  
        m6 = torch.cat([up6, c4], dim=1)
        c6 = self.conv6(m6) + self.d_emd_1(t) + self.d_context_emd_1(context)
        # c6 = self.att6(c6)

        up7 = self.up7(c6) 
        m7 = torch.cat([up7, c3], dim=1)
        c7 = self.conv7(m7) + self.d_emd_2(t) + self.d_context_emd_2(context)
        # c7 = self.att7(c7)

        up8 = self.up8(c7)  
        m8 = torch.cat([up8, c2], dim=1)
        c8 = self.conv8(m8) + self.d_emd_3(t) + self.d_context_emd_3(context)
        # c8 = self.att8(c8)

        up9 = self.up9(c8)  
        m9 = torch.cat([up9, c1], dim=1)
        c9 = self.conv9(m9) + self.d_emd_4(t) + self.d_context_emd_4(context)
        # c9 = self.att9(c9)

        c10 = self.conv10(c9)
        return c10