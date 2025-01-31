import torch.nn as nn

from compressai.layers import *
from compressai.utils import *

class Encoder(nn.Module):
    def __init__(self,image_shape,patch_size,embed_dim,window_size,head_num,shift_size,out_channel_m):
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