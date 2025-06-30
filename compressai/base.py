import math
import torch
from torch.utils.data import DataLoader
import os
from torch import optim
from abc import abstractmethod
from tqdm import tqdm
import torch.nn.functional as F
from typing import cast
import random
from compressai.utils import *
from compressai.entropy_models import *
from compressai.modules import *
from third_party.DL_Pipeline.src.model import ModelBase

class ModelCompressionBase(ModelBase):
    def __init__(
        self,
        image_channel,
        image_height,
        image_weight, 
        out_channel_m, 
        out_channel_n, 
        lamda = None, 
        finetune_model_dir = None, 
        device = "cuda"
    ):
        super().__init__(device)
        self.finetune_model_dir = finetune_model_dir
        self.image_channel = image_channel
        self.image_height = image_height
        self.image_weight = image_weight
        self.image_shape = [image_channel, image_height, image_weight]
        self.lamda = lamda
        self.out_channel_m = out_channel_m
        self.out_channel_n = out_channel_n
        self.entropy_bottleneck = EntropyBottleneck(out_channel_n).to(self.device)
        self.gaussian_conditional = GaussianConditional(None).to(self.device)

    def compute_loss(self, input):
        lamda = input["lamda"]
        N, _, H, W = input["image"].size()
        output = {}
        num_pixels = N * H * W

        output["bpp_loss"] = sum(
            (torch.log(likelihoods).sum() / (-math.log(2) * num_pixels))
            for likelihoods in input["likelihoods"].values()
        )
        output["reconstruction_loss"] = F.mse_loss(input["reconstruction_image"], input["image"]) * 255**2
        output["total_loss"] = output["bpp_loss"] + lamda * output["reconstruction_loss"]
        return output

    def configure_optimizers(self, optimizer_name, lr, weight_decay):
        parameters = {
            n
            for n, p in self.named_parameters()
            if not n.endswith(".quantiles")
        }
        aux_parameters = {
            n
            for n, p in self.named_parameters()
            if n.endswith(".quantiles")
        }

        # Make sure we don't have an intersection of parameters
        params_dict = dict(self.named_parameters())
        inter_params = parameters & aux_parameters
        union_params = parameters | aux_parameters

        assert len(inter_params) == 0
        assert len(union_params) - len(params_dict.keys()) == 0

        if optimizer_name == "Adam":
            self.optimizer = optim.Adam(
                (params_dict[n] for n in sorted(parameters)),
                lr = lr,
            )
            self.aux_optimizer = optim.Adam(
                (params_dict[n] for n in sorted(aux_parameters)),
                lr = 0.001,
            )
            
        elif optimizer_name == "AdamW":
            self.optimizer = optim.AdamW(
                (params_dict[n] for n in sorted(parameters)),
                lr = lr,
            )
            self.aux_optimizer = optim.AdamW(
                (params_dict[n] for n in sorted(aux_parameters)),
                lr = 0.001,
            )
            
        else:
            self.optimizer = optim.Adam(
                (params_dict[n] for n in sorted(parameters)),
                lr = lr,
            )
            self.aux_optimizer = optim.Adam(
                (params_dict[n] for n in sorted(aux_parameters)),
                lr = 0.001,
            )

    def configure_train_set(self, save_model_dir):
        if self.lamda != None:
            self.save_model_dir = save_model_dir  + "/" + str(self.lamda)
        else:
            self.save_model_dir = save_model_dir
        self.first_trainning = True
        self.check_point_path = self.save_model_dir  + "/checkpoint.pth"
        self.log_path = self.save_model_dir + "/train.log"
        
    def init_model(self):
        if  os.path.isdir(self.save_model_dir) and os.path.exists(self.check_point_path) and os.path.exists(self.log_path):
            self.load_pretrained(self.save_model_dir)  
            self.first_trainning = False
        else:
            os.makedirs(self.save_model_dir, exist_ok = True)
            with open(self.log_path, "w") as f:
                pass  
            self.first_trainning = True
            if self.finetune_model_dir:
                self.load_pretrained(self.finetune_model_dir)
    
    def configure_train_log(self):
        if self.first_trainning:
            self.best_loss = float("inf")
            self.last_epoch = 0
        else:
            checkpoint = torch.load(self.check_point_path, map_location = self.device)
            self.optimizer.load_state_dict(checkpoint["optimizer"])
            self.aux_optimizer.load_state_dict(checkpoint["aux_optimizer"])
            self.lr_scheduler.load_state_dict(checkpoint["lr_scheduler"])
            self.best_loss = checkpoint["loss"]
            self.last_epoch = checkpoint["epoch"]
    
    def save_train_log(self):
        torch.save({
                "epoch": self.epoch,
                "loss": self.best_loss,
                "optimizer": self.optimizer.state_dict(),
                "aux_optimizer": self.aux_optimizer.state_dict(),
                "lr_scheduler": self.lr_scheduler.state_dict()
            }, 
            self.check_point_path
        )
        print("model saved !")
    
    def train_one_epoch(self):
        self.train().to(self.device)
        pbar = tqdm(self.train_dataloader,desc="Processing epoch "+str(self.epoch), unit="batch")
        total_loss = AverageMeter()
        
        for batch_id, inputs in enumerate(self.train_dataloader):
            """ grad zeroing """
            self.optimizer.zero_grad()
            self.aux_optimizer.zero_grad()

            """ forward """
            used_memory = 0 if self.device != "cuda" else torch.cuda.memory_allocated(torch.cuda.current_device()) / (1024 ** 3)  
            output = self.forward(inputs)

            """ calculate loss """
            out_criterion = self.compute_loss(output)
            out_criterion["total_loss"].backward()
            total_loss.update(out_criterion["total_loss"].item())

            """ grad clip """
            torch.nn.utils.clip_grad_norm_(self.parameters(), self.clip_max_norm)
                
            aux_loss = self.aux_loss()
            aux_loss.backward()
            self.aux_optimizer.step()

            """ modify parameters """
            self.optimizer.step()
            after_used_memory = 0 if self.device != "cuda" else torch.cuda.memory_allocated(torch.cuda.current_device()) / (1024 ** 3) 
            postfix_str = "Train Epoch: {:d}, total_loss: {:.4f}, use_memory: {:.1f}G".format(
                self.epoch,
                total_loss.avg, 
                after_used_memory - used_memory
            )
            pbar.set_postfix_str(postfix_str)
            pbar.update()
            
        self.logger.info(postfix_str)
    
    def eval_model(self, val_dataloader):
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
    
        print("PSNR = {:.4f}, BPP = {:.2f}\n".format(psnr.avg, bpp.avg))
        output = {
            "PSNR": psnr.avg,
            "bpp": bpp.avg
        }
        return output
    
    def aux_loss(self) -> Tensor:
        loss = sum(m.loss() for m in self.modules() if isinstance(m, EntropyBottleneck))
        return cast(Tensor, loss)

    def load_pretrained(self, save_model_dir, lamda = None):
        if lamda == None:
            self.load_state_dict(torch.load(save_model_dir  + "/model.pth"))
        else:
            self.load_state_dict(torch.load(save_model_dir + str(lamda) + "/model.pth"))

    def save_pretrained(self, save_model_dir, lamda = None):
        if lamda == None:
            torch.save(self.state_dict(), save_model_dir + "/model.pth")
        else:
            torch.save(self.state_dict(), save_model_dir + str(lamda) + "/model.pth")
  
class ModelQVRFBase(ModelCompressionBase):
    def __init__(
        self,
        image_channel,
        image_height,
        image_weight, 
        out_channel_m, 
        out_channel_n, 
        stage = 1,
        device = "cuda"
    ):
        super().__init__(image_channel, image_height, image_weight, out_channel_m, out_channel_n, None, None, device)
        self.Gain = torch.nn.Parameter(torch.tensor(
            [1.0000, 1.3944, 1.9293, 2.6874, 3.7268, 5.1801, 7.1957, 10.0000]), requires_grad = True)
        self.stage = stage
        self.lmbda = [0.0018, 0.0035, 0.0067, 0.0130, 0.025, 0.0483, 0.0932, 0.18]
        self.levels = len(self.lmbda) 

    def get_scale(self, s, is_train):
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
        return scale, rescale, s
    
    def test_one_epoch(self):
        total_loss = AverageMeter()
        # self.eval()
        self.to(self.device)
        with torch.no_grad():
            for s in range(self.levels):
                for batch_id, inputs in enumerate(self.test_dataloader):
                    """ forward """
                    output = self.forward(inputs, s = s, is_train = False)

                    """ calculate loss """
                    out_criterion = self.compute_loss(output)
                    total_loss.update(out_criterion["total_loss"].item())

        self.logger.info("Test Epoch: {:d}, total_loss: {:.4f}".format(self.epoch, total_loss.avg))
        return total_loss.avg
    
    def eval_model(self, val_dataloader):
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
            log_message = log_message + f"lamda = {lamda}, s = {index}, stage {self.stage}, PSNR = {psnrs[index].avg},  BPP = {bpps[index].avg}\n"
            # log_message = log_message + f"lamda = {lamda}, s = {index}, scale: {self.Gain.data[index].cpu().numpy():0.4f}, stage {self.stage}, PSNR = {psnrs[index].avg},  BPP = {bpps[index].avg}\n"
        print(log_message)
        
        psnrs = [each.avg for each in psnrs]
        bpps = [float(each.avg.cpu().numpy()) for each in bpps]
        del psnrs[1]
        del bpps[1]
        
        output = {
            "PSNR": psnrs,
            "bpp": bpps
        }
        return output

    def load_pretrained(self, save_model_dir):
        self.load_state_dict(torch.load(save_model_dir + "/model.pth"))

    def save_pretrained(self, save_model_dir):
        torch.save(self.state_dict(), save_model_dir + "/model.pth")
        
class ModelCQVRBase(ModelQVRFBase):        
    def __init__(
        self,
        image_channel,
        image_height,
        image_weight, 
        time_dim,
        out_channel_m, 
        out_channel_n, 
        stage = 1,
        device = "cuda"
    ):
        super().__init__(image_channel, image_height, image_weight, out_channel_m, out_channel_n, stage, device)
        self.time_dim = time_dim
        self.Gain = torch.nn.Parameter(torch.tensor(
             [[1.0000, 1.3944, 1.9293, 2.6874, 3.7268, 5.1801, 7.1957, 10.0000]] * self.out_channel_m, dtype=torch.float32), requires_grad=True)
        
        self.predict_model = Unet(
            width = (int)(self.image_weight / 16), 
            height = (int)(self.image_height / 16), 
            in_c = self.out_channel_m, 
            out_c = self.out_channel_m, 
            time_dim = self.time_dim
        ).to(self.device)
    
    def get_scale(self, s, is_train):
        if is_train == True:
            if self.stage > 1:
                s = random.randint(0, self.levels - 1)
                if s != 0:
                    if self.stage == 2:
                        scale = torch.max(self.Gain[:, s], torch.tensor(1e-4)) + 1e-9
                    else:
                        scale = torch.max(self.Gain[:, s], torch.tensor(1e-4)) + 1e-9
                        scale = scale.detach().clone()
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
        return scale, rescale, s
    
    def y_hat_enhance(self, y, scale, rescale, s, batch):
        if self.stage != 3:
            y_hat = self.gaussian_conditional.quantize(y * scale, "noise" if self.training else "dequantize") * rescale
            noisy = None
            predict_noisy = None
        else:
            y_hat, noisy = self.gaussian_conditional.quantize(y * scale, "noise" if self.training else "dequantize", return_noisy = True)
            y_hat = y_hat * rescale
            t = torch.ones((batch)) * (self.levels - s)
            t = t.to(self.device)
            predict_noisy = self.predict_model(y_hat, t)
            y_hat = y_hat - predict_noisy * rescale
        return y_hat, noisy, predict_noisy
           
    def compute_loss(self, input):
        output = super().compute_loss(input)
        if self.stage == 3 and self.training:
            output["noisy_loss"] = F.mse_loss(input["noisy"], input["predict_noisy"])
            output["total_loss"] = output["total_loss"] + output["noisy_loss"]
        return output
    
class ModelDiffusionBase(ModelBase):
    def __init__(
        self,
        width,
        height,
        channel = 3,
        time_dim = 256, 
        noise_steps = 500, 
        beta_start = 1e-4, 
        beta_end = 0.02,
        device = "cuda"
    ):
        super().__init__(device)
        self.width = width
        self.height = height
        self.channel = channel
        self.beta_start = beta_start
        self.beta_end = beta_end
        self.noise_steps = noise_steps
        self.time_dim = time_dim
        
        for k, v in self.ddpm_schedules(self.beta_start, self.beta_end, self.noise_steps).items():
            self.register_buffer(k, v)
    
    def ddpm_schedules(self, beta1, beta2, T):
        """
        Returns pre-computed schedules for DDPM sampling, training process.
        """
        assert beta1 < beta2 < 1.0, "beta1 and beta2 must be in (0, 1)"

        beta_t = (beta2 - beta1) * torch.arange(0, T + 1, dtype=torch.float32) / T + beta1
        sqrt_beta_t = torch.sqrt(beta_t)
        alpha_t = 1 - beta_t
        log_alpha_t = torch.log(alpha_t)
        alphabar_t = torch.cumsum(log_alpha_t, dim=0).exp()

        sqrtab = torch.sqrt(alphabar_t)
        oneover_sqrta = 1 / torch.sqrt(alpha_t)

        sqrtmab = torch.sqrt(1 - alphabar_t)
        mab_over_sqrtmab_inv = (1 - alpha_t) / sqrtmab

        return {
            "alpha_t": alpha_t,  # \alpha_t
            "oneover_sqrta": oneover_sqrta,  # 1/\sqrt{\alpha_t}
            "sqrt_beta_t": sqrt_beta_t,  # \sqrt{\beta_t}
            "alphabar_t": alphabar_t,  # \bar{\alpha_t}
            "sqrtab": sqrtab,  # \sqrt{\bar{\alpha_t}}
            "sqrtmab": sqrtmab,  # \sqrt{1-\bar{\alpha_t}}
            "mab_over_sqrtmab": mab_over_sqrtmab_inv,  # (1-\alpha_t)/\sqrt{1-\bar{\alpha_t}}
        }

    def load_pretrained(self, save_model_dir):
        self.load_state_dict(torch.load(save_model_dir + "/model.pth"))

    def save_pretrained(self,  save_model_dir):
        torch.save(self.state_dict(), save_model_dir + "/model.pth")    

class ModelDDPM(ModelDiffusionBase):
    def __init__(
        self,
        width,
        height,
        channel = 3,
        time_dim = 32, 
        noise_steps = 500, 
        beta_start = 1e-4, 
        beta_end = 0.02,
        device = "cuda"
    ):
        super().__init__(width, height, channel, time_dim, noise_steps, beta_start, beta_end, device)
        self.predict_model = Unet(
            width = self.width,
            height = self.height, 
            in_c = self.channel, 
            out_c = self.channel, 
            time_dim = self.time_dim
        ).to(self.device)
    
    def sample(self, sample_num = 1, context = None, guide_w = 0.0):
        self.eval()
        with torch.no_grad():
            x_i = torch.zeros(sample_num, self.channel, self.height, self.width).to(self.device)
            for i in range(self.noise_steps, 1, -1): 
                print(f'sampling timestep {i}',end='\r')

                x_i = x_i.repeat(2,1,1,1)
                t = torch.tensor([i]).repeat(sample_num).to(self.device)
                t_i = t.repeat(2)
                
                context_s = context.repeat(2,1,1,1)
                z = torch.randn(sample_num, self.channel, self.height, self.width).to(self.device) if i > 1 else 0
                
                eps = self.predict_model(x_i, t_i, context_s)
                eps1 = eps[:sample_num]
                eps2 = eps[sample_num:]
                eps = (1+guide_w) * eps1 - guide_w * eps2
                x_i = x_i[:sample_num]
                
                x_i = (
                    self.oneover_sqrta[i] * (x_i - eps * self.mab_over_sqrtmab[i])
                    + self.sqrt_beta_t[i] * z
                )
                # return x_0

            x_i = torch.clamp(x_i, min = 0, max = 1)
        return x_i

    def forward(self, x, context):
        _ts = torch.randint(1, self.noise_steps + 1, (x.shape[0],)).to(self.device)  # t ~ Uniform(0, n_T)
        noise = torch.randn_like(x)  # eps ~ N(0, 1)

        x_t = (
            self.sqrtab[_ts, None, None, None] * x
            + self.sqrtmab[_ts, None, None, None] * noise
        )  
        
        z_t = self.predict_model(x_t, _ts, context)
        x_0 = (x_t - self.sqrtmab[_ts, None, None, None] * z_t) / self.sqrtab[_ts, None, None, None]
        alphabar_t = self.alphabar_t[_ts]
        return x_0, alphabar_t, z_t, noise

           