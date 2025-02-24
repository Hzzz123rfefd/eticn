import math
import torch
from torch.utils.data import DataLoader
import torch.nn as nn
import os
from torch import optim
from abc import abstractmethod
from tqdm import tqdm
import torch.nn.functional as F
from typing import cast

from compressai.utils import *
from compressai.entropy_models import *


def configure_optimizers(net, lr):
    """Separate parameters for the main optimizer and the auxiliary optimizer.
    Return two optimizers"""

    parameters = {
        n
        for n, p in net.named_parameters()
        if not n.endswith(".quantiles") and p.requires_grad
    }
    aux_parameters = {
        n
        for n, p in net.named_parameters()
        if n.endswith(".quantiles") and p.requires_grad
    }

    # Make sure we don't have an intersection of parameters
    params_dict = dict(net.named_parameters())
    inter_params = parameters & aux_parameters
    union_params = parameters | aux_parameters

    assert len(inter_params) == 0
    assert len(union_params) - len(params_dict.keys()) == 0

    optimizer = optim.Adam(
        (params_dict[n] for n in sorted(parameters)),
        lr = lr,
    )
    aux_optimizer = optim.Adam(
        (params_dict[n] for n in sorted(aux_parameters)),
        lr = 0.001,
    )
    return optimizer, aux_optimizer

class ModelBase(nn.Module):
    def __init__(
        self,
        device = "cuda"
    ):
        super().__init__()
        self.device = device if torch.cuda.is_available() else "cpu"
    
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
            self.to(self.device)
            ## 1 trainning log path 
            first_trainning = True
            check_point_path = save_model_dir  + "/checkpoint.pth"
            log_path = save_model_dir + "/train.log"

            ## 2 get net pretrain parameters if need 
            if  os.path.isdir(save_model_dir) and os.path.exists(check_point_path) and os.path.exists(log_path):
                self.load_pretrained(save_model_dir)  
                first_trainning = False

            else:
                if not os.path.isdir(save_model_dir):
                    os.makedirs(save_model_dir)
                with open(log_path, "w") as file:
                    pass

            ##  3 get optimizer
            if optimizer_name == "Adam":
                optimizer = optim.Adam(self.parameters(),lr,weight_decay = weight_decay)
            elif optimizer_name == "AdamW":
                optimizer = optim.AdamW(self.parameters(),lr,weight_decay = weight_decay)
            else:
                optimizer = optim.Adam(self.parameters(),lr,weight_decay = weight_decay)
                
            lr_scheduler = optim.lr_scheduler.ReduceLROnPlateau(
                optimizer = optimizer, 
                mode = "min", 
                factor = factor, 
                patience = patience
            )

            ## 4 init trainng log
            if first_trainning:
                best_loss = float("inf")
                last_epoch = 0
            else:
                checkpoint = torch.load(check_point_path, map_location=self.device)
                optimizer.load_state_dict(checkpoint["optimizer"])
                lr_scheduler.load_state_dict(checkpoint["lr_scheduler"])
                best_loss = checkpoint["loss"]
                last_epoch = checkpoint["epoch"] + 1
                # optimizer.param_groups[0]['lr'] = 0.0001

            try:
                for epoch in range(last_epoch, total_epoch):
                    print(f"Learning rate: {optimizer.param_groups[0]['lr']}")
                    train_loss = self.train_one_epoch(epoch,train_dataloader, optimizer,clip_max_norm, log_path)
                    test_loss = self.test_epoch(epoch,test_dataloader,log_path)
                    loss = test_loss
                    lr_scheduler.step(loss)
                    is_best = loss < best_loss
                    best_loss = min(loss, best_loss)
                    torch.save(                
                        {
                            "epoch": epoch,
                            "loss": loss,
                            "optimizer": None,
                            "lr_scheduler": None
                        },
                        check_point_path
                    )
                    if epoch % eval_interval == 0:
                        self.eval_epoch(val_dataloader, log_path)
                    
                    if is_best:
                        self.save_pretrained(save_model_dir)

            # interrupt trianning
            except KeyboardInterrupt:
                    torch.save(                
                        {
                            "epoch": epoch,
                            "loss": loss,
                            "optimizer": optimizer.state_dict(),
                            "lr_scheduler": lr_scheduler.state_dict()
                        },
                        check_point_path
                    )
    
    def train_one_epoch(self, epoch, train_dataloader, optimizer, clip_max_norm, log_path = None):
        self.train().to(self.device)
        pbar = tqdm(train_dataloader,desc="Processing epoch "+str(epoch), unit="batch")
        total_loss = AverageMeter()
        
        for batch_id, inputs in enumerate(train_dataloader):
            """ grad zeroing """
            optimizer.zero_grad()

            """ forward """
            used_memory = 0 if self.device != "cuda" else torch.cuda.memory_allocated(torch.cuda.current_device()) / (1024 ** 3)  
            output = self.forward(inputs)

            """ calculate loss """
            out_criterion = self.compute_loss(output)
            out_criterion["total_loss"].backward()
            total_loss.update(out_criterion["total_loss"].item())

            """ grad clip """
            if clip_max_norm > 0:
                torch.nn.utils.clip_grad_norm_(self.parameters(), clip_max_norm)

            """ modify parameters """
            optimizer.step()
            after_used_memory = 0 if self.device != "cuda" else torch.cuda.memory_allocated(torch.cuda.current_device()) / (1024 ** 3) 
            postfix_str = "Train Epoch: {:d}, total_loss: {:.4f}, use_memory: {:.1f}G".format(
                epoch,
                total_loss.avg, 
                after_used_memory - used_memory
            )
            pbar.set_postfix_str(postfix_str)
            pbar.update()
            
        with open(log_path, "a") as file:
            file.write(postfix_str+"\n")

        return total_loss.avg

    def test_epoch(self, epoch, test_dataloader, log_path = None):
        total_loss = AverageMeter()
        self.eval()
        self.to(self.device)
        with torch.no_grad():
            for batch_id, inputs in enumerate(test_dataloader):
                """ forward """
                output = self.forward(inputs)

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
    
    @abstractmethod
    def eval_epoch(self, val_dataloader = None, log_path = None):
        pass
        
    @abstractmethod    
    def compute_loss(self, input):
        pass
    
    @abstractmethod
    def load_pretrained(self, save_model_dir):
        pass

    @abstractmethod
    def save_pretrained(self, save_model_dir):
        pass

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
            if self.lamda != None:
                save_model_dir = save_model_dir  + "/" + str(self.lamda)
            if self.finetune_model_dir:
                self.load_pretrained(self.finetune_model_dir)
                
            self.to(self.device)
            ## 1 trainning log path 
            first_trainning = True
            check_point_path = save_model_dir  + "/checkpoint.pth"
            log_path = save_model_dir + "/train.log"

            ## 2 get net pretrain parameters if need 
            if  os.path.isdir(save_model_dir) and os.path.exists(check_point_path) and os.path.exists(log_path):
                self.load_pretrained(save_model_dir)  
                first_trainning = False

            else:
                if not os.path.isdir(save_model_dir):
                    os.makedirs(save_model_dir)
                with open(log_path, "w") as file:
                    pass

            ##  3 get optimizer
            optimizer, aux_optimizer = configure_optimizers(self, lr)
            
            lr_scheduler = optim.lr_scheduler.ReduceLROnPlateau(
                optimizer = optimizer, 
                mode = "min", 
                factor = factor, 
                patience = patience
            )

            ## 4 init trainng log
            if first_trainning:
                best_loss = float("inf")
                last_epoch = 0
            else:
                checkpoint = torch.load(check_point_path, map_location = self.device)
                optimizer.load_state_dict(checkpoint["optimizer"])
                aux_optimizer.load_state_dict(checkpoint["aux_optimizer"])
                lr_scheduler.load_state_dict(checkpoint["lr_scheduler"])
                best_loss = checkpoint["loss"]
                last_epoch = checkpoint["epoch"] + 1
                # optimizer.param_groups[0]['lr'] = 0.0001

            try:
                for epoch in range(last_epoch, total_epoch):
                    print(f"Learning rate: {optimizer.param_groups[0]['lr']}")
                    self.train_one_epoch(epoch,train_dataloader, optimizer,clip_max_norm, aux_optimizer, log_path)
                    test_loss = self.test_epoch(epoch,test_dataloader,log_path)
                    loss = test_loss
                    lr_scheduler.step(loss)
                    is_best = loss < best_loss
                    best_loss = min(loss, best_loss)
                    torch.save(                
                        {
                            "epoch": epoch,
                            "loss": loss,
                            "optimizer": None,
                            "aux_optimizer": None,
                            "lr_scheduler": None
                        },
                        check_point_path
                    )
                    if epoch % eval_interval == 0:
                        self.eval_epoch(val_dataloader, log_path)
                    
                    if is_best:
                        self.save_pretrained(save_model_dir)

            # interrupt trianning
            except KeyboardInterrupt:
                    torch.save(                
                        {
                            "epoch": epoch,
                            "loss": loss,
                            "optimizer": optimizer.state_dict(),
                            "aux_optimizer": aux_optimizer.state_dict(),
                            "lr_scheduler": lr_scheduler.state_dict()
                        },
                        check_point_path
                    )
    
    def load_pretrained(self, save_model_dir):
        self.load_state_dict(torch.load(save_model_dir + "/model.pth"))
    
    def save_pretrained(self, save_model_dir):
        torch.save(self.state_dict(), save_model_dir + "/model.pth")

    def train_one_epoch(self, epoch, train_dataloader, optimizer, clip_max_norm, aux_optimizer = None, log_path = None):
        self.train().to(self.device)
        pbar = tqdm(train_dataloader,desc="Processing epoch "+str(epoch), unit="batch")
        total_loss = AverageMeter()
        
        for batch_id, inputs in enumerate(train_dataloader):
            """ grad zeroing """
            optimizer.zero_grad()
            aux_optimizer.zero_grad()

            """ forward """
            used_memory = 0 if self.device != "cuda" else torch.cuda.memory_allocated(torch.cuda.current_device()) / (1024 ** 3)  
            output = self.forward(inputs)

            """ calculate loss """
            out_criterion = self.compute_loss(output)
            out_criterion["total_loss"].backward()
            total_loss.update(out_criterion["total_loss"].item())

            """ grad clip """
            if clip_max_norm > 0:
                torch.nn.utils.clip_grad_norm_(self.parameters(), clip_max_norm)
                
            aux_loss = self.aux_loss()
            aux_loss.backward()
            aux_optimizer.step()

            """ modify parameters """
            optimizer.step()
            after_used_memory = 0 if self.device != "cuda" else torch.cuda.memory_allocated(torch.cuda.current_device()) / (1024 ** 3) 
            postfix_str = "Train Epoch: {:d}, total_loss: {:.4f}, use_memory: {:.1f}G".format(
                epoch,
                total_loss.avg, 
                after_used_memory - used_memory
            )
            pbar.set_postfix_str(postfix_str)
            pbar.update()
            
        with open(log_path, "a") as file:
            file.write(postfix_str+"\n")

        return total_loss.avg

    def test_epoch(self, epoch, test_dataloader, log_path = None):
        total_loss = AverageMeter()
        self.eval()
        self.to(self.device)
        with torch.no_grad():
            for batch_id, inputs in enumerate(test_dataloader):
                """ forward """
                output = self.forward(inputs)

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
    
    @abstractmethod
    def eval_epoch(self, val_dataloader = None, log_path = None):
        pass
    
    def aux_loss(self) -> Tensor:
        r"""Returns the total auxiliary loss over all ``EntropyBottleneck``\s.

        In contrast to the primary "net" loss used by the "net"
        optimizer, the "aux" loss is only used by the "aux" optimizer to
        update *only* the ``EntropyBottleneck.quantiles`` parameters. In
        fact, the "aux" loss does not depend on image data at all.

        The purpose of the "aux" loss is to determine the range within
        which most of the mass of a given distribution is contained, as
        well as its median (i.e. 50% probability). That is, for a given
        distribution, the "aux" loss converges towards satisfying the
        following conditions for some chosen ``tail_mass`` probability:

        * ``cdf(quantiles[0]) = tail_mass / 2``
        * ``cdf(quantiles[1]) = 0.5``
        * ``cdf(quantiles[2]) = 1 - tail_mass / 2``

        This ensures that the concrete ``_quantized_cdf``\s operate
        primarily within a finitely supported region. Any symbols
        outside this range must be coded using some alternative method
        that does *not* involve the ``_quantized_cdf``\s. Luckily, one
        may choose a ``tail_mass`` probability that is sufficiently
        small so that this rarely occurs. It is important that we work
        with ``_quantized_cdf``\s that have a small finite support;
        otherwise, entropy coding runtime performance would suffer.
        Thus, ``tail_mass`` should not be too small, either!
        """
        loss = sum(m.loss() for m in self.modules() if isinstance(m, EntropyBottleneck))
        return cast(Tensor, loss)
    
class ModelVBRCompressionBase(ModelCompressionBase):
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
        self.stage = stage
        self.lmbda = [0.0018, 0.0035, 0.0067, 0.0130, 0.025, 0.0483, 0.0932, 0.18]
        self.levels = len(self.lmbda) 

    def test_epoch(self, epoch, test_dataloader, log_path = None):
        total_loss = AverageMeter()
        self.eval()
        self.to(self.device)
        with torch.no_grad():
            for s in range(self.levels):
                for batch_id, inputs in enumerate(test_dataloader):
                    """ forward """
                    output = self.forward(inputs, s = s, is_train = False)

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
            log_message = log_message + f"lamda = {lamda}, s = {index}, stage {self.stage}, PSNR = {psnrs[index].avg},  BPP = {bpps[index].avg}\n"
            
        print(log_message)
        if log_path != None:
            with open(log_path, "a") as file:
                file.write(log_message+"\n")
                
        output = {
            "log_message":log_message,
            "PSNR": psnrs,
            "bpp": bpps
        }