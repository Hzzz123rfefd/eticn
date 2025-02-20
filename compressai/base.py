import math
import torch
from torch.utils.data import DataLoader
import torch.nn as nn
import os
from torch import optim
from abc import abstractmethod
from tqdm import tqdm
import torch.nn.functional as F

from compressai.utils import *
from compressai.entropy_models import *

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

            try:
                for epoch in range(last_epoch, total_epoch):
                    print(f"Learning rate: {optimizer.param_groups[0]['lr']}")
                    train_loss = self.train_one_epoch(epoch,train_dataloader, optimizer,clip_max_norm,log_path)
                    test_loss = self.test_epoch(epoch,test_dataloader,log_path)
                    loss = train_loss + test_loss
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
                clip_gradient(optimizer,clip_max_norm)

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
    def __init__(self,
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
        self.image_shape = [image_channel,image_height,image_weight]
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
        output["reconstruction_loss"] = F.mse_loss(input["reconstruction_image"], input["image"])
        output["total_loss"] = lamda * output["bpp_loss"] + output["reconstruction_loss"]
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
                
            super().trainning(train_dataloader, test_dataloader, val_dataloader, optimizer_name, weight_decay, clip_max_norm, factor, patience, lr, total_epoch, eval_interval, save_model_dir)

    def load_pretrained(self, save_model_dir):
        self.load_state_dict(torch.load(save_model_dir + "/model.pth"))
    
    def save_pretrained(self, save_model_dir):
        torch.save(self.state_dict(), save_model_dir + "/model.pth")