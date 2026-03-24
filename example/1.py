import argparse
import sys
import os

from matplotlib import pyplot as plt
sys.path.append(os.getcwd())
from compressai.utils import *
from compressai import models




def main(args):
    config = load_config(args.model_config_path)
    """ get model"""
    net = models[config["model_type"]](**config["model"]).to(config["model"]["device"])

    net.load_pretrained(
        save_model_dir = args.model_path
    )
    mean_gain = net.Gain.mean(dim=0)
    print(mean_gain)
    y = net.Gain[0:32, :].mean(dim=1)
    index = range(32)
# 画折线图
    plt.figure()
    plt.plot(index, y.detach().cpu().numpy(), marker='o')

    plt.xlabel("Index")
    plt.ylabel("Value")
    plt.title("Line Plot of x[0:32, 7]")

    plt.show()
        
if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_config_path", type=str, default = "config/eticn/eticn_qeevrf_stage3.yml")
    parser.add_argument("--model_path", type=str, default = "saved_model/eticn-/eticn_qeevrf/stage3")
    parser.add_argument("--save_dir", type=str, default = "result/2-fig22")
    args = parser.parse_args()
    main(args)