import argparse
import sys
import os
from matplotlib import pyplot as plt
sys.path.append(os.getcwd())
from compressai.utils import *
from compressai import models
import numpy as np
import matplotlib.pyplot as plt

# 设置格式
plt.rcParams['pdf.fonttype'] = 42   
plt.rcParams['ps.fonttype'] = 42
plt.rcParams['font.family'] = 'Times New Roman'  
plt.rcParams.update({
'font.family': ['Times New Roman', 'SimSun'],
'axes.unicode_minus': False,                 
'axes.labelweight': 'bold',
'axes.labelsize': 14,
'xtick.labelsize': 12,
'ytick.labelsize': 12,
'legend.fontsize': 12,
'mathtext.fontset': 'stix'
})

def plot_all_result(x, y):
    z = np.sqrt(y / 0.0018)
    a, b = np.polyfit(x, z, 1)
    x_fit = np.linspace(min(x), max(x), 200)
    z_fit = a * x_fit + b
    fig, ax = plt.subplots(figsize=(6,4))
    # 红色散点
    ax.scatter(x, z, color='red', s=20, label=r'各通道上的均值')
    # 黑色实线
    ax.plot(x_fit, z_fit, color='black', linewidth=1.2, label='线性拟合')
    ax.set_xlabel(r"$\bar{a}$")
    ax.set_ylabel(r'$\sqrt{\frac{\lambda}{\lambda_{ref}}}$')
    # 顶部公式
    formula = rf"$\sqrt{{\frac{{\lambda}}{{\lambda_{{ref}}}}}} = {a:.4f}\bar{{a}} {b:+.4f}$"
    ax.text(
        0.5, 0.95, formula,
        transform=ax.transAxes,
        ha='center',
        va='top',
        fontsize=14
    )
    # 图例
    legend = ax.legend(loc='lower right', frameon=True)
    legend.get_frame().set_linewidth(1.8)  # 图例边框加粗
    # 坐标轴
    for spine in ax.spines.values():
        spine.set_linewidth(2)
    plt.tight_layout()
    plt.savefig(os.path.join(args.save_dir, "result1.pdf"), bbox_inches='tight')
    plt.show()
    plt.close()

def plot_example(x0, x1, x2, y):
    z = np.sqrt(y / 0.0018)
    a0, b0 = np.polyfit(x0, z, 1)
    x_fit0 = np.linspace(min(x0), max(x0), 200)
    z_fit0 = a0 * x_fit0 + b0
    a1, b1 = np.polyfit(x1, z, 1)
    x_fit1 = np.linspace(min(x1), max(x1), 200)
    z_fit1 = a1 * x_fit1 + b1
    a2, b2 = np.polyfit(x2, z, 1)
    x_fit2 = np.linspace(min(x2), max(x2), 200)
    z_fit2 = a2 * x_fit2 + b2
    fig, ax = plt.subplots(figsize=(6,4))
    ax.set_xlim(0, 11)   
    ax.set_ylim(0, 11) 
    # print("a0 = ", a0, "a1 = ", a1, "a2 = ", a2, "b0 = ", b0, "b1 = ", b1, "b2 = ", b2)
    ax.plot(x_fit0, z_fit0, color='black', linewidth=1.2, label='channel = 0')
    ax.plot(x_fit1, z_fit1, color='red', linewidth=1.2, label='channel = 1')
    ax.plot(x_fit2, z_fit2, color='blue', linewidth=1.2, label='channel = 3')
    y_target = 8

    # 计算交点
    x0_cross = (y_target - b0) / a0
    x1_cross = (y_target - b1) / a1
    x2_cross = (y_target - b2) / a2

    # 横向虚线（只画到最大交点）
    x_max_cross = max(x0_cross, x1_cross, x2_cross)
    ax.plot([0, x_max_cross], [y_target, y_target],
            linestyle='--', color='gray', linewidth=1)

    # 竖向虚线（只画到 y_target）
    ax.plot([x0_cross, x0_cross], [0, y_target],
            linestyle='--', color='black', linewidth=1)

    ax.plot([x1_cross, x1_cross], [0, y_target],
            linestyle='--', color='red', linewidth=1)

    ax.plot([x2_cross, x2_cross], [0, y_target],
            linestyle='--', color='blue', linewidth=1)

    # 标注 a 的值
    # ax.text(x0_cross, 0.2, f"a0 = {x0_cross:.2f}", color='black', ha='center')
    # ax.text(x1_cross, 0.2, f"a1 = {x1_cross:.2f}", color='red', ha='center')
    # ax.text(x2_cross, 0.2, f"a3 = {x2_cross:.2f}", color='blue', ha='center')
    ax.set_xlabel(r"$a$")
    ax.set_ylabel(r'$\sqrt{\frac{\lambda}{\lambda_{ref}}}$')
    # 图例
    legend = ax.legend(loc='lower right', frameon=True)
    legend.get_frame().set_linewidth(1.8)  # 图例边框加粗
    # 坐标轴
    for spine in ax.spines.values():
        spine.set_linewidth(2)
    plt.tight_layout()
    plt.savefig(os.path.join(args.save_dir, "result2.pdf"), bbox_inches='tight')
    plt.show()
    plt.close()
     
def main(args):
    os.makedirs(args.save_dir, exist_ok = True)
    # 获取量化控制器
    config = load_config(args.model_config_path)
    net = models[config["model_type"]](**config["model"]).to(config["model"]["device"])
    net.load_pretrained(
        save_model_dir = args.model_path
    )
    gain = net.Gain.detach()
    mean_gain = net.Gain.mean(dim=0)
    x = mean_gain.detach().cpu().numpy()
    gain = net.Gain.detach()
    x2 = gain[0,:]
    x2 = x2.cpu().numpy()
    x3 = gain[1,:]
    x3 = x3.cpu().numpy()
    x26 = gain[3,:]
    x26 = x26.cpu().numpy()
    # 预定义拉格朗日集合
    y = np.array([0.0018,0.0035,0.0067,0.0130,0.025,0.0483,0.0932,0.18])
    plot_all_result(x, y)
    plot_example(x2, x3, x26, y)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_config_path", type=str, default = "config/eticn/eticn_qeevrf_stage3.yml")
    parser.add_argument("--model_path", type=str, default = "saved_model/eticn/eticn_qeevrf/stage3")
    parser.add_argument("--save_dir", type=str, default = "result/fig4-9")
    args = parser.parse_args()
    main(args)