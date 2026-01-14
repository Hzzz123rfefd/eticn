import argparse
import json
import os
import matplotlib.pyplot as plt
import sys
import os
sys.path.append(os.getcwd())

def R_D_SSIM(bpp_lists, msssim_lists, models):
    plt.rcParams.update({
        'font.family': 'serif',
        'font.serif': 'Times New Roman',
        'axes.labelweight': 'bold',
        'axes.labelsize': '14',
        'xtick.labelsize': '12',
        'ytick.labelsize': '12',
        'legend.fontsize': '12'
    })

    # ================== 虚线方法定义 ==================
    dashed_models = {
        "BPG", "JEPG2000", "VVC",
        "[BPG]", "[JEPG2000]", "[VVC]"
    }

    # ================== 绘图并保存句柄 ==================
    lines_dict = {}

    plt.figure(figsize=(7, 5))

    for bpp, psnr, model in zip(bpp_lists, msssim_lists, models):
        if model in dashed_models:
            line, = plt.plot(bpp, psnr, marker='o', linestyle='--', label=model)
        else:
            line, = plt.plot(bpp, psnr, marker='o', linestyle='-', label=model)

        lines_dict[model] = line

    plt.xlabel('bpp')
    plt.ylabel('PSNR(dB)')
    plt.grid(True)

    # ================== 固定前排（自动判断是否存在） ==================
    proposed_name = "[Ours]"
    baseline_name = "Baseline [Qian (ICLR2022)]"

    fixed_front = []

    if baseline_name in lines_dict:
        fixed_front.append(baseline_name)

    if proposed_name in lines_dict:
        fixed_front.append(proposed_name)

    # ================== Legend 顺序构建 ==================
    remaining_models = [m for m in lines_dict if m not in fixed_front]

    solid_models = [m for m in remaining_models if m not in dashed_models]
    dashed_models_sorted = [m for m in remaining_models if m in dashed_models]

    final_order = fixed_front + solid_models + dashed_models_sorted

    ordered_handles = [lines_dict[m] for m in final_order]

    plt.legend(
        ordered_handles,
        final_order,
        loc='lower right',
        frameon=True
    )

    plt.savefig(args.dir_path + "/bpp_ssim.png", dpi = 700)
    plt.show()

def R_D_PSNR(bpp_lists, psnr_lists, models):
    plt.rcParams.update({
        'font.family': 'serif',
        'font.serif': 'Times New Roman',
        'axes.labelweight': 'bold',
        'axes.labelsize': '14',
        'xtick.labelsize': '12',
        'ytick.labelsize': '12',
        'legend.fontsize': '12'
    })

    # ================== 虚线方法定义 ==================
    dashed_models = {
        "BPG", "JEPG2000", "VVC",
        "[BPG]", "[JEPG2000]", "[VVC]"
    }

    # ================== 绘图并保存句柄 ==================
    lines_dict = {}

    plt.figure(figsize=(7, 5))

    for bpp, psnr, model in zip(bpp_lists, psnr_lists, models):
        if model in dashed_models:
            line, = plt.plot(bpp, psnr, marker='o', linestyle='--', label=model)
        else:
            line, = plt.plot(bpp, psnr, marker='o', linestyle='-', label=model)

        lines_dict[model] = line

    plt.xlabel('bpp')
    plt.ylabel('PSNR(dB)')
    plt.grid(True)

    # ================== 固定前排（自动判断是否存在） ==================
    proposed_name = "[Ours]"
    baseline_name = "Baseline [Qian (ICLR2022)]"

    fixed_front = []

    # ---- Baseline：模糊匹配 ----
    baseline_key = next(
        (name for name in lines_dict if name.startswith("Baseline")),
        None
    )

    if baseline_key is not None:
        fixed_front.append(baseline_key)

    # ---- Ours ----
    if proposed_name in lines_dict:
        fixed_front.append(proposed_name)

    # ================== Legend 顺序构建 ==================
    remaining_models = [m for m in lines_dict if m not in fixed_front]

    solid_models = [m for m in remaining_models if m not in dashed_models]
    dashed_models_sorted = [m for m in remaining_models if m in dashed_models]

    final_order = fixed_front + solid_models + dashed_models_sorted

    ordered_handles = [lines_dict[m] for m in final_order]

    plt.legend(
        ordered_handles,
        final_order,
        loc='lower right',
        frameon=True
    )

    
    plt.savefig(args.dir_path + "/bpp_psnr.png", dpi = 700)
    plt.show()


def main(args):
    models = []
    bpp_lists = []
    psnr_lists = []
    ssim_lists = []
    # load data
    for filename in os.listdir(args.dir_path):
        if filename.endswith('.json'):
            model_name = os.path.splitext(filename)[0]
            models.append(model_name)
            file_path = os.path.join(args.dir_path, filename)
            with open(file_path, 'r') as file:
                data = json.load(file)
                bpp_lists.append(data['bpp'])
                psnr_lists.append(data['PSNR'])
                ssim_lists.append(data['ms-ssim'])
    sorted_indices = sorted(range(len(models)), key=lambda i: models[i])
    models = [models[i] for i in sorted_indices]
    bpp_lists = [bpp_lists[i] for i in sorted_indices]
    psnr_lists = [psnr_lists[i] for i in sorted_indices]
    ssim_lists = [ssim_lists[i] for i in sorted_indices]
    R_D_PSNR(bpp_lists, psnr_lists, models)
    R_D_SSIM(bpp_lists, ssim_lists, models)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--dir_path',type=str, default="result/R-D/fbr/camvid")
    args = parser.parse_args()
    main(args)