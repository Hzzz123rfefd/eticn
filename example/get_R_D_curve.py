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
    for bpp, psnr, model in zip(bpp_lists, msssim_lists, models):
        if model == "BPG" or model == "JEPG2000" or model == "VVC" or model == "[BPG]" or model == "[JEPG2000]" or model == "[VVC]" or model == "[Baseline]":
            plt.plot(bpp, psnr, marker='o', label=model, linestyle='--')
        else:
            plt.plot(bpp, psnr, marker='o', label=model) 
    plt.xlabel('bpp')
    plt.ylabel('MS-SSIM(db)')
    plt.legend(loc='lower right')
    plt.grid(True)
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
    for bpp, psnr, model in zip(bpp_lists, psnr_lists, models):
        if model == "BPG" or model == "JEPG2000" or model == "VVC"or model == "[BPG]" or model == "[JEPG2000]" or model == "[VVC]" or model == "[Baseline]":
            plt.plot(bpp, psnr, marker='o', label=model, linestyle='--')
        else:
            plt.plot(bpp, psnr, marker='o', label=model)
    plt.xlabel('bpp')
    plt.ylabel('PSNR(db)')
    plt.legend(loc='lower right')
    plt.grid(True)
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
    parser.add_argument('--dir_path',type=str, default="result/2-fig5/vic")
    args = parser.parse_args()
    main(args)