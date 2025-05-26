import json
import numpy as np
from scipy import interpolate
from scipy.integrate import quad
import argparse


def load_data(json_path):
    with open(json_path, 'r') as f:
        data = json.load(f)
    return data['bpp'], data['psnr']


def bd_rate(bpp1, psnr1, bpp2, psnr2):
    log_bpp1 = np.log(bpp1)
    log_bpp2 = np.log(bpp2)

    p1 = interpolate.PchipInterpolator(psnr1, log_bpp1)
    p2 = interpolate.PchipInterpolator(psnr2, log_bpp2)

    psnr_min = max(min(psnr1), min(psnr2))
    psnr_max = min(max(psnr1), max(psnr2))
    if psnr_max <= psnr_min:
        raise ValueError("PSNR 范围没有重叠，无法计算 BD-Rate")

    int1, _ = quad(p1, psnr_min, psnr_max)
    int2, _ = quad(p2, psnr_min, psnr_max)
    avg_diff = (int2 - int1) / (psnr_max - psnr_min)

    return (np.exp(avg_diff) - 1) * 100  

def bd_psnr(bpp1, psnr1, bpp2, psnr2):
    log_bpp1 = np.log(bpp1)
    log_bpp2 = np.log(bpp2)

    p1 = interpolate.PchipInterpolator(log_bpp1, psnr1)
    p2 = interpolate.PchipInterpolator(log_bpp2, psnr2)

    min_int = max(min(log_bpp1), min(log_bpp2))
    max_int = min(max(log_bpp1), max(log_bpp2))
    if max_int <= min_int:
        raise ValueError("BPP 范围没有重叠，无法计算 BD-PSNR")

    int1, _ = quad(p1, min_int, max_int)
    int2, _ = quad(p2, min_int, max_int)
    return (int2 - int1) / (max_int - min_int)

def main(args):
    bpp_ref, psnr_ref = load_data(args.base_result_path)
    bpp_test, psnr_test = load_data(args.target_result_path)

    bdpsnr = bd_psnr(bpp_ref, psnr_ref, bpp_test, psnr_test)
    bdrate = bd_rate(bpp_ref, psnr_ref, bpp_test, psnr_test)

    print(f"BD-PSNR: {bdpsnr:.3f} dB")
    print(f"BD-Rate: {bdrate:.2f} %")    



if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--base_result_path", type=str, default = "ref_result.json")
    parser.add_argument("--target_result_path", type=str, default = "test_result.json")
    args = parser.parse_args()
    main(args)