import argparse
import os
import matplotlib.pyplot as plt


def main(args):
    os.makedirs(args.save_dir, exist_ok = True)
    years = [2023, 2024, 2025, 2026, 2027, 2028, 2029, 2030, 2031, 2032]
    market_size = [7.9, 8.4, 9.0, 9.6, 10.3, 11.1, 12.0, 13.0, 14.0, 15.0]

    plt.rcParams.update({
    'font.family': ['Times New Roman', 'SimSun'],
    'axes.unicode_minus': False,                 
    'axes.labelweight': 'bold',
    'axes.labelsize': 14,
    'xtick.labelsize': 12,
    'ytick.labelsize': 12,
    'legend.fontsize': 12
    })

    # 创建图像
    plt.figure(figsize=(8, 5))

    # 绘制柱状图（不指定颜色）
    bars = plt.bar(years, market_size)

    # 添加数值标签
    for bar in bars:
        height = bar.get_height()
        plt.text(bar.get_x() + bar.get_width() / 2,
                height,
                f'{height:.1f}',
                ha='center',
                va='bottom',
                fontsize=10)

    # 坐标轴和标题
    plt.xlabel("年份")
    plt.ylabel("市场规模（十亿美元）")
    # plt.title("Global Traffic Camera Market Size Forecast (2023–2032)", fontsize=15, fontweight='bold')

    plt.xticks(years, rotation=45)
    plt.tight_layout()

    plt.savefig(os.path.join(args.save_dir, "fig1-1.png"), dpi=300)
    plt.show()


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--save_dir", type=str, default = "result/fig1-1")
    args = parser.parse_args()
    main(args)