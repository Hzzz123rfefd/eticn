import cv2
import os

def resize_images(input_folder, output_folder, width=500, height=375):
    os.makedirs(output_folder, exist_ok=True)
    for filename in os.listdir(input_folder):
        input_path = os.path.join(input_folder, filename)
        if os.path.isfile(input_path):
            img = cv2.imread(input_path)
            if img is None:
                print(f"跳过非图片文件: {filename}")
                continue
            resized_img = cv2.resize(img, (width, height))
            output_path = os.path.join(output_folder, filename)
            cv2.imwrite(output_path, resized_img)
            print(f"已处理: {filename}")

if __name__ == "__main__":
    input_folder = "datasets/JPEG/TEST"  # 修改为你的输入文件夹路径
    output_folder = "datasets/JPEG/test"  # 修改为你的输出文件夹路径
    resize_images(input_folder, output_folder)