import os
import json

# 设置输入文件夹（a）和输出文件夹（b）
input_dir = 'datasets/JPEG/test'
output_dir = 'JPEG_train'
output_file = os.path.join(output_dir, 'vaild.jsonl')

# 确保输出目录存在
os.makedirs(output_dir, exist_ok=True)

# 构建 image_path 列表
with open(output_file, 'w') as f:
    for filename in sorted(os.listdir(input_dir)):
        if filename.lower().endswith(('.jpg', '.jpeg', '.png', '.bmp')):
            rel_path = os.path.join(input_dir, filename)
            json_line = {"image_path": rel_path}
            f.write(json.dumps(json_line) + '\n')

print(f"JSONL 文件已保存到：{output_file}")