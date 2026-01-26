import os
import json

# ================== 配置 ==================
BDD_ROOT = "datasets/bdd"
OUTPUT_DIR = os.path.join(BDD_ROOT, "bdd_train")

SPLITS = {
    "train": "train.jsonl",
    "test": "test.jsonl",
    "val": "val.jsonl",   # 如果你的是 val，就改成 "val"
}

IMAGE_EXTS = (".jpg", ".jpeg", ".png")
DATASET_PREFIX = "datasets/bdd"  # 写入 jsonl 的路径前缀
# =========================================


def collect_images(folder):
    images = []
    for root, _, files in os.walk(folder):
        for f in files:
            if f.lower().endswith(IMAGE_EXTS):
                images.append(os.path.join(root, f))
    return sorted(images)


def write_jsonl(image_paths, output_file, split_name):
    with open(output_file, "w", encoding="utf-8") as f:
        for img_path in image_paths:
            rel_path = os.path.relpath(img_path, BDD_ROOT)
            json_line = {
                "image_path": f"{DATASET_PREFIX}/{rel_path.replace(os.sep, '/')}"
            }
            f.write(json.dumps(json_line, ensure_ascii=False) + "\n")


def main():
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    for split, jsonl_name in SPLITS.items():
        split_dir = os.path.join(BDD_ROOT, split)
        if not os.path.isdir(split_dir):
            print(f"[WARN] 跳过不存在的目录: {split_dir}")
            continue

        images = collect_images(split_dir)
        output_path = os.path.join(OUTPUT_DIR, jsonl_name)
        write_jsonl(images, output_path, split)

        print(f"[OK] {split}: {len(images)} images -> {output_path}")


if __name__ == "__main__":
    main()
