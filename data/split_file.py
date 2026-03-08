import os
import shutil
import random
from pathlib import Path

# ===================== CONFIG =====================
DATASET_DIR = r"D:/normal_file/coder/PSU-Code/Sophomore-Project/module/Aj_Nikom/project/data/real_data"
VALID_COUNT = 5  # จำนวนภาพที่ต้องการแยกเป็น valid
TEST_COUNT = 6  # จำนวนภาพที่ต้องการแยกเป็น test
RANDOM_SEED = 42
# ==================================================

random.seed(RANDOM_SEED)

src_img = Path(DATASET_DIR) / "train" / "images"
src_lbl = Path(DATASET_DIR) / "train" / "labels"

# สร้างโฟลเดอร์ปลายทาง (Roboflow/YOLOv8 standard)
for split in ("valid", "test"):
    (Path(DATASET_DIR) / split / "images").mkdir(parents=True, exist_ok=True)
    (Path(DATASET_DIR) / split / "labels").mkdir(parents=True, exist_ok=True)

# รายชื่อภาพทั้งหมดใน train
all_imgs = sorted(src_img.glob("*.jpg"))
total = len(all_imgs)

if VALID_COUNT + TEST_COUNT >= total:
    print(
        f"[X] รูปใน train มีแค่ {total} ภาพ แต่ขอ valid={VALID_COUNT} + test={TEST_COUNT} = {VALID_COUNT+TEST_COUNT}"
    )
    print("    กรุณาลด VALID_COUNT หรือ TEST_COUNT ลง")
    exit()

# สุ่มเลือก
selected = random.sample(all_imgs, VALID_COUNT + TEST_COUNT)
valid_set = selected[:VALID_COUNT]
test_set = selected[VALID_COUNT:]


def move_files(file_list, split_name):
    dst_img = Path(DATASET_DIR) / split_name / "images"
    dst_lbl = Path(DATASET_DIR) / split_name / "labels"
    moved = 0
    for img_path in file_list:
        lbl_path = src_lbl / (img_path.stem + ".txt")
        if not lbl_path.exists():
            print(f"  [skip] ไม่มี label: {img_path.name}")
            continue
        shutil.move(str(img_path), dst_img / img_path.name)
        shutil.move(str(lbl_path), dst_lbl / lbl_path.name)
        moved += 1
    return moved


valid_moved = move_files(valid_set, "valid")
test_moved = move_files(test_set, "test")
train_left = len(list(src_img.glob("*.jpg")))

print(f"[OK] แบ่งเสร็จแล้ว!")
print(f"     train : {train_left} ภาพ")
print(f"     valid : {valid_moved} ภาพ")
print(f"     test  : {test_moved}  ภาพ")

# อัพเดต data.yaml
yaml_path = Path(DATASET_DIR) / "data.yaml"
content = (
    f"path: {DATASET_DIR}\n"
    f"train: train/images\n"
    f"val:   valid/images\n"
    f"test:  test/images\n\n"
    f"nc: 1\n"
    f"names: ['fish']\n"
)
with open(yaml_path, "w") as f:
    f.write(content)
print(f"\n[OK] data.yaml อัพเดตแล้ว -> {yaml_path}")
print(f"\n--- Fine-tune command ---")
print(
    f"yolo detect train data={DATASET_DIR}/data.yaml model=best.pt epochs=50 imgsz=640 batch=16"
)
