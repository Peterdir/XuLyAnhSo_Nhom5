import cv2
import shutil
import numpy as np
from pathlib import Path
from tqdm import tqdm
from concurrent.futures import ThreadPoolExecutor, as_completed

# =====================================
# CẤU HÌNH
# =====================================

DATASET_DIR = Path("D:\Subject\XuLiAnhSo\Remake Final Project\dataset_final\dataset_final")
OUTPUT_DIR = Path("D:\Subject\XuLiAnhSo\Remake Final Project\dataset_preprocessed_v1")
SPLITS = ["train", "valid", "test"]
IMAGE_EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff"}

# =====================================
# HÀM HỖ TRỢ ĐỌC/GHI ẢNH AN TOÀN (UNICODE)
# =====================================

def imread_utf8(path):
    """Đọc ảnh hỗ trợ đường dẫn tiếng Việt/Unicode"""
    img_array = np.fromfile(path, np.uint8)
    return cv2.imdecode(img_array, cv2.IMREAD_COLOR)

def imwrite_utf8(path, img):
    """Ghi ảnh hỗ trợ đường dẫn tiếng Việt/Unicode"""
    is_success, im_buf_arr = cv2.imencode(path.suffix, img)
    if is_success:
        im_buf_arr.tofile(path)

# =====================================
# TIỀN XỬ LÝ ẢNH
# =====================================

def apply_unsharp_mask(image):
    """Hàm áp dụng Unsharp Masking để làm nét ảnh"""
    blurred = cv2.GaussianBlur(image, (5, 5), 0)
    alpha = 1.5
    beta = -0.5
    sharpened = cv2.addWeighted(image, alpha, blurred, beta, 0)
    return sharpened

def preprocess_image(img):
    img = cv2.resize(img, (640, 640))
    """Kết hợp khử nhiễu và làm nét"""
    # 1. Khử nhiễu bằng Median Filter (Giữ nguyên như ban đầu)
    filtered_img = cv2.medianBlur(img, 3)

    # 2. Làm nét bằng hàm Unsharp Mask tối ưu của bạn
    processed_img = apply_unsharp_mask(filtered_img)
    
    return processed_img

# =====================================
# WORKER CHO ĐA LUỒNG
# =====================================

def process_single_image(args):
    img_path, label_dir, out_image_dir, out_label_dir = args

    # 1. Đọc ảnh an toàn
    img = imread_utf8(img_path)
    if img is None:
        return f"Lỗi đọc ảnh: {img_path}"

    # 2. Thực hiện toàn bộ quy trình tiền xử lý
    processed = preprocess_image(img)

    # 3. Lưu ảnh an toàn
    save_img_path = out_image_dir / img_path.name
    imwrite_utf8(save_img_path, processed)

    # 4. Copy label tương ứng
    label_file = label_dir / f"{img_path.stem}.txt"
    if label_file.exists():
        shutil.copy2(label_file, out_label_dir / label_file.name)
        
    return None # Thành công trả về None

# =====================================
# XỬ LÝ TỪNG SPLIT
# =====================================

def process_split(split):
    image_dir = DATASET_DIR / "images" / split
    label_dir = DATASET_DIR / "labels" / split

    out_image_dir = OUTPUT_DIR / "images" / split
    out_label_dir = OUTPUT_DIR / "labels" / split

    out_image_dir.mkdir(parents=True, exist_ok=True)
    out_label_dir.mkdir(parents=True, exist_ok=True)

    if not image_dir.exists():
        return

    image_files = [f for f in image_dir.iterdir() if f.suffix.lower() in IMAGE_EXTS]
    print(f"\n[{split.upper()}] Đang xử lý {len(image_files)} ảnh...")

    # Đóng gói tham số cho đa luồng
    tasks = [(img_path, label_dir, out_image_dir, out_label_dir) for img_path in image_files]

    # Sử dụng ThreadPoolExecutor để chạy song song
    with ThreadPoolExecutor() as executor:
        futures = {executor.submit(process_single_image, task): task for task in tasks}
        
        # Kết hợp với tqdm để hiển thị thanh tiến trình
        for future in tqdm(as_completed(futures), total=len(tasks)):
            error_msg = future.result()
            if error_msg:
                print(error_msg)

# =====================================
# MAIN
# =====================================

if __name__ == "__main__":
    for split in SPLITS:
        process_split(split)

    print("\n✅ Hoàn tất tiền xử lý dataset!")