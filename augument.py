import os
import cv2

def is_empty_txt(file_path):
    """Hàm kiểm tra file txt rỗng (đã loại bỏ các ký tự ẩn như khoảng trắng, tab, null bytes)."""
    if not os.path.exists(file_path):
        return False
        
    if os.path.getsize(file_path) == 0:
        return True
    try:
        with open(file_path, 'r', encoding='utf-8-sig') as f:
            content = f.read()
            # Xóa bỏ null bytes do lỗi phần mềm gán nhãn và gọt sạch khoảng trắng/dòng trống
            cleaned_content = content.replace('\x00', '').strip()
            if not cleaned_content:
                return True
    except UnicodeDecodeError:
        pass
    return False

def rotate_image(image, angle):
    """Hàm xoay ảnh theo góc chỉ định bằng OpenCV (giữ nguyên kích thước khung)."""
    (h, w) = image.shape[:2]
    center = (w // 2, h // 2)
    
    # Tính toán ma trận xoay
    M = cv2.getRotationMatrix2D(center, angle, 1.0)
    
    # Thực hiện xoay ảnh
    rotated = cv2.warpAffine(image, M, (w, h))
    return rotated

# ==========================================
# 1. KHAI BÁO CÁC ĐƯỜNG DẪN THƯ MỤC
# ==========================================
images_dir = './dataset_split/images/train'             # Thư mục chứa ảnh gốc
labels_dir = './dataset_split/labels/train'             # Thư mục chứa file nhãn (.txt) gốc

aug_images_dir = './aug_images'     # Thư mục mới để lưu ảnh đã xoay
aug_labels_dir = './aug_labels'     # Thư mục mới để lưu file nhãn rỗng của ảnh xoay

# Tự động tạo thư mục xuất nếu chúng chưa tồn tại
os.makedirs(aug_images_dir, exist_ok=True)
os.makedirs(aug_labels_dir, exist_ok=True)

# ==========================================
# 2. XỬ LÝ TĂNG CƯỜNG DỮ LIỆU
# ==========================================
augmented_count = 0

print(f"Đang quét '{labels_dir}' để tìm và tách các ảnh có nhãn rỗng...")

for label_filename in os.listdir(labels_dir):
    if label_filename.endswith('.txt'):
        label_path = os.path.join(labels_dir, label_filename)
        
        # Kiểm tra xem đây có phải là file nhãn rỗng (ảnh nền) không
        if is_empty_txt(label_path):
            base_name, _ = os.path.splitext(label_filename)
            
            # Tìm file ảnh tương ứng (hỗ trợ nhiều định dạng ảnh phổ biến)
            img_path = None
            img_ext_found = ""
            for ext in ['.jpg', '.jpeg', '.png', '.webp']:
                temp_path = os.path.join(images_dir, base_name + ext)
                if os.path.exists(temp_path):
                    img_path = temp_path
                    img_ext_found = ext
                    break
            
            if img_path:
                # Đọc ảnh gốc bằng thư viện OpenCV
                img = cv2.imread(img_path)
                
                if img is not None:
                    # Xoay +45 độ (qua trái) và -45 độ (qua phải)
                    img_rot_left = rotate_image(img, 45)
                    img_rot_right = rotate_image(img, -45)
                    
                    # Đặt tên cho các file mới
                    name_left = f"{base_name}_rotL45"
                    name_right = f"{base_name}_rotR45"
                    
                    # LƯU ẢNH MỚI: Xuất thẳng vào thư mục aug_images
                    cv2.imwrite(os.path.join(aug_images_dir, f"{name_left}{img_ext_found}"), img_rot_left)
                    cv2.imwrite(os.path.join(aug_images_dir, f"{name_right}{img_ext_found}"), img_rot_right)
                    
                    # LƯU NHÃN MỚI: Tạo file txt rỗng tương ứng vào thư mục aug_labels
                    open(os.path.join(aug_labels_dir, f"{name_left}.txt"), 'w').close()
                    open(os.path.join(aug_labels_dir, f"{name_right}.txt"), 'w').close()
                    
                    augmented_count += 1
                    print(f"[-] Đã xử lý và xuất ra thư mục riêng: {base_name}{img_ext_found}")

# ==========================================
# 3. TỔNG KẾT
# ==========================================
print("-" * 40)
print(f"Hoàn tất! Đã tăng cường thành công {augmented_count} ảnh nền rỗng.")
print(f"Tổng cộng {augmented_count * 2} file ảnh mới đã được lưu tại: '{aug_images_dir}'")
print(f"Tổng cộng {augmented_count * 2} file nhãn mới đã được lưu tại: '{aug_labels_dir}'")