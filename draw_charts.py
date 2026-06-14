import pandas as pd
import matplotlib.pyplot as plt
import os

print("=== BẮT ĐẦU VẼ BIỂU ĐỒ CHUẨN STYLE YOLOv8 ===")

csv_path = "results.csv"

if os.path.exists(csv_path):

    print("✅ Đã đọc thành công file results.csv!")
    df = pd.read_csv(csv_path)

    # Xóa khoảng trắng thừa
    df.columns = df.columns.str.strip()

    epochs = df["epoch"]

    plt.rcParams.update({
        "font.size": 10
    })

    # ==========================
    # Hàm làm mượt EMA
    # ==========================
    def create_smooth_line(y):
        return y.ewm(span=5, adjust=False).mean()

    # ==========================
    # Hàm vẽ biểu đồ
    # ==========================
    def draw_yolo_chart(col_name, save_filename):

        if col_name not in df.columns:
            print(f"⚠️ Không tìm thấy cột: {col_name}")
            return

        y_raw = df[col_name]
        y_smooth = create_smooth_line(y_raw)

        # Tự động điều chỉnh kích thước marker
        num_epochs = len(epochs)

        if num_epochs > 200:
            marker_size = 1
        elif num_epochs > 100:
            marker_size = 2
        elif num_epochs > 50:
            marker_size = 3
        else:
            marker_size = 5

        plt.figure(figsize=(8, 6))

        # Đường gốc
        plt.plot(
            epochs,
            y_raw,
            marker='o',
            linestyle='-',
            color='#1f77b4',
            markersize=marker_size,
            linewidth=2,
            label='results'
        )

        # Đường làm mượt
        plt.plot(
            epochs,
            y_smooth,
            linestyle=':',
            color='#ff7f0e',
            linewidth=3,
            label='smooth'
        )

        # Style YOLOv8
        plt.title(col_name, fontsize=12)
        plt.xlabel("Epoch")
        plt.ylabel(col_name)

        plt.legend(loc="best")
        plt.grid(False)

        ax = plt.gca()
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)

        plt.tight_layout()

        plt.savefig(
            save_filename,
            dpi=300,
            bbox_inches="tight",
            facecolor="white"
        )

        plt.close()

        print(f"✅ Đã lưu: {save_filename}")

    # ==========================
    # VẼ BIỂU ĐỒ
    # ==========================
    print("⏳ Đang xuất các file ảnh...")

    draw_yolo_chart(
        "train/box_loss",
        "1a_Train_Box_Loss.png"
    )

    draw_yolo_chart(
        "val/box_loss",
        "1b_Val_Box_Loss.png"
    )

    draw_yolo_chart(
        "metrics/precision(B)",
        "2_Precision.png"
    )

    draw_yolo_chart(
        "metrics/recall(B)",
        "3_Recall.png"
    )

    draw_yolo_chart(
        "metrics/mAP50(B)",
        "4a_mAP50.png"
    )

    draw_yolo_chart(
        "metrics/mAP50-95(B)",
        "4b_mAP50_95.png"
    )

    print("\n🎉 HOÀN THÀNH!")
    print("Các biểu đồ đã được lưu trong thư mục hiện tại.")

else:
    print("❌ LỖI: Không tìm thấy file results.csv")