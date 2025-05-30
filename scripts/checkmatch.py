import cv2
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from pathlib import Path
import random # 追加
from matplotlib.widgets import Button # 追加

# グローバル変数で現在の状態を管理
current_image_display_index = 0
images_to_display_paths = []
labels_to_display_paths = []
class_names_for_display = None
fig_main_display, ax_main_display = None, None
button_next_img = None
button_prev_img = None

def visualize_yolo_annotation(ax, image_path_str, label_path_str, class_names=None, current_idx=-1, total_images=-1): # ax とインデックス情報を引数に追加
    """
    指定されたmatplotlibのAxesに画像とYOLO形式のラベルからバウンディングボックスを描画する。

    Args:
        ax (matplotlib.axes.Axes): 描画対象のAxes。
        image_path_str (str): 画像ファイルのパス。
        label_path_str (str): YOLO形式のラベルファイルのパス。
        class_names (list, optional): クラス名のリスト。
        current_idx (int, optional): 現在表示している画像のインデックス。
        total_images (int, optional): 表示する総画像数。
    """
    ax.clear() # 描画前にクリア
    image_path = Path(image_path_str)
    label_path = Path(label_path_str)

    if not image_path.exists():
        print(f"エラー: 画像ファイルが見つかりません: {image_path}")
        ax.text(0.5, 0.5, f"エラー: 画像ファイルが見つかりません:\n{image_path}", ha='center', va='center')
        ax.set_title(f"エラー: {image_path.name}")
        if fig_main_display: fig_main_display.canvas.draw_idle()
        return
    if not label_path.exists():
        # ラベルファイルがなくても画像は表示するが、警告は出す
        print(f"警告: ラベルファイルが見つかりません: {label_path} (画像のみ表示)")

    img_bgr = cv2.imread(str(image_path))
    if img_bgr is None:
        print(f"エラー: 画像を読み込めません: {image_path}")
        ax.text(0.5, 0.5, f"エラー: 画像を読み込めません:\n{image_path}", ha='center', va='center')
        ax.set_title(f"エラー: {image_path.name}")
        if fig_main_display: fig_main_display.canvas.draw_idle()
        return
    img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
    h, w, _ = img_rgb.shape

    ax.imshow(img_rgb)
    title_str = f"Image: {image_path.name}"
    if current_idx != -1 and total_images != -1:
        title_str += f" ({current_idx + 1}/{total_images})"
    if label_path.exists():
        title_str += f"\nLabel: {label_path.name}"
    else:
        title_str += f"\nLabel: (ファイルなし - {label_path.name})"
    ax.set_title(title_str)


    if label_path.exists():
        with open(label_path, 'r') as f:
            for line in f:
                parts = line.strip().split()
                if len(parts) < 5:
                    print(f"警告: ラベルファイル {label_path.name} の行 '{line.strip()}' は形式が不正です。")
                    continue
                
                try:
                    class_id = int(parts[0])
                    x_center_norm = float(parts[1])
                    y_center_norm = float(parts[2])
                    width_norm = float(parts[3])
                    height_norm = float(parts[4])
                except ValueError:
                    print(f"警告: ラベルファイル {label_path.name} の行 '{line.strip()}' の数値変換に失敗しました。")
                    continue

                # 正規化された座標を絶対座標に変換
                box_w = width_norm * w
                box_h = height_norm * h
                x_min = (x_center_norm * w) - (box_w / 2)
                y_min = (y_center_norm * h) - (box_h / 2)

                # バウンディングボックスを描画
                rect = patches.Rectangle((x_min, y_min), box_w, box_h,
                                         linewidth=1, edgecolor='r', facecolor='none')
                ax.add_patch(rect)

                # クラス名を表示 (オプション)
                if class_names and 0 <= class_id < len(class_names):
                    label_text = class_names[class_id]
                    ax.text(x_min, y_min - 5, label_text, color='red', fontsize=8, 
                            bbox=dict(facecolor='white', alpha=0.5, pad=0))
                else:
                    ax.text(x_min, y_min - 5, f"ID: {class_id}", color='red', fontsize=8,
                            bbox=dict(facecolor='white', alpha=0.5, pad=0))
    else:
        # ラベルファイルがない場合のメッセージを画像中央に表示（オプション）
        # ax.text(0.5, 0.1, "ラベルファイルが見つかりません", ha='center', va='center', transform=ax.transAxes, color='orange', fontsize=10)
        pass # タイトルで通知済み

    if fig_main_display: fig_main_display.canvas.draw_idle()


def show_current_image_in_viewer():
    global current_image_display_index, images_to_display_paths, labels_to_display_paths, class_names_for_display, ax_main_display
    if not images_to_display_paths:
        ax_main_display.clear()
        ax_main_display.text(0.5,0.5, "表示する画像がありません。", ha="center", va="center")
        if fig_main_display: fig_main_display.canvas.draw_idle()
        return

    if 0 <= current_image_display_index < len(images_to_display_paths):
        img_path = images_to_display_paths[current_image_display_index]
        lbl_path = labels_to_display_paths[current_image_display_index]
        visualize_yolo_annotation(ax_main_display, str(img_path), str(lbl_path), 
                                  class_names=class_names_for_display,
                                  current_idx=current_image_display_index,
                                  total_images=len(images_to_display_paths))
    else:
        ax_main_display.clear()
        ax_main_display.text(0.5,0.5, "画像の範囲外です。", ha="center", va="center")
        if fig_main_display: fig_main_display.canvas.draw_idle()

    # ボタンの状態更新
    if button_prev_img:
        button_prev_img.set_active(current_image_display_index > 0)
    if button_next_img:
        button_next_img.set_active(current_image_display_index < len(images_to_display_paths) - 1)


def on_next_image(event):
    global current_image_display_index, images_to_display_paths
    if current_image_display_index < len(images_to_display_paths) - 1:
        current_image_display_index += 1
        show_current_image_in_viewer()
    else:
        print("これが最後の画像です。")

def on_prev_image(event):
    global current_image_display_index
    if current_image_display_index > 0:
        current_image_display_index -= 1
        show_current_image_in_viewer()
    else:
        print("これが最初の画像です。")


if __name__ == '__main__':
    # --- 設定 ---
    dataset_base_dir = Path("C:/Users/akama/AppData/Local/Programs/Python/Python310/python_file/projects/tennisvision/data/processed/datasets/tracking1") # データセットのベースパス
    split_type = 'train' # 'train', 'val', または 'test' を指定
    num_images_to_check = 10 # 確認する画像の枚数

    # (オプション) data.yaml からクラス名を取得する場合
    # data_yaml_path = dataset_base_dir / 'data.yaml'
    # class_names_list = []
    # if data_yaml_path.exists():
    #     import yaml
    #     with open(data_yaml_path, 'r') as f:
    #         data_cfg = yaml.safe_load(f)
    #         if 'names' in data_cfg:
    #             class_names_list = data_cfg['names']
    # else:
    #     print(f"警告: data.yaml が見つかりません: {data_yaml_path}")

    # class_names_for_display = class_names_list 
    class_names_for_display = None # ここで設定

    # --- 画像とラベルのリストを準備 ---
    image_dir = dataset_base_dir / split_type / 'images'
    label_dir = dataset_base_dir / split_type / 'labels'
    
    all_image_files = []
    if image_dir.exists():
        all_image_files = sorted([f for f in image_dir.iterdir() if f.is_file() and f.suffix.lower() in ['.png', '.jpg', '.jpeg']])
        random.shuffle(all_image_files) # リストをシャッフル
        images_to_display_paths = all_image_files[:num_images_to_check] # 指定枚数だけ選択
        
        for img_path in images_to_display_paths:
            labels_to_display_paths.append(label_dir / (img_path.stem + ".txt"))
    
    if images_to_display_paths:
        # MatplotlibのFigureとAxesを準備
        fig_main_display, ax_main_display = plt.subplots()
        plt.subplots_adjust(bottom=0.2) # ボタン用のスペースを確保

        # 「前へ」ボタン
        ax_prev = plt.axes([0.7, 0.05, 0.1, 0.075]) # [left, bottom, width, height]
        button_prev_img = Button(ax_prev, 'Prev')
        button_prev_img.on_clicked(on_prev_image)

        # 「次へ」ボタン
        ax_next = plt.axes([0.81, 0.05, 0.1, 0.075])
        button_next_img = Button(ax_next, 'Next')
        button_next_img.on_clicked(on_next_image)
        
        current_image_display_index = 0 # 最初の画像から表示
        show_current_image_in_viewer() # 最初の画像を表示
        
        plt.show()
    else:
        print(f"{split_type}/images ディレクトリに画像が見つからないか、処理対象の画像がありません。")
