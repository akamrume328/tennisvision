import os
import shutil
from tqdm import tqdm

# _collect_files_from_dataset は内部ヘルパー関数として残ります
def _collect_files_from_dataset(images_dir, labels_dir, dataset_name_prefix, split_name):
    dataset_identifier = f"{dataset_name_prefix} ({split_name})"
    if not os.path.isdir(images_dir):
        print(f"警告: {dataset_identifier} の画像ディレクトリが見つかりません: {images_dir}")
        return []
    
    collected_items = []
    # print(f"{dataset_identifier} からファイルを収集中...") # tqdmが同様の情報を提供するためコメントアウトも可
    for img_filename in tqdm(os.listdir(images_dir), desc=f"Scanning {dataset_identifier} images", leave=False):
        supported_image_extensions = ('.png', '.jpg', '.jpeg')
        if img_filename.lower().endswith(supported_image_extensions):
            image_path = os.path.join(images_dir, img_filename)
            base_name, _ = os.path.splitext(img_filename)
            label_filename = base_name + '.txt'
            label_path = os.path.join(labels_dir, label_filename)
            
            if not os.path.exists(labels_dir):
                # ラベルディレクトリの存在確認と警告の管理
                # この警告管理は関数スコープの属性を使って一度だけ表示するようにする
                if not hasattr(_collect_files_from_dataset, 'warned_missing_label_dirs'):
                    _collect_files_from_dataset.warned_missing_label_dirs = set()
                if labels_dir not in _collect_files_from_dataset.warned_missing_label_dirs:
                    print(f"情報: {dataset_identifier} のラベルディレクトリが見つかりません: {labels_dir}。画像のみ処理します。")
                    _collect_files_from_dataset.warned_missing_label_dirs.add(labels_dir)
                collected_items.append((image_path, None))
            elif os.path.exists(label_path):
                collected_items.append((image_path, label_path))
            else:
                collected_items.append((image_path, None))
    return collected_items

def _perform_merge_for_split_multiple(datasets, dest_merged_images_dir, dest_merged_labels_dir, split_name):
    """
    複数のデータセットを1つに融合する（特定スプリット用）。
    datasets は (images_dir, labels_dir, dataset_name_prefix) のタプルを要素としたリストを想定。
    """
    # 出力ディレクトリ作成
    for dir_path in [dest_merged_images_dir, dest_merged_labels_dir]:
        if not os.path.exists(dir_path):
            os.makedirs(dir_path)
            print(f"作成されたディレクトリ ({split_name}): {dir_path}")

    items_to_merge = []
    for images_dir, labels_dir, dataset_name_prefix in datasets:
        items_to_merge.extend(_collect_files_from_dataset(images_dir, labels_dir, dataset_name_prefix, split_name))

    if not items_to_merge:
        print(f"スプリット '{split_name}' で融合するアイテムがありません。")
        return

    print(f"スプリット '{split_name}': 合計 {len(items_to_merge)} 個のアイテムを融合します。")
    saved_item_count = 0
    for i, (image_path, label_path) in enumerate(tqdm(items_to_merge, desc=f"Merging {split_name} datasets", leave=False)):
        try:
            if not os.path.exists(image_path):
                print(f"警告 ({split_name}): 画像ファイルが見つかりません: {image_path}")
                continue
            
            # 元の拡張子を保持
            _, original_ext = os.path.splitext(image_path)
            new_base_filename = f'frame_{saved_item_count:06d}'
            
            image_save_path = os.path.join(dest_merged_images_dir, new_base_filename + original_ext)
            shutil.copy2(image_path, image_save_path)

            if label_path:
                label_save_path = os.path.join(dest_merged_labels_dir, new_base_filename + '.txt')
                shutil.copy2(label_path, label_save_path)
            
            saved_item_count += 1
        except Exception as e:
            print(f"エラー処理中 ({split_name}) {image_path}" + (f" と {label_path}" if label_path else "") + f": {e}")

    print(f"スプリット '{split_name}' の融合処理完了。合計 {saved_item_count} 個のアイテムをそれぞれのディレクトリに保存しました。")
    print(f"  画像保存先: {dest_merged_images_dir}")
    print(f"  ラベル保存先: {dest_merged_labels_dir}")

def merge_datasets_with_splits_multiple(datasets, merged_base_dir, splits=["train", "test", "val"]):
    """
    複数のデータセットを指定されたスプリットごとに融合する。
    datasets は (base_dir, dataset_name_prefix) のタプルを要素としたリストを想定。
    """
    # _collect_files_from_dataset の警告セットリセット
    if hasattr(_collect_files_from_dataset, 'warned_missing_label_dirs'):
        del _collect_files_from_dataset.warned_missing_label_dirs

    for split in splits:
        print(f"\n--- スプリット '{split}' のマルチデータセット処理を開始 ---")
        merge_targets = []
        for base_dir, dataset_prefix in datasets:
            img_dir = os.path.join(base_dir, split, 'images')
            lbl_dir = os.path.join(base_dir, split, 'labels')
            merge_targets.append((img_dir, lbl_dir, dataset_prefix))

        merged_images_dir = os.path.join(merged_base_dir, split, 'images')
        merged_labels_dir = os.path.join(merged_base_dir, split, 'labels')
        _perform_merge_for_split_multiple(merge_targets, merged_images_dir, merged_labels_dir, split)
        print(f"--- スプリット '{split}' の処理完了 ---")

def create_data_yaml(merged_base_dir, splits=["train", "test", "val"]):
    """
    YOLOv8用のdata.yamlファイルを作成する（Google Driveマウント対応）
    """
    yaml_content = """# YOLOv8 dataset configuration for tennis ball detection
# This configuration is designed for use with Google Drive mounted directories

# Dataset paths - relative to this data.yaml file
# These paths assume the data.yaml is located in the root of the mounted dataset
train: train/images
val: val/images
test: test/images

# Class configuration
nc: 3  # number of classes
names: 
  0: player_front
  1: player_back
  2: tennis_ball

# Optional: dataset root path (relative to data.yaml location)
path: .

"""
    
    yaml_path = os.path.join(merged_base_dir, 'data.yaml')
    try:
        with open(yaml_path, 'w', encoding='utf-8') as f:
            f.write(yaml_content)
        print(f"data.yamlファイルを作成しました: {yaml_path}")
        print("注意: このdata.yamlは相対パスを使用しています。")
        print("Google Driveにアップロード後、Colabでマウントして使用してください。")
        print("例: /content/drive/MyDrive/datasets/data.yaml として配置した場合、")
        print("     train/images -> /content/drive/MyDrive/datasets/train/images")
        print("     val/images   -> /content/drive/MyDrive/datasets/val/images")
        print("     test/images  -> /content/drive/MyDrive/datasets/test/images")
    except Exception as e:
        print(f"data.yamlファイルの作成に失敗しました: {e}")

def create_colab_data_yaml(merged_base_dir, mount_point="/content/drive/MyDrive", dataset_subpath="datasets/final_merged_dataset"):
    """
    Google Colab + Google Drive用の絶対パス指定data.yamlファイルを作成する
    """
    dataset_base_path = f"{mount_point}/{dataset_subpath}"
    
    yaml_content = f"""# YOLOv8 dataset configuration for tennis ball detection
# Google Colab + Google Drive specific configuration

# Dataset paths - absolute paths for Google Drive mounted directories
train: {dataset_base_path}/train/images
val: {dataset_base_path}/val/images
test: {dataset_base_path}/test/images

# Class configuration
nc: 3  # number of classes
names: 
  0: player_front
  1: player_back
  2: tennis_ball

# Dataset root path
path: {dataset_base_path}

"""
    
    yaml_path = os.path.join(merged_base_dir, 'data_colab.yaml')
    try:
        with open(yaml_path, 'w', encoding='utf-8') as f:
            f.write(yaml_content)
        print(f"Colab用data.yamlファイルを作成しました: {yaml_path}")
        print(f"このファイルは以下のGoogle Drive構成を想定しています:")
        print(f"  マウントポイント: {mount_point}")
        print(f"  データセットサブパス: {dataset_subpath}")
        print(f"  最終的なデータセットパス: {dataset_base_path}")
    except Exception as e:
        print(f"Colab用data.yamlファイルの作成に失敗しました: {e}")

if __name__ == "__main__":
    # --- 設定項目 ---
    # 融合するデータセットのリスト (ベースパス, 識別名)
    datasets_to_merge = [
        ("../data/processed/datasets/tracking2", "tracking2"),
        ("../data/processed/datasets/tracking1", "tracking1"),
        ("../data/processed/datasets/tracking4", "tracking4"),
        ("../data/processed/datasets/tracking3", "tracking3"),

        # 必要に応じて追加
        # ("../data/processed/datasets/another_dataset", "AnotherDataset"),
    ]
    
    # 融合後のデータセットの保存先
    merged_dataset_base_path = '../data/processed/datasets/final_merged_dataset'
    
    # 処理するスプリット
    target_splits = ["train", "test", "val"] 
    # --- 設定項目ここまで ---

    if not datasets_to_merge:
        print("エラー: 融合するデータセットが指定されていません。")
    else:
        merge_datasets_with_splits_multiple(datasets_to_merge, merged_dataset_base_path, splits=target_splits)
        
        # 相対パス版とColab絶対パス版の両方のdata.yamlファイルを作成
        create_data_yaml(merged_dataset_base_path, splits=target_splits)
        create_colab_data_yaml(merged_dataset_base_path, mount_point="/content/drive/MyDrive", dataset_subpath="datasets/final_merged_dataset")
        
        print(f"\n=== 全ての処理が完了しました ===")
        print(f"融合データセット保存先: {merged_dataset_base_path}")
        print(f"作成されたファイル:")
        print(f"  - data.yaml (相対パス版): Google Driveアップロード用")
        print(f"  - data_colab.yaml (絶対パス版): Colab実行時の参考用")
