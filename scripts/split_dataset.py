# filepath: split_dataset.py
import os
import random
from pathlib import Path
import shutil
import yaml

def reorganize_dataset_for_yolo(dataset_base_dir, data_yaml_filename='data.yaml', train_ratio=0.7, val_ratio=0.15):
    """
    Moves images and labels from specified directories into train/images, train/labels, 
    val/images, val/labels, test/images, test/labels directories, splits them according to ratios, 
    and updates data.yaml.
    Assumes original images are in `dataset_base_dir/images` and labels in `dataset_base_dir/labels`.
    """
    dataset_base_dir = Path(dataset_base_dir)
    data_yaml_path = dataset_base_dir / data_yaml_filename

    original_images_dir = dataset_base_dir / 'images'
    original_labels_dir = dataset_base_dir / 'labels'

    # Define new directory structure paths
    new_train_images_dir = dataset_base_dir / 'train' / 'images'
    new_train_labels_dir = dataset_base_dir / 'train' / 'labels'
    new_val_images_dir = dataset_base_dir / 'val' / 'images'
    new_val_labels_dir = dataset_base_dir / 'val' / 'labels'
    new_test_images_dir = dataset_base_dir / 'test' / 'images'
    new_test_labels_dir = dataset_base_dir / 'test' / 'labels'

    # Create new directories
    new_train_images_dir.mkdir(parents=True, exist_ok=True)
    new_train_labels_dir.mkdir(parents=True, exist_ok=True)
    new_val_images_dir.mkdir(parents=True, exist_ok=True)
    new_val_labels_dir.mkdir(parents=True, exist_ok=True)
    new_test_images_dir.mkdir(parents=True, exist_ok=True)
    new_test_labels_dir.mkdir(parents=True, exist_ok=True)

    image_files = []
    if original_images_dir.exists():
        image_files = [f for f in original_images_dir.iterdir() if f.is_file() and f.suffix.lower() in ['.png', '.jpg', '.jpeg']]
    else:
        print(f"Warning: Original images directory not found: {original_images_dir}")
        return

    if not image_files:
        print(f"No image files found in {original_images_dir}")
        return

    # Create a list of (image_path, label_path) tuples
    dataset_pairs = []
    for img_path in image_files:
        label_filename = img_path.stem + '.txt'
        label_path = original_labels_dir / label_filename
        if label_path.exists():
            dataset_pairs.append((img_path, label_path))
        else:
            print(f"Warning: Label file not found for image {img_path.name} at {label_path}")

    if not dataset_pairs:
        print("No image-label pairs found.")
        return

    random.shuffle(dataset_pairs)

    total_pairs = len(dataset_pairs)
    train_end_idx = int(total_pairs * train_ratio)
    val_end_idx = train_end_idx + int(total_pairs * val_ratio)

    train_pairs = dataset_pairs[:train_end_idx]
    val_pairs = dataset_pairs[train_end_idx:val_end_idx]
    test_pairs = dataset_pairs[val_end_idx:]

    def move_pairs(pairs, target_img_dir, target_lbl_dir):
        for img_path, lbl_path in pairs:
            try:
                shutil.move(str(img_path), str(target_img_dir / img_path.name))
                shutil.move(str(lbl_path), str(target_lbl_dir / lbl_path.name))
            except Exception as e:
                print(f"Error moving file: {e}")

    print(f"Moving {len(train_pairs)} pairs to training set...")
    move_pairs(train_pairs, new_train_images_dir, new_train_labels_dir)
    print(f"Moving {len(val_pairs)} pairs to validation set...")
    move_pairs(val_pairs, new_val_images_dir, new_val_labels_dir)
    if test_pairs:
        print(f"Moving {len(test_pairs)} pairs to test set...")
        move_pairs(test_pairs, new_test_images_dir, new_test_labels_dir)
    else:
        print("No pairs for the test set.")

    # Update data.yaml
    if data_yaml_path.exists():
        with open(data_yaml_path, 'r') as f:
            data_config = yaml.safe_load(f)

        data_config['train'] = '../train/images'  # Relative to 'path'
        data_config['val'] = '../val/images'    # Relative to 'path'
        data_config['test'] = '../test/images' 

        # Update test entry
        if test_pairs: # テストセットが実際に作成された場合
            data_config['test'] = 'test/images'
            print("Updated 'test' entry in data.yaml to point to test/images.")
        elif 'test' in data_config: # テストセットが作成されず、yamlに既存のエントリがある場合
            del data_config['test']
            print("Removed 'test' entry from data.yaml as no test set was created.")


        with open(data_yaml_path, 'w') as f:
            yaml.dump(data_config, f, sort_keys=False, default_flow_style=None)
        print(f"Updated {data_yaml_path}")
    else:
        print(f"Warning: data.yaml not found at {data_yaml_path}. Cannot update.")

    print("Dataset reorganization complete.")
    
    # Optional: Clean up old train.txt, val.txt, and test.txt if they exist
    for txt_file_name in ['train.txt', 'val.txt', 'test.txt']:
        txt_file_path = dataset_base_dir / txt_file_name
        if txt_file_path.exists():
            try:
                txt_file_path.unlink()
                print(f"Removed old {txt_file_name}")
            except OSError as e:
                print(f"Error removing {txt_file_name}: {e}")

    # Optional: Attempt to remove original 'images' and 'labels' directories
    # if they are now empty.
    for original_dir in [original_images_dir, original_labels_dir]:
        try:
            if original_dir.exists() and not any(original_dir.iterdir()):
                original_dir.rmdir()
                print(f"Removed empty original directory: {original_dir}")
        except OSError as e:
            print(f"Could not remove {original_dir} (it might not be empty or other issues): {e}")


if __name__ == "__main__":
    # --- 設定 ---
    # \'data.yaml\' が配置されるディレクトリ (通常は dataset/)
    # このディレクトリ内に元の images/ および labels/ ディレクトリがあると仮定
    dataset_root_dir = Path("C:/Users/akama/AppData/Local/Programs/Python/Python310/python_file/projects/tennisvision/data/processed/dataset")
    
    # 分割比率 (train_ratio + val_ratio <= 1.0 であること)
    # 残りが test_ratio となります。
    train_split_ratio = 0.7
    val_split_ratio = 0.15
    # test_split_ratio は自動的に 1.0 - train_split_ratio - val_split_ratio になります。
    # --- 設定終了 ---

    if not dataset_root_dir.exists() or not (dataset_root_dir / 'images').exists() or not (dataset_root_dir / 'labels').exists():
        print(f"エラー: データセットのルートディレクトリ、またはその中の \'images\'/\'labels\' ディレクトリが見つかりません: {dataset_root_dir}")
        print("パスが正しいことを確認してください。\'images\' および \'labels\' ディレクトリが {dataset_root_dir} 直下にある必要があります。")
    elif (train_split_ratio + val_split_ratio) > 1.0:
        print(f"エラー: train_ratio ({train_split_ratio}) と val_ratio ({val_split_ratio}) の合計が1.0を超えています。")
    else:
        print(f"データセットを再編成します: {dataset_root_dir}")
        print(f"訓練セットの割合: {train_split_ratio}, 検証セットの割合: {val_split_ratio}")
        test_split_ratio = 1.0 - train_split_ratio - val_split_ratio
        print(f"テストセットの割合 (自動計算): {test_split_ratio:.2f}")
        
        reorganize_dataset_for_yolo(dataset_root_dir, train_ratio=train_split_ratio, val_ratio=val_split_ratio)
        
        print("\\n--- データセットの再編成処理が完了しました。 ---")
        print(f"ディレクトリ構造を確認してください: {dataset_root_dir}")
        print("data.yaml が正しく \'train/images\', \'val/images\', \'test/images\' (存在する場合) を指し、\'path: .\' となっていることを確認してください。")
