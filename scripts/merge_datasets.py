import os
import shutil
import cv2
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

def _perform_merge_for_split(src_dataset1_images_dir, src_dataset1_labels_dir,
                             src_dataset2_images_dir, src_dataset2_labels_dir,
                             dest_merged_images_dir, dest_merged_labels_dir, split_name):
    """
    特定のスプリットについて、2つのデータセットを1つに融合する。
    """
    # 出力ディレクトリの作成
    for dir_path in [dest_merged_images_dir, dest_merged_labels_dir]:
        if not os.path.exists(dir_path):
            os.makedirs(dir_path)
            print(f"作成されたディレクトリ ({split_name}): {dir_path}")

    items_to_merge = []
    items_to_merge.extend(_collect_files_from_dataset(src_dataset1_images_dir, src_dataset1_labels_dir, "データセット1", split_name))
    items_to_merge.extend(_collect_files_from_dataset(src_dataset2_images_dir, src_dataset2_labels_dir, "データセット2", split_name))

    if not items_to_merge:
        print(f"スプリット '{split_name}' で融合するアイテム（画像/ラベル）が見つかりませんでした。")
        return

    print(f"スプリット '{split_name}': 合計 {len(items_to_merge)} 個のアイテムを融合します。")

    saved_item_count = 0 # スプリットごとにカウンターをリセット
    for i, (image_path, label_path) in enumerate(tqdm(items_to_merge, desc=f"Merging {split_name} datasets", leave=False)):
        try:
            img = cv2.imread(image_path)
            if img is None:
                print(f"警告 ({split_name}): 画像を読み込めませんでした: {image_path}")
                continue
            
            new_base_filename = f'frame_{saved_item_count:06d}'
            
            image_save_path = os.path.join(dest_merged_images_dir, new_base_filename + '.png')
            cv2.imwrite(image_save_path, img)

            if label_path:
                label_save_path = os.path.join(dest_merged_labels_dir, new_base_filename + '.txt')
                shutil.copy2(label_path, label_save_path)
            
            saved_item_count += 1
        except Exception as e:
            print(f"エラー処理中 ({split_name}) {image_path}" + (f" と {label_path}" if label_path else "") + f": {e}")

    print(f"スプリット '{split_name}' の融合処理完了。合計 {saved_item_count} 個のアイテムをそれぞれのディレクトリに保存しました。")
    print(f"  画像保存先: {dest_merged_images_dir}")
    print(f"  ラベル保存先: {dest_merged_labels_dir}")

def merge_datasets_with_splits(dataset1_base_dir, dataset2_base_dir, merged_base_dir, splits=["train", "test", "val"]):
    """
    指定されたスプリット（train, test, valなど）ごとにデータセットを融合する。
    """
    # _collect_files_from_dataset の警告セットをリセット（実行ごとにクリアするため）
    if hasattr(_collect_files_from_dataset, 'warned_missing_label_dirs'):
        del _collect_files_from_dataset.warned_missing_label_dirs

    for split in splits:
        print(f"\n--- スプリット '{split}' の処理を開始 ---")

        d1_images_dir = os.path.join(dataset1_base_dir, split, 'images')
        d1_labels_dir = os.path.join(dataset1_base_dir, split, 'labels')
        d2_images_dir = os.path.join(dataset2_base_dir, split, 'images')
        d2_labels_dir = os.path.join(dataset2_base_dir, split, 'labels')
        
        merged_images_dir = os.path.join(merged_base_dir, split, 'images')
        merged_labels_dir = os.path.join(merged_base_dir, split, 'labels')

        _perform_merge_for_split(
            d1_images_dir, d1_labels_dir,
            d2_images_dir, d2_labels_dir,
            merged_images_dir, merged_labels_dir,
            split_name=split
        )
        print(f"--- スプリット '{split}' の処理が完了 ---")

if __name__ == "__main__":
    # --- 設定項目 ---
    # 1つ目のデータセットのベースパス
    dataset1_base_path = '../data/processed/datasets/tracking3' 
    # 2つ目のデータセットのベースパス
    dataset2_base_path = '../data/processed/datasets/merged_dataset_split' 
    # 融合後のデータセットの保存先ベースパス
    merged_dataset_base_path = '../data/processed/datasets/merged_dataset_split' # 新しい出力先名
    
    # 処理するスプリットのリスト
    # 必要に応じて変更してください (例: ["train", "val"] のみなど)
    target_splits = ["train", "test", "val"] 
    # --- 設定項目ここまで ---

    # ディレクトリパスは実際の環境に合わせて変更してください
    # 例:
    # dataset1_base_path = 'C:/path/to/your/first_dataset_base'
    # dataset2_base_path = 'C:/path/to/your/second_dataset_base'
    # merged_dataset_base_path = 'C:/path/to/your/merged_dataset_base'

    merge_datasets_with_splits(dataset1_base_path, dataset2_base_path, 
                               merged_dataset_base_path, splits=target_splits)
