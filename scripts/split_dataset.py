# filepath: split_dataset.py
import os
import random
from pathlib import Path
import shutil
import yaml

def reorganize_dataset_for_yolo(dataset_base_dir, data_yaml_filename='data.yaml', train_ratio=0.7, val_ratio=0.15):
    """
    指定されたディレクトリから画像とラベルを train/images, train/labels,
    val/images, val/labels, test/images, test/labels ディレクトリに移動し、指定された比率で分割し、
    data.yaml を更新します。
    元の画像は `dataset_base_dir/images` に、ラベルは `dataset_base_dir/labels` にあると仮定します。
    """
    dataset_base_dir = Path(dataset_base_dir)
    data_yaml_path = dataset_base_dir / data_yaml_filename

    original_images_dir = dataset_base_dir / 'images'
    original_labels_dir = dataset_base_dir / 'labels'

    # 新しいディレクトリ構造のパスを定義
    new_train_images_dir = dataset_base_dir / 'train' / 'images'
    new_train_labels_dir = dataset_base_dir / 'train' / 'labels'
    new_val_images_dir = dataset_base_dir / 'val' / 'images'
    new_val_labels_dir = dataset_base_dir / 'val' / 'labels'
    new_test_images_dir = dataset_base_dir / 'test' / 'images'
    new_test_labels_dir = dataset_base_dir / 'test' / 'labels'

    # 新しいディレクトリを作成
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
        print(f"警告: 元の画像ディレクトリが見つかりません: {original_images_dir}")
        return

    if not image_files:
        print(f"画像ファイルが {original_images_dir} に見つかりません")
        return

    # (画像パス, ラベルパス) のタプルのリストを作成
    dataset_pairs = []
    for img_path in image_files:
        label_filename = img_path.stem + '.txt'
        label_path = original_labels_dir / label_filename
        if label_path.exists():
            dataset_pairs.append((img_path, label_path))
        else:
            print(f"警告: 画像 {img_path.name} のラベルファイルが見つかりません {label_path}")

    if not dataset_pairs:
        print("画像とラベルのペアが見つかりません。")
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
                print(f"ファイルの移動中にエラーが発生しました: {e}")

    print(f"{len(train_pairs)} 個のペアを訓練セットに移動中...")
    move_pairs(train_pairs, new_train_images_dir, new_train_labels_dir)
    print(f"{len(val_pairs)} 個のペアを検証セットに移動中...")
    move_pairs(val_pairs, new_val_images_dir, new_val_labels_dir)
    if test_pairs:
        print(f"{len(test_pairs)} 個のペアをテストセットに移動中...")
        move_pairs(test_pairs, new_test_images_dir, new_test_labels_dir)
    else:
        print("テストセット用のペアはありません。")

    # data.yaml を更新
    if data_yaml_path.exists():
        with open(data_yaml_path, 'r') as f:
            data_config = yaml.safe_load(f)

        data_config['train'] = './train/images'  # 'path' からの相対パス
        data_config['val'] = './val/images'    # 'path' からの相対パス
        # data_config['test'] = '../test/images' # この行は下のロジックで設定されるため、一旦コメントアウトまたは削除しても良い

        # test エントリを更新
        if test_pairs: # テストセットが実際に作成された場合
            data_config['test'] = './test/images' # YOLOv5の規約に合わせて'../test/images'に修正
            print("data.yaml の 'test' エントリを test/images を指すように更新しました。")
        elif 'test' in data_config: # テストセットが作成されず、yamlに既存のエントリがある場合
            del data_config['test']
            print("テストセットが作成されなかったため、data.yaml から 'test' エントリを削除しました。")


        with open(data_yaml_path, 'w') as f:
            yaml.dump(data_config, f, sort_keys=False, default_flow_style=None)
        print(f"{data_yaml_path} を更新しました")
    else:
        print(f"警告: {data_yaml_path} に data.yaml が見つかりません。更新できません。")

    print("データセットの再編成が完了しました。")
    
    # オプション: 既存の train.txt, val.txt, test.txt があれば削除
    for txt_file_name in ['train.txt', 'val.txt', 'test.txt']:
        txt_file_path = dataset_base_dir / txt_file_name
        if txt_file_path.exists():
            try:
                txt_file_path.unlink()
                print(f"古い {txt_file_name} を削除しました")
            except OSError as e:
                print(f"{txt_file_name} の削除中にエラーが発生しました: {e}")

    # オプション: 元の 'images' および 'labels' ディレクトリが空であれば削除を試みる
    for original_dir in [original_images_dir, original_labels_dir]:
        try:
            if original_dir.exists() and not any(original_dir.iterdir()): # ディレクトリが存在し、かつ空であるかを確認
                original_dir.rmdir()
                print(f"空の元のディレクトリを削除しました: {original_dir}")
        except OSError as e: # 例: ディレクトリが空でない、権限がないなど
            print(f"{original_dir} を削除できませんでした (空でないか、他の問題がある可能性があります): {e}")


if __name__ == "__main__":
    # --- 設定 ---
    # 'data.yaml' が配置されるディレクトリ (通常は dataset/)
    # このディレクトリ内に元の images/ および labels/ ディレクトリがあると仮定
    dataset_root_dir = Path("C:/Users/akama/AppData/Local/Programs/Python/Python310/python_file/projects/tennisvision/data/processed/dataset")
    
    # 分割比率 (train_ratio + val_ratio <= 1.0 であること)
    # 残りが test_ratio となります。
    train_split_ratio = 0.7
    val_split_ratio = 0.15
    # test_split_ratio は自動的に 1.0 - train_split_ratio - val_split_ratio になります。
    # --- 設定終了 ---

    # 元の画像ファイル数を事前にカウント
    original_images_path = dataset_root_dir / 'images'
    initial_image_count = 0
    if original_images_path.exists():
        initial_image_count = len([f for f in original_images_path.iterdir() if f.is_file() and f.suffix.lower() in ['.png', '.jpg', '.jpeg']])
        print(f"処理前の元の画像ファイル数: {initial_image_count}")
    else:
        print(f"警告: 元の画像ディレクトリ {original_images_path} が見つかりません。ファイル数の事前カウントはスキップします。")


    if not dataset_root_dir.exists() or not (dataset_root_dir / 'images').exists() or not (dataset_root_dir / 'labels').exists():
        print(f"エラー: データセットのルートディレクトリ、またはその中の 'images'/'labels' ディレクトリが見つかりません: {dataset_root_dir}")
        print(f"パスが正しいことを確認してください。'images' および 'labels' ディレクトリが {dataset_root_dir} 直下にある必要があります。")
    elif (train_split_ratio + val_split_ratio) > 1.0:
        print(f"エラー: train_ratio ({train_split_ratio}) と val_ratio ({val_split_ratio}) の合計が1.0を超えています。")
    else:
        print(f"データセットを再編成します: {dataset_root_dir}")
        print(f"訓練セットの割合: {train_split_ratio}, 検証セットの割合: {val_split_ratio}")
        test_split_ratio = 1.0 - train_split_ratio - val_split_ratio
        print(f"テストセットの割合 (自動計算): {test_split_ratio:.2f}")
        
        reorganize_dataset_for_yolo(dataset_root_dir, train_ratio=train_split_ratio, val_ratio=val_split_ratio)
        
        print("\n--- データセットの再編成処理が完了しました。 ---")
        print(f"ディレクトリ構造を確認してください: {dataset_root_dir}")
        print("data.yaml が正しく 'train: ../train/images', 'val: ../val/images', 'test: ../test/images' (存在する場合) のように、")
        print(f"data.yaml内の 'path' エントリ (通常は '{dataset_root_dir.name}' や '.') からの相対パスで指定されていることを確認してください。")
        print("\n--- ファイル数の確認 ---")
        print(f"処理前の合計画像数 (目安): {initial_image_count}")

        # 処理後の各セットのファイル数を確認
        total_processed_images = 0
        for split in ['train', 'val', 'test']:
            img_dir = dataset_root_dir / split / 'images'
            lbl_dir = dataset_root_dir / split / 'labels'
            img_count = 0
            if img_dir.exists():
                img_count = len([f for f in img_dir.iterdir() if f.is_file()])
                print(f"{split}/images のファイル数: {img_count}")
                total_processed_images += img_count
            else:
                # testセットは存在しない場合があるので、警告ではなく情報として表示
                if split == 'test' and test_split_ratio == 0:
                    print(f"{split}/images ディレクトリは作成されませんでした (テストセットの割合が0のため)。")
                else:
                    print(f"警告: {split}/images ディレクトリが存在しません。")
            
            if lbl_dir.exists():
                lbl_count = len([f for f in lbl_dir.iterdir() if f.is_file()])
                print(f"{split}/labels のファイル数: {lbl_count}")
            else:
                if split == 'test' and test_split_ratio == 0:
                     print(f"{split}/labels ディレクトリは作成されませんでした (テストセットの割合が0のため)。")
                else:
                    print(f"警告: {split}/labels ディレクトリが存在しません。")
        
        print(f"\n処理後の合計画像ファイル数 (train + val + test): {total_processed_images}")

        print("\n期待されるファイル分割数 (目安):")
        if initial_image_count > 0:
            expected_train = round(initial_image_count * train_split_ratio)
            expected_val = round(initial_image_count * val_split_ratio)
            expected_test = round(initial_image_count * test_split_ratio)
            # ラベルファイルがない等の理由で実際のファイル数は少なくなる可能性があるため、合計がinitial_image_countと一致しない場合がある
            # ここでは単純な比率での期待値を示す
            print(f"  訓練セット (画像): 約 {expected_train}")
            print(f"  検証セット (画像): 約 {expected_val}")
            print(f"  テストセット (画像): 約 {expected_test}")
            print(f"  期待される合計 (train+val+test): 約 {expected_train + expected_val + expected_test}")
        else:
            print("  元の画像ファイル数が0または不明なため、期待値は計算できません。")

        print("\n注意:")
        print("- 上記の期待値は、全ての画像に対応するラベルファイルが存在する場合の目安です。")
        print("- ラベルファイルが見つからない画像は、そのペアがデータセットから除外されるため、実際のファイル数は期待値より少なくなることがあります。")
        print("- 各セット (train, val, test) の画像ファイルが重複していないか、合計数が元の有効なペア数とおおよそ一致するかを確認してください。")
        print("- もし train と test の両方に全ての（または大部分の）フレームが含まれているように見える場合、分割比率の設定や、スクリプトが意図せず複数回実行されていないか等を確認してください。")
