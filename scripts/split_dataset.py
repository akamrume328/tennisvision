# filepath: split_dataset.py
import os
import random
from pathlib import Path

def split_frame_identifiers(original_train_txt_path, output_dir, train_ratio=0.8):
    """
    元の train.txt から画像ファイルパスを読み込み、シャッフルし、
    新しい train.txt と val.txt ファイルに分割します。
    """
    original_train_txt_path = Path(original_train_txt_path)
    output_dir = Path(output_dir)
    # 出力ディレクトリを作成（親ディレクトリも含む、既に存在する場合はエラーにしない）
    output_dir.mkdir(parents=True, exist_ok=True)

    new_train_file = output_dir / "train.txt"
    new_val_file = output_dir / "val.txt"

    # 元の train.txt が存在するか確認し、存在しない場合はエラーメッセージを表示して終了
    if not original_train_txt_path.exists():
        print(f"Error: Original train.txt not found at {original_train_txt_path}")
        return

    image_file_paths = [] # 変数名を変更
    # 元の train.txt を読み込みモードで開く
    with open(original_train_txt_path, 'r') as f:
        for line in f:
            # 各行の前後の空白を削除
            line = line.strip()
            # 空行でない場合のみ処理
            if line:
                # 行全体（ファイルパス）をリストに追加
                image_file_paths.append(line)

    # 画像ファイルパスが見つからなかった場合はメッセージを表示して終了
    if not image_file_paths:
        print("No image file paths found in the original train.txt.")
        return

    # 画像ファイルパスのリストをランダムにシャッフル
    random.shuffle(image_file_paths)

    # train_ratio に基づいて訓練データと検証データの分割点を計算
    split_index = int(len(image_file_paths) * train_ratio)
    # 訓練データ用のパスリストを作成
    train_paths = image_file_paths[:split_index]
    # 検証データ用のパスリストを作成
    val_paths = image_file_paths[split_index:]

    # 新しい train.txt ファイルに訓練データパスを書き込む
    with open(new_train_file, 'w') as f:
        for path_entry in train_paths: # 変数名を変更
            f.write(f"{path_entry}\n") # 完全なパスを書き込む
    print(f"新しい train.txt を作成しました。エントリ数: {len(train_paths)}, パス: {new_train_file}")

    # 新しい val.txt ファイルに検証データパスを書き込む
    with open(new_val_file, 'w') as f:
        for path_entry in val_paths: # 変数名を変更
            f.write(f"{path_entry}\n") # 完全なパスを書き込む
    print(f"新しい val.txt を作成しました。エントリ数: {len(val_paths)}, パス: {new_val_file}")

if __name__ == "__main__":
    # --- 設定 ---
    # 現在の train.txt ファイルへのパス（フルパスが記述されているもの）
    # スクリプトがプロジェクトルートにない場合は、このパスを調整してください
    current_train_txt = Path("C:/Users/akama/AppData/Local/Programs/Python/Python310/python_file/projects/tennisvision/data/processed/dataset/train.txt")

    # 新しい train.txt, val.txt, data.yaml が配置されるディレクトリ
    target_dataset_dir = Path("C:/Users/akama/AppData/Local/Programs/Python/Python310/python_file/projects/tennisvision/data/processed/dataset")
    # --- 設定終了 ---

    # プロジェクトルートからの相対パスで元の train.txt が存在するか確認
    # このスクリプトがプロジェクトルート 'tennisvision/' から実行されることを想定
    project_root = Path().cwd() # 'tennisvision/' であるべき
    original_file_path_check = project_root / current_train_txt

    if not original_file_path_check.exists():
        print(f"エラー: 元の train.txt が見つかりませんでした: {original_file_path_check}")
        print("パスが正しいこと、およびスクリプトがプロジェクトルートから実行されていることを確認してください。")
    else:
        split_frame_identifiers(original_file_path_check, target_dataset_dir)
        print("\n注意: 新しい train.txt および val.txt と同じディレクトリに 'data.yaml' があることを確認し、")
        print("'val: val.txt' を含むように更新してください（必要な場合）。")
        print("また、画像ファイルで学習する場合、これらの .txt ファイルにリストされた各画像ファイルに対応する")
        print("ラベルファイル（例: 'labels/' ディレクトリ内に同じファイル名で拡張子が .txt のファイル）が")
        print(f"'{target_dataset_dir}' を基準として正しく配置されていることを確認してください。")
        print("例: 'train.txt' 内の '../images/frame_0001.png' に対応するラベルは '../labels/frame_0001.txt' のようになります。")
