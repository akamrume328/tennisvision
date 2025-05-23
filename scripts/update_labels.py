import os

def update_label_file(file_path):
    """
    単一のラベルファイルを読み込み、クラスIDを変換して上書き保存する。
    ルール:
    - クラスID 3 -> 0
    - クラスID 4 -> 1
    - クラスID 5 -> 2
    - 元のクラスID 0, 1, 2 の行は削除
    """
    updated_lines = []
    try:
        with open(file_path, 'r') as f:
            lines = f.readlines()

        for line in lines:
            parts = line.strip().split()
            if not parts:
                continue

            try:
                class_id = int(parts[0])
                rest_of_line = " ".join(parts[1:])
            except ValueError:
                print(f"警告: ファイル {file_path} の行 '{line.strip()}' でクラスIDを数値に変換できませんでした。スキップします。")
                updated_lines.append(line) # 変換できない行はそのまま保持（またはエラー処理）
                continue

            if class_id == 3:
                updated_lines.append(f"0 {rest_of_line}\n")
            elif class_id == 4:
                updated_lines.append(f"1 {rest_of_line}\n")
            elif class_id == 5:
                updated_lines.append(f"2 {rest_of_line}\n")
            elif class_id in [0, 1, 2]:
                # 元のクラスID 0, 1, 2 は削除するので何もしない
                pass
            else:
                # 上記以外のクラスIDはそのまま保持
                updated_lines.append(line)
        
        with open(file_path, 'w') as f:
            f.writelines(updated_lines)
        # print(f"更新完了: {file_path}")

    except Exception as e:
        print(f"エラー: ファイル {file_path} の処理中にエラーが発生しました - {e}")

def process_label_directory(directory_path):
    """
    指定されたディレクトリ内のすべての .txt ラベルファイルを処理する。
    """
    if not os.path.isdir(directory_path):
        print(f"エラー: ディレクトリが見つかりません - {directory_path}")
        return

    print(f"処理中: {directory_path}")
    for filename in os.listdir(directory_path):
        if filename.endswith(".txt"):
            file_path = os.path.join(directory_path, filename)
            update_label_file(file_path)
    print(f"完了: {directory_path}")


if __name__ == "__main__":
    # データセットのベースパス (環境に合わせて変更してください)
    dataset_base_path = r"c:\Users\akama\AppData\Local\Programs\Python\Python310\python_file\projects\tennisvision\data\processed\dataset"

    # 処理対象のラベルディレクトリ
    label_directories = [
        os.path.join(dataset_base_path, "test", "labels"),
        os.path.join(dataset_base_path, "train", "labels"),
        os.path.join(dataset_base_path, "val", "labels"),
    ]

    # バックアップの警告
    print("警告: このスクリプトはラベルファイルを直接上書きします。")
    print("実行前に必ずデータセットのバックアップを取ってください。")
    confirm = input("処理を続行しますか？ (yes/no): ")

    if confirm.lower() == 'yes':
        for directory in label_directories:
            process_label_directory(directory)
        print("すべてのラベルファイルの処理が完了しました。")
    else:
        print("処理はキャンセルされました。")

"""
**使用方法:**

1.  上記のコードを、例えば `c:\Users\akama\AppData\Local\Programs\Python\Python310\python_file\projects\tennisvision\scripts\update_labels.py` という名前で保存します。
2.  スクリプト内の `dataset_base_path` が、あなたのデータセットの正しい場所を指していることを確認してください。
3.  ターミナルまたはコマンドプロンプトを開き、スクリプトを保存したディレクトリに移動します。
    ````bash
    cd c:\Users\akama\AppData\Local\Programs\Python\Python310\python_file\projects\tennisvision\scripts
    ````
4.  スクリプトを実行します。
    ````bash
    python update_labels.py
    ````
5.  実行すると、バックアップに関する警告と確認メッセージが表示されます。内容を理解した上で `yes` と入力して Enter キーを押すと、処理が開始されます。

このスクリプトは、`test/labels`、`train/labels`、`val/labels` 内のすべての `.txt` ファイルに対して、指定されたクラスIDの変換ルールを適用します。
"""