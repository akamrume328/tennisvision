import os
import zipfile
from ultralytics import YOLO

# --- ★★★ ユーザー設定項目 (ローカル環境に合わせて修正してください) ★★★ ---

# データセットのdata.yamlファイルへのパス
# 例: 'C:/Users/YourUser/datasets/my_yolo_dataset/data.yaml'
# 例: '../datasets/my_yolo_dataset/data.yaml' (このスクリプトからの相対パス)
# ★★★ この `data.yaml` は、訓練画像/ラベルの場所、クラス名、そして
# ★★★ `train: path/to/train.txt` や `val: path/to/val.txt` のように、
# ★★★ 訓練用/検証用テキストファイルへのパス（通常はdata.yamlからの相対パス）を定義している必要があります。
# ★★★ `train.txt` や `val.txt` は、画像ファイルへのパスのリスト（例: `images/train/frame_000001.png` や `../images/frame_000001.png`）を含みます。
dataset_yaml_path = '../data/processed/dataset/data.yaml'  # ★★★ ここをあなたのdata.yamlへの実際のパスに修正してください ★★★

# トレーニング結果（重み、ログなど）を保存するディレクトリ
# 例: 'C:/Users/YourUser/YOLOv8_Tennis_Project_Outputs'
# 例: '../models/YOLOv8_Training_Outputs' (このスクリプトからの相対パス)
project_output_dir = '../models/project_outputs'  # ★★★ ここを修正してください ★★★

# この特定のトレーニング実行の名前（project_output_dirのサブディレクトリとして作成されます）
experiment_name = 'tennis_detection_local_run1'

# --- チェックポイントと再開設定 ---
# トレーニングを再開する場合、ここにチェックポイントファイル (.pt) のパスを指定します。
# 例: 'path/to/your/project_outputs/tennis_detection_local_run1/weights/last.pt'
# 新規トレーニングの場合は None または空文字列（''）のままにします。
checkpoint_to_resume = None  # ★★★ 再開する場合、ここにチェックポイントのパスを設定 ★★★

# 何エポックごとにチェックポイントを保存するか (例: 10エポックごと)
# 0または負の値を指定すると、エポックごとの追加保存は行われません (last.ptとbest.ptは常に保存されます)。
save_every_n_epochs = 10  # ★★★ 必要に応じて変更 ★★★

# --- ZIP展開設定 (オプション) ---
# データセットがZIPファイルとしてローカルに保存されている場合、そのパスと展開先を指定します。
# ZIP展開機能を使用しない場合は、zip_file_path_local を None または空文字列にしてください。
# その場合、dataset_yaml_path は展開済みのデータセット内の data.yaml を直接指すようにしてください。
zip_file_path_local = None  # 例: 'C:/path/to/your/dataset.zip'  # ★★★ ZIPを使用する場合に設定 ★★★
# Colabの /content/datasets に相当するローカルの一時展開ディレクトリ
# dataset_yaml_path は、この展開先ディレクトリ内の data.yaml を指すように後で調整する必要があります。
local_extract_to_dir = 'temp_extracted_datasets' # ★★★ 必要であれば変更してください ★★★

# --- モデルとトレーニングパラメータ ---
# 事前学習済みモデル名: 'yolov8n.pt', 'yolov8s.pt', 'yolov8m.pt', 'yolov8l.pt', 'yolov8x.pt' など
# 新規トレーニングの場合にロードするモデル。checkpoint_to_resume が指定されていればそちらが優先されます。
base_model_name_if_new = 'yolov8s.pt'

num_epochs = 100  # トレーニングエポック数
batch_size = 6   # バッチサイズ (GPUメモリに応じて調整)
img_size = 1920    # 入力画像サイズ (例: 640, 1280)

# その他のYOLOv8 train()メソッドのパラメータ (必要に応じてコメントアウト解除して設定)
# device_setting = 0  # 0 for CUDA device 0, 'cpu' for CPU. YOLOv8が自動検出することも多い
# workers_setting = 8 # データローダーのワーカー数 (CPUコア数に応じて調整)
# patience_setting = 30 # 早期終了の忍耐エポック数
# lr0_setting = 0.01    # 初期学習率
# optimizer_setting = 'AdamW' # オプティマイザ: 'SGD', 'Adam', 'AdamW'

# --- ★★★ 設定項目終了 ★★★ ---


def extract_dataset_if_needed(zip_path, extraction_path):
    """
    指定されたZIPファイルを指定された場所に展開します。
    """
    if not zip_path or not os.path.exists(zip_path):
        print(f"情報: ZIPファイルパスが指定されていないか、ファイル '{zip_path}' が存在しません。展開をスキップします。")
        return False, None

    if not extraction_path:
        print("エラー: 展開先パスが指定されていません。展開をスキップします。")
        return False, None

    if not os.path.exists(extraction_path):
        os.makedirs(extraction_path)
        print(f"展開先ディレクトリを作成しました: {extraction_path}")
    else:
        print(f"展開先ディレクトリは既に存在します: {extraction_path}")
        # 注意: 既存のファイルがある場合、上書きされる可能性があります。
        # 必要であれば、展開前に既存のディレクトリをクリアする処理を追加してください。

    print(f"ZIPファイル {zip_path} を {extraction_path} に展開しています...")
    try:
        with zipfile.ZipFile(zip_path, 'r') as zip_ref:
            zip_ref.extractall(extraction_path)
        print(f"データセットの展開が完了しました。展開先: {extraction_path}")

        # 展開された data.yaml のパス候補を探す
        found_yaml_paths = []
        for root, _, files in os.walk(extraction_path):
            if "data.yaml" in files:
                found_yaml_paths.append(os.path.join(root, "data.yaml"))
        
        if found_yaml_paths:
            print("\n展開された可能性のある data.yaml のパス:")
            for p_yaml in found_yaml_paths:
                print(f"- {p_yaml}")
            print(f"これらのいずれかを `dataset_yaml_path` グローバル変数に設定するか、")
            print(f"この関数の戻り値として取得したパスを使用してください。")
            # 最初の候補を返す (より洗練されたロジックが必要な場合あり)
            return True, found_yaml_paths[0]
        else:
            print(f"\n警告: 展開されたディレクトリ内に 'data.yaml' が見つかりませんでした。")
            print(f"ZIPファイルの内容と展開後の構造を確認し、`dataset_yaml_path` を手動で正しく設定してください。")
            return True, None # 展開は成功したがyamlは見つからず

    except zipfile.BadZipFile:
        print(f"エラー: {zip_path} は有効なZIPファイルではありません。")
    except Exception as e:
        print(f"展開中にエラーが発生しました: {e}")
    return False, None


def train_yolo_model(
    data_yaml,
    output_dir,
    exp_name,
    resume_chkpt,
    base_model_name,
    epochs,
    batch,
    imgsz,
    save_period,
    **kwargs # その他のYOLO train引数用
):
    """
    YOLOv8モデルのトレーニングを実行します。
    """
    # プロジェクト出力ディレクトリが存在しない場合は作成
    os.makedirs(output_dir, exist_ok=True)
    print(f"データセットのYAMLパス: {data_yaml}")
    print(f"プロジェクト出力ディレクトリ: {output_dir}")
    print(f"実験名: {exp_name}")

    if not os.path.exists(data_yaml):
        print(f"エラー: データセットのYAMLファイルが見つかりません: {data_yaml}")
        print("パスを再確認してください。ZIP展開機能を使用した場合は、展開後のローカルパスを指定しているか確認してください。")
        return

    model_to_load = base_model_name
    if resume_chkpt and os.path.exists(resume_chkpt):
        print(f"チェックポイントからトレーニングを再開します: {resume_chkpt}")
        model_to_load = resume_chkpt
    elif resume_chkpt: # パスは指定されたが見つからなかった場合
        print(f"警告: 指定された再開用チェックポイント '{resume_chkpt}' が見つかりませんでした。")
        print(f"新しい事前学習済みモデル ({base_model_name}) からトレーニングを開始します。")
    else:
        print(f"新しい事前学習済みモデル ({base_model_name}) からトレーニングを開始します。")

    model = YOLO(model_to_load)

    print(f"\nモデルのトレーニングを開始します。")
    print(f"エポック数: {epochs}, バッチサイズ: {batch}, 画像サイズ: {imgsz}")
    if save_period > 0:
        print(f"チェックポイントは {save_period} エポックごとに保存されます。")
    
    # kwargs に設定されているYOLOの追加パラメータをログに出力
    if kwargs:
        print("追加のトレーニングパラメータ:")
        for key, value in kwargs.items():
            print(f"  {key}: {value}")

    try:
        results = model.train(
            data=data_yaml,
            epochs=epochs,
            imgsz=imgsz,
            batch=batch,
            project=output_dir,
            name=exp_name,
            exist_ok=True,  # experiment_nameディレクトリが既に存在する場合は上書き/追記を許可
            save_period=save_period if save_period > 0 else -1, # 指定エポックごとに保存 (-1で無効)
            **kwargs # device, workers, patienceなどを渡す
        )
        print("\nトレーニングが正常に完了しました！")
        final_results_path = os.path.join(output_dir, exp_name)
        print(f"結果、ログ、モデルの重みは次の場所に保存されました: {final_results_path}")

        # トレーニング結果の確認
        weights_dir = os.path.join(final_results_path, 'weights')
        if os.path.exists(weights_dir):
            print(f"\n重みディレクトリ内のファイル ({weights_dir}):")
            for f_name in sorted(os.listdir(weights_dir)): # ソートして表示
                print(f"- {f_name}")
        
        best_model_path = os.path.join(weights_dir, 'best.pt')
        print(f"\n最良のモデルへのパス: {best_model_path}")
        if os.path.exists(best_model_path):
            print("最良のモデルファイル (best.pt) が見つかりました。")
        else:
            print("警告: 最良のモデルファイル (best.pt) が見つかりません。トレーニングが失敗したか、生成されなかった可能性があります。")

    except Exception as e:
        print(f"\nトレーニング中にエラーが発生しました: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    print("ローカル環境でのYOLOv8モデルトレーニングスクリプト")
    print("--- 設定値の確認 ---")
    print(f"データセットYAML: {dataset_yaml_path}")
    print(f"プロジェクト出力先: {project_output_dir}")
    print(f"実験名: {experiment_name}")
    print(f"再開用チェックポイント: {checkpoint_to_resume if checkpoint_to_resume else 'なし'}")
    print(f"新規の場合のベースモデル: {base_model_name_if_new}")
    print(f"エポック数: {num_epochs}, バッチサイズ: {batch_size}, 画像サイズ: {img_size}")
    print(f"チェックポイント保存頻度: {save_every_n_epochs} エポックごと")

    actual_dataset_yaml_path = dataset_yaml_path

    # オプション: ローカルZIPデータセットの展開
    if zip_file_path_local and local_extract_to_dir:
        print("\n--- ローカルデータセット展開処理 ---")
        展開成功, 展開後のyaml候補パス = extract_dataset_if_needed(zip_file_path_local, local_extract_to_dir)
        if 展開成功:
            if 展開後のyaml候補パス:
                print(f"ZIP展開後の data.yaml 候補: {展開後のyaml候補パス}")
                print(f"このパスを dataset_yaml_path として使用します。")
                actual_dataset_yaml_path = 展開後のyaml候補パス
            else:
                print(f"ZIP展開は行われましたが、data.yaml が自動で見つかりませんでした。")
                print(f"手動で `dataset_yaml_path` (現在: {actual_dataset_yaml_path}) が正しいか確認してください。")
        else:
            print("ZIP展開に失敗しました。設定を確認し、手動でデータセットを準備してください。")
            # 展開が必須の場合はここで終了することも検討
            # exit(1)
    
    # その他のYOLO trainパラメータを辞書として準備
    additional_train_params = {}
    # if 'device_setting' in locals() and device_setting is not None:
    #     additional_train_params['device'] = device_setting
    # if 'workers_setting' in locals() and workers_setting is not None:
    #     additional_train_params['workers'] = workers_setting
    # if 'patience_setting' in locals() and patience_setting is not None:
    #     additional_train_params['patience'] = patience_setting
    # if 'lr0_setting' in locals() and lr0_setting is not None:
    #     additional_train_params['lr0'] = lr0_setting
    # if 'optimizer_setting' in locals() and optimizer_setting is not None:
    #     additional_train_params['optimizer'] = optimizer_setting
    # 例: additional_train_params['device'] = 'cpu' # CPUで実行する場合

    print(f"\n--- トレーニング開始 (使用するYAML: {actual_dataset_yaml_path}) ---")
    train_yolo_model(
        data_yaml=actual_dataset_yaml_path,
        output_dir=project_output_dir,
        exp_name=experiment_name,
        resume_chkpt=checkpoint_to_resume,
        base_model_name=base_model_name_if_new,
        epochs=num_epochs,
        batch=batch_size,
        imgsz=img_size,
        save_period=save_every_n_epochs,
        **additional_train_params
    )

    print("\nスクリプトの実行が終了しました。")
