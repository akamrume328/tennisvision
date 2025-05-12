# テニスビジョンプロジェクト

## 概要
テニスビジョンプロジェクトは、コンピュータビジョン技術を使用してテニスの試合を分析するために設計されたアプリケーションです。YOLOv8とOpenCVを活用することで、このプロジェクトはビデオ映像から選手、ボール、コートラインを検出・追跡し、選手のパフォーマンスや試合のダイナミクスに関する洞察を提供することを目的としています。

## プロジェクト構成
```
tennis-vision-project
├── data
│   ├── processed          # モデルのトレーニングと評価に使用できる状態に処理されたデータセット
│   └── raw               # 元のビデオファイルや画像などの生データ
├── models                 # 学習済みモデルファイルを格納するディレクトリ
├── notebooks              # データ処理とモデルトレーニング用のJupyter Notebook
│   ├── 1_data_preprocessing.ipynb
│   └── 2_model_training.ipynb
├── scripts                # データ前処理とモデルトレーニング用のPythonスクリプト
│   ├── preprocess_data.py
│   └── train.py
├── src                    # アプリケーションのソースコード
│   ├── __init__.py
│   ├── config.py         # プロジェクトの設定
│   ├── detection.py      # 物体検出用の関数とクラス
│   ├── main.py           # アプリケーションのメインエントリーポイント
│   ├── tracking.py       # 物体追跡用の関数とクラス
│   └── utils.py          # プロジェクト全体で使用されるユーティリティ関数
├── tests                  # アプリケーションの単体テスト
│   ├── __init__.py
│   ├── test_detection.py
│   └── test_tracking.py
├── README.md              # プロジェクトのドキュメント
└── requirements.txt       # プロジェクトに必要な依存関係のリスト
```

## セットアップ手順
1.  **リポジトリをクローンします:**
    ```
    git clone <repository-url>
    cd tennis-vision-project
    ```

2.  **仮想環境を作成します:**
    ```
    python -m venv venv
    source venv/bin/activate  # Windowsでは `venv\Scripts\activate` を使用します
    ```

3.  **必要なパッケージをインストールします:**
    ```
    pip install -r requirements.txt
    ```

## 利用ガイドライン
-   **データ前処理:** Jupyter Notebook `notebooks/1_data_preprocessing.ipynb` を使用して生データを読み込み、前処理します。
-   **モデルトレーニング:** `notebooks/2_model_training.ipynb` を使用してモデルをトレーニングし、必要に応じてハイパーパラメータを調整します。
-   **アプリケーションの実行:** `src/main.py` を実行してメインアプリケーションを実行します。これにより、データの前処理、モデルのトレーニング、評価のワークフローが調整されます。

## 貢献
貢献を歓迎します！機能強化やバグ修正については、Issueを開くか、プルリクエストを送信してください。

## ライセンス
このプロジェクトはMITライセンスの下でライセンスされています - 詳細についてはLICENSEファイルを参照してください。