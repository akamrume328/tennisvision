# TensorFlow GPU環境セットアップ手順

## 概要
このガイドはWindows環境でTensorFlow GPUを使用するためのセットアップ手順です。

## 前提条件
- Windows 10/11
- NVIDIA GPU (CUDA対応)
- Python 3.8-3.11

## 手順

### 1. 自動セットアップ（推奨）
```bash
python install_gpu_requirements.py
```

### 2. 手動セットアップ

#### 2.1 既存のTensorFlowを削除
```bash
pip uninstall tensorflow tensorflow-gpu -y
```

#### 2.2 TensorFlow 2.12をインストール
```bash
pip install tensorflow==2.12.0
```

#### 2.3 その他パッケージをインストール
```bash
pip install -r requirements_gpu.txt
```

### 3. CUDA環境セットアップ（GPUが検出されない場合）

#### 3.1 NVIDIA ドライバー更新
1. [NVIDIA公式サイト](https://www.nvidia.com/drivers/)から最新ドライバーをダウンロード
2. インストール後再起動

#### 3.2 CUDA Toolkit 11.8 インストール
1. [CUDA 11.8ダウンロード](https://developer.nvidia.com/cuda-11-8-0-download-archive)
2. Windows x86_64 → local installer を選択
3. インストール実行（デフォルト設定推奨）

#### 3.3 cuDNN 8.6 インストール
1. [cuDNNダウンロード](https://developer.nvidia.com/cudnn)（要登録）
2. "Download cuDNN v8.6.0 for CUDA 11.x" を選択
3. ZIPファイルを解凍
4. 以下のフォルダにファイルをコピー:
   ```
   解凍先/bin → C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v11.8\bin
   解凍先/include → C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v11.8\include
   解凍先/lib → C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v11.8\lib
   ```

#### 3.4 環境変数確認
システムのPATHに以下が含まれていることを確認:
```
C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v11.8\bin
C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v11.8\libnvvp
```

### 4. 動作確認
```bash
python -c "import tensorflow as tf; print(tf.config.list_physical_devices('GPU'))"
```

## トラブルシューティング

### よくある問題

#### "Could not load dynamic library 'cudart64_110.dll'"
- CUDA Toolkitが正しくインストールされていない
- 環境変数PATHの設定を確認

#### "Could not load dynamic library 'cudnn64_8.dll'"
- cuDNNが正しくインストールされていない
- ファイルの配置場所を確認

#### GPU が検出されない
1. NVIDIA GPUがCUDA対応か確認
2. ドライバーが最新か確認
3. CUDA/cuDNNのバージョン互換性を確認

### バージョン互換性表
| TensorFlow | Python | CUDA | cuDNN |
|------------|--------|------|-------|
| 2.12.0     | 3.8-3.11 | 11.8 | 8.6   |

## 使用方法
セットアップ完了後、以下を実行:
```bash
python train_lstm_model.py
```

GPU使用状況は学習開始時に表示されます。
