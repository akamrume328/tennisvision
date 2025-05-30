# CUDA環境およびPyTorchインストールの確認方法

PyTorchでGPUを利用するためには、NVIDIAドライバ、CUDA Toolkit、cuDNN、そしてCUDA対応のPyTorchが正しくインストールされ、連携している必要があります。以下にそれぞれの確認手順を示します。

## 1. NVIDIAドライバの確認

### 確認方法:
   - **NVIDIAコントロールパネル:**
     1. デスクトップで右クリックし、「NVIDIAコントロールパネル」を選択します。
     2. 左下の「システム情報」をクリックします。
     3. 「コンポーネント」タブで、表示されているNVIDIA GPUとドライバのバージョンを確認します。
   - **コマンドプロンプトまたはPowerShell:**
     ```shell
     nvidia-smi
     ```
     このコマンドを実行すると、ドライバのバージョン、CUDAのバージョン（ドライバがサポートする最新版）、認識されているGPUの情報が表示されます。

### 確認ポイント:
   - GPUが正しく認識されているか。
   - 最新または安定版のドライバがインストールされているか。

## 2. CUDA Toolkitの確認

### 確認方法:
   - **コマンドプロンプトまたはPowerShell:**
     ```shell
     nvcc --version
     ```
     このコマンドが成功すれば、CUDA Toolkitがインストールされており、そのバージョンが表示されます。
   - **環境変数:**
     `CUDA_PATH` や `CUDA_HOME` といった環境変数が設定されているか確認します（例: `C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v11.8`）。
   - **インストール済みプログラムの一覧:**
     Windowsの「設定」 > 「アプリ」 > 「インストールされているアプリ」で "NVIDIA CUDA Toolkit" があるか確認します。

### 確認ポイント:
   - インストールされているCUDA Toolkitのバージョン。
   - このバージョンが、インストールしようとしている（またはインストール済みの）PyTorchバージョンと互換性があるか（PyTorch公式サイトで確認）。

## 3. cuDNNの確認

cuDNNはCUDA Toolkitの特定のバージョンに対応しています。通常、手動でCUDA Toolkitのディレクトリにファイルをコピーしてインストールします。

### 確認方法:
   1. CUDA Toolkitのインストールディレクトリ（例: `C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\vX.Y` ここで `vX.Y` はCUDAバージョン）を開きます。
   2. 以下のファイルが存在するか確認します:
      - `include\cudnn.h`
      - `bin\cudnn*.dll` (例: `cudnn64_8.dll`)
      - `lib\x64\cudnn*.lib`
   3. `cudnn.h` ファイルを開き、`CUDNN_MAJOR`, `CUDNN_MINOR`, `CUDNN_PATCHLEVEL` の定義からバージョンを確認できます。

### 確認ポイント:
   - cuDNNのファイルが適切な場所に存在するか。
   - cuDNNのバージョンが、インストールされているCUDA Toolkitのバージョンと互換性があるか。
   - PyTorchが要求するcuDNNバージョンと適合しているか。

## 4. PyTorchのインストールとCUDAサポートの確認

### 確認方法:
   Python環境（スクリプトを実行しているのと同じvenv環境など）でPythonインタプリタを起動し、以下のコマンドを実行します。

   ```python
   import torch

   # PyTorchのバージョン
   print(f"PyTorch Version: {torch.__version__}")

   # PyTorchがビルドされたCUDAのバージョン (CPU版の場合はNoneや表示なし)
   print(f"PyTorch CUDA Version: {torch.version.cuda}")

   # CUDAが利用可能か (これがTrueになる必要がある)
   print(f"CUDA Available: {torch.cuda.is_available()}")

   if torch.cuda.is_available():
       # 利用可能なGPUの数
       print(f"Number of GPUs: {torch.cuda.device_count()}")
       # 現在のGPUデバイス番号
       print(f"Current GPU ID: {torch.cuda.current_device()}")
       # 現在のGPUデバイス名
       print(f"Current GPU Name: {torch.cuda.get_device_name(torch.cuda.current_device())}")
   else:
       print("PyTorch cannot find CUDA. Ensure your PyTorch installation supports CUDA and your environment is correctly set up.")
   ```

### 確認ポイント:
   - `torch.__version__` でPyTorchのバージョンを確認。
   - `torch.version.cuda` でPyTorchがどのCUDAバージョンでビルドされたかを確認。これが表示されない、または `None` の場合はCPU版のPyTorchである可能性が高いです。
   - **`torch.cuda.is_available()` が `True` を返すこと。これが `False` の場合、PyTorchはGPUを認識・利用できていません。**

## トラブルシューティングのヒント:

- **仮想環境:** Anacondaやvenvなどの仮想環境を使用している場合、その環境に正しくPyTorch（CUDA対応版）がインストールされていることを確認してください。
- **PyTorchの再インストール:** 問題が解決しない場合、PyTorchを再インストールすることを検討してください。PyTorch公式サイト ([https://pytorch.org/get-started/locally/](https://pytorch.org/get-started/locally/)) で、お使いのOS、パッケージマネージャ（pipやconda）、CUDAバージョンに合ったインストールコマンドを確認して実行してください。
  例 (pipでCUDA 11.8対応版をインストールする場合):
  ```shell
  pip uninstall torch torchvision torchaudio
  pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118
  ```
  例 (pipでCUDA 12.1対応版をインストールする場合):
  ```shell
  pip uninstall torch torchvision torchaudio
  pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121
  ```
- **パス設定:** CUDA Toolkitのパス (`bin` や `libnvvp`) がシステムの環境変数 `PATH` に正しく設定されているか確認してください。
- **再起動:** ドライバやツールキットをインストール・更新した後は、システムを再起動すると問題が解決することがあります。

これらの手順で問題箇所を特定し、修正することで、PyTorchでGPUを利用できるようになるはずです。
