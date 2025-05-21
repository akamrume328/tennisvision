import os
import cv2
import numpy as np
# import pandas as pd # pandas は使用されていないため削除
import subprocess # subprocessモジュールをインポート
import shutil # shutilモジュールをインポート
from tqdm import tqdm # tqdmをインポート
import re # ffmpegの進捗解析のためにreをインポート

def _get_video_total_frames(video_path):
    """ffprobeを使用して動画の総フレーム数を取得する。"""
    command = [
        'ffprobe', '-v', 'error', '-select_streams', 'v:0',
        '-show_entries', 'stream=nb_frames', '-of', 'default=nokey=1:noprint_wrappers=1',
        video_path
    ]
    try:
        result = subprocess.run(command, capture_output=True, text=True, check=True)
        return int(result.stdout.strip())
    except (subprocess.CalledProcessError, FileNotFoundError, ValueError) as e:
        print(f"警告: {video_path} の総フレーム数取得に失敗: {e}")
        return None

def load_raw_data(data_dir):
    raw_data = []
    # 一時的なフレーム保存用ディレクトリを data ディレクトリの親に作成
    # 例: data_dir が '../data/raw' の場合、'../data/temp_frames' を作成
    temp_frame_base_dir = os.path.join(os.path.dirname(os.path.abspath(data_dir)), 'temp_frames')
    
    if not os.path.exists(temp_frame_base_dir):
        os.makedirs(temp_frame_base_dir, exist_ok=True)

    # data_dir内のファイルリストを取得
    files_in_data_dir = os.listdir(data_dir)
    print("Loading raw data...")
    for filename in tqdm(files_in_data_dir, desc="Processing files in raw_data_dir"):
        file_path = os.path.join(data_dir, filename)
        if filename.endswith('.mp4'):
            video_name = os.path.splitext(filename)[0]
            output_frame_dir = os.path.join(temp_frame_base_dir, video_name)
            
            if not os.path.exists(output_frame_dir):
                os.makedirs(output_frame_dir, exist_ok=True)
            else:
                # 既存のフレームがあれば一度クリアする（オプション）
                for f in os.listdir(output_frame_dir):
                    os.remove(os.path.join(output_frame_dir, f))

            total_frames = _get_video_total_frames(file_path)

            # ffmpegコマンドでフレームを抽出
            # -qscale:v 1 は高画質 (ほぼロスレスに近い)
            # リサイズフィルターを削除
            # -start_number を '0' に変更して、期待するフレーム番号に近づける
            command = [
                'ffmpeg', '-hide_banner', '-nostats', # 余計な出力を抑制
                '-i', file_path,
                '-start_number', '0',   # フレーム番号を0から開始するように指定 (エラー回避のため元に戻す)
                # '-vf', 'scale=640:480', # リサイズフィルターを削除
                '-qscale:v', '1',       # 最高画質に近い設定でフレームを抽出
                os.path.join(output_frame_dir, 'frame_%04d.png'),
                '-progress', 'pipe:1' # 進捗情報を標準出力へ
            ]
            
            process = subprocess.Popen(command, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True, universal_newlines=True, bufsize=1)
            
            with tqdm(total=total_frames, desc=f"Extracting {video_name}", unit="frame", leave=False) as pbar:
                if process.stdout:
                    for line in process.stdout:
                        match = re.search(r"frame=\s*(\d+)", line)
                        if match:
                            frame_num = int(match.group(1))
                            # pbar.n は現在の進捗なので、差分をupdateするか、直接セットする
                            pbar.update(frame_num - pbar.n) 
                        # 他の進捗情報（例: 'fps=', 'size=', 'time='）もここで解析可能

                stdout_data, stderr_data = process.communicate() # プロセス終了まで待機し、残りの出力を取得

                if process.returncode != 0:
                    print(f"Error processing video {filename} with ffmpeg. Return code: {process.returncode}")
                    if stderr_data:
                        print(f"FFmpeg stderr:\n{stderr_data}")
                elif pbar.total and pbar.n < pbar.total: # 完了時にプログレスバーを100%にする
                     pbar.update(pbar.total - pbar.n)


            # ffmpeg実行後、フレームファイルを集める (この部分のtqdmは削除)
            extracted_frame_files = sorted([f for f in os.listdir(output_frame_dir) if f.lower().endswith('.png')])
            
            if len(extracted_frame_files) > 0:
                frames_to_add = extracted_frame_files[1:] 
            else:
                frames_to_add = []

            for frame_filename in frames_to_add: 
                raw_data.append(os.path.join(output_frame_dir, frame_filename))
            # except subprocess.CalledProcessError as e:
            #     print(f"Error processing video {filename} with ffmpeg. Return code: {e.returncode}")
            #     # エラーメッセージを表示したい場合は、stdout/stderrをキャプチャする
            #     # print(f"FFmpeg stdout: {e.stdout.decode() if e.stdout else 'N/A'}")
            #     # print(f"FFmpeg stderr: {e.stderr.decode() if e.stderr else 'N/A'}")

            # except FileNotFoundError:
            #     print("Error: ffmpeg command not found. Please ensure ffmpeg is installed and in your PATH.")

        elif filename.lower().endswith(('.jpg', '.jpeg', '.png')): # .jpegも追加、小文字で比較
            raw_data.append(file_path)
    return raw_data, temp_frame_base_dir # 一時ディレクトリのパスも返す

def preprocess_image(image_path):
    image = cv2.imread(image_path)
    if image is None:
        print(f"警告: 画像ファイルが読み込めませんでした: {image_path}")
        return None # 読み込めない場合は None を返す
    # 画像を指定されたサイズにリサイズする処理をコメントアウトまたは削除
    # image = cv2.resize(image, (640, 480)) 
    # 画像を正規化し、float32 型に変換してメモリ使用量を抑える
    image = image.astype(np.float32) / 255.0
    return image

def main(raw_data_dir, processed_data_dir):
    raw_data_paths, temp_dir_to_delete = load_raw_data(raw_data_dir) # 一時ディレクトリのパスを受け取る
    
    if not raw_data_paths:
        print("処理対象のRAW画像パスが見つかりませんでした。処理を終了します。")
        if temp_dir_to_delete and os.path.exists(temp_dir_to_delete):
            try:
                shutil.rmtree(temp_dir_to_delete)
                print(f"一時ディレクトリを正常に削除しました: {temp_dir_to_delete}")
            except OSError as e:
                print(f"一時ディレクトリ {temp_dir_to_delete} の削除中にエラー: {e.strerror}")
        return

    if not os.path.exists(processed_data_dir):
        os.makedirs(processed_data_dir)
        print(f"作成されたディレクトリ: {processed_data_dir}")

    print("Preprocessing and saving images...")
    saved_image_count = 0
    for image_path in tqdm(raw_data_paths, desc="Processing images"):
        processed_img_float32 = preprocess_image(image_path)
        
        if processed_img_float32 is not None:
            # 画像を0-255の範囲に戻し、uint8に変換して保存
            image_to_save = (processed_img_float32 * 255.0).astype(np.uint8)
            
            # ファイル名を frame_XXXXXX.png 形式に変更
            save_path = os.path.join(processed_data_dir, f'frame_{saved_image_count:06d}.png')
            cv2.imwrite(save_path, image_to_save)
            saved_image_count += 1
            
    if saved_image_count == 0:
        print("処理できる画像がありませんでした。")
    else:
        print(f"合計 {saved_image_count} 枚の画像を処理し保存しました。")

    # 一時ディレクトリを削除
    if temp_dir_to_delete and os.path.exists(temp_dir_to_delete):
        try:
            shutil.rmtree(temp_dir_to_delete)
            print(f"Successfully deleted temporary directory: {temp_dir_to_delete}")
        except OSError as e:
            print(f"Error deleting temporary directory {temp_dir_to_delete}: {e.strerror}")

if __name__ == "__main__":
    raw_data_directory = '../data/raw'  # 必要に応じてパスを調整
    processed_data_directory = '../data/processed/dataset/images'  # 保存先を images サブディレクトリに変更
    main(raw_data_directory, processed_data_directory)