import os
import cv2
import numpy as np
import pandas as pd
import subprocess # subprocessモジュールをインポート
import shutil # shutilモジュールをインポート

def load_raw_data(data_dir):
    raw_data = []
    # 一時的なフレーム保存用ディレクトリを data ディレクトリの親に作成
    # 例: data_dir が '../data/raw' の場合、'../data/temp_frames' を作成
    temp_frame_base_dir = os.path.join(os.path.dirname(os.path.abspath(data_dir)), 'temp_frames')
    
    if not os.path.exists(temp_frame_base_dir):
        os.makedirs(temp_frame_base_dir, exist_ok=True)

    for filename in os.listdir(data_dir):
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

            # ffmpegコマンドでフレームを抽出し、同時にリサイズ (640x480)
            command = [
                'ffmpeg',
                '-i', file_path,
                '-vf', 'scale=640:480', # リサイズフィルターを追加
                '-qscale:v', '2',       # 高画質でフレームを抽出
                os.path.join(output_frame_dir, 'frame_%04d.png')
            ]
            try:
                # ffmpegコマンドの標準出力と標準エラー出力を抑制する
                subprocess.run(command, check=True, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
                # 抽出されたフレームのパスをリストに追加
                for frame_filename in sorted(os.listdir(output_frame_dir)): # フレーム順を保証するためにソート
                    if frame_filename.lower().endswith('.png'): # ffmpegの出力形式に合わせる
                        raw_data.append(os.path.join(output_frame_dir, frame_filename))
            except subprocess.CalledProcessError as e:
                print(f"Error processing video {filename} with ffmpeg. Return code: {e.returncode}")
                # エラーメッセージを表示したい場合は、stdout/stderrをキャプチャする
                # print(f"FFmpeg stdout: {e.stdout.decode() if e.stdout else 'N/A'}")
                # print(f"FFmpeg stderr: {e.stderr.decode() if e.stderr else 'N/A'}")

            except FileNotFoundError:
                print("Error: ffmpeg command not found. Please ensure ffmpeg is installed and in your PATH.")

        elif filename.lower().endswith(('.jpg', '.jpeg', '.png')): # .jpegも追加、小文字で比較
            raw_data.append(file_path)
    return raw_data, temp_frame_base_dir # 一時ディレクトリのパスも返す

def preprocess_image(image_path):
    image = cv2.imread(image_path)
    # 画像を指定されたサイズにリサイズ (YOLOv8学習のため、このリサイズは一般的に推奨されます。ただし、対象物が非常に小さい場合、この解像度で検出が困難な場合は、解像度の調整や他の手法の検討が必要になることがあります)
    image = cv2.resize(image, (640, 480))
    # 画像を正規化
    image = image / 255.0
    return image

def save_processed_data(processed_data, save_dir):
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)
    for i, image in enumerate(processed_data):
        # 画像を0-255の範囲に戻す
        image_to_save = (image * 255.0).astype(np.uint8)
        # ファイル名を frame_XXXXXX.png 形式に変更 (XXXXXX は6桁の0埋め連番)
        save_path = os.path.join(save_dir, f'frame_{i:06d}.png')
        cv2.imwrite(save_path, image_to_save) # cv2.imwriteを使用して保存

def main(raw_data_dir, processed_data_dir):
    raw_data_paths, temp_dir_to_delete = load_raw_data(raw_data_dir) # 一時ディレクトリのパスを受け取る
    processed_data = [preprocess_image(image_path) for image_path in raw_data_paths]
    save_processed_data(processed_data, processed_data_dir)

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