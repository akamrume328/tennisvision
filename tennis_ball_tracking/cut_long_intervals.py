import cv2
import pandas as pd
from pathlib import Path
import numpy as np
import subprocess
import tempfile # 一時ファイル/ディレクトリの管理のため追加
import shutil # shutil.rmtree のため

def get_video_fps(video_path_str: str) -> float:
    """FFprobe を使用してビデオのFPSを取得します."""
    cmd = [
        "ffprobe",
        "-v", "error",
        "-select_streams", "v:0",
        "-show_entries", "stream=r_frame_rate",
        "-of", "default=noprint_wrappers=1:nokey=1",
        str(video_path_str)
    ]
    try:
        result = subprocess.run(cmd, capture_output=True, text=True, check=True)
        num, den = map(int, result.stdout.strip().split('/'))
        return num / den
    except FileNotFoundError:
        print("エラー: ffprobe が見つかりません。FFmpegがインストールされ、PATHが通っているか確認してください。")
        return 0.0
    except subprocess.CalledProcessError as e:
        print(f"エラー: ffprobe の実行に失敗しました: {e}")
        print(f"ffprobe stderr: {e.stderr}")
        return 0.0
    except ValueError as e:
        print(f"エラー: ffprobe からFPSの解析に失敗しました: {e}. Output: {result.stdout.strip()}")
        return 0.0

def cut_video_by_point_interval_ffmpeg(
    video_path_str: str, 
    csv_path_str: str, 
    output_video_path_str: str, 
    # fps: float, # FPSはffprobeから取得するため不要に
    threshold_seconds: float = 2.0, 
    interval_phase_name: str = "point_interval"
):
    """
    FFmpegを使用して、ビデオから指定されたフェーズが閾値秒以上続く部分をカットします。

    Args:
        video_path_str (str): 入力ビデオファイルのパス。
        csv_path_str (str): フレームごとのフェーズ情報を含むCSVファイルのパス。
        output_video_path_str (str): 出力ビデオファイルのパス。
        threshold_seconds (float): カットするインターバルの閾値（秒）。
        interval_phase_name (str): カット対象のフェーズ名。
    """
    video_path = Path(video_path_str)
    csv_path = Path(csv_path_str)
    output_video_path = Path(output_video_path_str)

    if not video_path.exists():
        print(f"エラー: 入力ビデオファイルが見つかりません: {video_path}")
        return False
    if not csv_path.exists():
        print(f"エラー: CSVファイルが見つかりません: {csv_path}")
        return False

    # FFmpeg/ffprobeの存在確認 (簡易的)
    try:
        subprocess.run(["ffmpeg", "-version"], capture_output=True, check=True, shell=True)
        subprocess.run(["ffprobe", "-version"], capture_output=True, check=True, shell=True)
    except (FileNotFoundError, subprocess.CalledProcessError):
        print("エラー: FFmpeg または ffprobe が見つかりません。インストールしてPATHを通してください。")
        return False

    fps = get_video_fps(str(video_path))
    if fps == 0.0:
        print("エラー: ビデオのFPSを取得できませんでした。処理を中止します。")
        return False
    print(f"ビデオのFPS: {fps:.2f}")

    try:
        df = pd.read_csv(csv_path)
    except Exception as e:
        print(f"エラー: CSVファイルの読み込みに失敗しました: {csv_path} - {e}")
        return False

    if 'frame_number' not in df.columns or 'predicted_phase' not in df.columns:
        print(f"エラー: CSVファイルに必要な列 ('frame_number', 'predicted_phase') がありません: {csv_path}")
        return False

    df['is_interval'] = (df['predicted_phase'] == interval_phase_name)
    df['block'] = (df['is_interval'].diff(1) != 0).astype('int').cumsum()
    
    frames_to_cut = set()
    for _, group in df.groupby('block'):
        if not group['is_interval'].all():
            continue
        start_frame = group['frame_number'].min()
        end_frame = group['frame_number'].max()
        duration_frames = end_frame - start_frame + 1
        duration_seconds = duration_frames / fps
        
        if duration_seconds >= threshold_seconds:
            print(f"カット対象のインターバル: フレーム {start_frame} から {end_frame} (期間: {duration_seconds:.2f}秒)")
            for frame_num in range(int(start_frame), int(end_frame) + 1):
                frames_to_cut.add(frame_num)

    if not frames_to_cut:
        print("カット対象の長いインターバルは見つかりませんでした。ビデオは変更されません。")
        try:
            shutil.copy(video_path, output_video_path)
            print(f"元のビデオをコピーしました: {output_video_path}")
            return True
        except Exception as e:
            print(f"エラー: 元のビデオのコピーに失敗しました: {e}")
            return False

    # 総フレーム数を取得 (CSVの最大フレーム番号から推定、またはffprobeで取得も可能)
    # ここではCSVの最大フレーム番号を使用
    total_frames_from_csv = df['frame_number'].max() + 1 

    frames_to_keep_segments = []
    current_segment_start = -1
    for i in range(int(total_frames_from_csv)):
        if i not in frames_to_cut:
            if current_segment_start == -1:
                current_segment_start = i
        else:
            if current_segment_start != -1:
                frames_to_keep_segments.append((current_segment_start, i - 1))
                current_segment_start = -1
    if current_segment_start != -1:
        frames_to_keep_segments.append((current_segment_start, int(total_frames_from_csv) - 1))

    if not frames_to_keep_segments:
        print("エラー: 保持するフレームセグメントがありません。出力ビデオは生成されません。")
        # 空のビデオを作成するか、エラーとするか。
        # ここではエラーとしてFalseを返す。
        return False

    temp_dir = Path(tempfile.mkdtemp(prefix="ffmpeg_cut_"))
    segment_files = []
    concat_file_list_path = temp_dir / "concat_list.txt"

    print(f"一時ディレクトリを作成しました: {temp_dir}")

    try:
        print(f"保持するセグメント数: {len(frames_to_keep_segments)}")
        for i, (start_f, end_f) in enumerate(frames_to_keep_segments):
            segment_output_path = temp_dir / f"segment_{i}.mp4"
            print(f"セグメント {i+1}/{len(frames_to_keep_segments)} (フレーム {start_f}-{end_f}) を抽出中... -> {segment_output_path}")
            
            # FFmpegコマンドでセグメントを抽出
            # selectフィルター: 'gte(n,START_FRAME)*lte(n,END_FRAME)'
            # setpts=N/FRAME_RATE/TB だと元のタイムスタンプが失われるため、PTS-STARTPTS を使う
            # ただし、最初のセグメント以外では、前のセグメントの長さを考慮してPTSを調整する必要があるかもしれないが、
            # concatフィルターがタイムスタンプを再計算してくれるはず。
            # より安全なのは、各セグメントの開始タイムスタンプを0にリセットし、concatで繋ぐこと。
            cmd_segment = [
                "ffmpeg", "-y",
                "-i", str(video_path),
                "-vf", f"select='gte(n,{start_f})*lte(n,{end_f})',setpts=PTS-STARTPTS",
                "-af", f"aselect='gte(n,{start_f})*lte(n,{end_f})',asetpts=PTS-STARTPTS",
                "-c:v", "libx264", # 必要に応じてコーデック指定
                "-preset", "medium", # 速度と品質のバランス
                "-crf", "23", # 品質 (低いほど高品質、ファイルサイズ大)
                "-c:a", "aac", # 音声コーデック
                "-b:a", "128k", # 音声ビットレート
                str(segment_output_path)
            ]
            try:
                # FFmpegの出力を抑制し、エラー時のみ表示
                process = subprocess.run(cmd_segment, capture_output=True, text=True, check=True)
                segment_files.append(segment_output_path)
            except subprocess.CalledProcessError as e:
                print(f"エラー: セグメント {i+1} の抽出に失敗しました (フレーム {start_f}-{end_f})。")
                print(f"FFmpeg コマンド: {' '.join(e.cmd)}")
                print(f"FFmpeg stderr: {e.stderr}")
                # 1つのセグメント抽出失敗で全体を中止
                raise # try-finallyブロックでクリーンアップされる

        if not segment_files:
            print("エラー: 抽出されたセグメントがありません。処理を中止します。")
            return False

        # concat用のファイルリストを作成
        with open(concat_file_list_path, 'w') as f_concat:
            for seg_path in segment_files:
                f_concat.write(f"file '{seg_path.resolve().as_posix()}'\n") # resolveで絶対パス、as_posixでffmpegフレンドリなパス

        print(f"\n全 {len(segment_files)} セグメントの抽出が完了しました。結合処理を開始します...")
        output_video_path.parent.mkdir(parents=True, exist_ok=True)
        cmd_concat = [
            "ffmpeg", "-y",
            "-f", "concat",
            "-safe", "0", # 絶対パスを許可
            "-i", str(concat_file_list_path),
            "-c", "copy", # ビデオと音声を再エンコードせずにコピー (高速)
            str(output_video_path)
        ]
        try:
            process = subprocess.run(cmd_concat, capture_output=True, text=True, check=True)
            print(f"ビデオのカット処理 (FFmpeg) が完了しました。出力先: {output_video_path}")
            return True
        except subprocess.CalledProcessError as e:
            print(f"エラー: セグメントの結合に失敗しました。")
            print(f"FFmpeg コマンド: {' '.join(e.cmd)}")
            print(f"FFmpeg stderr: {e.stderr}")
            return False

    finally:
        # 一時ディレクトリとファイルをクリーンアップ
        if temp_dir.exists():
            print(f"一時ディレクトリ {temp_dir} をクリーンアップします。")
            try:
                shutil.rmtree(temp_dir)
            except Exception as e:
                print(f"警告: 一時ディレクトリのクリーンアップに失敗しました: {e}")

# 元のOpenCVベースの関数名を変更 (または削除)
# def cut_video_by_point_interval_opencv(...): ...

# run_tennis_pipeline.py から呼び出される関数名を cut_video_by_point_interval に統一するため、
# 古い関数はリネームまたは削除し、新しいFFmpegベースの関数を cut_video_by_point_interval とする。
# ここでは、元の関数を cut_video_by_point_interval_opencv とリネームし、
# 新しい関数を cut_video_by_point_interval とします。

# 元の関数 (OpenCVベース) をリネーム
def cut_video_by_point_interval_opencv(
    video_path_str: str, csv_path_str: str, output_video_path_str: str, fps: float, 
    threshold_seconds: float = 2.0, interval_phase_name: str = "point_interval"
):
    video_path = Path(video_path_str)
    csv_path = Path(csv_path_str)
    output_video_path = Path(output_video_path_str)

    if not video_path.exists():
        print(f"エラー: 入力ビデオファイルが見つかりません: {video_path}")
        return False
    if not csv_path.exists():
        print(f"エラー: CSVファイルが見つかりません: {csv_path}")
        return False

    try:
        df = pd.read_csv(csv_path)
    except Exception as e:
        print(f"エラー: CSVファイルの読み込みに失敗しました: {csv_path} - {e}")
        return False

    if 'frame_number' not in df.columns or 'predicted_phase' not in df.columns:
        print(f"エラー: CSVファイルに必要な列 ('frame_number', 'predicted_phase') がありません: {csv_path}")
        return False

    df['is_interval'] = (df['predicted_phase'] == interval_phase_name)
    df['block'] = (df['is_interval'].diff(1) != 0).astype('int').cumsum()
    
    frames_to_cut = set()
    for _, group in df.groupby('block'):
        if not group['is_interval'].all():
            continue
        start_frame = group['frame_number'].min()
        end_frame = group['frame_number'].max()
        duration_frames = end_frame - start_frame + 1
        duration_seconds = duration_frames / fps
        
        if duration_seconds >= threshold_seconds:
            print(f"カット対象のインターバル: フレーム {start_frame} から {end_frame} (期間: {duration_seconds:.2f}秒)")
            for frame_num in range(int(start_frame), int(end_frame) + 1):
                frames_to_cut.add(frame_num)

    if not frames_to_cut:
        print("カット対象の長いインターバルは見つかりませんでした。ビデオは変更されません。")
        try:
            shutil.copy(video_path, output_video_path)
            print(f"元のビデオをコピーしました: {output_video_path}")
            return True
        except Exception as e:
            print(f"エラー: 元のビデオのコピーに失敗しました: {e}")
            return False

    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        print(f"エラー: 入力ビデオファイルを開けませんでした: {video_path}")
        return False

    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out_fps = cap.get(cv2.CAP_PROP_FPS)
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    
    output_video_path.parent.mkdir(parents=True, exist_ok=True)
    writer = cv2.VideoWriter(str(output_video_path), fourcc, out_fps, (width, height))

    if not writer.isOpened():
        print(f"エラー: 出力ビデオファイルを開けませんでした: {output_video_path}")
        cap.release()
        return False

    print(f"ビデオのカット処理 (OpenCV) を開始します。出力先: {output_video_path}")
    current_frame_num = 0
    frames_written = 0
    total_frames_video = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    progress_interval = max(1, total_frames_video // 20) # 約5%ごとに進捗表示
    
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        
        if current_frame_num not in frames_to_cut:
            writer.write(frame)
            frames_written +=1
        
        if total_frames_video > 0 and current_frame_num % progress_interval == 0:
            print(f"OpenCVカット処理進捗: {current_frame_num}/{total_frames_video} フレーム処理済み ({current_frame_num/total_frames_video*100:.1f}%)")
        current_frame_num += 1

    cap.release()
    writer.release()
    print(f"ビデオのカット処理 (OpenCV) が完了しました。{frames_written} フレームが書き込まれました。")
    return True

# エイリアスまたはメインの関数としてFFmpeg版を指定
cut_video_by_point_interval = cut_video_by_point_interval_ffmpeg

if __name__ == '__main__':
    print("cut_long_intervals.py のテスト実行 (FFmpegベース)")
    
    test_data_dir = Path("./test_data_cut_script_ffmpeg")
    test_data_dir.mkdir(exist_ok=True)
    dummy_csv_path = test_data_dir / "dummy_predictions.csv"
    dummy_video_path = test_data_dir / "dummy_video.mp4"
    dummy_output_video_path = test_data_dir / "dummy_video_cut_ffmpeg.mp4"
    
    # ffprobeでFPSを取得するため、ダミービデオのFPSはここで定義
    fps_test = 30.0 
    data = []
    for i in range(300): # 10秒のビデオ @ 30fps
        phase = "rally"
        if 50 <= i < 120: phase = "point_interval" # 70フレーム = 2.33秒 @ 30fps
        elif 180 <= i < 210: phase = "point_interval" # 30フレーム = 1秒
        elif 250 <= i < 300: phase = "point_interval" # 50フレーム = 1.66秒
        data.append({'frame_number': i, 'predicted_phase': phase})
    
    df_test = pd.DataFrame(data)
    df_test.to_csv(dummy_csv_path, index=False)
    print(f"ダミーCSVを作成: {dummy_csv_path}")

    # ダミービデオの作成 (OpenCVを使用するが、テスト用なので許容)
    width_test, height_test = 640, 480
    # OpenCVでビデオを作成する場合、FPSは整数である必要がある場合がある
    cap_test_out = cv2.VideoWriter(str(dummy_video_path), cv2.VideoWriter_fourcc(*'mp4v'), int(fps_test), (width_test, height_test))
    if not cap_test_out.isOpened():
        print(f"エラー: ダミービデオライターを開けませんでした: {dummy_video_path}")
    else:
        for i in range(300):
            frame_test = np.zeros((height_test, width_test, 3), dtype=np.uint8)
            cv2.putText(frame_test, f"Frame: {i}", (50, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)
            cap_test_out.write(frame_test)
        cap_test_out.release()
        print(f"ダミービデオを作成: {dummy_video_path} (FPS: {fps_test})")

        # FFmpegベースのカット関数をテスト
        success = cut_video_by_point_interval_ffmpeg(
            video_path_str=str(dummy_video_path),
            csv_path_str=str(dummy_csv_path),
            output_video_path_str=str(dummy_output_video_path),
            # fps=fps_test, # FFmpeg版では不要
            threshold_seconds=2.0,
            interval_phase_name="point_interval"
        )
        if success:
            print(f"テスト成功。カットされたビデオ (FFmpeg): {dummy_output_video_path}")
            # カット後のビデオの長さを確認 (オプション)
            # ffprobe -v error -show_entries format=duration -of default=noprint_wrappers=1:nokey=1 dummy_video_cut_ffmpeg.mp4
        else:
            print("テスト失敗 (FFmpeg)。")
