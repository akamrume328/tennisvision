import cv2
import pandas as pd
from pathlib import Path
import numpy as np
import subprocess
import tempfile
import shutil
import sys

def get_video_fps(video_path_str: str) -> float:
    """FFprobe を使用してビデオのFPSを取得します。"""
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

def cut_non_rally_segments(
    video_path_str: str, 
    csv_path_str: str, 
    output_video_path_str: str, 
    rally_phase_name: str = "rally",
    buffer_seconds: float = 2.0,
    min_rally_duration_seconds: float = 2.0  # 新しいパラメータ
):
    """
    FFmpegを使用して、rally区間とその前後の指定秒数以外をビデオからカットします。
    指定された最小長より短いrally区間は無視されます。

    Args:
        video_path_str (str): 入力ビデオファイルのパス。
        csv_path_str (str): フレームごとのフェーズ情報を含むCSVファイルのパス。
                               'frame_number' と 'predicted_phase' 列が必要です。
        output_video_path_str (str): 出力ビデオファイルのパス。
        rally_phase_name (str): 保持対象のフェーズ名（デフォルト: "rally"）。
        buffer_seconds (float): rally区間の前後に保持する秒数。
        min_rally_duration_seconds (float): 保持対象とするrally区間の最小秒数（デフォルト: 2.0秒）。
                                            これより短いrally区間は無視されます。
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

    # FFmpeg/ffprobeの存在確認
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

    # rally フレームを特定
    df['is_rally'] = (df['predicted_phase'] == rally_phase_name)
    
    # 連続するrallyブロックを識別
    df['block'] = (df['is_rally'].diff(1) != 0).astype('int').cumsum()
    
    buffer_frames = int(buffer_seconds * fps)
    frames_to_keep = set()
    
    print(f"Rally区間の検出と前後{buffer_seconds}秒（{buffer_frames}フレーム）の保護範囲設定 (最小ラリー長: {min_rally_duration_seconds}秒):")
    
    for block_num, group in df.groupby('block'):
        if not group['is_rally'].all():  # このブロックがrallyでなければスキップ
            continue
            
        start_frame = group['frame_number'].min()
        end_frame = group['frame_number'].max()
        duration_frames = end_frame - start_frame + 1
        duration_seconds = duration_frames / fps
        
        # Rally区間の長さが最小要件を満たしているか確認
        if duration_seconds <= min_rally_duration_seconds:
            print(f"Rally区間: フレーム {start_frame}-{end_frame} (期間: {duration_seconds:.2f}秒) は短すぎるためスキップします (最小期間: {min_rally_duration_seconds}秒)。")
            continue
            
        # 前後のバッファを追加した保護範囲を計算
        protected_start = max(0, int(start_frame) - buffer_frames)
        protected_end = int(end_frame) + buffer_frames
        
        print(f"Rally区間: フレーム {start_frame}-{end_frame} (期間: {duration_seconds:.2f}秒)")
        print(f"  -> 保護範囲: フレーム {protected_start}-{protected_end}")
        
        for frame_num in range(protected_start, protected_end + 1):
            frames_to_keep.add(frame_num)

    if not frames_to_keep:
        print(f"エラー: {rally_phase_name} 区間が見つかりませんでした。処理を中止します。")
        return False

    # 総フレーム数を取得
    total_frames_from_csv = df['frame_number'].max() + 1
    
    # 保持するフレームを連続セグメントにまとめる
    frames_to_keep_sorted = sorted(frames_to_keep)
    frames_to_keep_segments = []
    current_segment_start = -1
    
    for i, frame_num in enumerate(frames_to_keep_sorted):
        if current_segment_start == -1:
            current_segment_start = frame_num
            current_segment_end = frame_num
        elif frame_num == current_segment_end + 1:
            current_segment_end = frame_num
        else:
            # 現在のセグメント完了
            frames_to_keep_segments.append((current_segment_start, current_segment_end))
            current_segment_start = frame_num
            current_segment_end = frame_num
    
    # 最後のセグメントを追加
    if current_segment_start != -1:
        frames_to_keep_segments.append((current_segment_start, current_segment_end))

    if not frames_to_keep_segments:
        print("エラー: 保持するフレームセグメントがありません。")
        return False

    temp_dir = Path(tempfile.mkdtemp(prefix="ffmpeg_rally_cut_"))
    segment_files = []
    concat_file_list_path = temp_dir / "concat_list.txt"

    print(f"一時ディレクトリを作成しました: {temp_dir}")

    try:
        print(f"Rally区間とバッファを含む保持セグメント数: {len(frames_to_keep_segments)}")
        
        # 元のビデオの総フレーム数に基づいてセグメントの範囲を調整
        for i, (start_f, end_f) in enumerate(frames_to_keep_segments):
            # CSVの範囲を超えないように調整
            end_f = min(end_f, int(total_frames_from_csv) - 1)
            
            if start_f > end_f:
                print(f"警告: セグメント {i+1} の範囲が無効です (start={start_f}, end={end_f})。スキップします。")
                continue
                
            # Calculate segment start time and duration for faster copy
            start_time = start_f / fps
            duration = (end_f - start_f + 1) / fps
            segment_output_path = temp_dir / f"rally_segment_{i}.mp4"
            print(f"セグメント {i+1}/{len(frames_to_keep_segments)} (時間 {start_time:.2f}s - {start_time+duration:.2f}s) を抽出中... -> {segment_output_path}")

            cmd_segment = [
                "ffmpeg", "-y",
                "-ss", str(start_time),
                "-i", str(video_path),
                "-t", str(duration),
                "-c", "copy",
                str(segment_output_path)
            ]
            try:
                subprocess.run(cmd_segment, check=True)
                segment_files.append(segment_output_path)
            except subprocess.CalledProcessError as e:
                print(f"エラー: セグメント {i+1} の抽出に失敗しました (時間 {start_time:.2f}s - {start_time+duration:.2f}s)。")
                print(f"FFmpeg コマンド: {' '.join(e.cmd)}")
                print(f"FFmpeg stderr: {e.stderr}")
                raise

        if not segment_files:
            print("エラー: 抽出されたセグメントがありません。処理を中止します。")
            return False

        # concat用のファイルリストを作成
        with open(concat_file_list_path, 'w') as f_concat:
            for seg_path in segment_files:
                f_concat.write(f"file '{seg_path.resolve().as_posix()}'\n")

        print(f"\n全 {len(segment_files)} セグメントの抽出が完了しました。結合処理を開始します...")
        output_video_path.parent.mkdir(parents=True, exist_ok=True)
        
        cmd_concat = [
            "ffmpeg", "-y",
            "-f", "concat",
            "-safe", "0",
            "-i", str(concat_file_list_path),
            "-c", "copy",
            str(output_video_path)
        ]
        try:
            process = subprocess.run(cmd_concat, capture_output=True, text=True, check=True)
            print(f"Rally区間抽出処理 (FFmpeg) が完了しました。出力先: {output_video_path}")
            
            # 元のビデオと出力ビデオの長さを比較表示
            original_frames = len(frames_to_keep)
            kept_frames = len(frames_to_keep)
            removed_frames = int(total_frames_from_csv) - kept_frames
            print(f"処理結果: 全{int(total_frames_from_csv)}フレーム中、{kept_frames}フレームを保持、{removed_frames}フレームを削除")
            
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

if __name__ == '__main__':
    print("cut_non_rally_segments.py のテスト実行")
    
    test_data_dir = Path("./test_data_rally_cut_script")
    test_data_dir.mkdir(exist_ok=True)
    dummy_csv_path = test_data_dir / "dummy_predictions_rally.csv"
    dummy_video_path = test_data_dir / "dummy_video_rally.mp4"
    dummy_output_video_path = test_data_dir / "dummy_video_rally_only.mp4"
    
    fps_test = 30.0 
    data = []
    # テストデータ: 10秒のビデオ (300フレーム @ 30fps)
    # Rally 1: フレーム 30-59 (1秒間) -> スキップされるべき (min_rally_duration_seconds = 2.0 の場合)
    # Rally 2: フレーム 120-209 (3秒間) -> 保持されるべき
    total_test_frames = 300
    for i in range(total_test_frames):
        phase = "point_interval"  # デフォルトは非rally
        if 30 <= i < 60:   # フレーム 30-59: rally (1秒間)
            phase = "rally"
        elif 120 <= i < 210:  # フレーム 120-209: rally (3秒間)
            phase = "rally"
        data.append({'frame_number': i, 'predicted_phase': phase})
    
    df_test = pd.DataFrame(data)
    df_test.to_csv(dummy_csv_path, index=False)
    print(f"ダミーCSVを作成: {dummy_csv_path}")
    print("テスト用Rally区間:")
    print(f"  Rally 1 (短): フレーム 30-59 ({(60-30)/fps_test:.2f}秒)")
    print(f"  Rally 2 (長): フレーム 120-209 ({(210-120)/fps_test:.2f}秒)")

    # ダミービデオの作成
    width_test, height_test = 640, 480
    # シェル経由での実行を避けるため、FFmpegコマンドをリストで直接指定
    cmd_create_dummy_video = [
        "ffmpeg", "-y", "-f", "lavfi",
        "-i", f"color=c=blue:s={width_test}x{height_test}:d={total_test_frames/fps_test}",
        "-vf", f"[in]drawtext=fontfile=arial.ttf:text='Frame %{{n}}':x=50:y=50:fontsize=24:fontcolor=white," +
               f"drawtext=fontfile=arial.ttf:text='CSV Phase: point_interval':x=50:y=100:fontsize=20:fontcolor=red:enable='between(n,0,29)+between(n,60,119)+between(n,210,{total_test_frames-1})'," +
               f"drawtext=fontfile=arial.ttf:text='CSV Phase: rally':x=50:y=100:fontsize=20:fontcolor=green:enable='between(n,30,59)+between(n,120,209)'[out]",
        "-r", str(int(fps_test)),
        str(dummy_video_path)
    ]
    try:
        print(f"ダミービデオを作成中: {dummy_video_path} (FPS: {fps_test})")
        subprocess.run(cmd_create_dummy_video, check=True, capture_output=True, text=True)
        print(f"ダミービデオを作成しました。")
    except FileNotFoundError:
        print("エラー: FFmpeg が見つかりません。ダミービデオの作成をスキップします。OpenCVフォールバックを使用します。")
        # OpenCVフォールバック (FFmpegが使えない環境向け)
        cap_test_out = cv2.VideoWriter(str(dummy_video_path), cv2.VideoWriter_fourcc(*'mp4v'), int(fps_test), (width_test, height_test))
        if not cap_test_out.isOpened():
            print(f"エラー: ダミービデオライターを開けませんでした: {dummy_video_path}")
        else:
            for i in range(total_test_frames):
                frame_test = np.zeros((height_test, width_test, 3), dtype=np.uint8)
                current_phase_in_csv = "point_interval"
                if (30 <= i < 60) or (120 <= i < 210):
                    current_phase_in_csv = "rally"
                
                cv2.putText(frame_test, f"Frame: {i}", (50, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)
                cv2.putText(frame_test, f"CSV Phase: {current_phase_in_csv}", (50, 100), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0) if current_phase_in_csv == "rally" else (0, 0, 255), 2)
                cap_test_out.write(frame_test)
            cap_test_out.release()
            print(f"ダミービデオを作成 (OpenCV): {dummy_video_path} (FPS: {fps_test})")
    except subprocess.CalledProcessError as e:
        print(f"エラー: ダミービデオの作成に失敗しました (FFmpeg)。")
        print(f"FFmpeg コマンド: {' '.join(e.cmd)}")
        print(f"FFmpeg stderr: {e.stderr}")
        print("テストを続行できません。")
        sys.exit(1)


    if not dummy_video_path.exists():
        print(f"エラー: ダミービデオファイルが存在しません: {dummy_video_path}。テストを中止します。")
        sys.exit(1)

    # Rally区間抽出のテスト
    success = cut_non_rally_segments(
        video_path_str=str(dummy_video_path),
        csv_path_str=str(dummy_csv_path),
        output_video_path_str=str(dummy_output_video_path),
        rally_phase_name="rally",
        buffer_seconds=1.0, # バッファを1秒に設定してテスト
        min_rally_duration_seconds=2.0 # 2秒未満のラリーは無視
    )
    if success:
        print(f"テスト成功。Rally区間のみのビデオ: {dummy_output_video_path}")
        print(f"期待される動作: Rally 1 (1秒) は無視され、Rally 2 (3秒) + 前後1秒バッファが抽出される。")
    else:
        print("テスト失敗。")