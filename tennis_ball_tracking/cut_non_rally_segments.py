import cv2
import pandas as pd
from pathlib import Path
import numpy as np
import subprocess
import tempfile
import shutil
import sys

FFMPEG_SEEK_BUFFER_FRAMES = 5  # キーフレーム境界での高速カットのためのバッファフレーム数
RALLY_PHASE_NAME = "rally"  # ラリー局面の名前

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

def smooth_phase_predictions(df: pd.DataFrame, fps: float, min_phase_duration_seconds: float, rally_phase_name: str) -> pd.DataFrame:
    """
    Rally区間の前後にある短い誤検出を修正し、局面予測をインテリジェントにスムージングします。
    """
    df_smooth = df.copy()
    min_phase_frames = int(min_phase_duration_seconds * fps)
    
    if min_phase_frames <= 1:
        return df_smooth

    print(f"局面予測スムージング中... ({min_phase_duration_seconds}秒未満の局面をインテリジェントに修正)")
    
    df_smooth['phase_block'] = (df_smooth['predicted_phase'] != df_smooth['predicted_phase'].shift(1)).cumsum()
    
    phase_block_info = df_smooth.groupby('phase_block').agg(
        phase=('predicted_phase', 'first'),
        duration_frames=('frame_number', 'count')
    ).reset_index()

    for i in range(1, len(phase_block_info) - 1):
        current_block = phase_block_info.loc[i]
        
        if current_block['duration_frames'] < min_phase_frames:
            prev_phase = phase_block_info.loc[i - 1, 'phase']
            next_phase = phase_block_info.loc[i + 1, 'phase']
            
            phase_to_correct = None
            
            # ケース1: Rally -> 短い非Rally -> Rally
            if (current_block['phase'] != rally_phase_name and
                prev_phase == rally_phase_name and 
                next_phase == rally_phase_name):
                phase_to_correct = rally_phase_name

            # ケース2: 非Rally -> 短いRally -> 非Rally
            elif (current_block['phase'] == rally_phase_name and
                  prev_phase == next_phase and
                  prev_phase != rally_phase_name):
                phase_to_correct = prev_phase

            if phase_to_correct:
                block_id_to_correct = current_block['phase_block']
                original_phase = current_block['phase']
                df_smooth.loc[df_smooth['phase_block'] == block_id_to_correct, 'predicted_phase'] = phase_to_correct
                print(f"  スムージング: ブロック {block_id_to_correct} ('{original_phase}') を '{phase_to_correct}' に修正しました。")

    df_smooth = df_smooth.drop(columns=['phase_block'], errors='ignore')
    return df_smooth

def cut_rally_segments(
    video_path: Path,
    csv_path: Path,
    output_path: Path,
    buffer_before: float,
    buffer_after: float,
    min_rally_duration: float,
    min_phase_duration: float
):
    """FFmpegを使用して、ラリー区間のみを切り出して結合するメイン関数。"""
    try:
        subprocess.run(["ffmpeg", "-version"], capture_output=True, check=True)
    except (FileNotFoundError, subprocess.CalledProcessError):
        print("エラー: FFmpegまたはffprobeが見つかりません。インストールしてPATHを通してください。", file=sys.stderr)
        return False

    fps = get_video_fps(str(video_path))
    if fps == 0.0: return False
    print(f"ビデオFPS: {fps:.2f}")

    df = pd.read_csv(csv_path)
    if 'frame_number' not in df.columns or 'predicted_phase' not in df.columns:
        print(f"エラー: CSVに必須列('frame_number', 'predicted_phase')がありません。", file=sys.stderr)
        return False

    # ★★★ 修正点: smooth_phase_predictions 呼び出し時に引数を追加 ★★★
    df = smooth_phase_predictions(df, fps, min_phase_duration, rally_phase_name=RALLY_PHASE_NAME)

    # is_rally と block の計算をスムージング後に移動
    df['is_rally'] = (df['predicted_phase'] == RALLY_PHASE_NAME)
    df['block'] = (df['is_rally'].diff(1) != 0).astype('int').cumsum()
    
    buffer_before_frames = int(buffer_before * fps)
    buffer_after_frames = int(buffer_after * fps)
    
    rally_intervals = []
    for _, group in df[df['is_rally']].groupby('block'):
        start_frame, end_frame = group['frame_number'].min(), group['frame_number'].max()
        if (end_frame - start_frame + 1) / fps >= min_rally_duration:
            protected_start = max(0, start_frame - buffer_before_frames)
            protected_end = end_frame + buffer_after_frames
            rally_intervals.append((protected_start, protected_end))
        else:
            print(f"情報: 短いラリー区間 (フレーム {start_frame}-{end_frame}) はスキップされました。")

    if not rally_intervals:
        print(f"エラー: 保持対象の '{RALLY_PHASE_NAME}' 区間が見つかりませんでした。", file=sys.stderr)
        return False

    rally_intervals.sort()
    
    merged_intervals = []
    if rally_intervals:
        current_start, current_end = rally_intervals[0]
        for next_start, next_end in rally_intervals[1:]:
            if next_start <= current_end + 1:
                current_end = max(current_end, next_end)
            else:
                merged_intervals.append((current_start, current_end))
                current_start, current_end = next_start, next_end
        merged_intervals.append((current_start, current_end))

    temp_dir = Path(tempfile.mkdtemp(prefix="rally_cut_"))
    segment_files = []
    print(f"\n一時ディレクトリを作成しました: {temp_dir}")

    try:
        total_frames_from_csv = df['frame_number'].max() + 1
        print(f"保持セグメント数: {len(merged_intervals)}")
        for i, (start_f, end_f) in enumerate(merged_intervals):
            end_f = min(end_f, total_frames_from_csv - 1)
            if start_f >= end_f: continue

            start_time = max(0, (start_f - FFMPEG_SEEK_BUFFER_FRAMES) / fps)
            duration = ((end_f + FFMPEG_SEEK_BUFFER_FRAMES) / fps) - start_time
            if duration <= 0: continue

            segment_path = temp_dir / f"segment_{i}.mp4"
            print(f"セグメント {i+1}/{len(merged_intervals)} (時間 {start_time:.2f}s - {start_time+duration:.2f}s) を抽出中...")
            cmd = [
                "ffmpeg", "-y", "-ss", f"{start_time:.3f}", "-i", str(video_path),
                "-t", f"{duration:.3f}", "-c", "copy", "-avoid_negative_ts", "make_zero", str(segment_path)
            ]
            result = subprocess.run(cmd, capture_output=True, text=True)
            if result.returncode == 0 and segment_path.exists() and segment_path.stat().st_size > 0:
                segment_files.append(segment_path)
            else:
                print(f"警告: セグメント {i+1} の抽出に失敗しました。スキップします。", file=sys.stderr)
                print(f"FFmpeg stderr: {result.stderr}", file=sys.stderr)

        if not segment_files:
            print("エラー: 抽出された有効なビデオセグメントがありません。", file=sys.stderr)
            return False

        concat_list_path = temp_dir / "concat_list.txt"
        with open(concat_list_path, 'w') as f:
            for seg_path in segment_files:
                f.write(f"file '{seg_path.resolve().as_posix()}'\n")

        print(f"\n全 {len(segment_files)} セグメントを結合中...")
        output_path.parent.mkdir(parents=True, exist_ok=True)
        cmd_concat = [
            "ffmpeg", "-y", "-f", "concat", "-safe", "0", "-i", str(concat_list_path),
            "-c", "copy", str(output_path)
        ]
        result = subprocess.run(cmd_concat, capture_output=True, text=True)
        if result.returncode != 0:
            print("エラー: セグメントの結合に失敗しました。", file=sys.stderr)
            print(f"FFmpeg stderr: {result.stderr}", file=sys.stderr)
            return False

        print(f"\n✅ 処理完了！ Rally区間のみのビデオを保存しました: {output_path}")
        return True

    finally:
        if temp_dir.exists():
            shutil.rmtree(temp_dir)
            print(f"一時ディレクトリをクリーンアップしました: {temp_dir}")
            
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
    # Rally 1: フレーム 30-59 (1秒間) -> スキップされるべき
    # Rally 2: フレーム 120-209 (3秒間) -> 保持されるべき
    # 一瞬の誤検出: フレーム 150-154 (5フレーム、0.17秒) で point_interval -> スムージングで無視されるべき
    total_test_frames = 300
    for i in range(total_test_frames):
        phase = "point_interval"  # デフォルトは非rally
        if 30 <= i < 60:   # フレーム 30-59: rally (1秒間)
            phase = "rally"
        elif 120 <= i < 210:  # フレーム 120-209: rally (3秒間)
            phase = "rally"
            # 一瞬の誤検出をシミュレート
            if 150 <= i <= 154:  # 5フレーム (0.17秒) の誤検出
                phase = "point_interval"
        data.append({'frame_number': i, 'predicted_phase': phase})
    
    df_test = pd.DataFrame(data)
    df_test.to_csv(dummy_csv_path, index=False)
    print(f"ダミーCSVを作成: {dummy_csv_path}")
    print("テスト用Rally区間:")
    print(f"  Rally 1 (短): フレーム 30-59 ({(60-30)/fps_test:.2f}秒)")
    print(f"  Rally 2 (長): フレーム 120-209 ({(210-120)/fps_test:.2f}秒)")
    print(f"  誤検出: フレーム 150-154 ({5/fps_test:.2f}秒) - スムージングで修正されるべき")

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
    success = cut_rally_segments(
        video_path_str=str(dummy_video_path),
        csv_path_str=str(dummy_csv_path),
        output_video_path_str=str(dummy_output_video_path),
        rally_phase_name="rally",
        buffer_before_seconds=1.5,  # Rally前1.5秒
        buffer_after_seconds=0.5,   # Rally後0.5秒
        min_rally_duration_seconds=2.0,
        min_phase_duration_seconds=0.5  # 0.5秒未満の局面変更を無視
    )
    if success:
        print(f"テスト成功。Rally区間のみのビデオ: {dummy_output_video_path}")
        print(f"期待される動作: Rally 1 (1秒) は無視され、Rally 2 (3秒) + 前1.5秒・後0.5秒バッファが抽出される。")
        print(f"0.5秒未満の誤検出は修正される。")
    else:
        print("テスト失敗。")