import cv2
import pandas as pd
from pathlib import Path
import tempfile
import subprocess
import shutil
from tqdm import tqdm
import json # ffprobeの出力解析に使う可能性
import re # 正規表現による時間パース用

class PredictionOverlay:
    def __init__(
        self,
        predictions_csv_dir,
        input_video_dir,
        output_video_dir,
        video_fps=None,
        total_frames=None,
        frame_skip=1,
        ffmpeg_path='ffmpeg',
        ffprobe_path='ffprobe'
    ):
        self.predictions_csv_dir = Path(predictions_csv_dir)
        self.input_video_dir = Path(input_video_dir)
        self.output_video_dir = Path(output_video_dir)
        self.video_fps_from_pipeline = video_fps
        self.total_frames_from_pipeline = total_frames
        self.frame_skip_from_pipeline = frame_skip
        self.ffmpeg_path = ffmpeg_path
        self.ffprobe_path = ffprobe_path

        self.font_settings = {
            'FontName': 'Arial',
            'FontSize': '20', # SRTのフォントサイズ
            'PrimaryColour': '&H00FFFFFF',  # White
            'BoxColour': '&H99000000',  # Semi-transparent black for box
            'BorderStyle': '3',  # Box border
            'Alignment': '7',  # Top-left
            'MarginL': '50',
            'MarginV': '50',
            'Shadow': '0',
            'Outline': '0'
        }

    def _get_video_metadata_ffprobe(self, video_path: Path):
        """ffprobeを使用して動画のFPSと総フレーム数を取得する。タイムアウトを設定."""
        # パイプラインからFPSと総フレーム数が渡されていれば、それを使用
        if self.video_fps_from_pipeline is not None and self.total_frames_from_pipeline is not None:
            print(f"ℹ️ パイプラインからの動画情報を使用: FPS={self.video_fps_from_pipeline}, 総フレーム数={self.total_frames_from_pipeline}")
            return self.video_fps_from_pipeline, self.total_frames_from_pipeline, None # durationはNoneで返す

        # ffprobeのコマンド
        cmd_fps_duration = [
            'ffprobe', '-v', 'error', '-select_streams', 'v:0',
            '-show_entries', 'stream=r_frame_rate,duration,nb_frames', '-of', 'json', str(video_path)
        ]
        cmd_count_frames = [ # nb_framesが信頼できない場合があるため、別途count_framesも試みる
            'ffprobe', '-v', 'error', '-select_streams', 'v:0', '-count_frames',
            '-show_entries', 'stream=nb_read_frames', '-of', 'default=noprint_wrappers=1:nokey=1', str(video_path)
        ]

        fps, duration_sec, total_frames = None, None, None
        timeout_seconds = 60 # タイムアウトを60秒に設定

        try:
            process_fps_duration = subprocess.run(cmd_fps_duration, capture_output=True, text=True, check=True, timeout=timeout_seconds)
            metadata = json.loads(process_fps_duration.stdout)
            if metadata['streams']:
                stream_data = metadata['streams'][0]
                # FPS (r_frame_rateは "num/den" 形式)
                if 'r_frame_rate' in stream_data:
                    num, den = map(int, stream_data['r_frame_rate'].split('/'))
                    if den > 0:
                        fps = num / den
                # Duration
                if 'duration' in stream_data:
                    duration_sec = float(stream_data['duration'])
                # Total frames (nb_frames) - これはしばしば不正確か、存在しない
                if 'nb_frames' in stream_data and stream_data['nb_frames'] != 'N/A':
                    total_frames = int(stream_data['nb_frames'])
                
                print(f"  FPS (ffprobe r_frame_rate): {fps if fps else 'N/A'}")
                print(f"  動画の長さ (ffprobe duration): {duration_sec if duration_sec else 'N/A'} 秒")
                print(f"  総フレーム数 (ffprobe nb_frames): {total_frames if total_frames else 'N/A'}")

        except subprocess.TimeoutExpired:
            print(f"⚠️  ffprobeでFPS/Durationの取得に失敗 (タイムアウト): {cmd_fps_duration}")
        except (subprocess.CalledProcessError, json.JSONDecodeError, KeyError) as e:
            print(f"⚠️  ffprobeでFPS/Durationの取得中にエラー: {e}")

        # nb_framesが取得できなかった、または不正確な場合があるため、count_framesを試す
        if total_frames is None or total_frames == 0:
            try:
                process_count_frames = subprocess.run(cmd_count_frames, capture_output=True, text=True, check=True, timeout=timeout_seconds)
                counted_frames_str = process_count_frames.stdout.strip()
                if counted_frames_str and counted_frames_str != 'N/A':
                    total_frames = int(counted_frames_str)
                    print(f"  総フレーム数 (ffprobe count_frames): {total_frames}")
                else:
                    print(f"⚠️  ffprobeでnb_read_framesの取得に失敗 (N/Aまたは空)")
            except subprocess.TimeoutExpired:
                print(f"⚠️  ffprobeでnb_read_framesの取得に失敗 (タイムアウトの可能性): {cmd_count_frames}")
            except (subprocess.CalledProcessError, ValueError) as e:
                print(f"⚠️  ffprobeでnb_read_framesの取得中にエラー: {e}")

        # フォールバック: durationとfpsから推定 (total_framesがまだNoneの場合)
        if total_frames is None and fps and duration_sec:
            total_frames = int(fps * duration_sec)
            print(f"  推定総フレーム数 (fps * duration): {total_frames}")
        
        if fps is None or total_frames is None:
            print("⚠️ 動画のFPSまたは総フレーム数を特定できませんでした。OpenCVでのフォールバックを試みます。")
            cap = cv2.VideoCapture(str(video_path))
            if cap.isOpened():
                if fps is None:
                    fps = cap.get(cv2.CAP_PROP_FPS)
                    print(f"  FPS (OpenCV fallback): {fps if fps else 'N/A'}")
                if total_frames is None:
                    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
                    print(f"  総フレーム数 (OpenCV fallback): {total_frames if total_frames else 'N/A'}")
                cap.release()
            else:
                print(f"❌ OpenCVでもビデオを開けませんでした: {video_path}")
                return None, None, None

        return fps, total_frames, duration_sec


    def _frame_to_srt_time(self, frame_number, fps):
        """フレーム番号をSRT形式のタイムスタンプ (HH:MM:SS,ms) に変換する"""
        if fps == 0: return "00:00:00,000"
        total_seconds = frame_number / fps
        hours = int(total_seconds / 3600)
        minutes = int((total_seconds % 3600) / 60)
        seconds = int(total_seconds % 60)
        milliseconds = int((total_seconds - int(total_seconds)) * 1000)
        return f"{hours:02}:{minutes:02}:{seconds:02},{milliseconds:03}"

    def _generate_srt_content(self, predictions_df, fps, total_video_frames):
        """予測DataFrameからSRTファイルの内容を生成する"""
        srt_content = ""
        # 'frame_number' 列が存在し、予測がそのフレームで開始すると仮定
        # 予測CSVの各行が動画の1フレームに対応していると仮定（特徴量抽出で展開済みの場合）
        # 'predicted_phase' 列に予測ラベルが含まれていると仮定
        # HMM処理後の列名が 'hmm_predicted_phase' の場合、そちらを優先
        prediction_col = 'hmm_predicted_phase' if 'hmm_predicted_phase' in predictions_df.columns else 'predicted_phase'

        # 予測データフレームの行数が動画の総フレーム数と大きく異なる場合、警告
        # ただし、LSTMのシーケンス処理により、予測行数は total_video_frames - (sequence_length - 1) になることがある
        # ここでは、予測DFの各行がフレームに対応し、'frame_number'列でインデックスされると仮定
        # もし'frame_number'がなければ、indexをフレーム番号として扱う
        if 'frame_number' not in predictions_df.columns:
            predictions_df['frame_number'] = predictions_df.index

        # 最後の予測が動画の終わりまで続くと仮定
        # または、各予測にdurationを持たせるか、次の予測の開始までとする
        # ここでは簡略化のため、各予測は1フレーム期間表示されると仮定し、
        # 連続する同じ予測は結合しない（FFmpegが処理する）
        
        # 予測データはフレームスキップ適用後のトラッキングデータから生成され、
        # 特徴量抽出で元の動画のフレームレートに展開されているはず。
        # そのため、predictions_dfの各行は元の動画の1フレームに対応すると期待される。
        
        # 警告: 予測データフレームの行数と動画の総フレーム数の比較
        # LSTMのシーケンス長を考慮した予測行数
        # (例: sequence_length=30なら、予測行数は total_video_frames - 29)
        # ここでは単純に比較するが、厳密な比較は難しい場合がある
        if abs(len(predictions_df) - total_video_frames) > 50 and len(predictions_df) < total_video_frames : # 50フレーム以上の差があり、予測が短い場合
            print(f"⚠️  予測データフレームの行数 ({len(predictions_df)}) が動画の総フレーム数 ({total_video_frames}) と大きく異なります。")
            print(f"     予測は {len(predictions_df)} フレーム分のみオーバーレイされます。")
        
        # SRTエントリ番号
        entry_num = 1
        for idx, row in predictions_df.iterrows():
            frame_num = int(row['frame_number']) # 0-indexed frame number
            
            # 予測が動画の総フレーム数を超えないようにする
            if frame_num >= total_video_frames:
                continue

            start_time_srt = self._frame_to_srt_time(frame_num, fps)
            # 次のフレームまで表示、または短い固定時間 (例: 1/fps 秒)
            # ここでは、各予測が次のフレームの開始直前まで表示されると仮定
            end_time_srt = self._frame_to_srt_time(frame_num + 1, fps) 
            
            prediction_text = str(row[prediction_col]) if pd.notna(row[prediction_col]) else ""
            if not prediction_text: # 空の予測はスキップ
                continue

            srt_content += f"{entry_num}\n"
            srt_content += f"{start_time_srt} --> {end_time_srt}\n"
            srt_content += f"{prediction_text}\n\n"
            entry_num += 1
            
        return srt_content

    def load_predictions(self, csv_path: Path):
        """指定されたCSVファイルから予測データを読み込む"""
        if not csv_path.exists():
            print(f"エラー: 予測CSVファイルが見つかりません: {csv_path}")
            return None
        try:
            df = pd.read_csv(csv_path)
            print(f"✅ 予測データを読み込みました: {csv_path} ({len(df)}行)")
            return df
        except Exception as e:
            print(f"エラー: 予測CSVファイルの読み込み中にエラーが発生しました: {e}")
            return None

    def process_video(self, video_path: Path, predictions_df: pd.DataFrame, csv_filename_stem: str):
        """FFmpegを使用して動画に予測をオーバーレイする"""
        print(f"--- 動画処理開始 (FFmpeg使用): {video_path.name} ---")
        
        fps, total_video_frames, _ = self._get_video_metadata_ffprobe(video_path)

        if fps is None or total_video_frames is None:
            print(f"❌ 動画メタデータの取得に失敗したため、オーバーレイ処理を中止します: {video_path.name}")
            return False
        if fps == 0:
            print(f"❌ FPSが0のため、オーバーレイ処理を中止します: {video_path.name}")
            return False

        srt_content = self._generate_srt_content(predictions_df, fps, total_video_frames)
        if not srt_content:
            print("⚠️ 生成されたSRTコンテンツが空です。オーバーレイは行われません。")
            return False # もしくはTrueで、空の動画を出力する選択も

        with tempfile.NamedTemporaryFile(mode="w", delete=False, suffix=".srt", encoding='utf-8') as tmp_srt_file:
            tmp_srt_file.write(srt_content)
            temp_srt_path = tmp_srt_file.name
        print(f"  一時SRTファイル生成: {temp_srt_path}")

        output_filename = f"{video_path.stem}_overlay_ffmpeg_{Path(csv_filename_stem).stem}.mp4"
        output_filepath = self.output_video_dir / output_filename
        
        # FFmpegコマンドの構築 (SRT字幕としてオーバーレイ)
        # force_styleでフォント、サイズ、色、背景ボックス、配置を指定
        style_string = ",".join([f"{k}={v}" for k,v in self.font_settings.items()])
        
        # SRTパス内のバックスラッシュをスラッシュに置き換え、コロンをエスケープ
        escaped_srt_path = str(Path(temp_srt_path).resolve()).replace('\\', '/')
        # WindowsではCドライブのコロンもエスケープが必要
        if escaped_srt_path[1] == ':':
            escaped_srt_path = escaped_srt_path[0] + '\\:' + escaped_srt_path[2:]

        ffmpeg_cmd = [
            'ffmpeg', '-y', '-i', str(video_path),
            '-vf', f"subtitles='{escaped_srt_path}':force_style='{style_string}'",
            '-c:v', 'h264_nvenc',  # GPUエンコーダを指定 (利用可能な場合)
            '-preset', 'p3',      # エンコードプリセット (p1-p7, p7が最高品質)
            '-cq', '23',          # 固定品質 (1-51, 低いほど高品質)
            '-c:a', 'aac', '-strict', 'experimental', # オーディオコーデック
            str(output_filepath)
        ]
        
        # CPUエンコーダへのフォールバック (h264_nvencが利用できない場合)
        # この部分は、実際にエンコーダの可用性をチェックするロジックを追加するか、
        # ユーザーが選択できるようにするのが望ましい。ここではh264_nvencを試行する。
        print(f"  GPUエンコーダ (h264_nvenc) を使用します。")
        print(f"  実行コマンド: {' '.join(ffmpeg_cmd)}")

        try:
            # FFmpegの実行と進捗表示 (tqdmを使用)
            process = subprocess.Popen(ffmpeg_cmd, stderr=subprocess.PIPE, universal_newlines=True, encoding='utf-8')
            
            progress_bar = tqdm(total=total_video_frames, unit="frame", desc="動画処理中 (FFmpeg)")
            
            # FFmpegのstderrから進捗情報を読み取るための正規表現
            #例: frame=  123 fps= 25.0 q=28.0 size=   1234kB time=00:00:05.00 bitrate=2048.0kbits/s speed=1.0x
            progress_regex = re.compile(r"frame=\s*(\d+)")

            for line in process.stderr:
                match = progress_regex.search(line)
                if match:
                    current_frame = int(match.group(1))
                    progress_bar.update(current_frame - progress_bar.n) # nは現在の進捗
                # print(line, end="") # FFmpegの全出力を表示したい場合

            process.wait()
            progress_bar.close()

            if process.returncode == 0:
                print(f"✅ 動画処理完了: {output_filepath}")
                success = True
            else:
                print(f"❌ FFmpeg処理エラー (リターンコード: {process.returncode})")
                # エラー出力を表示 (既に上で表示している場合は不要かも)
                # for line in process.stderr: # stderrは既に読み終わっている
                #     print(line.strip())
                success = False
        except FileNotFoundError:
            print(f"❌ FFmpegが見つかりません: {self.ffmpeg_path}。パスを確認してください。")
            success = False
        except Exception as e:
            print(f"❌ FFmpeg実行中に予期せぬエラー: {e}")
            success = False
        finally:
            # 一時SRTファイルの削除
            Path(temp_srt_path).unlink()
            print(f"  一時SRTファイル削除: {temp_srt_path}")
            
        return success

    def display_video_with_predictions_realtime(self, video_path: Path, predictions_df: pd.DataFrame):
        """OpenCVを使用してビデオと予測をリアルタイムで表示する"""
        print(f"--- リアルタイム表示開始: {video_path.name} ---")
        cap = cv2.VideoCapture(str(video_path))
        if not cap.isOpened():
            print(f"エラー: ビデオファイルを開けませんでした: {video_path}")
            return

        fps = cap.get(cv2.CAP_PROP_FPS)
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        
        # パイプラインからのFPSと総フレーム数があれば、それらを優先
        if self.video_fps_from_pipeline is not None:
            fps = self.video_fps_from_pipeline
        if self.total_frames_from_pipeline is not None:
            total_frames = self.total_frames_from_pipeline
            
        if fps == 0:
            print("FPSが0です。リアルタイム表示を中止します。")
            cap.release()
            return

        print(f"  ビデオ情報: FPS={fps}, 総フレーム数={total_frames}")
        
        # 予測列の選択
        prediction_col = 'hmm_predicted_phase' if 'hmm_predicted_phase' in predictions_df.columns else 'predicted_phase'
        if 'frame_number' not in predictions_df.columns:
             predictions_df['frame_number'] = predictions_df.index # フレーム番号がない場合はインデックスを使用

        # 予測をフレーム番号で検索できるように辞書化
        # predictions_dict = predictions_df.set_index('frame_number')[prediction_col].to_dict()
        # 最後の予測が動画の終わりまで続くように、またはシーケンス長を考慮
        # ここでは、各フレームに対応する予測があるか、前方フィルで補完する
        predictions_df = predictions_df.set_index('frame_number')
        # 全フレームに対する予測行を作成 (前方フィルで欠損値を補完)
        # 元の動画の全フレームインデックスを作成
        all_frames_index = pd.RangeIndex(start=0, stop=total_frames, step=1)
        # 既存の予測をリインデックスし、前方フィル
        # ただし、予測はシーケンス長分短くなることがあるので、最後の有効な予測でフィルする
        # predictions_for_display = predictions_df[prediction_col].reindex(all_frames_index).fillna(method='ffill')
        
        # より安全な方法: 予測データフレームの範囲内でのみ予測を表示
        # predictions_dict はフレーム番号をキー、予測ラベルを値とする
        predictions_dict = {}
        for _, row in predictions_df.iterrows(): # indexがframe_numberになっているはず
            frame_idx = int(row.name) # row.name がインデックス (frame_number)
            predictions_dict[frame_idx] = row[prediction_col]


        frame_idx = 0
        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                break

            current_prediction = predictions_dict.get(frame_idx, "") # 現在のフレームの予測を取得
            
            # テキスト描画設定
            font = cv2.FONT_HERSHEY_SIMPLEX
            font_scale = 0.7
            font_color = (255, 255, 255)  # White
            line_type = 2
            text_x, text_y = 30, 50 # 表示位置 (左上)

            # テキスト背景用のボックス設定
            (text_width, text_height), baseline = cv2.getTextSize(str(current_prediction), font, font_scale, line_type)
            box_coords = ((text_x, text_y + baseline), (text_x + text_width, text_y - text_height - baseline))
            
            # 背景ボックスを描画 (半透明)
            if current_prediction: # 予測がある場合のみ
                overlay = frame.copy()
                cv2.rectangle(overlay, (box_coords[0][0]-5, box_coords[0][1]+5), (box_coords[1][0]+5, box_coords[1][1]-5), (0, 0, 0), -1) # Black box
                alpha = 0.6 # 透明度
                cv2.addWeighted(overlay, alpha, frame, 1 - alpha, 0, frame)
                
                # テキストを描画
                cv2.putText(frame, str(current_prediction), (text_x, text_y), font, font_scale, font_color, line_type)

            cv2.imshow(f"Realtime Overlay - {video_path.name}", frame)
            
            # FPSに基づいた待機時間
            if cv2.waitKey(int(1000/fps if fps > 0 else 1)) & 0xFF == ord('q'):
                break
            frame_idx += 1

        cap.release()
        cv2.destroyAllWindows()
        print("--- リアルタイム表示終了 ---")


def main():
    overlay_processor = PredictionOverlay()
    
    selected_csv_path = overlay_processor.select_prediction_csv_file()
    if not selected_csv_path:
        print("CSVファイルが選択されませんでした。処理を終了します。")
        return

    predictions_df = overlay_processor.load_predictions(selected_csv_path)
    if predictions_df is None:
        print("推論データの読み込みに失敗しました。処理を終了します。")
        return

    selected_video_path = overlay_processor.select_video_file()
    if not selected_video_path:
        print("動画ファイルが選択されませんでした。処理を終了します。")
        return

    print("\n=== 処理方法選択 ===")
    print("  1. 動画に予測をオーバーレイしてファイル保存 (FFmpeg使用)")
    print("  2. 動画と予測をリアルタイムで表示 (OpenCV使用)")
    
    while True:
        try:
            choice = input("処理方法を選択してください (1-2), Enterでキャンセル: ").strip()
            if not choice:
                print("キャンセルしました。")
                return
            choice_num = int(choice)
            if choice_num == 1:
                print("\n--- FFmpegによるオーバーレイ処理を開始します ---")
                success = overlay_processor.process_video(selected_video_path, predictions_df, selected_csv_path.name)
                if success:
                    print("\n✅ FFmpegオーバーレイ処理が正常に完了しました。")
                else:
                    print("\n❌ FFmpegオーバーレイ処理中にエラーが発生しました。")
                break
            elif choice_num == 2:
                print("\n--- OpenCVによるリアルタイム表示を開始します ---")
                overlay_processor.display_video_with_predictions_realtime(selected_video_path, predictions_df)
                print("\n✅ リアルタイム表示が終了しました。")
                break
            else:
                print("無効な選択です。1または2を入力してください。")
        except ValueError:
            print("無効な入力です。数字で入力してください。")
        except EOFError:
            print("入力がキャンセルされました。")
            return


if __name__ == "__main__":
    main()
