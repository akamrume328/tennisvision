import cv2
import pandas as pd
from pathlib import Path
from datetime import datetime
from typing import Optional, Tuple
import subprocess # FFmpeg実行用
import tempfile # 一時SRTファイル用
import os # 一時ファイル削除用
import re # 進捗解析用
from tqdm import tqdm # tqdmをインポート

class PredictionOverlay:
    def __init__(self, 
                 predictions_csv_dir: str = "./training_data/predictions",
                 input_video_dir: str = "../data/raw", # 元動画が格納されていると仮定するディレクトリ
                 output_video_dir: str = "./training_data/video_outputs"):
        self.predictions_csv_dir = Path(predictions_csv_dir)
        self.input_video_dir = Path(input_video_dir)
        self.output_video_dir = Path(output_video_dir)

        self.predictions_csv_dir.mkdir(parents=True, exist_ok=True)
        self.input_video_dir.mkdir(parents=True, exist_ok=True) # ユーザーが動画を置くことを期待
        self.output_video_dir.mkdir(parents=True, exist_ok=True)

        # 表示設定
        self.font = cv2.FONT_HERSHEY_SIMPLEX
        self.font_scale = 1
        self.font_color = (255, 255, 255) # White
        self.line_type = 2
        self.text_position = (50, 50) # (x, y) from top-left
        self.confidence_threshold = 0.0 # 表示する信頼度の閾値 (0.0なら全て表示)
        
        self.use_gpu_encoder = self._check_gpu_encoder_availability()
        if self.use_gpu_encoder:
            print("ℹ️  NVIDIA GPUエンコーダ (h264_nvenc) が利用可能です。")
        else:
            print("ℹ️  NVIDIA GPUエンコーダが見つからないため、CPUエンコーダ (libx264) を使用します。")

    def _check_gpu_encoder_availability(self) -> bool:
        try:
            result = subprocess.run(['ffmpeg', '-hide_banner', '-encoders'], capture_output=True, text=True, timeout=5)
            if "h264_nvenc" in result.stdout:
                return True
        except (FileNotFoundError, subprocess.TimeoutExpired, Exception) as e:
            print(f"⚠️  GPUエンコーダの確認中にエラー: {e}")
        return False

    def select_file_from_dir(self, target_dir: Path, file_pattern: str, description: str) -> Optional[Path]:
        print(f"\n=== {description} ファイル選択 ===")
        files = sorted(list(target_dir.glob(file_pattern)), key=lambda p: p.stat().st_mtime, reverse=True)
        
        if not files:
            print(f"❌ {file_pattern} ファイルが見つかりません in {target_dir}")
            return None

        for i, f_path in enumerate(files, 1):
            try:
                mtime = datetime.fromtimestamp(f_path.stat().st_mtime).strftime('%Y-%m-%d %H:%M')
            except Exception:
                mtime = "N/A"
            print(f"  {i}. {f_path.name} (更新日時: {mtime})")
        
        while True:
            try:
                choice = input(f"選択してください (1-{len(files)}), Enterでキャンセル: ").strip()
                if not choice: return None # Enterでキャンセル
                choice_num = int(choice)
                if 1 <= choice_num <= len(files):
                    return files[choice_num - 1]
                else:
                    print("無効な選択です。")
            except ValueError:
                print("無効な入力です。数字で入力してください。")
            except EOFError: 
                print("入力がキャンセルされました。")
                return None

    def select_prediction_csv_file(self) -> Optional[Path]:
        return self.select_file_from_dir(self.predictions_csv_dir, "*_predictions_*.csv", "推論結果CSV")

    def select_video_file(self) -> Optional[Path]:
        video_patterns = ["*.mp4", "*.avi", "*.mov", "*.mkv"]
        all_video_files = []
        for pattern in video_patterns:
            all_video_files.extend(list(self.input_video_dir.glob(pattern)))
        
        unique_video_files = sorted(list(set(all_video_files)), key=lambda p: p.stat().st_mtime, reverse=True)

        print(f"\n=== 元動画ファイル選択 ===")
        if not unique_video_files:
            print(f"❌ 動画ファイルが見つかりません in {self.input_video_dir} (対応拡張子: .mp4, .avi, .mov, .mkv)")
            print(f"ℹ️  動画ファイルを {self.input_video_dir} に配置してください。")
            return None

        for i, f_path in enumerate(unique_video_files, 1):
            try:
                mtime = datetime.fromtimestamp(f_path.stat().st_mtime).strftime('%Y-%m-%d %H:%M')
            except Exception:
                mtime = "N/A"
            print(f"  {i}. {f_path.name} (更新日時: {mtime})")
        
        while True:
            try:
                choice = input(f"選択してください (1-{len(unique_video_files)}), Enterでキャンセル: ").strip()
                if not choice: return None
                choice_num = int(choice)
                if 1 <= choice_num <= len(unique_video_files):
                    return unique_video_files[choice_num - 1]
                else:
                    print("無効な選択です。")
            except ValueError:
                print("無効な入力です。数字で入力してください。")
            except EOFError:
                print("入力がキャンセルされました。")
                return None

    def load_predictions(self, csv_path: Path) -> Optional[pd.DataFrame]:
        print(f"\n--- 推論結果読み込み: {csv_path.name} ---")
        try:
            df = pd.read_csv(csv_path)
            required_cols = ['predicted_phase', 'prediction_confidence']
            missing_cols = [col for col in required_cols if col not in df.columns]
            if missing_cols:
                print(f"❌ CSVに必要な列がありません: {missing_cols}")
                return None
            
            df['predicted_phase'] = df['predicted_phase'].fillna('')
            df['prediction_confidence'] = df['prediction_confidence'].fillna(0.0)

            print(f"✅ 推論結果読み込み完了: {len(df)}フレーム分のデータ")
            return df
        except Exception as e:
            print(f"❌ CSV読み込みエラー: {e}")
            return None

    def _format_time_for_srt(self, seconds: float) -> str:
        # Helper to format time for SRT files (HH:MM:SS,mmm)
        millis = int((seconds - int(seconds)) * 1000)
        seconds_int = int(seconds)
        hrs = seconds_int // 3600
        mins = (seconds_int % 3600) // 60
        secs = seconds_int % 60
        return f"{hrs:02d}:{mins:02d}:{secs:02d},{millis:03d}"

    def _generate_srt_content(self, predictions_df: pd.DataFrame, fps: float) -> str:
        srt_entries = []
        num_predictions = len(predictions_df)
        if fps <= 0: # FPSが不正な場合は警告を出し、デフォルトのフレーム期間を使用
            print(f"⚠️  FPS値 ({fps}) が不正です。各字幕の表示期間を0.1秒に設定します。")
            frame_duration = 0.1 
        else:
            frame_duration = 1.0 / fps

        for i in range(num_predictions):
            prediction_row = predictions_df.iloc[i]
            phase = prediction_row['predicted_phase']
            confidence = prediction_row['prediction_confidence']

            if pd.notna(phase) and phase != "" and confidence >= self.confidence_threshold:
                text = f"{phase} ({confidence:.2f})"
                
                start_time = i * frame_duration if fps > 0 else i * 0.1 # FPSが不正な場合のフォールバック
                end_time = (i + 1) * frame_duration if fps > 0 else (i + 1) * 0.1

                # end_timeがstart_timeと同じかそれより前にならないように微調整
                if end_time <= start_time:
                    end_time = start_time + 0.001 # 非常に短い期間

                srt_entries.append(f"{i + 1}")
                srt_entries.append(f"{self._format_time_for_srt(start_time)} --> {self._format_time_for_srt(end_time)}")
                srt_entries.append(text)
                srt_entries.append("") # Blank line separator

        return "\n".join(srt_entries)

    def process_video(self, video_path: Path, predictions_df: pd.DataFrame, 
                      prediction_csv_filename: str) -> bool:
        print(f"\n--- 動画処理開始 (FFmpeg使用): {video_path.name} ---")
        
        fps = 30.0 # デフォルトFPS
        video_duration_seconds = 0.0
        try:
            ffprobe_cmd = [
                'ffprobe', '-v', 'error', '-select_streams', 'v:0',
                '-show_entries', 'stream=r_frame_rate,duration', '-of', 'default=noprint_wrappers=1:nokey=1',
                str(video_path)
            ]
            # ffprobeのタイムアウトを延長
            result = subprocess.run(ffprobe_cmd, capture_output=True, text=True, check=True, timeout=60) # 10秒から60秒へ
            output_lines = result.stdout.strip().split('\n')
            if len(output_lines) >= 1 and '/' in output_lines[0]: # r_frame_rate
                num_str, den_str = output_lines[0].split('/')
                num, den = int(num_str), int(den_str)
                if den != 0:
                    fps = num / den
                print(f"  FPS (ffprobe): {fps:.2f}")
            else:
                 print(f"⚠️  ffprobeでFPSの取得に失敗。デフォルトFPS {fps:.2f} を使用します。")


            if len(output_lines) >= 2 and output_lines[1] != "N/A": # duration
                video_duration_seconds = float(output_lines[1])
                print(f"  動画の長さ (ffprobe): {video_duration_seconds:.2f} 秒")
            else:
                print(f"⚠️  ffprobeで動画の長さを取得できませんでした。")

        except (subprocess.CalledProcessError, FileNotFoundError, ValueError, subprocess.TimeoutExpired) as e:
            print(f"⚠️  ffprobeでFPS/Durationを取得できませんでした: {e}.")
            print(f"ℹ️  FFmpeg (と ffprobe) がシステムパスにインストールされていることを確認してください。")
            # OpenCVフォールバックは維持
            try:
                cap_cv2 = cv2.VideoCapture(str(video_path))
                if cap_cv2.isOpened():
                    cv2_fps = cap_cv2.get(cv2.CAP_PROP_FPS)
                    cv2_total_frames = int(cap_cv2.get(cv2.CAP_PROP_FRAME_COUNT))
                    cap_cv2.release()
                    if cv2_fps > 0:
                        fps = cv2_fps
                        print(f"ℹ️  OpenCV経由でFPSを取得しました: {fps:.2f}")
                        if cv2_total_frames > 0 and video_duration_seconds == 0.0:
                             video_duration_seconds = cv2_total_frames / cv2_fps
                             print(f"ℹ️  OpenCV経由で動画の長さを計算しました: {video_duration_seconds:.2f} 秒")
                    else:
                        print(f"ℹ️  OpenCVでも有効なFPSを取得できず。デフォルトFPS {fps:.2f} を使用します。")
                else:
                    print(f"❌ OpenCVでも動画を開けず。デフォルトFPS {fps:.2f} を使用します。")
            except Exception as cv_e:
                print(f"❌ OpenCVでのFPS/Duration取得試行中にエラー: {cv_e}. デフォルトFPS {fps:.2f} を使用します。")
        
        total_frames_video = 0
        if video_duration_seconds > 0 and fps > 0:
            total_frames_video = int(video_duration_seconds * fps)
            print(f"  推定総フレーム数: {total_frames_video}")
        
        # ffprobeで直接nb_read_framesを取得する試み（タイムアウトしやすいので注意）
        try:
            ffprobe_frames_cmd = [
                'ffprobe', '-v', 'error', '-select_streams', 'v:0',
                '-count_frames', '-show_entries', 'stream=nb_read_frames',
                '-of', 'default=noprint_wrappers=1:nokey=1', str(video_path)
            ]
            result_frames = subprocess.run(ffprobe_frames_cmd, capture_output=True, text=True, check=True, timeout=60) 
            if result_frames.stdout.strip() and result_frames.stdout.strip() != "N/A":
                 nb_read_frames = int(result_frames.stdout.strip())
                 if nb_read_frames > 0 :
                    total_frames_video = nb_read_frames # より正確な値で上書き
                    print(f"  動画の総フレーム数 (ffprobe nb_read_frames): {total_frames_video}")
        except (subprocess.CalledProcessError, FileNotFoundError, ValueError, subprocess.TimeoutExpired) as e:
            print(f"⚠️  ffprobeでnb_read_framesの取得に失敗 (タイムアウトの可能性): {e}")


        if total_frames_video > 0 and len(predictions_df) != total_frames_video:
             print(f"⚠️  予測データフレームの行数 ({len(predictions_df)}) と動画の総フレーム数 ({total_frames_video}) が異なります。")


        srt_content = self._generate_srt_content(predictions_df, fps)
        if not srt_content.strip():
            print("ℹ️  表示する予測テキストがありません。オーバーレイ処理をスキップします。")
            # 必要であれば、ここで元動画をコピーするなどの処理を追加できる
            return True # スキップも成功とみなすか、Falseにするかは要件次第

        temp_srt_file_path = "" # finallyブロックで参照するため初期化
        try:
            # NamedTemporaryFileは自動削除されるため、FFmpegがアクセスできるようdelete=Falseにする
            with tempfile.NamedTemporaryFile(mode="w", suffix=".srt", delete=False, encoding='utf-8') as f:
                f.write(srt_content)
                temp_srt_file_path = f.name 
            
            print(f"  一時SRTファイル生成: {temp_srt_file_path}")

            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            base_video_name = video_path.stem
            csv_stem = Path(prediction_csv_filename).stem
            parts = csv_stem.split('_predictions_')
            original_input_base = parts[0].replace("tennis_inference_features_", "") if len(parts) > 0 else "unknown_input"
            
            output_filename = f"{base_video_name}_overlay_ffmpeg_{original_input_base}_{timestamp}.mp4"
            output_path = self.output_video_dir / output_filename

            b, g, r = self.font_color
            primary_colour = f"&H00{r:02X}{g:02X}{b:02X}" 
            box_colour = "&H99000000" # 40%不透明の黒 (ASS/SSAアルファ: 00=不透明, FF=透明。0.6*255=153 -> 99)

            font_size = max(10, int(self.font_scale * 20)) # スケール1で20px程度と仮定
            margin_l = self.text_position[0]
            margin_v = self.text_position[1]
            font_name = 'Arial' 

            style_options = [
                f"FontName={font_name}",
                f"FontSize={font_size}",
                f"PrimaryColour={primary_colour}",
                f"BoxColour={box_colour}",
                "BorderStyle=3", 
                "Alignment=7",   
                f"MarginL={margin_l}",
                f"MarginV={margin_v}",
                "Shadow=0",
                "Outline=0" # または縁取りの太さを指定 (例: Outline=1)
            ]
            force_style = ",".join(style_options)
            
            # FFmpegはパス内のバックスラッシュをエスケープする必要がある場合があるため、as_posix()でスラッシュ区切りにする
            # WindowsのパスをFFmpegフィルタ内で安全に使用するための処理
            # 1. pathlib.Pathで正規化し、as_posix()でスラッシュ区切りにする (例: C:/Users/...)
            # 2. ドライブレターの後のコロン ':' をエスケープする (例: C\:/Users/...)
            #    Pythonの文字列内でバックスラッシュをリテラルとして扱うには r'\' を使用
            srt_path_for_filter = Path(temp_srt_file_path).as_posix()
            # srt_path_for_filter = srt_path_for_filter.replace(':', r'\:') # FFmpeg 7以降では不要な場合が多い
            # ':'のエスケープは環境やFFmpegのバージョンによって挙動が異なるため、
            # 問題が発生する場合は有効化を検討してください。
            # 一般的には、as_posix()でスラッシュ区切りにすれば十分なことが多いです。
            # Windowsで `subtitles` フィルタに渡すパスは、`C\\:/path/to/file.srt` のように
            # コロンをエスケープし、さらにバックスラッシュもエスケープする必要がある場合があります。
            # もしくは、`filename=C\\:/path/to/file.srt` の代わりに `filename='C:/path/to/file.srt'` のように
            # シングルクォートで囲むことで、多くのケースで正しく解釈されます。
            # ここでは、シングルクォートで囲む戦略を採用します。
            # srt_path_for_filter = Path(temp_srt_file_path).as_posix().replace(":", "\\:") # より安全なエスケープ
            
            # FFmpegのsubtitlesフィルタは、パス内の特殊文字の扱いに注意が必要です。
            # Windowsでは、ドライブレター後のコロンをエスケープする必要がある場合があります。
            # 例: 'C\:/path/to/subtitle.srt'
            # pathlib.as_posix() はスラッシュ区切りを返しますが、コロンはそのままです。
            # フィルタの引数文字列内でこれを正しく扱うため、手動でエスケープするか、
            # FFmpegが解釈できる形式に整形します。
            # `filename`パラメータをシングルクォートで囲むことで、多くのケースでパスが正しく解釈されます。
            # それでも問題が起きる場合は、コロンのエスケープを試みます。
            escaped_srt_path = Path(temp_srt_file_path).as_posix()
            if os.name == 'nt': # Windowsの場合のみコロンのエスケープを検討
                 # C:/foo/bar.srt -> C\:/foo/bar.srt
                 escaped_srt_path = re.sub(r'^([a-zA-Z]):', r'\1\\:', escaped_srt_path)


            video_encoder = 'libx264'
            encoder_preset = 'medium'
            encoder_options = ['-crf', '23']

            if self.use_gpu_encoder:
                video_encoder = 'h264_nvenc'
                encoder_preset = 'p3' # NVIDIAのプリセット例 (p1-p7, slow-fast)
                encoder_options = ['-cq', '23'] # 固定品質モード
                print("  GPUエンコーダ (h264_nvenc) を使用します。")
            else:
                print("  CPUエンコーダ (libx264) を使用します。")

            ffmpeg_cmd = [
                'ffmpeg', '-y', '-i', str(video_path),
                '-vf', f"subtitles='{escaped_srt_path}':force_style='{force_style}'",
                '-c:v', video_encoder,
                '-preset', encoder_preset,
            ] + encoder_options + [
                '-c:a', 'aac', # 音声コーデック (必要に応じて変更)
                '-strict', 'experimental', # aac を使用する場合に必要になることがある
                str(output_path)
            ]
            
            print(f"  実行コマンド: {' '.join(ffmpeg_cmd)}")

            # FFmpegの実行と進捗表示
            process = subprocess.Popen(ffmpeg_cmd, stderr=subprocess.PIPE, universal_newlines=True, encoding='utf-8')
            
            # tqdmで進捗バーを表示するための準備
            # total_frames_video が0の場合は、進捗バーのtotalが不明になるため、表示を工夫する
            progress_bar_total = total_frames_video if total_frames_video > 0 else None
            pbar_desc = "動画処理中 (FFmpeg)"
            if progress_bar_total is None:
                pbar_desc += " (総フレーム数不明)"

            with tqdm(total=progress_bar_total, unit="frame", desc=pbar_desc, dynamic_ncols=True) as pbar:
                current_frame = 0
                for line in process.stderr:
                    # print(line, end='') # FFmpegの全出力を表示したい場合
                    match = re.search(r"frame=\s*(\d+)", line)
                    if match:
                        new_frame = int(match.group(1))
                        if progress_bar_total: # 総フレーム数が分かっている場合のみ更新
                            pbar.update(new_frame - current_frame)
                            current_frame = new_frame
                        else: # 総フレーム数が不明な場合は、処理されたフレーム数を表示
                            pbar.n = new_frame
                            pbar.refresh()
                    if "error" in line.lower() and "output" not in line.lower(): # "Output"行のエラーは無視
                        print(f"⚠️ FFmpeg Error: {line.strip()}")


            process.wait() # FFmpegプロセスが終了するのを待つ

            if process.returncode == 0:
                print(f"✅ 動画処理完了: {output_path}")
                return True
            else:
                print(f"❌ FFmpeg処理エラー (リターンコード: {process.returncode})")
                # エラー出力はすでに出力されているはず
                return False

        except Exception as e:
            print(f"❌ 動画処理中に予期せぬエラー: {e}")
            import traceback
            traceback.print_exc()
            return False
        finally:
            if temp_srt_file_path and Path(temp_srt_file_path).exists():
                try:
                    os.remove(temp_srt_file_path)
                    print(f"  一時SRTファイル削除: {temp_srt_file_path}")
                except OSError as e:
                    print(f"⚠️ 一時SRTファイルの削除に失敗: {e}")

    def display_video_with_predictions_realtime(self, video_path: Path, predictions_df: pd.DataFrame):
        print(f"\n--- リアルタイム表示開始: {video_path.name} ---")
        
        cap = cv2.VideoCapture(str(video_path))
        if not cap.isOpened():
            print(f"❌ 動画ファイルを開けませんでした: {video_path}")
            return

        fps = cap.get(cv2.CAP_PROP_FPS)
        if fps <= 0:
            print("⚠️  動画のFPSが取得できませんでした。再生速度が正しくない可能性があります。デフォルトで30ms待機します。")
            wait_time_ms = 30
        else:
            wait_time_ms = int(1000 / fps)

        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        print(f"  動画FPS: {fps:.2f}, 総フレーム数: {total_frames}")
        print("  再生ウィンドウで 'q' を押すと終了します。")

        frame_idx = 0
        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                print("ℹ️  動画の終端に達したか、読み込みエラーです。")
                break

            current_frame_num = int(cap.get(cv2.CAP_PROP_POS_FRAMES)) -1 # 0-indexed

            if current_frame_num < len(predictions_df):
                prediction_row = predictions_df.iloc[current_frame_num]
                phase = prediction_row['predicted_phase']
                confidence = prediction_row['prediction_confidence']

                if pd.notna(phase) and phase != "" and confidence >= self.confidence_threshold:
                    text = f"{phase} ({confidence:.2f})"
                    
                    # テキストの背景を描画 (オプション)
                    (text_width, text_height), baseline = cv2.getTextSize(text, self.font, self.font_scale, self.line_type)
                    
                    # 背景の矩形座標を計算 (text_positionを左上とする)
                    rect_start_x = self.text_position[0]
                    rect_start_y = self.text_position[1] - text_height - baseline // 2 
                    rect_end_x = self.text_position[0] + text_width
                    rect_end_y = self.text_position[1] + baseline // 2

                    # 画面外に出ないように調整
                    frame_h, frame_w = frame.shape[:2]
                    rect_start_x = max(0, rect_start_x)
                    rect_start_y = max(0, rect_start_y)
                    rect_end_x = min(frame_w, rect_end_x)
                    rect_end_y = min(frame_h, rect_end_y)
                    
                    # テキスト描画位置も調整 (背景矩形の左下基準でテキストを描画するため)
                    text_draw_pos_x = rect_start_x
                    text_draw_pos_y = rect_start_y + text_height + baseline // 4 # 微調整

                    if rect_end_x > rect_start_x and rect_end_y > rect_start_y : # 有効な矩形の場合のみ描画
                        # 半透明の背景 (黒)
                        sub_img = frame[rect_start_y:rect_end_y, rect_start_x:rect_end_x]
                        black_rect =_img = cv2.rectangle(frame.copy(), (rect_start_x, rect_start_y), (rect_end_x, rect_end_y), (0,0,0), -1)
                        res = cv2.addWeighted(sub_img, 0.6, black_rect[rect_start_y:rect_end_y, rect_start_x:rect_end_x], 0.4, 1.0)
                        frame[rect_start_y:rect_end_y, rect_start_x:rect_end_x] = res
                    
                    cv2.putText(frame, text, (text_draw_pos_x, text_draw_pos_y), self.font, self.font_scale, self.font_color, self.line_type, cv2.LINE_AA)


            cv2.imshow(f"Video with Predictions - {video_path.name}", frame)
            
            key = cv2.waitKey(wait_time_ms) & 0xFF
            if key == ord('q'):
                print("ℹ️  'q' キーが押されたため、再生を終了します。")
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
