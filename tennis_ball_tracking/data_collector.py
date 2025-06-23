import cv2
import json
import os
import numpy as np
from datetime import datetime
from court_calibrator import CourtCalibrator
import tkinter as tk

class PhaseAnnotator:
    """
    テニス局面ラベリング専用クラス
    
    用途:
    - 動画の局面分析と高速なラベリング作業
    - train_phase_model.py用のデータ作成
    - コート座標設定（オプション）
    """
    def __init__(self):
        # --- 状態管理 ---
        self.video_cap = None
        self.current_frame_number = 0
        self.total_frames = 0
        self.fps = 30.0
        self.is_playing = False
        self.playback_speed = 1.0
        
        # --- 局面データ ---
        self.phases = [
            "point_interval", "rally", "serve_front_deuce", "serve_front_ad",
            "serve_back_deuce", "serve_back_ad", "changeover"
        ]
        self.current_phase = None
        self.phase_changes = []
        
        # --- コート座標 ---
        self.court_coordinates = {}
        self.show_court_overlay = False
        
        # --- UI設定 ---
        self.display_scale = 1.0
        self.window_width = 1280
        self.window_height = 720
        self.ui_font_scale = 0.6

    def annotate_video(self, video_path: str):
        """
        動画のアノテーション処理を開始します。
        セットアップ、メインループ、クリーンアップの順に処理を実行します。
        """
        if not self._setup(video_path):
            return False

        self._annotation_loop()
        
        return self._cleanup_and_save(video_path)

    # --- 初期化・セットアップ ---
    
    def _setup(self, video_path: str):
        """アノテーション開始前の準備を行います。"""
        cap = cv2.VideoCapture(video_path)
        if not cap.isOpened():
            print(f"Error: 動画ファイルを開けません: {video_path}")
            return False
        
        self.set_video_source(cap)
        self._detect_display_size()
        
        print(f"動画をロードしました: {os.path.basename(video_path)}")
        print(f"総フレーム数: {self.total_frames}, FPS: {self.fps:.2f}")
        print(f"表示設定: {self.window_width}x{self.window_height} (スケール: {self.display_scale})")

        if self.setup_court_coordinates(video_path):
            self.show_court_overlay = True

        cv2.namedWindow('Phase Annotation', cv2.WINDOW_NORMAL)
        cv2.resizeWindow('Phase Annotation', self.window_width, self.window_height)
        
        self._print_usage_instructions()
        return True
    
    def set_video_source(self, video_cap):
        """動画ソースを設定します。"""
        self.video_cap = video_cap
        if video_cap:
            self.total_frames = int(video_cap.get(cv2.CAP_PROP_FRAME_COUNT))
            self.fps = video_cap.get(cv2.CAP_PROP_FPS)

    def _detect_display_size(self):
        """ディスプレイサイズを検出し、適切な表示設定を決定します。"""
        try:
            root = tk.Tk()
            screen_width = root.winfo_screenwidth()
            screen_height = root.winfo_screenheight()
            root.destroy()
            
            # ディスプレイサイズに基づいた設定調整
            if screen_width >= 2560:
                self.display_scale, self.window_width, self.window_height, self.ui_font_scale = 1.5, 1920, 1080, 0.8
            elif screen_width >= 1920:
                self.display_scale, self.window_width, self.window_height, self.ui_font_scale = 1.2, 1600, 900, 0.7
            elif screen_width >= 1366:
                self.display_scale, self.window_width, self.window_height, self.ui_font_scale = 1.0, 1280, 720, 0.6
            elif screen_width >= 1024:
                self.display_scale, self.window_width, self.window_height, self.ui_font_scale = 0.8, 1000, 600, 0.45
            else:
                self.display_scale, self.window_width, self.window_height, self.ui_font_scale = 0.6, min(800, screen_width - 50), min(500, screen_height - 100), 0.35

        except Exception as e:
            print(f"ディスプレイサイズ検出エラー: {e}。デフォルト設定を使用します。")
            self.display_scale, self.window_width, self.window_height, self.ui_font_scale = 0.7, 900, 600, 0.4

    # --- メインループとイベント処理 ---

    def _annotation_loop(self):
        """アノテーションのメインループを実行します。"""
        while True:
            frame = self._get_current_frame()
            if frame is None:
                break
            
            self._draw_ui(frame)
            
            wait_time = max(1, int(1000 / (self.fps * self.playback_speed))) if self.is_playing else 0
            key = cv2.waitKey(wait_time) & 0xFF

            if key != 255: # キー入力があった場合
                if self._handle_key_input(key):
                    break

    def _get_current_frame(self):
        """現在のフレームを取得または読み込みます。"""
        if self.is_playing:
            ret, frame = self.video_cap.read()
            if not ret:
                print("動画の終端に達しました")
                self.is_playing = False
                self._seek_frame(self.total_frames - 1)
                ret, frame = self.video_cap.read()
            else:
                 self.current_frame_number = int(self.video_cap.get(cv2.CAP_PROP_POS_FRAMES)) - 1
        else:
            self.video_cap.set(cv2.CAP_PROP_POS_FRAMES, self.current_frame_number)
            ret, frame = self.video_cap.read()

        return frame if ret else None

    def _handle_key_input(self, key):
        """キー入力を処理し、ループを終了するかどうかを返します。"""
        # 局面選択 (1-7)
        if ord('1') <= key <= ord('7'):
            self._record_phase_change(self.phases[key - ord('1')])
        # 再生/停止
        elif key == ord(' '):
            self.is_playing = not self.is_playing
        # フレーム移動
        elif key in [ord('d'), 83]: self._seek_frame(self.current_frame_number + 1, stop_playback=True)
        elif key in [ord('a'), 81]: self._seek_frame(self.current_frame_number - 1, stop_playback=True)
        elif key in [ord('w'), 82]: self._seek_frame(self.current_frame_number + 10, stop_playback=True)
        elif key in [ord('s'), 84]: self._seek_frame(self.current_frame_number - 10, stop_playback=True)
        elif key == ord('z'): self._seek_frame(self.current_frame_number - 100, stop_playback=True)
        elif key == ord('x'): self._seek_frame(self.current_frame_number + 100, stop_playback=True)
        elif key == 2: self._seek_frame(0, stop_playback=True) # HOME
        elif key == 3: self._seek_frame(self.total_frames - 1, stop_playback=True) # END
        # 再生速度
        elif key in [ord('-'), ord('_')]: self._change_playback_speed(direction='down')
        elif key in [ord('+'), ord('=')]: self._change_playback_speed(direction='up')
        elif key == ord('0'): self.playback_speed = 1.0
        # コート座標
        elif key == ord('c'): self._run_court_calibration()
        elif key == ord('o'): self.show_court_overlay = not self.show_court_overlay
        # その他
        elif key == ord('r'): self._reset_annotations()
        elif key == ord('u'): self._undo_last_phase_change()
        # 終了
        elif key == ord('q'):
            return self._confirm_exit()

        return False

    def _seek_frame(self, target_frame, stop_playback=False):
        """指定フレームに移動し、現在の局面を更新します。"""
        self.current_frame_number = max(0, min(target_frame, self.total_frames - 1))
        if stop_playback:
            self.is_playing = False
        self._update_current_phase()

    def _change_playback_speed(self, direction='up'):
        """再生速度を変更します。"""
        speed_levels = [0.25, 0.5, 1.0, 2.0, 4.0]
        try:
            current_index = speed_levels.index(self.playback_speed)
            if direction == 'up' and current_index < len(speed_levels) - 1:
                self.playback_speed = speed_levels[current_index + 1]
            elif direction == 'down' and current_index > 0:
                self.playback_speed = speed_levels[current_index - 1]
            print(f"再生速度: {self.playback_speed}x")
        except ValueError:
            self.playback_speed = 1.0 # リストにない場合はリセット

    def _run_court_calibration(self):
        """対話的なコート座標設定を起動します。"""
        print("\nコート座標設定を開始します...")
        self.is_playing = False
        
        ret, frame = self.video_cap.read()
        if not ret:
            print("❌ フレーム読み込みエラー")
            return

        calibrator = CourtCalibrator()
        if calibrator.calibrate(frame, self.video_cap):
            self.court_coordinates = calibrator.get_coordinates()
            self.show_court_overlay = True
            print("✅ コート座標設定完了")
        else:
            print("❌ コート座標設定がキャンセルされました")
        
        # ウィンドウを再生成してフォーカスを戻す
        cv2.destroyAllWindows()
        cv2.namedWindow('Phase Annotation', cv2.WINDOW_NORMAL)
        cv2.resizeWindow('Phase Annotation', self.window_width, self.window_height)

    # --- UI描画 ---

    def _draw_ui(self, frame):
        """フレームにUI要素を描画します。"""
        display_frame = self._resize_frame_for_display(frame.copy())
        if self.show_court_overlay and self.court_coordinates:
            display_frame = self.draw_court_overlay(display_frame)
        self._draw_annotation_ui(display_frame)
        cv2.imshow('Phase Annotation', display_frame)

    def _resize_frame_for_display(self, frame):
        """表示用にフレームをリサイズします。"""
        h, w = frame.shape[:2]
        target_w = int(self.window_width * 0.6)
        target_h = int(self.window_height * 0.8)
        
        aspect_ratio = w / h
        if target_w / aspect_ratio <= target_h:
            new_w, new_h = target_w, int(target_w / aspect_ratio)
        else:
            new_h, new_w = target_h, int(target_h * aspect_ratio)
        
        return cv2.resize(frame, (new_w, new_h), interpolation=cv2.INTER_AREA)

    def _draw_annotation_ui(self, frame):
        """アノテーション用のUIを描画します。"""
        h, w = frame.shape[:2]
        font_scale = max(0.3, self.ui_font_scale * self.display_scale)
        thickness = max(1, int(1.5 * self.display_scale))
        
        # --- UI要素のY座標と高さを定義 ---
        y = int(20 * self.display_scale)
        dy = int(18 * self.display_scale)
        left_x, right_x = 10, w // 2 + 10
        
        # 各セクションのY座標を計算
        info_y = y
        status_y = info_y + dy
        phase_y = status_y + dy
        
        ph_dy = int(dy * 0.7)
        phases_list_y_start = phase_y + int(dy * 1.3)
        phases_list_y_end = phases_list_y_start + (len(self.phases) // 2) * ph_dy

        help_y_start = phases_list_y_end + int(dy * 0.5)
        help_texts = [
            "1-7:Phase SPACE:Play/Pause A/D:Frame W/S:10f Z/X:100f",
            "+/-:Speed 0:1x C:Court O:Overlay U:Undo Q:Save&Quit"
        ]
        help_y_end = help_y_start + (len(help_texts) - 1) * int(dy * 0.6)

        # コンテンツの高さに基づいてUI背景の高さを決定
        ui_background_height = help_y_end + int(20 * self.display_scale)

        # --- UI背景を描画 ---
        overlay = frame.copy()
        cv2.rectangle(overlay, (5, 5), (w - 5, min(h - 5, ui_background_height)), (0, 0, 0), -1)
        cv2.addWeighted(overlay, 0.75, frame, 0.25, 0, frame)

        # --- UIテキスト要素を描画 ---
        # フレーム情報
        progress = self.current_frame_number / self.total_frames if self.total_frames > 0 else 0
        cv2.putText(frame, f"Frame: {self.current_frame_number}/{self.total_frames}", (left_x, info_y), cv2.FONT_HERSHEY_SIMPLEX, font_scale, (255,255,255), thickness)
        cv2.putText(frame, f"Progress: {progress:.1%}", (left_x, status_y), cv2.FONT_HERSHEY_SIMPLEX, font_scale, (255,255,255), thickness)

        # 再生状態
        status, s_color = ("PLAY", (0,255,0)) if self.is_playing else ("PAUSE", (0,0,255))
        cv2.putText(frame, f"Status: {status}", (right_x, info_y), cv2.FONT_HERSHEY_SIMPLEX, font_scale, s_color, thickness)
        cv2.putText(frame, f"Speed: {self.playback_speed}x", (right_x, status_y), cv2.FONT_HERSHEY_SIMPLEX, font_scale, (0,255,255), thickness)

        # 現在の局面とコート状態
        cv2.putText(frame, f"Phase: {self.current_phase or 'None'}", (left_x, phase_y), cv2.FONT_HERSHEY_SIMPLEX, font_scale, (0,255,0), thickness)
        court_status = f"Court: {'ON' if self.court_coordinates else 'OFF'} Ovl: {'ON' if self.show_court_overlay else 'OFF'}"
        cv2.putText(frame, court_status, (right_x, phase_y), cv2.FONT_HERSHEY_SIMPLEX, font_scale, (255,255,0), thickness)

        # 局面リスト
        ph_font_scale = font_scale * 0.85
        phases_per_column = (len(self.phases) + 1) // 2
        for i, p in enumerate(self.phases):
            col, row = i // phases_per_column, i % phases_per_column
            x_pos = left_x if col == 0 else right_x
            y_pos = phases_list_y_start + row * ph_dy
            color = (0, 255, 0) if p == self.current_phase else (255, 255, 255)
            short_phase = p.replace("serve_", "").replace("_", " ")[:12]
            cv2.putText(frame, f"{i+1}: {short_phase}", (x_pos, y_pos), cv2.FONT_HERSHEY_SIMPLEX, ph_font_scale, color, max(1, thickness-1))

        # 操作ヘルプ
        for i, text in enumerate(help_texts):
            y_pos = help_y_start + i * int(dy * 0.6)
            if y_pos < h - 10:
                cv2.putText(frame, text, (left_x, y_pos), cv2.FONT_HERSHEY_SIMPLEX, font_scale * 0.6, (255,255,0), max(1, thickness-1))

    def draw_court_overlay(self, frame):
        """コートのオーバーレイを描画します。"""
        overlay_frame = frame.copy()
        thickness = max(1, int(1.5 * self.display_scale))
        
        points = [self.court_coordinates.get(name) for name in ["top_left_corner", "top_right_corner", "bottom_right_corner", "bottom_left_corner"]]
        if all(p is not None for p in points):
            cv2.polylines(overlay_frame, [np.array(points, np.int32)], True, (255, 255, 0), thickness)
        
        # ネット
        net_l, net_r = self.court_coordinates.get("net_left_ground"), self.court_coordinates.get("net_right_ground")
        if net_l and net_r:
            cv2.line(overlay_frame, tuple(net_l), tuple(net_r), (0, 255, 255), thickness + 1)
            
        for name, point in self.court_coordinates.items():
            cv2.circle(overlay_frame, tuple(point), max(2, int(4 * self.display_scale)), (0, 255, 0), -1)

        return overlay_frame

    # --- データ管理と状態更新 ---

    def _record_phase_change(self, new_phase):
        """局面の変更を記録します。"""
        frame_num = self.current_frame_number
        
        # 現在フレーム以降の変更を削除
        self.phase_changes = [c for c in self.phase_changes if c['frame_number'] <= frame_num]
        
        # 直前の変更と同じでなければ追加
        if not self.phase_changes or self.phase_changes[-1]['phase'] != new_phase:
            self.phase_changes.append({
                'frame_number': frame_num,
                'phase': new_phase,
                'timestamp': frame_num / self.fps
            })
            print(f"Frame {frame_num}: 新しい局面 -> {new_phase}")
        
        self.phase_changes.sort(key=lambda x: x['frame_number'])
        self.current_phase = new_phase

    def _update_current_phase(self):
        """現在のフレーム位置に基づいて局面を更新します。"""
        updated_phase = None
        for change in self.phase_changes:
            if change['frame_number'] <= self.current_frame_number:
                updated_phase = change['phase']
        self.current_phase = updated_phase

    def _undo_last_phase_change(self):
        """最後のアノテーションを取り消します。"""
        if self.phase_changes:
            removed = self.phase_changes.pop()
            print(f"取り消し: Frame {removed['frame_number']} - {removed['phase']}")
            self._update_current_phase()
        else:
            print("取り消す変更がありません")

    def _reset_annotations(self):
        """すべてのアノテーションをリセットします。"""
        if self.phase_changes and input("全ての局面変更を削除しますか？ (y/n): ").lower() == 'y':
            self.phase_changes = []
            self.current_phase = None
            print("全てのアノテーションを削除しました")

    # --- 終了と保存 ---

    def _cleanup_and_save(self, video_path):
        """リソースを解放し、データを保存します。"""
        self.video_cap.release()
        cv2.destroyAllWindows()
        
        if self.phase_changes:
            result = self._save_phase_data(video_path)
            if self.court_coordinates:
                self.save_court_coordinates(video_path)
            return result
        return False
    
    def _confirm_exit(self):
        """終了時の確認ダイアログを表示します。"""
        if not self.phase_changes:
            return True # 保存するものがなければそのまま終了
        
        save = input("変更を保存して終了しますか？ (y/n): ").lower()
        if save == 'y':
            return True
        else:
            return input("保存せずに終了しますか？ (y/n): ").lower() == 'y'

    def _save_phase_data(self, video_path):
        """局面データをJSONファイルに保存します。"""
        video_name = os.path.splitext(os.path.basename(video_path))[0]
        output_dir = "training_data"
        os.makedirs(output_dir, exist_ok=True)
        
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        phase_file = os.path.join(output_dir, f"phase_annotations_{video_name}_{timestamp}.json")
        
        annotation_data = {
            'video_path': video_path,
            'video_name': video_name,
            'total_frames': self.total_frames,
            'fps': self.fps,
            'duration_seconds': self.total_frames / self.fps,
            'phase_changes': self.phase_changes,
            'annotation_timestamp': timestamp,
            'phase_statistics': self._calculate_phase_statistics(),
            'court_coordinates': self.court_coordinates or None,
            'court_coordinates_available': bool(self.court_coordinates)
        }
        
        try:
            with open(phase_file, 'w', encoding='utf-8') as f:
                json.dump(annotation_data, f, indent=2, ensure_ascii=False)
            print(f"\n✅ 局面アノテーションを保存しました: {phase_file}")
            self._print_statistics(annotation_data)
            return True
        except Exception as e:
            print(f"❌ 局面データ保存エラー: {e}")
            return False

    def setup_court_coordinates(self, video_path):
        """既存のコート座標ファイルを読み込みます。"""
        video_name = os.path.splitext(os.path.basename(video_path))[0]
        coord_file = os.path.join("training_data", f"court_coords_{video_name}.json")
        if os.path.exists(coord_file):
            try:
                with open(coord_file, 'r') as f:
                    self.court_coordinates = json.load(f)
                print(f"✅ 既存のコート座標を読み込みました: {coord_file}")
                return True
            except Exception as e:
                print(f"❌ コート座標ファイル読み込みエラー: {e}")
        return False

    def save_court_coordinates(self, video_path):
        """コート座標をJSONファイルに保存します。"""
        video_name = os.path.splitext(os.path.basename(video_path))[0]
        output_dir = "training_data"
        os.makedirs(output_dir, exist_ok=True)
        coord_file = os.path.join(output_dir, f"court_coords_{video_name}.json")
        try:
            with open(coord_file, 'w') as f:
                json.dump(self.court_coordinates, f, indent=2)
            print(f"✅ コート座標を保存しました: {coord_file}")
            return True
        except Exception as e:
            print(f"❌ コート座標保存エラー: {e}")
            return False

    # --- 統計とヘルパー ---

    def _calculate_phase_statistics(self):
        """各局面の時間と割合を計算します。"""
        if not self.phase_changes: return {}
        
        durations = {phase: 0 for phase in self.phases}
        for i, change in enumerate(self.phase_changes):
            start_frame = change['frame_number']
            end_frame = self.phase_changes[i+1]['frame_number'] if i + 1 < len(self.phase_changes) else self.total_frames
            durations[change['phase']] += (end_frame - start_frame)
        
        total_duration_sec = self.total_frames / self.fps
        stats = {}
        for phase, frame_count in durations.items():
            if frame_count > 0:
                duration_sec = frame_count / self.fps
                stats[phase] = {
                    'duration': duration_sec,
                    'percentage': (duration_sec / total_duration_sec) * 100 if total_duration_sec > 0 else 0
                }
        return stats
    
    def _print_statistics(self, data):
        """統計情報をコンソールに出力します。"""
        print(f"記録された局面変更数: {len(data['phase_changes'])}")
        if data['court_coordinates_available']:
            print("🏟️  コート座標も含まれています")
        
        print("\n=== 局面統計 ===")
        stats = data.get('phase_statistics', {})
        for phase, info in stats.items():
            print(f"{phase:<20}: {info['duration']:.1f}秒 ({info['percentage']:.1f}%)")
    
    def _print_usage_instructions(self):
        """操作方法をコンソールに出力します。"""
        print("\n=== 局面アノテーション開始 ===")
        print("局面選択: 1-7キー")
        print("再生制御: SPACE (再生/停止), a/d (コマ送り), w/s (10コマ), z/x (100コマ)")
        print("再生速度: +/- (変更), 0 (標準速度)")
        print("コート座標: c (設定), o (オーバーレイON/OFF)")
        print("その他: u (元に戻す), r (リセット), q (保存して終了)")

# --- CLIメニューとファイル管理関数 ---

def get_video_files(data_dir="../data/raw"):
    """指定ディレクトリから動画ファイル一覧を取得します。"""
    video_extensions = ['.mp4', '.avi', '.mov', '.mkv']
    search_paths = [data_dir, "data/raw", "./data/raw"]
    
    for path in search_paths:
        abs_path = os.path.abspath(path)
        if os.path.exists(abs_path):
            try:
                files = [os.path.join(abs_path, f) for f in os.listdir(abs_path) if any(f.lower().endswith(ext) for ext in video_extensions)]
                if files:
                    print(f"動画ファイルを {abs_path} で発見しました。")
                    return files
            except Exception as e:
                print(f"ディレクトリ読み取りエラー: {e}")
    return []

def select_video_file():
    """ユーザーに動画ファイルを選択させます。"""
    video_files = get_video_files()
    if not video_files:
        print("\n動画ファイルが見つかりませんでした。")
        if input("手動でファイルパスを入力しますか？ (y/n): ").lower() == 'y':
            path = input("動画ファイルのフルパスを入力してください: ").strip()
            return path if os.path.exists(path) else None
        return None

    print("\n=== 動画ファイル一覧 ===")
    for i, video_path in enumerate(video_files, 1):
        print(f"{i}: {os.path.basename(video_path)}")
    
    try:
        choice = int(input(f"\n動画を選択してください (1-{len(video_files)}): "))
        return video_files[choice - 1] if 1 <= choice <= len(video_files) else None
    except ValueError:
        return None

def main():
    """メイン関数 - アノテーションツールを起動します。"""
    print("=== テニス局面アノテーションツール ===")
    print("1: 局面アノテーションを開始")
    print("2: 終了")
    
    choice = input("\n選択してください (1-2): ").strip()
    
    if choice == '1':
        video_path = select_video_file()
        if video_path:
            annotator = PhaseAnnotator()
            if annotator.annotate_video(video_path):
                print("\n✅ アノテーションが正常に完了し、データが保存されました。")
            else:
                print("\nアノテーションがキャンセルまたは中断されました。")
    
    print("プログラムを終了します。")

if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("\n\n操作が中断されました。プログラムを終了します。")
    except Exception as e:
        print(f"\n予期せぬエラーが発生しました: {e}")