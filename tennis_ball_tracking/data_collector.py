import cv2
import json
import os
import numpy as np
from datetime import datetime
from court_calibrator import CourtCalibrator
import tkinter as tk
import time

class PhaseAnnotator:
    """
    テニス局面ラベリング専用クラス
    
    用途:
    - 動画の局面分析と高速なラベリング作業
    - train_phase_model.py用のデータ作成
    - コート座標設定（オプション）
    - 既存アノテーションの編集
    """
    def __init__(self):
        # --- 状態管理 ---
        self.video_cap = None
        self.video_path = None
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

        # --- 編集モード用 ---
        self.editing_file_path = None

    def annotate_video(self, video_path: str, existing_annotation_path: str = None):
        """
        動画のアノテーション処理を開始します。
        既存のアノテーションパスが指定された場合は編集モードで開始します。
        """
        self.video_path = video_path
        self.editing_file_path = existing_annotation_path
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
        
        # 既存アノテーションの読み込み、または新規作成の準備
        if self.editing_file_path and os.path.exists(self.editing_file_path):
            print(f"\n📝 既存のアノテーションを編集中: {os.path.basename(self.editing_file_path)}")
            self._load_phase_data(self.editing_file_path)
        else:
            print("\n📝 新規アノテーションを作成します。")
            # 新規作成の場合のみ、独立したコート座標ファイルの読み込みを試みる
            if self.setup_court_coordinates(video_path):
                self.show_court_overlay = True

        self._update_current_phase()
        print(f"総フレーム数: {self.total_frames}, FPS: {self.fps:.2f}")
        print(f"表示設定: {self.window_width}x{self.window_height} (スケール: {self.display_scale})")

        cv2.namedWindow('Phase Annotation', cv2.WINDOW_NORMAL)
        cv2.resizeWindow('Phase Annotation', self.window_width, self.window_height)
        
        self._print_usage_instructions()
        return True

    def _load_phase_data(self, file_path: str):
        """既存のアノテーションファイルを読み込み、クラスの状態を復元します。"""
        try:
            with open(file_path, 'r', encoding='utf-8') as f:
                data = json.load(f)
            
            self.phase_changes = data.get('phase_changes', [])
            
            if 'court_coordinates' in data and data['court_coordinates']:
                self.court_coordinates = data['court_coordinates']
                self.show_court_overlay = True
            
            if 'fps' in data: # アノテーション時のFPSを優先
                self.fps = data['fps']
            
            print(f"✅ {len(self.phase_changes)}件の局面変更とコート座標を読み込みました。")
            return True
        except Exception as e:
            print(f"❌ アノテーションファイルの読み込みに失敗しました: {e}")
            self.editing_file_path = None # 失敗した場合は新規モードにフォールバック
            return False
    
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

    # --- (メインループとイベント処理は変更なし) ---
    def _annotation_loop(self):
            # --- ▼▼▼ 修正箇所（全体を置き換え）▼▼▼ ---
            
            # 高精度タイマーで、ループ開始前の時刻を記録
            loop_start_time = time.perf_counter()

            while True:
                # --- フレーム取得とUI描画 ---
                frame = self._get_current_frame()
                if frame is None:
                    break
                self._draw_ui(frame)

                # --- 待機時間の動的計算 ---
                wait_time_ms = 0  # 停止中 (is_playing=False) のデフォルト待機時間 (キー入力まで無制限に待つ)
                
                if self.is_playing:
                    # FPSが正常に取得できているか確認
                    effective_fps = self.fps if self.fps and self.fps > 1 else 30.0
                    # 1フレームあたりにかけたい目標時間（秒）
                    target_duration_sec = 1.0 / (effective_fps * self.playback_speed)

                    # 前回のループ開始から現在までの経過時間（実際の処理時間）を計算
                    processing_time_sec = time.perf_counter() - loop_start_time

                    # 待機すべき時間 = 目標時間 - 実際の処理時間
                    wait_duration_sec = target_duration_sec - processing_time_sec
                    
                    # 計算結果をミリ秒に変換。負の値（処理遅延）の場合は最低でも1msとする
                    wait_time_ms = max(1, int(wait_duration_sec * 1000))

                # --- キー入力の受付 ---
                key = cv2.waitKey(wait_time_ms) & 0xFF

                # 次のループのために、現在の時刻を記録する
                loop_start_time = time.perf_counter()

                # --- キー入力の処理 ---
                if key != 255 and self._handle_key_input(key):
                    break
            # --- ▲▲▲ 修正完了 ▲▲▲ ---

    def _get_current_frame(self):
            if self.is_playing:
                # --- 60fps対策: 1フレーム読み飛ばして処理負荷を半分にする ---
                self.video_cap.grab()  # 1フレーム目をデコードだけして読み飛ばす（描画しないので高速）
                ret, frame = self.video_cap.read() # 2フレーム目を読み込んで、これを表示対象とする
                
                if not ret:
                    print("動画の終端に達しました")
                    self.is_playing = False
                    self._seek_frame(self.total_frames - 1)
                    # 終端でもう一度読み込みを試みる
                    self.video_cap.set(cv2.CAP_PROP_POS_FRAMES, self.current_frame_number)
                    ret, frame = self.video_cap.read()
                else:
                    self.current_frame_number = int(self.video_cap.get(cv2.CAP_PROP_POS_FRAMES)) - 1
            else:
                # 停止中はフレームスキップしない
                self.video_cap.set(cv2.CAP_PROP_POS_FRAMES, self.current_frame_number)
                ret, frame = self.video_cap.read()
                
            return frame if ret else None
    def _handle_key_input(self, key):
        if ord('1') <= key <= ord('7'): self._record_phase_change(self.phases[key - ord('1')])
        elif key == ord(' '): self.is_playing = not self.is_playing
        elif key in [ord('d'), 83]: self._seek_frame(self.current_frame_number + 1, stop_playback=True)
        elif key in [ord('a'), 81]: self._seek_frame(self.current_frame_number - 1, stop_playback=True)
        elif key in [ord('w'), 82]: self._seek_frame(self.current_frame_number + 10, stop_playback=True)
        elif key in [ord('s'), 84]: self._seek_frame(self.current_frame_number - 10, stop_playback=True)
        elif key == ord('z'): self._seek_frame(self.current_frame_number - 100, stop_playback=True)
        elif key == ord('x'): self._seek_frame(self.current_frame_number + 100, stop_playback=True)
        elif key == 2: self._seek_frame(0, stop_playback=True)
        elif key == 3: self._seek_frame(self.total_frames - 1, stop_playback=True)
        elif key in [ord('-'), ord('_')]: self._change_playback_speed(direction='down')
        elif key in [ord('+'), ord('=')]: self._change_playback_speed(direction='up')
        elif key == ord('0'): self.playback_speed = 1.0
        elif key == ord('c'): self._run_court_calibration()
        elif key == ord('o'): self.show_court_overlay = not self.show_court_overlay
        elif key == ord('r'): self._reset_annotations()
        elif key == ord('u'): self._undo_last_phase_change()
        elif key == ord('q'): return self._confirm_exit()
        return False
    def _seek_frame(self, target_frame, stop_playback=False):
        self.current_frame_number = max(0, min(target_frame, self.total_frames - 1))
        if stop_playback: self.is_playing = False
        self._update_current_phase()
    def _change_playback_speed(self, direction='up'):
        speed_levels = [0.25, 0.5, 1.0, 2.0, 4.0]
        try:
            current_index = speed_levels.index(self.playback_speed)
            if direction == 'up' and current_index < len(speed_levels) - 1: self.playback_speed = speed_levels[current_index + 1]
            elif direction == 'down' and current_index > 0: self.playback_speed = speed_levels[current_index - 1]
            print(f"再生速度: {self.playback_speed}x")
        except ValueError: self.playback_speed = 1.0
    def _run_court_calibration(self):
            print("\nコート座標設定を開始します...")
            
            # 1. 現在の再生状態とフレーム番号を保存
            original_is_playing = self.is_playing
            original_frame_number = self.current_frame_number
            self.is_playing = False  # 安全のため再生を停止
            
            # 2. 現在表示しているフレームを正確に取得してキャリブレーションに使う
            self.video_cap.set(cv2.CAP_PROP_POS_FRAMES, original_frame_number)
            ret, frame_for_calib = self.video_cap.read()
            if not ret:
                print("❌ 現在のフレームの読み込みに失敗しました。")
                self.is_playing = original_is_playing # 元の状態に戻す
                return

            # 3. CourtCalibratorを準備し、既存の座標があれば渡す
            calibrator = CourtCalibrator()
            if self.court_coordinates:
                print("既存の座標を読み込んで編集モードで開始します。")
                calibrator.set_coordinates(self.court_coordinates)

            # 4. キャリブレーションを実行
            # calibrateメソッドがTrueを返せば、座標が設定または更新された
            if calibrator.calibrate(frame_for_calib, self.video_cap):
                self.court_coordinates = calibrator.get_coordinates()
                self.show_court_overlay = True
                print("✅ コート座標が更新されました。")
                if self.video_path:
                    try:
                        video_name = os.path.splitext(os.path.basename(self.video_path))[0]
                        output_dir = "training_data"
                        os.makedirs(output_dir, exist_ok=True)
                        coord_file = os.path.join(output_dir, f"court_coords_{video_name}.json")

                        with open(coord_file, 'w', encoding='utf-8') as f:
                            json.dump(self.court_coordinates, f, indent=2, ensure_ascii=False)
                            print(f"✅ コート座標を保存しました: {coord_file}")
                    except Exception as e:
                        print(f"❌ コート座標の保存に失敗しました: {e}")

            else:
                print("🟡 コート座標設定がキャンセルまたは中断されました。")

            # 5. アノテーション用のウィンドウを再生成
            # CourtCalibratorのウィンドウを閉じるため、こちらも再準備が必要
            cv2.destroyAllWindows() 
            cv2.namedWindow('Phase Annotation', cv2.WINDOW_NORMAL)
            cv2.resizeWindow('Phase Annotation', self.window_width, self.window_height)

            # 6. 元のフレームと再生状態に戻す
            # これにより、作業を中断したところからシームレスに再開できる
            self._seek_frame(original_frame_number, stop_playback=(not original_is_playing))

    # --- (UI描画は変更なし) ---
    def _draw_ui(self, frame):
        # --- ▼▼▼ 【テスト用】UI描画を極限まで軽量化 ▼▼▼ ---
        if self.is_playing:
            # 【テスト】再生中はリサイズして表示するだけ。UIは一切描画しない。
            # これで速度が改善するかどうかを確認します。
            display_frame, _ = self._resize_frame_for_display(frame)
            cv2.imshow('Phase Annotation', display_frame)
        else:
            # 停止中は元フレームを保護するためにコピーしてからUIを描画します
            display_frame, scale_factors = self._resize_frame_for_display(frame.copy())
            if self.show_court_overlay and self.court_coordinates:
                display_frame = self.draw_court_overlay(display_frame, scale_factors)
            
            # _draw_annotation_uiは前回修正した軽量版を呼び出します
            self._draw_annotation_ui(display_frame)
            cv2.imshow('Phase Annotation', display_frame)
        # --- ▲▲▲ テスト用コード終了 ▲▲▲ ---
    def _resize_frame_for_display(self, frame):
        h, w = frame.shape[:2]
        target_w, target_h = int(self.window_width * 0.7), int(self.window_height * 0.9)
        aspect_ratio = w / h if h > 0 else 1.0
        if target_w / aspect_ratio <= target_h: new_w, new_h = target_w, int(target_w / aspect_ratio)
        else: new_h, new_w = target_h, int(target_h * aspect_ratio)
        resized_frame = cv2.resize(frame, (new_w, new_h), interpolation=cv2.INTER_AREA)
        scale_x, scale_y = (new_w / w if w > 0 else 0), (new_h / h if h > 0 else 0)
        return resized_frame, (scale_x, scale_y)
    def _draw_annotation_ui(self, frame):
            # --- ▼▼▼ 修正箇所（全体を置き換え）▼▼▼ ---
            h, w = frame.shape[:2]
            font_scale, thickness = max(0.3, self.ui_font_scale * self.display_scale), max(1, int(1.5 * self.display_scale))
            y, dy, left_x, right_x = int(20 * self.display_scale), int(18 * self.display_scale), 10, w // 2 + 10

            # --- UI要素の座標などを事前に計算 ---
            info_y, status_y, phase_y = y, y + dy, y + dy * 2
            ph_dy = int(dy * 0.7)
            phases_list_y_start = phase_y + int(dy * 1.3)
            phases_per_column = (len(self.phases) + 1) // 2
            phases_list_y_end = phases_list_y_start + (phases_per_column -1) * ph_dy
            help_texts = ["1-7:Phase SPACE:Play/Pause A/D:Frame W/S:10f Z/X:100f", "+/-:Speed 0:1x C:Court O:Overlay U:Undo Q:Save&Quit"]
            help_y_start = phases_list_y_end + int(dy * 1.5)
            help_y_end = help_y_start + (len(help_texts) - 1) * int(dy * 0.6)

            # --- ステップ1: UI背景の描画を軽量化 ---
            # 再生中は最小限の背景、停止中は全てのUIが入る高さの背景を描画
            if self.is_playing:
                ui_background_height = phase_y + int(dy * 0.5)
            else:
                ui_background_height = help_y_end + int(20 * self.display_scale)

            # ★★★ 最も重要な軽量化: addWeightedを廃止し、単純な矩形描画に変更 ★★★
            cv2.rectangle(frame, (5, 5), (w - 5, min(h - 5, ui_background_height)), (0, 0, 0), -1)

            # --- 常に表示する情報（再生中/停止中共通） ---
            progress = self.current_frame_number / self.total_frames if self.total_frames > 0 else 0
            # FrameとProgressを1行にまとめて描画命令を削減
            cv2.putText(frame, f"Frame: {self.current_frame_number}/{self.total_frames} ({progress:.1%})", (left_x, info_y), cv2.FONT_HERSHEY_SIMPLEX, font_scale, (255,255,255), thickness)

            status, s_color = ("PLAY", (0,255,0)) if self.is_playing else ("PAUSE", (0,0,255))
            cv2.putText(frame, f"Status: {status}", (left_x, status_y), cv2.FONT_HERSHEY_SIMPLEX, font_scale, s_color, thickness)
            cv2.putText(frame, f"Speed: {self.playback_speed}x", (right_x, status_y), cv2.FONT_HERSHEY_SIMPLEX, font_scale, (0,255,255), thickness)
            cv2.putText(frame, f"Phase: {self.current_phase or 'None'}", (left_x, phase_y), cv2.FONT_HERSHEY_SIMPLEX, font_scale, (0,255,0), thickness)

            # --- ステップ2: 停止中 (is_playing=False) のみ詳細情報を描画 ---
            if not self.is_playing:
                court_info = f"ON ({len(self.court_coordinates)} pts)" if self.court_coordinates else "OFF"
                court_status = f"Court: {court_info} | Ovl: {'ON' if self.show_court_overlay else 'OFF'}"
                cv2.putText(frame, court_status, (right_x, phase_y), cv2.FONT_HERSHEY_SIMPLEX, font_scale, (255,255,0), thickness)

                # 局面リストの描画
                ph_font_scale = font_scale * 0.85
                for i, p in enumerate(self.phases):
                    col, row = i // phases_per_column, i % phases_per_column
                    x_pos, y_pos = (left_x if col == 0 else right_x), phases_list_y_start + row * ph_dy
                    color = (0, 255, 0) if p == self.current_phase else (255, 255, 255)
                    short_phase = p.replace("serve_", "").replace("_", " ")[:12]
                    cv2.putText(frame, f"{i+1}: {short_phase}", (x_pos, y_pos), cv2.FONT_HERSHEY_SIMPLEX, ph_font_scale, color, max(1, thickness-1))

                # ヘルプテキストの描画
                for i, text in enumerate(help_texts):
                    y_pos = help_y_start + i * int(dy * 0.6)
                    if y_pos < h - 10: cv2.putText(frame, text, (left_x, y_pos), cv2.FONT_HERSHEY_SIMPLEX, font_scale * 0.6, (255,255,0), max(1, thickness-1))
            # --- ▲▲▲ 修正完了 ▲▲▲ ---
    def draw_court_overlay(self, frame, scale_factors):
        overlay_frame, thickness, (scale_x, scale_y) = frame.copy(), max(1, int(1.5 * self.display_scale)), scale_factors
        scaled_coords = {name: (int(p[0] * scale_x), int(p[1] * scale_y)) for name, p in self.court_coordinates.items() if p and len(p) == 2}
        corner_keys = ["top_left_corner", "top_right_corner", "bottom_right_corner", "bottom_left_corner"]
        points = [scaled_coords.get(key) for key in corner_keys]
        if all(p is not None for p in points): cv2.polylines(overlay_frame, [np.array(points, dtype=np.int32)], isClosed=True, color=(255, 255, 0), thickness=thickness)
        net_l, net_r = scaled_coords.get("net_left_ground"), scaled_coords.get("net_right_ground")
        if net_l and net_r: cv2.line(overlay_frame, net_l, net_r, (0, 255, 255), thickness + 1)
        for point in scaled_coords.values(): cv2.circle(overlay_frame, point, max(2, int(4 * self.display_scale)), (0, 255, 0), -1)
        return overlay_frame

    # --- (データ管理は変更なし) ---
    def _record_phase_change(self, new_phase):
        frame_num = self.current_frame_number
        self.phase_changes = [c for c in self.phase_changes if c['frame_number'] <= frame_num]
        if self.phase_changes and self.phase_changes[-1]['frame_number'] == frame_num:
            if self.phase_changes[-1]['phase'] != new_phase:
                old_phase = self.phase_changes[-1]['phase']
                self.phase_changes[-1]['phase'] = new_phase
                print(f"Frame {frame_num}: 局面を上書き -> {new_phase} (旧: {old_phase})")
        else:
            if self.phase_changes and self.phase_changes[-1]['phase'] == new_phase: return
            self.phase_changes.append({'frame_number': frame_num, 'phase': new_phase, 'timestamp': frame_num / self.fps})
            print(f"Frame {frame_num}: 新しい局面 -> {new_phase}")
        self.phase_changes.sort(key=lambda x: x['frame_number'])
        self.current_phase = new_phase
    def _update_current_phase(self):
        updated_phase = None
        for change in reversed(self.phase_changes):
            if change['frame_number'] <= self.current_frame_number:
                updated_phase = change['phase']; break
        self.current_phase = updated_phase
    def _undo_last_phase_change(self):
        if self.phase_changes:
            removed = self.phase_changes.pop()
            print(f"取り消し: Frame {removed['frame_number']} - {removed['phase']}")
            self._update_current_phase()
        else: print("取り消す変更がありません")
    def _reset_annotations(self):
        if self.phase_changes and input("全ての局面変更を削除しますか？ (y/n): ").lower() == 'y':
            self.phase_changes, self.current_phase = [], None
            print("全てのアノテーションを削除しました")

    # --- 終了と保存 ---

    def _cleanup_and_save(self, video_path):
        """リソースを解放し、データを保存します。編集モードの場合は上書き確認を行います。"""
        self.video_cap.release()
        cv2.destroyAllWindows()
        
        if not self.phase_changes:
            print("変更がなかったため、保存せずに終了します。")
            return False

        if self.editing_file_path:
            overwrite = input(f"既存のファイルに上書き保存しますか？ (y/n): ").lower()
            if overwrite == 'y':
                return self._save_phase_data(video_path, output_path=self.editing_file_path)

        return self._save_phase_data(video_path)
    
    def _confirm_exit(self):
        """終了時の確認ダイアログを表示します。"""
        if not self.phase_changes:
            return True
        save = input("変更を保存して終了しますか？ (y/n): ").lower()
        if save == 'y': return True
        else: return input("保存せずに終了しますか？ (y/n): ").lower() == 'y'

    def _save_phase_data(self, video_path: str, output_path: str = None):
        """局面データをJSONファイルに保存します。output_pathが指定されていれば上書きします。"""
        if output_path:
            phase_file = output_path
        else:
            video_name = os.path.splitext(os.path.basename(video_path))[0]
            output_dir = "training_data"
            os.makedirs(output_dir, exist_ok=True)
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            phase_file = os.path.join(output_dir, f"phase_annotations_{video_name}_{timestamp}.json")
        
        annotation_data = {
            'video_path': os.path.abspath(video_path),
            'video_name': os.path.splitext(os.path.basename(video_path))[0],
            'total_frames': self.total_frames, 'fps': self.fps,
            'duration_seconds': self.total_frames / self.fps,
            'phase_changes': self.phase_changes,
            'annotation_timestamp': datetime.now().strftime("%Y%m%d_%H%M%S"),
            'phase_statistics': self._calculate_phase_statistics(),
            'court_coordinates': self.court_coordinates or None,
            'court_coordinates_available': bool(self.court_coordinates)
        }
        
        try:
            with open(phase_file, 'w', encoding='utf-8') as f:
                json.dump(annotation_data, f, indent=2, ensure_ascii=False)
            
            save_type = "上書き保存" if output_path else "新規保存"
            print(f"\n✅ 局面アノテーションを{save_type}しました: {phase_file}")
            self._print_statistics(annotation_data)
            return True
        except Exception as e:
            print(f"❌ 局面データ保存エラー: {e}")
            return False

    def setup_court_coordinates(self, video_path):
        """（新規作成時用）独立したコート座標ファイルを読み込みます。"""
        video_name = os.path.splitext(os.path.basename(video_path))[0]
        coord_file = os.path.join("training_data", f"court_coords_{video_name}.json")
        if os.path.exists(coord_file):
            try:
                with open(coord_file, 'r') as f: self.court_coordinates = json.load(f)
                print(f"✅ 既存のコート座標を読み込みました: {coord_file}")
                for name, point in self.court_coordinates.items(): print(f"   - {name}: {point}")
                return True
            except Exception as e:
                print(f"❌ コート座標ファイル読み込みエラー: {e}")
        return False
    # (save_court_coordinates, 統計、ヘルパー関数は変更なし)
    def save_court_coordinates(self, video_path):
        video_name = os.path.splitext(os.path.basename(video_path))[0]
        output_dir = "training_data"; os.makedirs(output_dir, exist_ok=True)
        coord_file = os.path.join(output_dir, f"court_coords_{video_name}.json")
        try:
            with open(coord_file, 'w') as f: json.dump(self.court_coordinates, f, indent=2)
            print(f"✅ コート座標を保存しました: {coord_file}"); return True
        except Exception as e:
            print(f"❌ コート座標保存エラー: {e}"); return False
    def _calculate_phase_statistics(self):
        if not self.phase_changes: return {}
        durations = {phase: 0 for phase in self.phases}
        for i, change in enumerate(self.phase_changes):
            start_frame, end_frame = change['frame_number'], self.phase_changes[i+1]['frame_number'] if i + 1 < len(self.phase_changes) else self.total_frames
            durations[change['phase']] += (end_frame - start_frame)
        total_duration_sec = self.total_frames / self.fps
        stats = {}
        for phase, frame_count in durations.items():
            if frame_count > 0:
                duration_sec = frame_count / self.fps
                stats[phase] = {'duration': duration_sec, 'percentage': (duration_sec / total_duration_sec) * 100 if total_duration_sec > 0 else 0}
        return stats
    def _print_statistics(self, data):
        print(f"記録された局面変更数: {len(data['phase_changes'])}")
        if data['court_coordinates_available']: print("🏟️  コート座標も含まれています")
        print("\n=== 局面統計 ===")
        stats = data.get('phase_statistics', {})
        for phase, info in stats.items(): print(f"{phase:<20}: {info['duration']:.1f}秒 ({info['percentage']:.1f}%)")
    def _print_usage_instructions(self):
        print("\n=== 局面アノテーション開始 ===")
        print("局面選択: 1-7キー | 再生制御: SPACE, a/d, w/s, z/x | 再生速度: +/-, 0")
        print("コート座標: c (設定), o (オーバーレイON/OFF) | その他: u (元に戻す), r (リセット), q (終了)")

# --- CLIメニューとファイル管理関数 ---

def select_annotation_file(video_path: str):
    """指定された動画に対応する既存のアノテーションファイルを選択させます。"""
    video_name = os.path.splitext(os.path.basename(video_path))[0]
    output_dir = "training_data"
    
    existing_files = []
    if os.path.exists(output_dir):
        for f in os.listdir(output_dir):
            if f.startswith(f"phase_annotations_{video_name}_") and f.endswith(".json"):
                existing_files.append(os.path.join(output_dir, f))

    if not existing_files:
        print("\n既存のアノテーションファイルは見つかりませんでした。")
        return None

    print("\n=== 既存のアノテーションファイルが見つかりました ===")
    existing_files.sort(key=os.path.getmtime, reverse=True)
    for i, f_path in enumerate(existing_files, 1):
        print(f"{i}: {os.path.basename(f_path)}")
    print(f"{len(existing_files) + 1}: 📝 新規作成")
    
    try:
        choice = int(input(f"\n編集するファイルを選択してください (1-{len(existing_files) + 1}): "))
        if 1 <= choice <= len(existing_files):
            return existing_files[choice - 1]
    except (ValueError, IndexError):
        pass
    
    print("新規作成を選択しました。")
    return None

def get_video_files(data_dir="../data/raw"):
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
            except Exception as e: print(f"ディレクトリ読み取りエラー: {e}")
    return []

def select_video_file():
    video_files = get_video_files()
    if not video_files:
        print("\n動画ファイルが見つかりませんでした。")
        if input("手動でファイルパスを入力しますか？ (y/n): ").lower() == 'y':
            path = input("動画ファイルのフルパスを入力してください: ").strip()
            return path if os.path.exists(path) else None
        return None
    print("\n=== 動画ファイル一覧 ===")
    for i, video_path in enumerate(video_files, 1): print(f"{i}: {os.path.basename(video_path)}")
    try:
        choice = int(input(f"\n動画を選択してください (1-{len(video_files)}): "))
        return video_files[choice - 1] if 1 <= choice <= len(video_files) else None
    except ValueError: return None

def main():
    """メイン関数 - アノテーションツールを起動します。"""
    print("=== テニス局面アノテーションツール (編集機能付き) ===")
    print("1: 局面アノテーションを開始")
    print("2: 終了")
    
    choice = input("\n選択してください (1-2): ").strip()
    
    if choice == '1':
        video_path = select_video_file()
        if video_path:
            annotation_path = select_annotation_file(video_path)
            annotator = PhaseAnnotator()
            annotator.annotate_video(video_path, existing_annotation_path=annotation_path)
    
    print("プログラムを終了します。")

if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("\n\n操作が中断されました。プログラムを終了します。")
    except Exception as e:
        print(f"\n予期せぬエラーが発生しました: {e}")