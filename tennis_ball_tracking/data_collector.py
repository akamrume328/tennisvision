import cv2
import json
import os
import numpy as np
from datetime import datetime
from court_calibrator import CourtCalibrator

class PhaseAnnotator:
    """
    テニス局面ラベリング専用クラス
    
    用途:
    - 動画の局面分析
    - 高速な局面ラベリング作業
    - 局面統計の作成
    - train_phase_model.py用のデータ作成
    - コート座標設定（オプション）
    """
    def __init__(self):
        self.phases = [
            "point_interval",           # ポイント間
            "rally",                   # ラリー中
            "serve_preparation",       # サーブ準備
            "serve_front_deuce",      # 手前デュースサイドからのサーブ
            "serve_front_ad",         # 手前アドサイドからのサーブ
            "serve_back_deuce",       # 奥デュースサイドからのサーブ
            "serve_back_ad",          # 奥アドサイドからのサーブ
            "changeover"              # チェンジコート間
        ]
        self.current_phase = None
        self.phase_changes = []
        self.video_cap = None
        self.current_frame_number = 0
        self.total_frames = 0
        self.is_playing = False
        self.playback_speed = 1.0
        self.fps = 30.0
        
        # コート座標設定機能（統合版）
        self.show_court_overlay = False
        self.court_coordinates = {}
        
    def set_video_source(self, video_cap):
        """動画ソースを設定"""
        self.video_cap = video_cap
        if video_cap:
            self.total_frames = int(video_cap.get(cv2.CAP_PROP_FRAME_COUNT))
            self.fps = video_cap.get(cv2.CAP_PROP_FPS)

    def setup_court_coordinates(self, video_path):
        """既存のコート座標を読み込み"""
        video_name = os.path.splitext(os.path.basename(video_path))[0]
        coord_file = os.path.join("training_data", f"court_coords_{video_name}.json")
        
        if os.path.exists(coord_file):
            try:
                with open(coord_file, 'r') as f:
                    self.court_coordinates = json.load(f)
                print(f"\n✅ 既存のコート座標を読み込みました: {coord_file}")
                for name, point in self.court_coordinates.items():
                    print(f"   {name}: {point}")
                return True
            except Exception as e:
                print(f"❌ コート座標ファイル読み込みエラー: {e}")
        
        return False

    def save_court_coordinates(self, video_path):
        """コート座標を保存"""
        if not self.court_coordinates:
            return False
            
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

    def setup_court_coordinates_interactive(self):
        """対話的なコート座標設定"""
        if not self.video_cap:
            print("❌ 動画が読み込まれていません")
            return False
            
        # 現在のフレームを取得
        self.video_cap.set(cv2.CAP_PROP_POS_FRAMES, self.current_frame_number)
        ret, frame = self.video_cap.read()
        if not ret:
            print("❌ フレーム読み込みエラー")
            return False
            
        # CourtCalibratorを使用
        calibrator = CourtCalibrator()
        success = calibrator.calibrate(frame, self.video_cap)
        
        if success:
            self.court_coordinates = calibrator.get_coordinates()
            self.show_court_overlay = True
            print("✅ コート座標設定完了")
            return True
        else:
            print("❌ コート座標設定がキャンセルされました")
            return False

    def draw_court_overlay(self, frame):
        """コート座標をフレームに描画"""
        if not self.court_coordinates or not self.show_court_overlay:
            return frame
        
        overlay_frame = frame.copy()
        
        # コート四隅を線で結ぶ
        points = []
        corner_names = ["top_left_corner", "top_right_corner", 
                       "bottom_right_corner", "bottom_left_corner"]
        
        for name in corner_names:
            if name in self.court_coordinates:
                points.append(self.court_coordinates[name])
        
        if len(points) == 4:
            pts = np.array(points, np.int32)
            cv2.polylines(overlay_frame, [pts], True, (255, 255, 0), 2)
        
        # ネットライン
        if ("net_left_ground" in self.court_coordinates and 
            "net_right_ground" in self.court_coordinates):
            net_left = self.court_coordinates["net_left_ground"]
            net_right = self.court_coordinates["net_right_ground"]
            cv2.line(overlay_frame, net_left, net_right, (0, 255, 255), 3)
        
        # 各点を描画
        for name, point in self.court_coordinates.items():
            cv2.circle(overlay_frame, point, 6, (0, 255, 0), -1)
            cv2.putText(overlay_frame, name[:8], (point[0]+8, point[1]-8), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.4, (0, 255, 0), 1)
        
        return overlay_frame

    def _record_phase_change(self, new_phase):
        """局面変更を記録"""
        if new_phase != self.current_phase:
            change_data = {
                'frame_number': self.current_frame_number,
                'phase': new_phase,
                'timestamp': self.current_frame_number / self.fps
            }
            self.phase_changes.append(change_data)
            self.current_phase = new_phase

    def _seek_frame(self, target_frame):
        """指定フレームに移動し、現在の局面を更新"""
        target_frame = max(0, min(target_frame, self.total_frames - 1))
        self.current_frame_number = target_frame
        if self.video_cap:
            self.video_cap.set(cv2.CAP_PROP_POS_FRAMES, target_frame)
        
        # フレーム移動時に現在の局面を更新
        self._update_current_phase()

    def _update_current_phase(self):
        """現在の局面を再計算（取り消し後やフレーム移動時に使用）"""
        if not self.phase_changes:
            self.current_phase = None
            return
            
        # 現在フレーム以下の最新の局面変更を探す
        current_phase = None
        for change in self.phase_changes:
            if change['frame_number'] <= self.current_frame_number:
                current_phase = change['phase']
        
        self.current_phase = current_phase

    def _calculate_phase_statistics(self):
        """局面統計を計算"""
        if not self.phase_changes:
            return {}
            
        # 各局面の時間を計算
        phase_durations = {}
        
        for i, change in enumerate(self.phase_changes):
            phase = change['phase']
            start_frame = change['frame_number']
            
            # 次の変更までの時間を計算
            if i + 1 < len(self.phase_changes):
                end_frame = self.phase_changes[i + 1]['frame_number']
            else:
                end_frame = self.total_frames
            
            duration_frames = end_frame - start_frame
            duration_seconds = duration_frames / self.fps
            
            if phase not in phase_durations:
                phase_durations[phase] = 0
            phase_durations[phase] += duration_seconds
        
        # パーセンテージを計算
        total_duration = self.total_frames / self.fps
        statistics = {}
        
        for phase, duration in phase_durations.items():
            percentage = (duration / total_duration) * 100 if total_duration > 0 else 0
            statistics[phase] = {
                'duration': duration,
                'percentage': percentage
            }
        
        return statistics

    def annotate_video(self, video_path: str):
        """動画全体の局面アノテーション"""
        cap = cv2.VideoCapture(video_path)
        if not cap.isOpened():
            print(f"Error: 動画ファイルを開けません: {video_path}")
            return False
        
        self.set_video_source(cap)
        print(f"動画をロードしました: {os.path.basename(video_path)}")
        print(f"総フレーム数: {self.total_frames}")
        print(f"FPS: {self.fps}")
        
        # 既存のコート座標をチェック・読み込み
        self.setup_court_coordinates(video_path)
        if self.court_coordinates:
            self.show_court_overlay = True
        
        print("\n=== 局面アノテーション開始 ===")
        print("局面選択キー (Numpad推奨):")
        for i, phase in enumerate(self.phases):
            print(f"  {i+1}: {phase}")
        print("\n再生制御:")
        print("  SPACE: 再生/停止")
        print("  →/d: 次のフレーム")
        print("  ←/a: 前のフレーム")
        print("  ↑/w: 10フレーム進む")
        print("  ↓/s: 10フレーム戻る")
        print("  z: 100フレーム戻る")
        print("  x: 100フレーム進む")
        print("  HOME: 最初のフレーム")
        print("  END: 最後のフレーム")
        print("\n再生速度:")
        print("  -(マイナス): 速度を下げる (0.25x → 0.5x → 1.0x)")
        print("  +(プラス): 速度を上げる (1.0x → 2.0x → 4.0x)")
        print("  0: 通常速度に戻す (1.0x)")
        print("\nコート座標:")
        print("  c: コート座標設定")
        print("  o: コートオーバーレイ ON/OFF")
        print("\nその他:")
        print("  r: リセット（全ての局面変更を削除）")
        print("  u: 最後の局面変更を取り消し")
        print("  q: 終了して保存")
        
        # 最初のフレームから開始
        self.current_frame_number = 0
        cap.set(cv2.CAP_PROP_POS_FRAMES, self.current_frame_number)
        
        while True:
            # フレーム読み取り
            if self.is_playing:
                ret, frame = cap.read()
                if not ret:
                    print("動画の終端に達しました")
                    self.is_playing = False
                    self.current_frame_number = self.total_frames - 1
                    cap.set(cv2.CAP_PROP_POS_FRAMES, self.current_frame_number)
                    ret, frame = cap.read()
                else:
                    self.current_frame_number = int(cap.get(cv2.CAP_PROP_POS_FRAMES)) - 1
            else:
                cap.set(cv2.CAP_PROP_POS_FRAMES, self.current_frame_number)
                ret, frame = cap.read()
            
            if not ret:
                print("フレーム読み取りエラー")
                break
            
            # フレーム情報を表示
            display_frame = frame.copy()
            
            # コートオーバーレイを追加
            display_frame = self.draw_court_overlay(display_frame)
            
            # UIを描画
            self._draw_annotation_ui(display_frame)
            
            cv2.imshow('Phase Annotation', display_frame)
            
            # キー入力待機時間を調整
            if self.is_playing:
                wait_time = max(1, int(1000 / (self.fps * self.playback_speed)))
            else:
                wait_time = 0
            
            key = cv2.waitKey(wait_time) & 0xFF
            
            if key == 255:  # タイムアウト（再生中）
                continue
            
            # 局面選択（1-8 数字キー）
            if ord('1') <= key <= ord('8'):
                phase_index = key - ord('1')
                if phase_index < len(self.phases):
                    new_phase = self.phases[phase_index]
                    self._record_phase_change(new_phase)
                    print(f"Frame {self.current_frame_number}: {new_phase}")
            
            # 再生制御
            elif key == ord(' '):  # スペース: 再生/停止
                self.is_playing = not self.is_playing
                status = "再生開始" if self.is_playing else "停止"
                print(f"{status} (Frame: {self.current_frame_number})")
            
            # フレーム移動（競合を避ける）
            elif key == ord('d') or key == 83:  # d または右矢印
                self._seek_frame(self.current_frame_number + 1)
                self.is_playing = False
            elif key == ord('a') or key == 81:  # a または左矢印
                self._seek_frame(self.current_frame_number - 1)
                self.is_playing = False
            elif key == ord('w') or key == 82:  # w または上矢印
                self._seek_frame(self.current_frame_number + 10)
                self.is_playing = False
            elif key == ord('s') or key == 84:  # s または下矢印
                self._seek_frame(self.current_frame_number - 10)
                self.is_playing = False
            elif key == ord('z'):  # 100フレーム戻る
                self._seek_frame(self.current_frame_number - 100)
                self.is_playing = False
            elif key == ord('x'):  # 100フレーム進む
                self._seek_frame(self.current_frame_number + 100)
                self.is_playing = False
            elif key == 2:  # HOME
                self._seek_frame(0)
                self.is_playing = False
                print("最初のフレームに移動")
            elif key == 3:  # END
                self._seek_frame(self.total_frames - 1)
                self.is_playing = False
                print("最後のフレームに移動")
            
            # 再生速度変更
            elif key == ord('-') or key == ord('_'):  # マイナスキー
                speed_levels = [0.25, 0.5, 1.0, 2.0, 4.0]
                current_index = speed_levels.index(self.playback_speed) if self.playback_speed in speed_levels else 2
                if current_index > 0:
                    self.playback_speed = speed_levels[current_index - 1]
                    print(f"再生速度: {self.playback_speed}x")
                else:
                    print(f"最低速度です: {self.playback_speed}x")
            
            elif key == ord('+') or key == ord('='):  # プラスキー
                speed_levels = [0.25, 0.5, 1.0, 2.0, 4.0]
                current_index = speed_levels.index(self.playback_speed) if self.playback_speed in speed_levels else 2
                if current_index < len(speed_levels) - 1:
                    self.playback_speed = speed_levels[current_index + 1]
                    print(f"再生速度: {self.playback_speed}x")
                else:
                    print(f"最高速度です: {self.playback_speed}x")
            
            elif key == ord('0'):  # 0キー: 通常速度
                self.playback_speed = 1.0
                print(f"再生速度を通常に戻しました: {self.playback_speed}x")
            
            # コート座標機能
            elif key == ord('c') or key == ord('C'):
                print("\nコート座標設定を開始します...")
                self.is_playing = False
                success = self.setup_court_coordinates_interactive()
                if success:
                    self.show_court_overlay = True
                print("アノテーション画面に戻ります...")
                # ウィンドウフォーカスを戻す
                cv2.destroyAllWindows()
                cv2.namedWindow('Phase Annotation', cv2.WINDOW_AUTOSIZE)
            elif key == ord('o') or key == ord('O'):
                self.show_court_overlay = not self.show_court_overlay
                status = "ON" if self.show_court_overlay else "OFF"
                print(f"コートオーバーレイ: {status}")
            
            # その他の機能
            elif key == ord('r'):
                if self.phase_changes:
                    print(f"現在の局面変更数: {len(self.phase_changes)}")
                    confirm = input("全ての局面変更を削除しますか？ (y/n): ").lower()
                    if confirm == 'y':
                        self.phase_changes = []
                        self.current_phase = None
                        print("全ての局面変更を削除しました")
                else:
                    print("削除する局面変更がありません")
            
            elif key == ord('u'):
                if self.phase_changes:
                    removed = self.phase_changes.pop()
                    print(f"局面変更を取り消しました: Frame {removed['frame_number']} - {removed['phase']}")
                    self._update_current_phase()
                else:
                    print("取り消す局面変更がありません")
            
            elif key == ord('q'):
                if self.phase_changes:
                    print(f"記録された局面変更数: {len(self.phase_changes)}")
                    save = input("変更を保存しますか？ (y/n): ").lower()
                    if save == 'y':
                        break
                    else:
                        confirm = input("保存せずに終了しますか？ (y/n): ").lower()
                        if confirm == 'y':
                            cap.release()
                            cv2.destroyAllWindows()
                            return False
                else:
                    break
        
        cap.release()
        cv2.destroyAllWindows()
        
        # データを保存
        result = self._save_phase_data(video_path)
        
        # コート座標も保存
        if self.court_coordinates:
            self.save_court_coordinates(video_path)
        
        return result
    
    def _draw_annotation_ui(self, frame):
        """アノテーション用UIを描画"""
        height, width = frame.shape[:2]
        
        # 背景オーバーレイ
        overlay = frame.copy()
        cv2.rectangle(overlay, (10, 10), (width-10, 300), (0, 0, 0), -1)
        cv2.addWeighted(overlay, 0.7, frame, 0.3, 0, frame)
        
        # フレーム情報
        cv2.putText(frame, f"Frame: {self.current_frame_number}/{self.total_frames}", 
                   (20, 40), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)
        
        progress = self.current_frame_number / self.total_frames if self.total_frames > 0 else 0
        cv2.putText(frame, f"Progress: {progress:.1%}", 
                   (20, 70), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)
        
        # 再生状態（速度表示を強調）
        status = "PLAYING" if self.is_playing else "PAUSED"
        status_color = (0, 255, 0) if self.is_playing else (0, 0, 255)
        speed_color = (0, 255, 255) if self.playback_speed != 1.0 else (255, 255, 255)
        cv2.putText(frame, f"Status: {status}", 
                   (20, 100), cv2.FONT_HERSHEY_SIMPLEX, 0.8, status_color, 2)
        cv2.putText(frame, f"Speed: {self.playback_speed}x", 
                   (20, 130), cv2.FONT_HERSHEY_SIMPLEX, 0.8, speed_color, 2)
        
        # 現在の局面
        current_text = f"Current Phase: {self.current_phase or 'None'}"
        cv2.putText(frame, current_text, (20, 160), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)
        
        # コート設定状態
        court_status = "Court: " + ("SET" if self.court_coordinates else "NOT SET")
        overlay_status = f" Overlay: {'ON' if self.show_court_overlay else 'OFF'}"
        cv2.putText(frame, court_status + overlay_status, (20, 190), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 0), 2)
        
        # 局面リスト
        y_start = 220
        for i, phase in enumerate(self.phases):
            color = (0, 255, 0) if phase == self.current_phase else (255, 255, 255)
            cv2.putText(frame, f"{i+1}: {phase}", (20, y_start + i*20), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 1)
        
        # 操作ヘルプ（右側） - 速度変更を追加
        help_x = width - 400
        help_texts = [
            "1-8: Phase selection",
            "SPACE: Play/Pause",
            "A/D: Prev/Next frame",  
            "W/S: +/-10 frames",
            "Z/X: +/-100 frames",
            "+/-: Speed up/down",
            "0: Normal speed (1x)",
            "C: Court setup",
            "O: Court overlay",
            "U: Undo last change",
            "Q: Quit and save"
        ]
        
        for i, text in enumerate(help_texts):
            cv2.putText(frame, text, (help_x, 40 + i*25), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 0), 1)

    def _save_phase_data(self, video_path):
        """局面データを保存"""
        if not self.phase_changes:
            print("保存する局面データがありません")
            return False
        
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        video_name = os.path.splitext(os.path.basename(video_path))[0]
        
        output_dir = "training_data"
        os.makedirs(output_dir, exist_ok=True)
        
        phase_file = os.path.join(output_dir, f"phase_annotations_{video_name}_{timestamp}.json")
        try:
            annotation_data = {
                'video_path': video_path,
                'video_name': video_name,
                'total_frames': self.total_frames,
                'fps': self.fps,
                'duration_seconds': self.total_frames / self.fps,
                'phase_changes': self.phase_changes,
                'annotation_timestamp': timestamp,
                'phase_statistics': self._calculate_phase_statistics(),
                'court_coordinates': self.court_coordinates if self.court_coordinates else None,
                'court_coordinates_available': bool(self.court_coordinates)
            }
            
            with open(phase_file, 'w', encoding='utf-8') as f:
                json.dump(annotation_data, f, indent=2, ensure_ascii=False)
            print(f"局面アノテーションを保存しました: {phase_file}")
            print(f"記録された局面変更数: {len(self.phase_changes)}")
            
            if self.court_coordinates:
                print(f"✅ コート座標も含まれています")
            
            # 統計情報を表示
            stats = annotation_data['phase_statistics']
            print("\n=== 局面統計 ===")
            for phase, info in stats.items():
                print(f"{phase}: {info['duration']:.1f}秒 ({info['percentage']:.1f}%)")
                
            return True
                
        except Exception as e:
            print(f"局面データ保存エラー: {e}")
            return False

def get_video_files(data_dir="../data/raw"):
    """data/rawフォルダ内の動画ファイル一覧を取得"""
    print(f"動画ファイルを検索中: {data_dir}")
    
    # 絶対パスに変換
    abs_data_dir = os.path.abspath(data_dir)
    print(f"絶対パス: {abs_data_dir}")
    
    video_extensions = ['.mp4', '.avi', '.mov', '.mkv', '.wmv', '.flv', '.webm']
    video_files = []
    
    if os.path.exists(abs_data_dir):
        print(f"ディレクトリが存在します: {abs_data_dir}")
        try:
            files = os.listdir(abs_data_dir)
            print(f"ディレクトリ内のファイル数: {len(files)}")
            
            for file in files:
                print(f"チェック中: {file}")
                if any(file.lower().endswith(ext) for ext in video_extensions):
                    video_path = os.path.join(abs_data_dir, file)
                    video_files.append(video_path)
                    print(f"動画ファイル発見: {file}")
        except Exception as e:
            print(f"ディレクトリ読み取りエラー: {e}")
    else:
        print(f"ディレクトリが存在しません: {abs_data_dir}")
        
        # 他の可能性のあるパスもチェック
        alternative_paths = [
            "data/raw",
            "./data/raw", 
            "../../data/raw",
            os.path.join(os.getcwd(), "data", "raw"),
            "C:/Users/akama/Desktop/tennis_videos"  # 追加の可能性
        ]
        
        print("\n代替パスを検索中...")
        for alt_path in alternative_paths:
            abs_alt_path = os.path.abspath(alt_path)
            print(f"チェック: {abs_alt_path}")
            if os.path.exists(abs_alt_path):
                print(f"代替パスが見つかりました: {abs_alt_path}")
                try:
                    files = os.listdir(abs_alt_path)
                    for file in files:
                        if any(file.lower().endswith(ext) for ext in video_extensions):
                            video_path = os.path.join(abs_alt_path, file)
                            video_files.append(video_path)
                            print(f"動画ファイル発見: {file}")
                    if video_files:
                        break
                except Exception as e:
                    print(f"代替パス読み取りエラー: {e}")
    
    print(f"合計動画ファイル数: {len(video_files)}")
    return video_files

def select_video_file():
    """動画ファイルを選択"""
    print("動画ファイル検索を開始...")
    video_files = get_video_files()
    
    if not video_files:
        print("\n動画ファイルが見つかりませんでした。")
        print("以下の場所に動画ファイルを配置してください:")
        print("- ../data/raw フォルダ")
        print("- data/raw フォルダ")
        print("- 現在のディレクトリ")
        
        # 手動でファイルパスを入力する選択肢を追加
        print("\n手動でファイルパスを入力しますか？ (y/n): ")
        manual_input = input().lower().strip()
        
        if manual_input == 'y':
            file_path = input("動画ファイルの完全パスを入力してください: ").strip()
            if os.path.exists(file_path):
                return file_path
            else:
                print(f"ファイルが見つかりません: {file_path}")
        
        return None
    
    print("\n=== 動画ファイル一覧 ===")
    for i, video_path in enumerate(video_files, 1):
        filename = os.path.basename(video_path)
        print(f"{i}: {filename}")
    
    try:
        choice = int(input(f"\n動画を選択してください (1-{len(video_files)}): "))
        if 1 <= choice <= len(video_files):
            return video_files[choice - 1]
        else:
            print("無効な選択です")
            return None
    except ValueError:
        print("数字を入力してください")
        return None

def get_training_data_files():
    """training_dataフォルダ内のファイル一覧を取得"""
    training_dir = "training_data"
    
    if not os.path.exists(training_dir):
        print(f"❌ {training_dir} フォルダが存在しません")
        return {
            'annotations': [],
            'features': [],
            'models': [],
            'court_coords': []
        }
    
    files = {
        'annotations': [],      # phase_annotations_*.json
        'features': [],         # tennis_features_dataset_*.csv
        'models': [],          # *.pkl, *.pth, *.h5
        'court_coords': []     # court_coords_*.json
    }
    
    try:
        for file in os.listdir(training_dir):
            file_path = os.path.join(training_dir, file)
            
            # アノテーションファイル
            if file.startswith("phase_annotations_") and file.endswith(".json"):
                files['annotations'].append(file_path)
            
            # 特徴量データセット
            elif file.startswith("tennis_features_dataset_") and file.endswith(".csv"):
                files['features'].append(file_path)
            
            # モデルファイル
            elif any(file.endswith(ext) for ext in ['.pkl', '.pth', '.h5', '.joblib']):
                if not file.startswith('tennis_') or 'model' in file.lower():
                    files['models'].append(file_path)
            
            # コート座標ファイル
            elif file.startswith("court_coords_") and file.endswith(".json"):
                files['court_coords'].append(file_path)
    
    except Exception as e:
        print(f"ディレクトリ読み取りエラー: {e}")
    
    # ファイルを日付順でソート（新しい順）
    for file_type in files:
        files[file_type].sort(key=lambda x: os.path.getmtime(x), reverse=True)
    
    return files

def select_training_data_file(file_type='annotations'):
    """訓練データファイルを選択"""
    files = get_training_data_files()
    
    file_type_names = {
        'annotations': '局面アノテーション',
        'features': '特徴量データセット', 
        'models': 'モデル',
        'court_coords': 'コート座標'
    }
    
    target_files = files.get(file_type, [])
    
    if not target_files:
        print(f"\n❌ {file_type_names[file_type]}ファイルが見つかりません")
        print(f"📍 場所: training_data/ フォルダ")
        return None
    
    print(f"\n=== {file_type_names[file_type]}ファイル一覧 ===")
    
    for i, file_path in enumerate(target_files, 1):
        filename = os.path.basename(file_path)
        
        # ファイル情報を表示
        try:
            stat = os.stat(file_path)
            size_mb = stat.st_size / (1024 * 1024)
            mtime = datetime.fromtimestamp(stat.st_mtime)
            
            print(f"{i}: {filename}")
            print(f"   📅 更新日時: {mtime.strftime('%Y-%m-%d %H:%M:%S')}")
            print(f"   📦 サイズ: {size_mb:.2f} MB")
            
            # ファイル内容の要約（JSONファイルの場合）
            if file_type == 'annotations' and filename.endswith('.json'):
                try:
                    with open(file_path, 'r', encoding='utf-8') as f:
                        data = json.load(f)
                    video_name = data.get('video_name', '不明')
                    phase_count = len(data.get('phase_changes', []))
                    duration = data.get('duration_seconds', 0)
                    print(f"   📹 動画: {video_name}")
                    print(f"   🎯 局面変更数: {phase_count}")
                    print(f"   ⏱️  時間: {duration:.1f}秒")
                except:
                    pass
            
            print()
            
        except Exception as e:
            print(f"{i}: {filename} (情報取得エラー)")
            print()
    
    # 全ファイル選択オプション
    print(f"{len(target_files) + 1}: 📁 全ファイル使用")
    print(f"{len(target_files) + 2}: 🔙 戻る")
    
    try:
        choice = input(f"\n選択してください (1-{len(target_files) + 2}): ").strip()
        choice_num = int(choice)
        
        if 1 <= choice_num <= len(target_files):
            selected_file = target_files[choice_num - 1]
            print(f"✅ 選択されたファイル: {os.path.basename(selected_file)}")
            return selected_file
        
        elif choice_num == len(target_files) + 1:
            print(f"✅ 全{len(target_files)}ファイルを使用します")
            return target_files
        
        elif choice_num == len(target_files) + 2:
            return None
        
        else:
            print("❌ 無効な選択です")
            return None
            
    except ValueError:
        print("❌ 数字を入力してください")
        return None

def show_file_management_menu():
    """ファイル管理メニューを表示"""
    print("\n=== 📁 ファイル管理 ===")
    
    files = get_training_data_files()
    
    print(f"📊 データファイル統計:")
    print(f"   局面アノテーション: {len(files['annotations'])}ファイル")
    print(f"   特徴量データセット: {len(files['features'])}ファイル") 
    print(f"   モデル: {len(files['models'])}ファイル")
    print(f"   コート座標: {len(files['court_coords'])}ファイル")
    
    print(f"\n1: 局面アノテーションファイル管理")
    print(f"2: 特徴量データセットファイル管理")
    print(f"3: モデルファイル管理")
    print(f"4: コート座標ファイル管理")
    print(f"5: ファイル削除")
    print(f"6: 戻る")
    
    while True:
        try:
            choice = input("\n選択してください (1-6): ").strip()
            
            if choice == '1':
                file_path = select_training_data_file('annotations')
                if file_path:
                    if isinstance(file_path, list):
                        print(f"選択された{len(file_path)}ファイルで処理を続行できます")
                    else:
                        print(f"選択されたファイル: {file_path}")
                break
                
            elif choice == '2':
                file_path = select_training_data_file('features')
                if file_path:
                    if isinstance(file_path, list):
                        print(f"選択された{len(file_path)}ファイルで処理を続行できます")
                    else:
                        print(f"選択されたファイル: {file_path}")
                break
                
            elif choice == '3':
                file_path = select_training_data_file('models')
                if file_path:
                    print(f"選択されたファイル: {file_path}")
                break
                
            elif choice == '4':
                file_path = select_training_data_file('court_coords')
                if file_path:
                    print(f"選択されたファイル: {file_path}")
                break
                
            elif choice == '5':
                delete_files_menu()
                break
                
            elif choice == '6':
                break
                
            else:
                print("❌ 無効な選択です。1-6を入力してください。")
                
        except KeyboardInterrupt:
            print("\n操作がキャンセルされました")
            break

def delete_files_menu():
    """ファイル削除メニュー"""
    print("\n=== 🗑️ ファイル削除 ===")
    print("⚠️  注意: 削除されたファイルは復元できません")
    
    files = get_training_data_files()
    all_files = []
    
    # 全ファイルをリストに統合
    for file_type, file_list in files.items():
        for file_path in file_list:
            all_files.append((file_path, file_type))
    
    if not all_files:
        print("❌ 削除可能なファイルがありません")
        return
    
    print(f"\n削除可能ファイル一覧:")
    for i, (file_path, file_type) in enumerate(all_files, 1):
        filename = os.path.basename(file_path)
        file_type_name = {
            'annotations': '局面アノテーション',
            'features': '特徴量データセット',
            'models': 'モデル',
            'court_coords': 'コート座標'
        }.get(file_type, file_type)
        
        print(f"{i}: [{file_type_name}] {filename}")
    
    print(f"{len(all_files) + 1}: 🔙 戻る")
    
    try:
        choice = input(f"\n削除するファイルを選択 (1-{len(all_files) + 1}): ").strip()
        choice_num = int(choice)
        
        if 1 <= choice_num <= len(all_files):
            file_path, file_type = all_files[choice_num - 1]
            filename = os.path.basename(file_path)
            
            print(f"\n🗑️  削除対象: {filename}")
            confirm = input("本当に削除しますか？ (yes/no): ").lower().strip()
            
            if confirm in ['yes', 'y']:
                try:
                    os.remove(file_path)
                    print(f"✅ ファイルを削除しました: {filename}")
                except Exception as e:
                    print(f"❌ 削除エラー: {e}")
            else:
                print("削除をキャンセルしました")
                
        elif choice_num == len(all_files) + 1:
            return
        else:
            print("❌ 無効な選択です")
            
    except ValueError:
        print("❌ 数字を入力してください")

def check_training_data_status():
    """訓練データの状況をチェック（更新版）"""
    print("=== 📊 訓練データ状況チェック ===")
    
    files = get_training_data_files()
    
    # アノテーションファイル
    if not files['annotations']:
        print("❌ 局面アノテーションファイルが見つかりません")
        print("📝 手順1: 局面アノテーション（選択肢1）を実行してデータを作成してください")
        return False
    
    print(f"✅ {len(files['annotations'])}個のアノテーションファイルが見つかりました:")
    for file_path in files['annotations']:
        filename = os.path.basename(file_path)
        try:
            with open(file_path, 'r', encoding='utf-8') as f:
                data = json.load(f)
            phase_count = len(data.get('phase_changes', []))
            duration = data.get('duration_seconds', 0)
            video_name = data.get('video_name', '不明')
            print(f"  📄 {filename}")
            print(f"     📹 動画: {video_name}")
            print(f"     🎯 局面変更数: {phase_count}, ⏱️ 時間: {duration:.1f}秒")
        except Exception as e:
            print(f"  ❌ {filename} (読み込みエラー: {e})")
    
    # 特徴量データセット
    if files['features']:
        print(f"\n✅ {len(files['features'])}個の特徴量データセットが見つかりました:")
        for file_path in files['features']:
            filename = os.path.basename(file_path)
            print(f"  📊 {filename}")
    
    # モデルファイル
    if files['models']:
        print(f"\n✅ {len(files['models'])}個のモデルファイルが見つかりました:")
        for file_path in files['models']:
            filename = os.path.basename(file_path)
            print(f"  🤖 {filename}")
    
    # train_lstm_model.pyの存在チェック
    model_script = "train_lstm_model.py"
    if os.path.exists(model_script):
        print(f"\n✅ {model_script} が見つかりました")
        print("🚀 実行準備完了！")
    else:
        print(f"\n❌ {model_script} が見つかりません")
        print("📝 train_lstm_model.pyファイルを作成する必要があります")
    
    return len(files['annotations']) > 0

def show_training_workflow():
    """訓練までのワークフローを表示"""
    print("\n=== train_phase_model.py実行までの手順 ===")
    print()
    print("📊 必要なデータ: 局面アノテーションファイル")
    print("📍 保存場所: training_data/phase_annotations_*.json")
    print()
    print("🔄 ワークフロー:")
    print("1️⃣ 動画を用意する")
    print("   - data/raw/ フォルダに.mp4ファイルを配置")
    print("   - または手動でパスを指定")
    print()
    print("2️⃣ 局面アノテーションを実行")
    print("   - このツールで「1: 局面アノテーション」を選択")
    print("   - 動画を再生しながら数字キー1-8で局面をラベリング")
    print("   - 最低10-20分程度の動画データを推奨")
    print()
    print("3️⃣ train_phase_model.pyを実行")
    print("   - python train_phase_model.py")
    print("   - 局面分類モデルが自動的に訓練される")
    print()
    print("💡 ヒント:")
    print("   - 複数の動画でアノテーションすると精度向上")
    print("   - 異なる試合・コート・カメラアングルのデータがあると良い")
    print("   - 最低でも各局面が30秒以上含まれるようにアノテーション")

def main():
    """メイン関数 - 局面アノテーションツールを実行"""
    print("=== テニス局面アノテーションツール ===")
    print()
    print("🎯 主な用途: train_lstm_model.py用のデータ作成")
    print("🏟️  新機能: 局面アノテーション中にコート座標も設定可能")
    print()
    print("1: 局面アノテーション")
    print("   - 動画の局面ラベリング")
    print("   - train_lstm_model.py用データ作成")
    print("   - コート座標設定（オプション）")
    print()
    print("2: コート座標設定のみ")
    print("3: 訓練データ状況チェック")
    print("4: train_lstm_model.py実行手順")
    print("5: 📁 ファイル管理")
    print("6: 終了")
    
    while True:
        try:
            choice = input("\n選択してください (1-6): ").strip()
            
            if choice == '1':
                # 局面アノテーション
                video_path = select_video_file()
                if video_path:
                    annotator = PhaseAnnotator()
                    success = annotator.annotate_video(video_path)
                    if success:
                        print("✅ 局面アノテーションが完了しました")
                        print("💾 データがtraining_data/フォルダに保存されました")
                        if annotator.court_coordinates:
                            print("🏟️  コート座標も一緒に保存されました")
                        print("🚀 train_lstm_model.pyを実行する準備ができました！")
                    else:
                        print("❌ 局面アノテーションがキャンセルされました")
                break
                
            elif choice == '2':
                # コート座標設定のみ
                print("📍 独立したコート座標設定ツールを起動します...")
                print("court_calibrator.py を実行してください")
                print("または以下のコマンドを実行:")
                print("python court_calibrator.py")
                break
            
            elif choice == '3':
                # 訓練データ状況チェック
                check_training_data_status()
                continue
                
            elif choice == '4':
                # 実行手順表示
                show_training_workflow()
                continue
                
            elif choice == '5':
                # ファイル管理
                show_file_management_menu()
                continue
                
            elif choice == '6':
                print("終了します")
                break
                
            else:
                print("無効な選択です。1-6を入力してください。")
                
        except KeyboardInterrupt:
            print("\n\n操作がキャンセルされました")
            break
        except Exception as e:
            print(f"エラーが発生しました: {e}")
            break

if __name__ == "__main__":
    main()