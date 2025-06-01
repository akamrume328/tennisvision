import cv2
import json
import os
import numpy as np
from datetime import datetime
from pathlib import Path
from typing import Dict, Tuple, Optional

class CourtCalibrator:
    """
    テニスコート座標設定ツール
    
    機能:
    - 動画フレームでのコート座標設定
    - フレーム間の移動・検索
    - 座標データの保存・読み込み
    - 座標の可視化・検証
    """
    
    def __init__(self):
        self.court_points = {}
        self.point_names = [
            "top_left_corner",      # 左上角
            "top_right_corner",     # 右上角
            "bottom_left_corner",   # 左下角
            "bottom_right_corner",  # 右下角
            "net_left_ground",      # ネット左端（地面）
            "net_right_ground"      # ネット右端（地面）
        ]
        self.current_point_index = 0
        self.calibration_complete = False
        self.temp_frame = None
        self.video_cap = None
        self.current_frame_number = 0
        self.total_frames = 0
        
        print("コート座標設定ツールを初期化しました")
        print("設定する座標点:")
        for i, name in enumerate(self.point_names):
            print(f"  {i+1}. {name}")
        
    def set_video_source(self, video_cap):
        """動画ソースを設定"""
        self.video_cap = video_cap
        if video_cap:
            self.total_frames = int(video_cap.get(cv2.CAP_PROP_FRAME_COUNT))
            self.current_frame_number = int(video_cap.get(cv2.CAP_PROP_POS_FRAMES))
            fps = video_cap.get(cv2.CAP_PROP_FPS)
            duration = self.total_frames / fps if fps > 0 else 0
            print(f"動画情報: {self.total_frames}フレーム, {fps:.1f}FPS, {duration:.1f}秒")
    
    def load_frame(self, frame_number: int) -> bool:
        """指定されたフレーム番号のフレームを読み込み"""
        if self.video_cap is None:
            return False
        
        frame_number = max(0, min(frame_number, self.total_frames - 1))
        self.video_cap.set(cv2.CAP_PROP_POS_FRAMES, frame_number)
        ret, frame = self.video_cap.read()
        
        if ret:
            self.temp_frame = frame.copy()
            self.current_frame_number = frame_number
            return True
        return False
    
    def mouse_callback(self, event, x, y, flags, param):
        """マウスクリック時のコールバック"""
        if event == cv2.EVENT_LBUTTONDOWN and not self.calibration_complete:
            point_name = self.point_names[self.current_point_index]
            self.court_points[point_name] = [x, y]  # リスト形式で保存（JSON互換）
            print(f"✅ {point_name}: ({x}, {y})")
            
            self.current_point_index += 1
            
            if self.current_point_index >= len(self.point_names):
                self.calibration_complete = True
                print("🎉 コート座標設定完了！")
                print("設定された座標:")
                for name, point in self.court_points.items():
                    print(f"  {name}: {point}")
            
            self.update_display_frame()
    
    def update_display_frame(self):
        """表示フレームを更新"""
        if self.temp_frame is None:
            return
            
        display_frame = self.temp_frame.copy()
        
        # 設定済みの点を描画
        for i, point_name in enumerate(self.point_names[:self.current_point_index]):
            point = tuple(self.court_points[point_name])
            cv2.circle(display_frame, point, 8, (0, 255, 0), -1)
            cv2.putText(display_frame, f"{i+1}:{point_name[:12]}", 
                       (point[0]+10, point[1]-10), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
        
        # コート形状を描画（4点以上設定されている場合）
        if len(self.court_points) >= 4:
            self.draw_court_shape(display_frame)
        
        # フレーム情報を表示
        cv2.putText(display_frame, f"Frame: {self.current_frame_number}/{self.total_frames}", 
                   (10, display_frame.shape[0] - 10), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
        
        # 進捗表示
        progress = self.current_point_index / len(self.point_names)
        cv2.putText(display_frame, f"Progress: {self.current_point_index}/{len(self.point_names)} ({progress:.0%})", 
                   (10, display_frame.shape[0] - 40), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
        
        # 現在設定中の点の説明を表示
        if not self.calibration_complete:
            current_name = self.point_names[self.current_point_index]
            instruction = f"点 {self.current_point_index+1}/6: {current_name} をクリック"
            cv2.putText(display_frame, instruction, (10, 30), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 255), 2)
            
            # フレーム変更のヘルプを表示
            help_texts = [
                "フレーム変更: A/D (10フレーム), S/W (1フレーム)",
                "J/L (100フレーム), HOME/END (最初/最後)",
                "数字+Enter: 指定フレームへジャンプ",
                "Enter: 完了, R: リセット, ESC: キャンセル"
            ]
            for i, text in enumerate(help_texts):
                cv2.putText(display_frame, text, (10, 70 + i*25), 
                           cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 0), 2)
        else:
            cv2.putText(display_frame, "設定完了！Enterで続行、Rでリセット", (10, 30), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)
        
        cv2.imshow('Court Calibration', display_frame)
    
    def draw_court_shape(self, frame):
        """設定されたコート座標でコート形状を描画"""
        try:
            # コート四隅を線で結ぶ
            corners = ["top_left_corner", "top_right_corner", "bottom_right_corner", "bottom_left_corner"]
            points = []
            
            for corner in corners:
                if corner in self.court_points:
                    points.append(tuple(self.court_points[corner]))
            
            if len(points) == 4:
                pts = np.array(points, np.int32)
                cv2.polylines(frame, [pts], True, (255, 255, 0), 2)
            
            # ネットライン
            if ("net_left_ground" in self.court_points and 
                "net_right_ground" in self.court_points):
                net_left = tuple(self.court_points["net_left_ground"])
                net_right = tuple(self.court_points["net_right_ground"])
                cv2.line(frame, net_left, net_right, (0, 255, 255), 3)
                
        except Exception as e:
            print(f"コート形状描画エラー: {e}")
    
    def calibrate(self, first_frame, video_cap=None) -> bool:
        """コート座標を設定"""
        self.temp_frame = first_frame.copy()
        self.set_video_source(video_cap)
        
        print("\n=== コート座標設定開始 ===")
        print("以下の順序で6点をクリックしてください：")
        for i, name in enumerate(self.point_names):
            print(f"{i+1}. {name}")
        
        print("\n🎮 キー操作:")
        print("📹 フレーム移動:")
        print("  A/D: 10フレーム戻る/進む")
        print("  S/W: 1フレーム戻る/進む") 
        print("  J/L: 100フレーム戻る/進む")
        print("  HOME: 最初のフレーム")
        print("  END: 最後のフレーム")
        print("  数字キー + Enter: 指定フレームにジャンプ")
        print("⚙️  設定:")
        print("  Enter: 完了")
        print("  R: リセット")
        print("  ESC: キャンセル")
        
        cv2.namedWindow('Court Calibration', cv2.WINDOW_AUTOSIZE)
        cv2.setMouseCallback('Court Calibration', self.mouse_callback)
        
        self.update_display_frame()
        
        frame_input = ""  # フレーム番号入力用
        
        while True:
            key = cv2.waitKey(1) & 0xFF
            
            if key == 13:  # Enter
                if frame_input:  # フレーム番号が入力されている場合
                    try:
                        target_frame = int(frame_input)
                        if self.load_frame(target_frame):
                            self.update_display_frame()
                            print(f"📍 フレーム {target_frame} に移動しました")
                        else:
                            print(f"❌ フレーム {target_frame} の読み込みに失敗しました")
                        frame_input = ""
                    except ValueError:
                        print("❌ 無効なフレーム番号です")
                        frame_input = ""
                elif self.calibration_complete:
                    if self.validate_coordinates():
                        cv2.destroyWindow('Court Calibration')
                        return True
                    else:
                        print("⚠️  座標の検証に失敗しました。設定を確認してください。")
                else:
                    print("⚠️  まだ設定が完了していません")
            
            elif key == 27:  # ESC
                print("❌ 設定がキャンセルされました")
                cv2.destroyWindow('Court Calibration')
                return False
            
            elif key == ord('r') or key == ord('R'):  # Reset
                self.reset()
                self.update_display_frame()
                print("🔄 設定をリセットしました")
                frame_input = ""
            
            # フレーム移動のキー処理
            elif key == ord('a') or key == ord('A'):  # 10フレーム戻る
                new_frame = max(0, self.current_frame_number - 10)
                if self.load_frame(new_frame):
                    self.update_display_frame()
                frame_input = ""
            
            elif key == ord('d') or key == ord('D'):  # 10フレーム進む
                new_frame = min(self.total_frames - 1, self.current_frame_number + 10)
                if self.load_frame(new_frame):
                    self.update_display_frame()
                frame_input = ""
            
            elif key == ord('s') or key == ord('S'):  # 1フレーム戻る
                new_frame = max(0, self.current_frame_number - 1)
                if self.load_frame(new_frame):
                    self.update_display_frame()
                frame_input = ""
            
            elif key == ord('w') or key == ord('W'):  # 1フレーム進む
                new_frame = min(self.total_frames - 1, self.current_frame_number + 1)
                if self.load_frame(new_frame):
                    self.update_display_frame()
                frame_input = ""
            
            elif key == ord('j') or key == ord('J'):  # 100フレーム戻る
                new_frame = max(0, self.current_frame_number - 100)
                if self.load_frame(new_frame):
                    self.update_display_frame()
                frame_input = ""
            
            elif key == ord('l') or key == ord('L'):  # 100フレーム進む
                new_frame = min(self.total_frames - 1, self.current_frame_number + 100)
                if self.load_frame(new_frame):
                    self.update_display_frame()
                frame_input = ""
            
            elif key == 2:  # HOME key
                if self.load_frame(0):
                    self.update_display_frame()
                    print("📍 最初のフレームに移動しました")
                frame_input = ""
            
            elif key == 3:  # END key
                if self.load_frame(self.total_frames - 1):
                    self.update_display_frame()
                    print("📍 最後のフレームに移動しました")
                frame_input = ""
            
            # 数字キーの処理（フレーム番号入力）
            elif key >= ord('0') and key <= ord('9'):
                frame_input += chr(key)
                print(f"📝 フレーム番号入力中: {frame_input} (Enterで移動)")
            
            elif key == 8:  # Backspace
                if frame_input:
                    frame_input = frame_input[:-1]
                    if frame_input:
                        print(f"📝 フレーム番号入力中: {frame_input}")
                    else:
                        print("🗑️  フレーム番号入力をクリアしました")

    def validate_coordinates(self) -> bool:
        """設定された座標の妥当性をチェック"""
        if len(self.court_points) != len(self.point_names):
            print(f"❌ 座標数が不正です: {len(self.court_points)}/{len(self.point_names)}")
            return False
        
        # 座標が画面内にあるかチェック
        if self.temp_frame is not None:
            height, width = self.temp_frame.shape[:2]
            for name, point in self.court_points.items():
                x, y = point
                if not (0 <= x < width and 0 <= y < height):
                    print(f"❌ {name} の座標が画面外です: ({x}, {y})")
                    return False
        
        # コート形状の妥当性をチェック
        if not self.validate_court_geometry():
            return False
        
        print("✅ 座標の検証に成功しました")
        return True
    
    def validate_court_geometry(self) -> bool:
        """コート形状の幾何学的妥当性をチェック"""
        try:
            # 四隅の座標を取得
            corners = ["top_left_corner", "top_right_corner", "bottom_right_corner", "bottom_left_corner"]
            points = []
            
            for corner in corners:
                if corner in self.court_points:
                    points.append(self.court_points[corner])
                else:
                    print(f"❌ {corner} が設定されていません")
                    return False
            
            # 基本的な形状チェック
            # 1. 上辺が下辺より上にある
            if points[0][1] >= points[3][1] or points[1][1] >= points[2][1]:
                print("⚠️  警告: 上辺が下辺より下にあります")
            
            # 2. 左辺が右辺より左にある
            if points[0][0] >= points[1][0] or points[3][0] >= points[2][0]:
                print("⚠️  警告: 左辺が右辺より右にあります")
            
            # 3. ネット位置の妥当性
            if ("net_left_ground" in self.court_points and 
                "net_right_ground" in self.court_points):
                net_left = self.court_points["net_left_ground"]
                net_right = self.court_points["net_right_ground"]
                
                # ネットが左右逆でないかチェック
                if net_left[0] >= net_right[0]:
                    print("⚠️  警告: ネットの左右が逆になっている可能性があります")
            
            print("✅ コート形状の検証に成功しました")
            return True
            
        except Exception as e:
            print(f"❌ コート形状の検証中にエラー: {e}")
            return False

    def reset(self):
        """設定をリセット"""
        self.court_points = {}
        self.current_point_index = 0
        self.calibration_complete = False
        print("🔄 全ての設定をリセットしました")
    
    def save_to_file(self, filepath: str) -> bool:
        """座標をファイルに保存"""
        try:
            # ディレクトリを作成
            os.makedirs(os.path.dirname(filepath), exist_ok=True)
            
            # メタデータを追加
            save_data = {
                **self.court_points,
                "_metadata": {
                    "creation_time": datetime.now().isoformat(),
                    "frame_number": self.current_frame_number,
                    "total_frames": self.total_frames,
                    "coordinate_count": len(self.court_points),
                    "point_names": self.point_names
                }
            }
            
            with open(filepath, 'w', encoding='utf-8') as f:
                json.dump(save_data, f, indent=2, ensure_ascii=False)
            print(f"💾 コート座標を保存しました: {filepath}")
            return True
            
        except Exception as e:
            print(f"❌ 保存エラー: {e}")
            return False
    
    def load_from_file(self, filepath: str) -> bool:
        """座標をファイルから読み込み"""
        if not os.path.exists(filepath):
            print(f"❌ ファイルが見つかりません: {filepath}")
            return False
            
        try:
            with open(filepath, 'r', encoding='utf-8') as f:
                data = json.load(f)
            
            # メタデータを除外して座標のみ取得
            self.court_points = {k: v for k, v in data.items() if not k.startswith('_')}
            
            # 設定完了状態をチェック
            self.calibration_complete = len(self.court_points) == len(self.point_names)
            self.current_point_index = len(self.court_points)
            
            print(f"📁 コート座標を読み込みました: {filepath}")
            print(f"読み込み座標数: {len(self.court_points)}/{len(self.point_names)}")
            
            for name, point in self.court_points.items():
                print(f"  {name}: {point}")
                
            return True
            
        except Exception as e:
            print(f"❌ 読み込みエラー: {e}")
            return False
    
    def get_coordinates(self) -> Dict:
        """設定された座標を取得"""
        return self.court_points.copy()
    
    def set_coordinates(self, coordinates: Dict):
        """座標を直接設定"""
        self.court_points = coordinates.copy()
        self.current_point_index = len(self.court_points)
        self.calibration_complete = len(self.court_points) == len(self.point_names)

def get_video_files(data_dir="../data/raw"):
    """動画ファイル一覧を取得"""
    print(f"動画ファイルを検索中: {data_dir}")
    
    abs_data_dir = os.path.abspath(data_dir)
    video_extensions = ['.mp4', '.avi', '.mov', '.mkv', '.wmv', '.flv', '.webm']
    video_files = []
    
    if os.path.exists(abs_data_dir):
        try:
            files = os.listdir(abs_data_dir)
            for file in files:
                if any(file.lower().endswith(ext) for ext in video_extensions):
                    video_path = os.path.join(abs_data_dir, file)
                    video_files.append(video_path)
        except Exception as e:
            print(f"ディレクトリ読み取りエラー: {e}")
    else:
        # 代替パスを検索
        alternative_paths = [
            "data/raw",
            "./data/raw", 
            "../../data/raw",
            os.path.join(os.getcwd(), "data", "raw")
        ]
        
        for alt_path in alternative_paths:
            abs_alt_path = os.path.abspath(alt_path)
            if os.path.exists(abs_alt_path):
                try:
                    files = os.listdir(abs_alt_path)
                    for file in files:
                        if any(file.lower().endswith(ext) for ext in video_extensions):
                            video_path = os.path.join(abs_alt_path, file)
                            video_files.append(video_path)
                    if video_files:
                        break
                except Exception as e:
                    print(f"代替パス読み取りエラー: {e}")
    
    return video_files

def select_video_file():
    """動画ファイルを選択"""
    video_files = get_video_files()
    
    if not video_files:
        print("\n❌ 動画ファイルが見つかりませんでした。")
        print("以下の場所に動画ファイルを配置してください:")
        print("- ../data/raw フォルダ")
        print("- data/raw フォルダ")
        
        # 手動入力
        manual_input = input("\n手動でファイルパスを入力しますか？ (y/n): ").lower().strip()
        if manual_input == 'y':
            file_path = input("動画ファイルの完全パスを入力してください: ").strip()
            if os.path.exists(file_path):
                return file_path
            else:
                print(f"❌ ファイルが見つかりません: {file_path}")
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
            print("❌ 無効な選択です")
            return None
    except ValueError:
        print("❌ 数字を入力してください")
        return None

def main():
    """メイン関数 - コート座標設定ツールを実行"""
    print("=== テニスコート座標設定ツール ===")
    print("🏟️  動画内のテニスコート座標を設定します")
    print()
    
    print("1: 新規コート座標設定")
    print("2: 既存座標ファイルの確認・編集")
    print("3: 座標ファイル一覧表示")
    print("4: 終了")
    
    while True:
        try:
            choice = input("\n選択してください (1-4): ").strip()
            
            if choice == '1':
                # 新規設定
                video_path = select_video_file()
                if video_path:
                    cap = cv2.VideoCapture(video_path)
                    if cap.isOpened():
                        ret, first_frame = cap.read()
                        if ret:
                            calibrator = CourtCalibrator()
                            
                            if calibrator.calibrate(first_frame, cap):
                                # 保存先を決定
                                video_name = os.path.splitext(os.path.basename(video_path))[0]
                                output_dir = "training_data"
                                coord_file = os.path.join(output_dir, f"court_coords_{video_name}.json")
                                
                                if calibrator.save_to_file(coord_file):
                                    print("✅ コート座標設定が完了しました")
                                    print(f"📁 保存先: {coord_file}")
                                else:
                                    print("❌ 保存に失敗しました")
                            else:
                                print("❌ コート座標設定がキャンセルされました")
                        else:
                            print("❌ 動画の最初のフレームを読み込めませんでした")
                        cap.release()
                    else:
                        print("❌ 動画ファイルを開けませんでした")
                break
                
            elif choice == '2':
                # 既存ファイルの編集
                training_dir = Path("training_data")
                coord_files = list(training_dir.glob("court_coords_*.json"))
                
                if not coord_files:
                    print("❌ 既存の座標ファイルが見つかりません")
                    continue
                
                print("\n=== 既存座標ファイル ===")
                for i, file_path in enumerate(coord_files, 1):
                    print(f"{i}: {file_path.name}")
                
                try:
                    file_choice = int(input(f"\n編集するファイルを選択 (1-{len(coord_files)}): "))
                    if 1 <= file_choice <= len(coord_files):
                        selected_file = coord_files[file_choice - 1]
                        
                        # 対応する動画ファイルを検索
                        video_name = selected_file.stem.replace("court_coords_", "")
                        video_files = get_video_files()
                        matching_video = None
                        
                        for video_path in video_files:
                            if video_name in os.path.basename(video_path):
                                matching_video = video_path
                                break
                        
                        if matching_video:
                            cap = cv2.VideoCapture(matching_video)
                            if cap.isOpened():
                                ret, first_frame = cap.read()
                                if ret:
                                    calibrator = CourtCalibrator()
                                    calibrator.load_from_file(str(selected_file))
                                    
                                    if calibrator.calibrate(first_frame, cap):
                                        calibrator.save_to_file(str(selected_file))
                                        print("✅ 座標ファイルを更新しました")
                                    else:
                                        print("❌ 編集がキャンセルされました")
                                cap.release()
                        else:
                            print("❌ 対応する動画ファイルが見つかりません")
                except ValueError:
                    print("❌ 無効な選択です")
                break
                
            elif choice == '3':
                # ファイル一覧表示
                training_dir = Path("training_data")
                coord_files = list(training_dir.glob("court_coords_*.json"))
                
                if not coord_files:
                    print("❌ 座標ファイルが見つかりません")
                else:
                    print(f"\n=== 座標ファイル一覧 ({len(coord_files)}件) ===")
                    for file_path in coord_files:
                        try:
                            with open(file_path, 'r', encoding='utf-8') as f:
                                data = json.load(f)
                            
                            coord_count = len([k for k in data.keys() if not k.startswith('_')])
                            creation_time = data.get('_metadata', {}).get('creation_time', '不明')
                            
                            print(f"📄 {file_path.name}")
                            print(f"   座標数: {coord_count}/6")
                            print(f"   作成日時: {creation_time}")
                            
                        except Exception as e:
                            print(f"❌ {file_path.name} (読み込みエラー: {e})")
                continue
                
            elif choice == '4':
                print("終了します")
                break
                
            else:
                print("❌ 無効な選択です。1-4を入力してください。")
                
        except KeyboardInterrupt:
            print("\n\n操作がキャンセルされました")
            break
        except Exception as e:
            print(f"❌ エラーが発生しました: {e}")
            break

if __name__ == "__main__":
    main()
