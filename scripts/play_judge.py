import numpy as np
import torch
import torch.nn as nn
from collections import deque
from typing import List, Tuple, Dict, Optional
import cv2
from ultralytics import YOLO
import os
import json
import sys
import time

# ボールトラッカーのインポート
sys.path.append(os.path.join(os.path.dirname(__file__), '..', 'tennis_ball_tracking'))
from balltracking import TennisBallTracker

class TennisPlayJudge:
    def __init__(self, yolo_model_path: str, ball_model_path: str = None, 
                 sequence_length: int = 30, confidence_threshold: float = 0.5):
        """
        テニスプレー判定クラス
        
        Args:
            yolo_model_path: プレイヤー検出用YOLOv8モデルのパス
            ball_model_path: ボール検出用YOLOv8モデルのパス（None の場合はプレイヤーモデルを使用）
            sequence_length: 時系列データの長さ（フレーム数）
            confidence_threshold: 検出の信頼度閾値
        """
        self.yolo_model = YOLO(yolo_model_path)
        self.sequence_length = sequence_length
        self.confidence_threshold = confidence_threshold
        
        # ボールトラッカーの初期化
        ball_model = ball_model_path if ball_model_path else yolo_model_path
        self.ball_tracker = TennisBallTracker(ball_model)
        
        # 時系列データを保存するバッファ
        self.feature_buffer = deque(maxlen=sequence_length)
        
        # LSTMモデルの初期化
        self.lstm_model = self._build_lstm_model()
        
        # 前フレームの情報
        self.prev_player_positions = {}
        
        # タイムスタンプベースのデータ収集
        self.timestamp_mode = False
        self.frame_features = []  # 全フレームの特徴量を保存
        self.frame_timestamps = []  # フレーム時刻を保存
        self.play_periods = []  # プレー期間 [(start_time, end_time), ...]
        self.current_play_start = None
        self.frame_count = 0
        self.fps = 30  # デフォルトFPS
        
        # 従来のリアルタイムデータ収集（互換性のため）
        self.data_collection_mode = False
        self.collected_data = []
        self.current_label = None
    
    def _build_lstm_model(self) -> nn.Module:
        """LSTM判定モデルの構築"""
        class PlayJudgeLSTM(nn.Module):
            def __init__(self, input_size=12, hidden_size=64, num_layers=2, num_classes=2):
                super(PlayJudgeLSTM, self).__init__()
                self.hidden_size = hidden_size
                self.num_layers = num_layers
                
                self.lstm = nn.LSTM(input_size, hidden_size, num_layers, batch_first=True, dropout=0.2)
                self.dropout = nn.Dropout(0.3)
                self.fc1 = nn.Linear(hidden_size, 32)
                self.fc2 = nn.Linear(32, num_classes)
                self.relu = nn.ReLU()
                self.softmax = nn.Softmax(dim=1)
                
            def forward(self, x):
                h0 = torch.zeros(self.num_layers, x.size(0), self.hidden_size)
                c0 = torch.zeros(self.num_layers, x.size(0), self.hidden_size)
                
                out, _ = self.lstm(x, (h0, c0))
                out = self.dropout(out[:, -1, :])  # 最後の出力のみ使用
                out = self.relu(self.fc1(out))
                out = self.fc2(out)
                return self.softmax(out)
        
        return PlayJudgeLSTM()
    
    def extract_features_from_frame(self, frame: np.ndarray) -> Optional[np.ndarray]:
        """
        フレームから特徴量を抽出（ボールトラッカーを使用）
        
        Args:
            frame: 入力フレーム
            
        Returns:
            特徴量ベクトル（12次元）
        """
        # ボールトラッキングを実行
        tracked_frame = self.ball_tracker.process_frame(frame)
        
        # プレイヤー検出実行
        results = self.yolo_model(frame, conf=self.confidence_threshold)
        
        features = np.zeros(12)  # 12次元の特徴量ベクトル
        
        player_front_boxes = []
        player_back_boxes = []
        
        # プレイヤー検出結果の解析
        for result in results:
            boxes = result.boxes
            if boxes is not None:
                for box in boxes:
                    class_id = int(box.cls[0])
                    conf = float(box.conf[0])
                    x1, y1, x2, y2 = box.xyxy[0].cpu().numpy()
                    
                    if class_id == 0:  # player_front
                        player_front_boxes.append([x1, y1, x2, y2, conf])
                    elif class_id == 1:  # player_back
                        player_back_boxes.append([x1, y1, x2, y2, conf])
        
        # ボール情報を取得
        ball_info = self._get_ball_tracking_info()
        
        # 特徴量の計算
        current_features = self._calculate_features(
            player_front_boxes, player_back_boxes, ball_info, frame.shape
        )
        
        return current_features
    
    def _get_ball_tracking_info(self) -> Dict:
        """ボールトラッキング情報を取得"""
        ball_info = {
            'detected': False,
            'position': None,
            'velocity': (0, 0),
            'trajectory_length': 0,
            'movement_score': 0.0
        }
        
        if self.ball_tracker.active_ball is not None:
            ball_data = self.ball_tracker.candidate_balls[self.ball_tracker.active_ball]
            if len(ball_data['position_history']) > 0:
                ball_info['detected'] = True
                ball_info['position'] = ball_data['position_history'][-1]
                ball_info['velocity'] = self.ball_tracker.calculate_velocity(
                    list(ball_data['position_history'])
                )
                ball_info['trajectory_length'] = len(self.ball_tracker.ball_trajectory)
                ball_info['movement_score'] = ball_data['movement_score']
        
        return ball_info
    
    def _calculate_features(self, player_front: List, player_back: List, 
                          ball_info: Dict, frame_shape: Tuple) -> np.ndarray:
        """具体的な特徴量の計算（12次元）"""
        features = np.zeros(12)
        
        # プレイヤー検出数
        features[0] = len(player_front)
        features[1] = len(player_back)
        
        # プレイヤー間距離
        if player_front and player_back:
            front_center = self._get_box_center(player_front[0])
            back_center = self._get_box_center(player_back[0])
            features[2] = np.linalg.norm(np.array(front_center) - np.array(back_center))
            features[2] /= np.sqrt(frame_shape[0]**2 + frame_shape[1]**2)  # 正規化
        
        # ボール関連特徴量
        features[3] = 1.0 if ball_info['detected'] else 0.0  # ボール検出フラグ
        
        if ball_info['detected'] and ball_info['position']:
            ball_pos = ball_info['position']
            features[4] = ball_pos[0] / frame_shape[1]  # x座標正規化
            features[5] = ball_pos[1] / frame_shape[0]  # y座標正規化
            
            # ボール速度
            velocity = ball_info['velocity']
            features[6] = np.linalg.norm(velocity) / np.sqrt(frame_shape[0]**2 + frame_shape[1]**2)
            
            # 軌跡の長さ
            features[7] = min(ball_info['trajectory_length'] / 30.0, 1.0)  # 正規化
            
            # ボールの動きの活発さ
            features[8] = min(ball_info['movement_score'] / 50.0, 1.0)  # 正規化
        
        # プレイヤーの動き
        if player_front:
            front_center = self._get_box_center(player_front[0])
            if 'front' in self.prev_player_positions:
                movement = np.array(front_center) - np.array(self.prev_player_positions['front'])
                features[9] = np.linalg.norm(movement) / np.sqrt(frame_shape[0]**2 + frame_shape[1]**2)
            self.prev_player_positions['front'] = front_center
        
        if player_back:
            back_center = self._get_box_center(player_back[0])
            if 'back' in self.prev_player_positions:
                movement = np.array(back_center) - np.array(self.prev_player_positions['back'])
                features[10] = np.linalg.norm(movement) / np.sqrt(frame_shape[0]**2 + frame_shape[1]**2)
            self.prev_player_positions['back'] = back_center
        
        # プレイヤーのサイズ（距離の推定）
        if player_front:
            box = player_front[0]
            features[11] = ((box[2] - box[0]) * (box[3] - box[1])) / (frame_shape[0] * frame_shape[1])
        
        return features
    
    def _get_box_center(self, box: List) -> Tuple[float, float]:
        """バウンディングボックスの中心座標を取得"""
        return ((box[0] + box[2]) / 2, (box[1] + box[3]) / 2)
    
    def enable_data_collection(self):
        """教師データ収集モードを有効化"""
        self.data_collection_mode = True
        print("教師データ収集モードが有効になりました")
        print("キー操作:")
        print("  'p' - プレー中としてラベル付け")
        print("  'n' - プレー中でないとしてラベル付け")
        print("  's' - データを保存")
        print("  'q' - 終了")
    
    def enable_timestamp_collection(self, video_fps: float = 30.0):
        """タイムスタンプベースのデータ収集モードを有効化"""
        self.timestamp_mode = True
        self.fps = video_fps
        self.frame_features = []
        self.frame_timestamps = []
        self.play_periods = []
        self.frame_count = 0
        
        print("=== タイムスタンプベースデータ収集モード ===")
        print("キー操作:")
        print("  's' - プレー開始をマーク")
        print("  'e' - プレー終了をマーク")
        print("  'u' - 直前のマークを取り消し")
        print("  'd' - 収集データを保存")
        print("  'q' - 終了")
        print("\n注意: プレー開始('s')とプレー終了('e')を交互に押してください")
        print("現在のステータス: 待機中")
    
    def handle_timestamp_key(self, key: int):
        """タイムスタンプモードでのキー操作処理"""
        if not self.timestamp_mode:
            return
        
        current_time = self.frame_count / self.fps
        
        if key == ord('s'):  # プレー開始
            if self.current_play_start is None:
                self.current_play_start = current_time
                print(f"プレー開始マーク: {current_time:.2f}秒")
                print("現在のステータス: プレー中 (終了は'e'キー)")
            else:
                print("既にプレー開始がマークされています。先に終了('e')をマークしてください。")
        
        elif key == ord('e'):  # プレー終了
            if self.current_play_start is not None:
                play_duration = current_time - self.current_play_start
                if play_duration > 1.0:  # 最小1秒のプレー時間
                    self.play_periods.append((self.current_play_start, current_time))
                    print(f"プレー終了マーク: {current_time:.2f}秒")
                    print(f"プレー時間: {play_duration:.2f}秒")
                    print(f"登録されたプレー期間数: {len(self.play_periods)}")
                    print("現在のステータス: 待機中 (開始は's'キー)")
                else:
                    print(f"プレー時間が短すぎます ({play_duration:.2f}秒 < 1秒)")
                self.current_play_start = None
            else:
                print("プレー開始がマークされていません。先に開始('s')をマークしてください。")
        
        elif key == ord('u'):  # 取り消し
            if self.current_play_start is not None:
                print(f"プレー開始マーク({self.current_play_start:.2f}秒)を取り消しました")
                self.current_play_start = None
                print("現在のステータス: 待機中")
            elif self.play_periods:
                removed_period = self.play_periods.pop()
                print(f"直前のプレー期間({removed_period[0]:.2f}s-{removed_period[1]:.2f}s)を削除しました")
                print(f"残りプレー期間数: {len(self.play_periods)}")
            else:
                print("取り消すものがありません")
        
        elif key == ord('d'):  # データ保存
            self.create_training_data_from_timestamps()
    
    def store_frame_features(self, features: np.ndarray):
        """フレームの特徴量を保存"""
        if self.timestamp_mode:
            self.frame_features.append(features.copy())
            self.frame_timestamps.append(self.frame_count / self.fps)
            self.frame_count += 1
    
    def create_training_data_from_timestamps(self):
        """タイムスタンプからシーケンスデータを作成"""
        if not self.frame_features or not self.play_periods:
            print("特徴量データまたはプレー期間が不足しています")
            return
        
        print(f"\n=== 訓練データ作成中 ===")
        print(f"総フレーム数: {len(self.frame_features)}")
        print(f"プレー期間数: {len(self.play_periods)}")
        
        training_sequences = []
        
        # 各フレーム位置でシーケンスを作成
        for i in range(self.sequence_length, len(self.frame_features)):
            # 現在のフレーム時刻
            current_time = self.frame_timestamps[i]
            
            # シーケンスの特徴量を取得
            sequence = np.array(self.frame_features[i-self.sequence_length:i])
            
            # ラベルを決定（プレー期間に含まれるかチェック）
            is_playing = self.is_time_in_play_periods(current_time)
            label = 1 if is_playing else 0
            
            training_sequences.append((sequence, label))
        
        # データの統計を表示
        play_count = sum(1 for _, label in training_sequences if label == 1)
        no_play_count = len(training_sequences) - play_count
        
        print(f"作成されたシーケンス数: {len(training_sequences)}")
        print(f"  プレー中: {play_count} ({play_count/len(training_sequences)*100:.1f}%)")
        print(f"  非プレー: {no_play_count} ({no_play_count/len(training_sequences)*100:.1f}%)")
        
        # バランス調整の提案
        if play_count < no_play_count * 0.3:
            print("警告: プレーデータが少なすぎます。より多くのプレー期間をマークしてください。")
        elif play_count > no_play_count * 3:
            print("警告: 非プレーデータが少なすぎます。より多くの非プレー期間が必要です。")
        
        # ファイル名の入力
        filename = input("保存ファイル名 (デフォルト: timestamp_training_data.json): ").strip()
        if not filename:
            filename = "timestamp_training_data.json"
        
        # メタデータと一緒に保存
        self.save_timestamp_training_data(training_sequences, filename)
    
    def is_time_in_play_periods(self, time: float) -> bool:
        """指定時刻がプレー期間に含まれるかチェック"""
        for start, end in self.play_periods:
            if start <= time <= end:
                return True
        return False
    
    def save_timestamp_training_data(self, training_sequences: List[Tuple[np.ndarray, int]], 
                                   filename: str):
        """タイムスタンプベースの訓練データを保存"""
        # データをシリアライズ可能な形式に変換
        data_to_save = {
            'metadata': {
                'total_frames': len(self.frame_features),
                'sequence_length': self.sequence_length,
                'fps': self.fps,
                'total_sequences': len(training_sequences),
                'play_periods': self.play_periods,
                'play_count': sum(1 for _, label in training_sequences if label == 1),
                'no_play_count': sum(1 for _, label in training_sequences if label == 0),
                'creation_time': time.strftime('%Y-%m-%d %H:%M:%S')
            },
            'sequences': []
        }
        
        for sequence, label in training_sequences:
            data_to_save['sequences'].append({
                'sequence': sequence.tolist(),
                'label': label
            })
        
        # ファイルに保存
        save_path = os.path.join(os.path.dirname(__file__), '..', 'data', 'training', filename)
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        
        with open(save_path, 'w') as f:
            json.dump(data_to_save, f, indent=2)
        
        print(f"\n訓練データを保存しました: {save_path}")
        print(f"メタデータも含めて保存されました")
        
        # プレー期間の詳細をログファイルに保存
        log_filename = filename.replace('.json', '_log.txt')
        log_path = os.path.join(os.path.dirname(save_path), log_filename)
        
        with open(log_path, 'w', encoding='utf-8') as f:
            f.write("=== Tennis Play Judge - データ収集ログ ===\n")
            f.write(f"作成日時: {time.strftime('%Y-%m-%d %H:%M:%S')}\n")
            f.write(f"総フレーム数: {len(self.frame_features)}\n")
            f.write(f"FPS: {self.fps}\n")
            f.write(f"総動画時間: {len(self.frame_features)/self.fps:.2f}秒\n\n")
            
            f.write("=== プレー期間 ===\n")
            total_play_time = 0
            for i, (start, end) in enumerate(self.play_periods, 1):
                duration = end - start
                total_play_time += duration
                f.write(f"{i:2d}. {start:7.2f}s - {end:7.2f}s (時間: {duration:5.2f}s)\n")
            
            f.write(f"\n総プレー時間: {total_play_time:.2f}秒\n")
            f.write(f"プレー時間率: {total_play_time/(len(self.frame_features)/self.fps)*100:.1f}%\n")
        
        print(f"ログファイルも保存しました: {log_path}")
    
    def load_training_data(self, filename: str = "training_data.json") -> List[Tuple[np.ndarray, int]]:
        """保存された教師データを読み込み（タイムスタンプ形式対応）"""
        load_path = os.path.join(os.path.dirname(__file__), '..', 'data', 'training', filename)
        
        if not os.path.exists(load_path):
            print(f"教師データファイルが見つかりません: {load_path}")
            return []
        
        with open(load_path, 'r') as f:
            data = json.load(f)
        
        # 新形式（メタデータ付き）かチェック
        if 'metadata' in data and 'sequences' in data:
            print("=== タイムスタンプ形式のデータを読み込み中 ===")
            metadata = data['metadata']
            print(f"作成日時: {metadata.get('creation_time', 'Unknown')}")
            print(f"総シーケンス数: {metadata.get('total_sequences', 0)}")
            print(f"プレー中: {metadata.get('play_count', 0)}")
            print(f"非プレー: {metadata.get('no_play_count', 0)}")
            print(f"FPS: {metadata.get('fps', 'Unknown')}")
            print(f"プレー期間数: {len(metadata.get('play_periods', []))}")
            
            sequences_data = data['sequences']
        else:
            print("=== 従来形式のデータを読み込み中 ===")
            sequences_data = data
        
        training_data = []
        for item in sequences_data:
            sequence = np.array(item['sequence'])
            label = item['label']
            training_data.append((sequence, label))
        
        print(f"教師データを読み込みました: {len(training_data)}サンプル")
        return training_data

    def judge_play_status(self, frame: np.ndarray) -> Tuple[bool, float]:
        """
        フレームからプレー状況を判定
        
        Args:
            frame: 入力フレーム
            
        Returns:
            (is_playing, confidence): プレー中かどうかと信頼度
        """
        # 特徴量抽出
        features = self.extract_features_from_frame(frame)
        if features is None:
            return False, 0.0
        
        # バッファに追加
        self.feature_buffer.append(features)
        
        # タイムスタンプモードの場合、特徴量を保存
        if self.timestamp_mode:
            self.store_frame_features(features)
        
        # 従来のデータ収集モードの場合
        if self.data_collection_mode:
            self.collect_training_sample()
        
        # シーケンス長に達していない場合
        if len(self.feature_buffer) < self.sequence_length:
            return False, 0.0
        
        # LSTM推論
        sequence = np.array(list(self.feature_buffer))
        sequence_tensor = torch.FloatTensor(sequence).unsqueeze(0)
        
        with torch.no_grad():
            self.lstm_model.eval()
            output = self.lstm_model(sequence_tensor)
            probabilities = output[0].numpy()
            
        is_playing = probabilities[1] > probabilities[0]  # クラス1がプレー中
        confidence = float(probabilities[1])
        
        return is_playing, confidence
    
    def train_model(self, training_data: List[Tuple[np.ndarray, int]], 
                   epochs: int = 100, lr: float = 0.001):
        """
        LSTMモデルの訓練
        
        Args:
            training_data: (特徴量シーケンス, ラベル)のリスト
            epochs: エポック数
            lr: 学習率
        """
        criterion = nn.CrossEntropyLoss()
        optimizer = torch.optim.Adam(self.lstm_model.parameters(), lr=lr)
        
        for epoch in range(epochs):
            total_loss = 0
            for sequences, labels in training_data:
                sequences_tensor = torch.FloatTensor(sequences).unsqueeze(0)
                labels_tensor = torch.LongTensor([labels])
                
                optimizer.zero_grad()
                outputs = self.lstm_model(sequences_tensor)
                loss = criterion(outputs, labels_tensor)
                loss.backward()
                optimizer.step()
                
                total_loss += loss.item()
            
            if epoch % 10 == 0:
                print(f'Epoch {epoch}, Loss: {total_loss/len(training_data):.4f}')
    
    def save_model(self, path: str):
        """モデルの保存"""
        torch.save(self.lstm_model.state_dict(), path)
    
    def load_model(self, path: str):
        """モデルの読み込み"""
        self.lstm_model.load_state_dict(torch.load(path))

# 使用例
if __name__ == "__main__":
    print("=== Tennis Play Judge ===")
    print()
    
    # モード選択
    print("実行モードを選択してください:")
    print("1. タイムスタンプデータ収集モード (timestamp_collect)")
    print("2. 従来データ収集モード (collect)")
    print("3. 訓練モード (train)")
    print("4. 推論モード (infer)")
    
    while True:
        try:
            mode_choice = int(input("モードを選択 (1-4): "))
            if mode_choice == 1:
                mode = 'timestamp_collect'
                break
            elif mode_choice == 2:
                mode = 'collect'
                break
            elif mode_choice == 3:
                mode = 'train'
                break
            elif mode_choice == 4:
                mode = 'infer'
                break
            else:
                print("1、2、3、または4を入力してください。")
        except ValueError:
            print("数字を入力してください。")
    
    print(f"選択されたモード: {mode}")
    print()
    
    # モデルパスの入力
    yolo_model = input("プレイヤー検出用YOLOモデルのパス: ").strip()
    if not yolo_model:
        yolo_model = "C:/Users/akama/AppData/Local/Programs/Python/Python310/python_file/projects/tennisvision/models/weights/best_5_25.pt"
        print(f"デフォルトパスを使用: {yolo_model}")
    
    ball_model = input("ボール検出用YOLOモデルのパス (空白でプレイヤーモデルと同じ): ").strip()
    if not ball_model:
        ball_model = None
        print("プレイヤーモデルと同じモデルを使用")
    
    # ビデオソースの入力
    if mode != 'train':
        video_input = input("ビデオファイルパス (0でWebカメラ、空白でデフォルト): ").strip()
        if not video_input:
            video_input = "C:/Users/akama/AppData/Local/Programs/Python/Python310/python_file/projects/tennisvision/data/raw/output4.mp4"
            print(f"デフォルト動画を使用: {video_input}")
        elif video_input == "0":
            video_input = 0
            print("Webカメラを使用")
    
    # LSTMモデルパスの入力（推論モードの場合）
    lstm_model = None
    if mode == 'infer':
        lstm_model_input = input("保存されたLSTMモデルのパス (空白でスキップ): ").strip()
        if lstm_model_input:
            lstm_model = lstm_model_input
            print(f"LSTMモデルを読み込み: {lstm_model}")
        else:
            print("未訓練のLSTMモデルで実行（推論結果は無意味です）")
    
    print("\n=== 初期化中 ===")
    
    # プレー判定モデルの初期化
    try:
        judge = TennisPlayJudge(yolo_model, ball_model)
        print("モデルの初期化が完了しました")
    except Exception as e:
        print(f"エラー: モデルの初期化に失敗しました - {e}")
        exit(1)
    
    if mode == 'timestamp_collect':
        # タイムスタンプベースデータ収集モード
        print("\n=== タイムスタンプベースデータ収集モード ===")
        
        # FPS情報を取得
        cap_temp = cv2.VideoCapture(video_input)
        if cap_temp.isOpened():
            video_fps = cap_temp.get(cv2.CAP_PROP_FPS)
            cap_temp.release()
        else:
            video_fps = 30.0
        
        judge.enable_timestamp_collection(video_fps)
        
    elif mode == 'collect':
        # 従来のデータ収集モード
        print("\n=== 従来データ収集モード ===")
        judge.enable_data_collection()
        
    elif mode == 'train':
        # 訓練モード
        print("\n=== 訓練モード ===")
        training_data = judge.load_training_data()
        if training_data:
            print(f"訓練データ: {len(training_data)}サンプル")
            
            # 訓練パラメータの入力
            try:
                epochs = int(input("エポック数 (デフォルト: 200): ") or "200")
                lr = float(input("学習率 (デフォルト: 0.001): ") or "0.001")
            except ValueError:
                print("デフォルト値を使用: エポック=200, 学習率=0.001")
                epochs = 200
                lr = 0.001
            
            print("モデルを訓練中...")
            judge.train_model(training_data, epochs=epochs, lr=lr)
            
            model_save_path = input("モデル保存パス (デフォルト: tennis_play_judge_model.pth): ").strip()
            if not model_save_path:
                model_save_path = "tennis_play_judge_model.pth"
            
            judge.save_model(model_save_path)
            print(f"訓練完了 - モデルを保存: {model_save_path}")
        else:
            print("訓練データがありません。まずデータ収集を行ってください。")
        exit()
        
    elif mode == 'infer' and lstm_model:
        # 事前訓練済みモデルを読み込み
        try:
            judge.load_model(lstm_model)
            print("訓練済みモデルを読み込みました")
        except Exception as e:
            print(f"警告: モデルの読み込みに失敗しました - {e}")
            print("未訓練のモデルで継続します")
    
    # ビデオ処理（データ収集・推論モード）
    if mode != 'train':
        print(f"\n=== {mode}モード開始 ===")
        try:
            cap = cv2.VideoCapture(video_input)
            
            if not cap.isOpened():
                print("エラー: ビデオソースを開けませんでした")
                exit(1)
            
            print("処理開始... 'q'キーで終了")
            
            while True:
                ret, frame = cap.read()
                if not ret:
                    print("ビデオの最後に到達しました")
                    break
                
                # プレー状況判定
                is_playing, confidence = judge.judge_play_status(frame)
                
                # 結果の表示
                status_text = f"Playing: {is_playing}, Confidence: {confidence:.2f}"
                color = (0, 255, 0) if is_playing else (0, 0, 255)
                cv2.putText(frame, status_text, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 
                           1, color, 2)
                
                # タイムスタンプモードの表示
                if judge.timestamp_mode:
                    time_text = f"Time: {judge.frame_count/judge.fps:.2f}s"
                    periods_text = f"Play Periods: {len(judge.play_periods)}"
                    status_text = "Playing" if judge.current_play_start else "Waiting"
                    
                    cv2.putText(frame, time_text, (10, 70), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
                    cv2.putText(frame, periods_text, (10, 100), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
                    cv2.putText(frame, f"Status: {status_text}", (10, 130), cv2.FONT_HERSHEY_SIMPLEX, 0.7, 
                               (0, 255, 255) if judge.current_play_start else (255, 255, 255), 2)
                
                # 従来のデータ収集モードの表示
                elif judge.data_collection_mode:
                    label_text = f"Current Label: {judge.current_label if judge.current_label is not None else 'None'}"
                    sample_text = f"Collected Samples: {len(judge.collected_data)}"
                    cv2.putText(frame, label_text, (10, 70), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 0), 2)
                    cv2.putText(frame, sample_text, (10, 100), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 0), 2)
                
                cv2.imshow('Tennis Play Judge', frame)
                
                key = cv2.waitKey(1) & 0xFF
                if key == ord('q'):
                    break
                
                # キー処理
                if judge.timestamp_mode:
                    judge.handle_timestamp_key(key)
                elif judge.data_collection_mode:
                    judge.handle_data_collection_key(key)
        
        except Exception as e:
            print(f"エラー: {e}")
        finally:
            if 'cap' in locals():
                cap.release()
            cv2.destroyAllWindows()
            
        # データ保存確認
        if mode == 'collect' and len(judge.collected_data) > 0:
            save_choice = input(f"\n{len(judge.collected_data)}個のサンプルを保存しますか？ (y/n): ").strip().lower()
            if save_choice == 'y':
                filename = input("保存ファイル名 (デフォルト: training_data.json): ").strip()
                if not filename:
                    filename = "training_data.json"
                judge.save_collected_data(filename)
        elif mode == 'timestamp_collect':
            if judge.play_periods:
                print(f"\n記録されたプレー期間: {len(judge.play_periods)}個")
                for i, (start, end) in enumerate(judge.play_periods, 1):
                    print(f"  {i}. {start:.2f}s - {end:.2f}s ({end-start:.2f}s)")
                save_choice = input("訓練データを作成して保存しますか？ (y/n): ").strip().lower()
                if save_choice == 'y':
                    judge.create_training_data_from_timestamps()
            else:
                print("プレー期間が記録されていません")
    
    print("処理完了")
