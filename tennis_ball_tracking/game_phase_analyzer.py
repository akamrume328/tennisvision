import cv2
import numpy as np
import math
from collections import deque
from typing import List, Tuple, Optional, Dict, Any
from datetime import datetime, timedelta
from enum import Enum
import csv
import os

class GamePhase(Enum):
    """ゲームの局面を定義"""
    POINT_INTERVAL = "point_interval"      # ポイント間
    SERVE_PREP = "serve_preparation"       # サーブ準備
    SERVE_FRONT = "serve_front"           # 手前プレイヤーのサーブ
    SERVE_BACK = "serve_back"             # 奥プレイヤーのサーブ
    RALLY = "rally"                       # ラリー中
    POINT_END = "point_end"               # ポイント終了
    CHANGEOVER = "changeover"             # チェンジオーバー
    UNKNOWN = "unknown"                   # 判定不能

class CourtSide(Enum):
    """コートサイドを定義"""
    DEUCE = "deuce"     # デュースサイド
    AD = "ad"           # アドサイド

class TennisGamePhaseAnalyzer:
    def __init__(self, court_width: int = 1920, court_height: int = 1080):
        """
        テニス局面判断システムの初期化
        
        Args:
            court_width: コート画像の幅
            court_height: コート画像の高さ
        """
        self.court_width = court_width
        self.court_height = court_height
        
        # コート座標（初期値、後でGUIで設定）
        self.court_points = {
            'front_left': None,      # 手前左角
            'front_right': None,     # 手前右角
            'back_left': None,       # 奥左角
            'back_right': None,      # 奥右角
            'net_left': None,        # ネット左端（地面）
            'net_right': None        # ネット右端（地面）
        }
        
        # コート領域の定義（GUIで設定後に更新）
        self.front_court_y = court_height * 0.6  # 手前コートのY座標閾値
        self.back_court_y = court_height * 0.4   # 奥コートのY座標閾値
        self.center_x = court_width // 2         # コート中央のX座標
        
        # サイドライン（デュース/アドサイド）
        self.deuce_side_x = court_width * 0.3    # デュースサイド境界
        self.ad_side_x = court_width * 0.7       # アドサイド境界
        
        # 現在の状態
        self.current_phase = GamePhase.UNKNOWN
        self.previous_phase = GamePhase.UNKNOWN
        self.phase_start_time = datetime.now()
        self.phase_duration = timedelta()
        
        # データ履歴管理
        self.ball_history = deque(maxlen=30)     # 30フレーム分のボール履歴
        self.player_history = deque(maxlen=30)   # 30フレーム分のプレイヤー履歴
        self.phase_history = []                  # 局面変更履歴
        
        # 判定パラメータ
        self.min_phase_duration = 1.0           # 最小局面継続時間（秒）
        self.player_movement_threshold = 20     # プレイヤー移動判定閾値
        self.ball_speed_threshold = 5           # ボール速度判定閾値
        self.serve_detection_frames = 15        # サーブ検出に必要なフレーム数
        self.rally_min_exchanges = 2            # ラリー判定に必要な最小交換回数
        
        # 状態カウンタ
        self.stationary_count = 0               # 静止状態カウンタ
        self.ball_missing_count = 0             # ボール未検出カウンタ
        self.serve_candidate_count = 0          # サーブ候補カウンタ
        
        # サーブ判定用
        self.current_server = None              # 現在のサーバー（'front'/'back'）
        self.serve_side = CourtSide.DEUCE      # サーブサイド
        
        # 時系列記録
        self.analysis_data = []
        self.frame_number = 0
    
    def analyze_frame(self, tracking_data: Dict[str, Any], 
                     player_detections: List[Tuple[int, int, int, int, int, float]] = None) -> Dict[str, Any]:
        """
        フレームごとの局面分析
        
        Args:
            tracking_data: ボールトラッキングデータ
            player_detections: プレイヤー検出データ
            
        Returns:
            分析結果辞書
        """
        self.frame_number += 1
        current_time = datetime.now()
        
        # データを履歴に追加
        self.ball_history.append(tracking_data)
        if player_detections:
            self.player_history.append(player_detections)
        
        # 局面判定を実行
        new_phase = self._determine_phase()
        
        # 局面変更の処理
        if new_phase != self.current_phase:
            self._handle_phase_change(new_phase, current_time)
        
        # 分析結果をまとめる
        analysis_result = {
            'frame_number': self.frame_number,
            'timestamp': current_time.isoformat(),
            'current_phase': self.current_phase.value,
            'phase_duration': self.phase_duration.total_seconds(),
            'server': self.current_server,
            'serve_side': self.serve_side.value,
            'ball_position': tracking_data.get('ball_position'),
            'ball_velocity': tracking_data.get('ball_velocity'),
            'ball_in_front_court': self._is_ball_in_front_court(tracking_data),
            'ball_in_back_court': self._is_ball_in_back_court(tracking_data),
            'players_moving': self._are_players_moving(),
            'ball_missing_count': self.ball_missing_count,
            'stationary_count': self.stationary_count,
            'confidence': self._calculate_confidence()
        }
        
        # プレイヤー位置情報を追加
        if player_detections:
            analysis_result.update(self._analyze_player_positions(player_detections))
        
        self.analysis_data.append(analysis_result)
        return analysis_result
    
    def _determine_phase(self) -> GamePhase:
        """現在の局面を判定"""
        
        # データが不十分な場合
        if len(self.ball_history) < 5:
            return GamePhase.UNKNOWN
        
        # 最新のボールデータ
        latest_ball = self.ball_history[-1]
        ball_active = latest_ball.get('ball_position') is not None
        
        if not ball_active:
            self.ball_missing_count += 1
        else:
            self.ball_missing_count = 0
        
        # プレイヤーの動きを分析
        players_moving = self._are_players_moving()
        
        if not players_moving:
            self.stationary_count += 1
        else:
            self.stationary_count = 0
        
        # 局面判定ロジック
        
        # 1. ポイント終了判定（プレイヤーが長時間静止 + ボール未検出）
        if (self.stationary_count > 30 and self.ball_missing_count > 20):
            return GamePhase.POINT_END
        
        # 2. ポイント間判定（プレイヤーが静止しているが、まだポイント終了ではない）
        if (self.stationary_count > 15 and self.ball_missing_count > 10):
            return GamePhase.POINT_INTERVAL
        
        # 3. サーブ判定
        serve_phase = self._detect_serve()
        if serve_phase != GamePhase.UNKNOWN:
            return serve_phase
        
        # 4. ラリー判定
        if self._is_rally_active():
            return GamePhase.RALLY
        
        # 5. チェンジオーバー判定（長期間の静止状態）
        if self.stationary_count > 180:  # 約6秒間静止
            return GamePhase.CHANGEOVER
        
        # デフォルト
        return self.current_phase if self.current_phase != GamePhase.UNKNOWN else GamePhase.POINT_INTERVAL
    
    def _detect_serve(self) -> GamePhase:
        """サーブ局面を検出"""
        if len(self.ball_history) < self.serve_detection_frames:
            return GamePhase.UNKNOWN
        
        # サーブパターンの検出
        recent_balls = list(self.ball_history)[-self.serve_detection_frames:]
        
        # ボールが手前コートから奥コートへ移動するパターン
        front_to_back_pattern = self._detect_front_to_back_movement(recent_balls)
        back_to_front_pattern = self._detect_back_to_front_movement(recent_balls)
        
        if front_to_back_pattern:
            self.current_server = 'front'
            return GamePhase.SERVE_FRONT
        elif back_to_front_pattern:
            self.current_server = 'back'
            return GamePhase.SERVE_BACK
        
        return GamePhase.UNKNOWN
    
    def _detect_front_to_back_movement(self, ball_data: List[Dict]) -> bool:
        """手前から奥への移動パターンを検出"""
        positions = []
        for data in ball_data:
            pos = data.get('ball_position')
            if pos:
                positions.append(pos)
        
        if len(positions) < 5:
            return False
        
        # Y座標の変化を確認（手前→奥は減少）
        start_y = positions[0][1]
        end_y = positions[-1][1]
        
        # 手前コートから始まって奥コートで終わる
        return (start_y > self.front_court_y and 
                end_y < self.back_court_y and
                (start_y - end_y) > 100)  # 十分な移動距離
    
    def _detect_back_to_front_movement(self, ball_data: List[Dict]) -> bool:
        """奥から手前への移動パターンを検出"""
        positions = []
        for data in ball_data:
            pos = data.get('ball_position')
            if pos:
                positions.append(pos)
        
        if len(positions) < 5:
            return False
        
        # Y座標の変化を確認（奥→手前は増加）
        start_y = positions[0][1]
        end_y = positions[-1][1]
        
        # 奥コートから始まって手前コートで終わる
        return (start_y < self.back_court_y and 
                end_y > self.front_court_y and
                (end_y - start_y) > 100)  # 十分な移動距離
    
    def _is_rally_active(self) -> bool:
        """ラリーが発生しているかを判定"""
        if len(self.ball_history) < 20:
            return False
        
        # 最近のボール移動を分析
        recent_balls = list(self.ball_history)[-20:]
        court_changes = 0
        
        prev_in_front = None
        for data in recent_balls:
            pos = data.get('ball_position')
            if pos:
                in_front = self._is_ball_in_front_court(data)
                if prev_in_front is not None and prev_in_front != in_front:
                    court_changes += 1
                prev_in_front = in_front
        
        # コート間の移動が複数回発生している場合はラリー
        return court_changes >= self.rally_min_exchanges
    
    def _are_players_moving(self) -> bool:
        """プレイヤーが動いているかを判定"""
        if len(self.player_history) < 5:
            return False
        
        recent_players = list(self.player_history)[-5:]
        
        for player_type in [0, 1]:  # front, back
            positions = []
            for frame_players in recent_players:
                for x1, y1, x2, y2, class_id, conf in frame_players:
                    if class_id == player_type:
                        center_x = (x1 + x2) // 2
                        center_y = (y1 + y2) // 2
                        positions.append((center_x, center_y))
                        break
            
            if len(positions) >= 2:
                # 位置の変化を計算
                total_movement = 0
                for i in range(1, len(positions)):
                    dx = positions[i][0] - positions[i-1][0]
                    dy = positions[i][1] - positions[i-1][1]
                    movement = math.sqrt(dx*dx + dy*dy)
                    total_movement += movement
                
                if total_movement > self.player_movement_threshold:
                    return True
        
        return False
    
    def _is_ball_in_front_court(self, ball_data: Dict) -> bool:
        """ボールが手前コートにあるかを判定"""
        pos = ball_data.get('ball_position')
        if not pos:
            return False
        return pos[1] > self.front_court_y
    
    def _is_ball_in_back_court(self, ball_data: Dict) -> bool:
        """ボールが奥コートにあるかを判定"""
        pos = ball_data.get('ball_position')
        if not pos:
            return False
        return pos[1] < self.back_court_y
    
    def _analyze_player_positions(self, player_detections: List[Tuple]) -> Dict:
        """プレイヤー位置を分析"""
        front_players = []
        back_players = []
        
        for x1, y1, x2, y2, class_id, conf in player_detections:
            center_x = (x1 + x2) // 2
            center_y = (y1 + y2) // 2
            
            if class_id == 0:  # player_front
                front_players.append((center_x, center_y))
            elif class_id == 1:  # player_back
                back_players.append((center_x, center_y))
        
        return {
            'front_player_count': len(front_players),
            'back_player_count': len(back_players),
            'front_player_positions': front_players,
            'back_player_positions': back_players
        }
    
    def _calculate_confidence(self) -> float:
        """判定の信頼度を計算"""
        confidence = 0.5  # ベース信頼度
        
        # データ量による信頼度調整
        if len(self.ball_history) >= 20:
            confidence += 0.2
        
        # プレイヤー検出による信頼度調整
        if len(self.player_history) >= 10:
            confidence += 0.2
        
        # 局面継続時間による信頼度調整
        if self.phase_duration.total_seconds() > self.min_phase_duration:
            confidence += 0.1
        
        return min(1.0, confidence)
    
    def _handle_phase_change(self, new_phase: GamePhase, current_time: datetime):
        """局面変更の処理"""
        self.phase_duration = current_time - self.phase_start_time
        
        # 最小継続時間をチェック
        if self.phase_duration.total_seconds() < self.min_phase_duration:
            return  # 変更しない
        
        # 局面履歴に記録
        if self.current_phase != GamePhase.UNKNOWN:
            self.phase_history.append({
                'phase': self.current_phase.value,
                'start_time': self.phase_start_time.isoformat(),
                'end_time': current_time.isoformat(),
                'duration': self.phase_duration.total_seconds(),
                'frame_start': max(0, self.frame_number - len(self.ball_history)),
                'frame_end': self.frame_number
            })
        
        # 状態を更新
        self.previous_phase = self.current_phase
        self.current_phase = new_phase
        self.phase_start_time = current_time
        
        # カウンタをリセット
        self.stationary_count = 0
        self.serve_candidate_count = 0
        
        print(f"Phase changed: {self.previous_phase.value} -> {new_phase.value}")
    
    def get_current_status(self) -> Dict[str, Any]:
        """現在の状態を取得"""
        return {
            'current_phase': self.current_phase.value,
            'phase_duration': self.phase_duration.total_seconds(),
            'server': self.current_server,
            'serve_side': self.serve_side.value,
            'confidence': self._calculate_confidence(),
            'ball_missing_count': self.ball_missing_count,
            'stationary_count': self.stationary_count
        }
    
    def save_analysis_data(self, output_path: str):
        """分析データをCSVファイルに保存"""
        if not self.analysis_data:
            print("保存するデータがありません")
            return
        
        os.makedirs(os.path.dirname(output_path), exist_ok=True)
        
        with open(output_path, 'w', newline='', encoding='utf-8') as csvfile:
            fieldnames = self.analysis_data[0].keys()
            writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
            
            writer.writeheader()
            writer.writerows(self.analysis_data)
        
        print(f"局面分析データを保存しました: {output_path}")
    
    def save_phase_history(self, output_path: str):
        """局面履歴をCSVファイルに保存"""
        if not self.phase_history:
            print("保存する局面履歴がありません")
            return
        
        os.makedirs(os.path.dirname(output_path), exist_ok=True)
        
        with open(output_path, 'w', newline='', encoding='utf-8') as csvfile:
            fieldnames = ['phase', 'start_time', 'end_time', 'duration', 'frame_start', 'frame_end']
            writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
            
            writer.writeheader()
            writer.writerows(self.phase_history)
        
        print(f"局面履歴を保存しました: {output_path}")
    
    def setup_court_coordinates(self, frame: np.ndarray) -> bool:
        """
        GUIでテニスコートの6点を設定
        
        Args:
            frame: 設定用の参考フレーム
            
        Returns:
            設定が完了したかどうか
        """
        print("\n=== テニスコート座標設定 ===")
        print("以下の順序でコートの6点をクリックして設定してください:")
        print("1. 手前左角 (Front Left)")
        print("2. 手前右角 (Front Right)")
        print("3. 奥左角 (Back Left)")
        print("4. 奥右角 (Back Right)")
        print("5. ネット左端（地面）(Net Left)")
        print("6. ネット右端（地面）(Net Right)")
        print("設定完了後、Enterキーを押してください")
        
        # 設定状態
        self.setup_state = {
            'points': [],
            'current_point': 0,
            'point_names': ['front_left', 'front_right', 'back_left', 'back_right', 'net_left', 'net_right'],
            'display_names': ['手前左角', '手前右角', '奥左角', '奥右角', 'ネット左端', 'ネット右端'],
            'completed': False
        }
        
        def mouse_callback(event, x, y, flags, param):
            if event == cv2.EVENT_LBUTTONDOWN and self.setup_state['current_point'] < 6:
                self.setup_state['points'].append((x, y))
                point_name = self.setup_state['display_names'][self.setup_state['current_point']]
                print(f"✓ {point_name}: ({x}, {y})")
                self.setup_state['current_point'] += 1
                
                if self.setup_state['current_point'] >= 6:
                    print("すべての点が設定されました。Enterキーを押して完了してください。")
        
        cv2.namedWindow('Court Setup')
        cv2.setMouseCallback('Court Setup', mouse_callback)
        
        try:
            while not self.setup_state['completed']:
                display_frame = frame.copy()
                
                # 設定済みの点を描画
                for i, point in enumerate(self.setup_state['points']):
                    cv2.circle(display_frame, point, 8, (0, 255, 0), -1)
                    cv2.putText(display_frame, f"{i+1}", (point[0]+10, point[1]-10),
                               cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
                
                # 次に設定する点の説明を表示
                if self.setup_state['current_point'] < 6:
                    next_point = self.setup_state['display_names'],self.setup_state['current_point']
                    cv2.putText(display_frame, f"Next: {next_point}", (10, 30),
                               cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 255), 2)
                    cv2.putText(display_frame, f"Point {self.setup_state['current_point']+1}/6", (10, 70),
                               cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 0, 0), 2)
                else:
                    cv2.putText(display_frame, "Setup Complete! Press Enter", (10, 30),
                               cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)
                
                # 設定済みの点をつなぐ線を描画
                if len(self.setup_state['points']) >= 4:
                    # コートの外周
                    pts = np.array(self.setup_state['points'][:4], np.int32)
                    pts = pts.reshape((-1, 1, 2))
                    cv2.polylines(display_frame, [pts], True, (255, 255, 0), 2)
                
                if len(self.setup_state['points']) >= 6:
                    # ネットライン
                    cv2.line(display_frame, self.setup_state['points'][4], self.setup_state['points'][5], (0, 255, 255), 2)
                
                cv2.imshow('Court Setup', display_frame)
                
                key = cv2.waitKey(1) & 0xFF
                if key == ord('\r') or key == ord('\n'):  # Enter
                    if self.setup_state['current_point'] >= 6:
                        self.setup_state['completed'] = True
                elif key == ord('r'):  # Reset
                    self.setup_state['points'] = []
                    self.setup_state['current_point'] = 0
                    print("リセットしました。最初から設定してください。")
                elif key == ord('q'):  # Quit
                    print("設定をキャンセルしました")
                    cv2.destroyWindow('Court Setup')
                    return False
        
        except Exception as e:
            print(f"座標設定エラー: {e}")
            cv2.destroyWindow('Court Setup')
            return False
        
        cv2.destroyWindow('Court Setup')
        
        # 設定された座標を保存
        if len(self.setup_state['points']) >= 6:
            for i, point_name in enumerate(self.setup_state['point_names']):
                self.court_points[point_name] = self.setup_state['points'][i]
            
            # コート領域の境界を更新
            self._update_court_boundaries()
            
            print("\n=== 設定完了 ===")
            for i, point_name in enumerate(self.setup_state['point_names']):
                display_name = self.setup_state['display_names'][i]
                point = self.setup_state['points'][i]
                print(f"{display_name}: {point}")
            
            return True
        
        return False
    
    def _update_court_boundaries(self):
        """設定されたコート座標から境界を計算"""
        if all(point is not None for point in self.court_points.values()):
            # ネットの位置を基準に手前/奥コートを設定
            net_y = (self.court_points['net_left'][1] + self.court_points['net_right'][1]) // 2
            
            # 手前コートと奥コートの境界
            front_y = (self.court_points['front_left'][1] + self.court_points['front_right'][1]) // 2
            back_y = (self.court_points['back_left'][1] + self.court_points['back_right'][1]) // 2
            
            # 境界の更新
            self.front_court_y = (net_y + front_y) // 2
            self.back_court_y = (net_y + back_y) // 2
            
            # サイドラインの更新
            left_x = (self.court_points['front_left'][0] + self.court_points['back_left'][0]) // 2
            right_x = (self.court_points['front_right'][0] + self.court_points['back_right'][0]) // 2
            court_width = right_x - left_x
            
            self.deuce_side_x = left_x + court_width * 0.3
            self.ad_side_x = left_x + court_width * 0.7
            self.center_x = (left_x + right_x) // 2
            
            print(f"コート境界更新: 手前Y={self.front_court_y}, 奥Y={self.back_court_y}, 中央X={self.center_x}")
    
    def draw_analysis_overlay(self, frame: np.ndarray, analysis_result: Dict[str, Any]) -> np.ndarray:
        """分析結果をフレームに描画（右上に表示）"""
        overlay_frame = frame.copy()
        height, width = frame.shape[:2]
        
        # 局面情報を右上に描画
        phase_text = f"Phase: {analysis_result['current_phase']}"
        duration_text = f"Duration: {analysis_result['phase_duration']:.1f}s"
        confidence_text = f"Confidence: {analysis_result['confidence']:.2f}"
        
        # サーバー情報
        server_text = f"Server: {analysis_result.get('server', 'Unknown')}"
        side_text = f"Side: {analysis_result.get('serve_side', 'Unknown')}"
        
        # テキストを右上に描画
        texts = [phase_text, duration_text, confidence_text, server_text, side_text]
        
        for i, text in enumerate(texts):
            # テキストサイズを取得
            text_size = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, 0.7, 2)[0]
            
            # 右上座標を計算
            x_pos = width - text_size[0] - 10
            y_pos = 30 + i * 30
            
            # 背景矩形を描画
            cv2.rectangle(overlay_frame, (x_pos - 5, y_pos - 25), 
                         (x_pos + text_size[0] + 5, y_pos + 5), (0, 0, 0), -1)
            
            # テキストを描画
            cv2.putText(overlay_frame, text, (x_pos, y_pos), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2)
        
        # 設定されたコート座標を描画
        if all(point is not None for point in self.court_points.values()):
            # コートの外周
            court_outline = [
                self.court_points['front_left'],
                self.court_points['front_right'], 
                self.court_points['back_right'],
                self.court_points['back_left']
            ]
            pts = np.array(court_outline, np.int32)
            pts = pts.reshape((-1, 1, 2))
            cv2.polylines(overlay_frame, [pts], True, (255, 255, 0), 2)
            
            # ネットライン
            cv2.line(overlay_frame, self.court_points['net_left'], 
                    self.court_points['net_right'], (0, 255, 255), 2)
            
            # 手前/奥コート境界線
            cv2.line(overlay_frame, (0, int(self.front_court_y)), 
                    (width, int(self.front_court_y)), (255, 255, 0), 1)
            cv2.line(overlay_frame, (0, int(self.back_court_y)), 
                    (width, int(self.back_court_y)), (255, 255, 0), 1)
            
            # 中央線
            cv2.line(overlay_frame, (int(self.center_x), 0), 
                    (int(self.center_x), height), (255, 255, 0), 1)
        else:
            # コート未設定の場合のデフォルト線
            cv2.line(overlay_frame, (0, int(self.front_court_y)), 
                    (width, int(self.front_court_y)), (128, 128, 128), 1)
            cv2.line(overlay_frame, (0, int(self.back_court_y)), 
                    (width, int(self.back_court_y)), (128, 128, 128), 1)
            cv2.line(overlay_frame, (int(self.center_x), 0), 
                    (int(self.center_x), height), (128, 128, 128), 1)
        
        return overlay_frame

def test_analyzer():
    """テスト用関数 - balltrackingと統合した使用例"""
    print("=== テニス局面分析テスト開始 ===")
    
    try:
        from balltracking import TennisBallTracker
        print("✓ balltrackingモジュールのインポート成功")
    except ImportError as e:
        print(f"✗ balltrackingモジュールのインポートエラー: {e}")
        print("balltracking.pyファイルが同じディレクトリにあることを確認してください")
        return
    
    # パス設定
    model_path = "C:/Users/akama/AppData/Local/Programs/Python/Python310/python_file/projects/tennisvision/models/weights/best_5_31.pt"
    video_path = "C:/Users/akama/AppData/Local/Programs/Python/Python310/python_file/projects/tennisvision/data/raw/output3.mp4"
    output_dir = "C:/Users/akama/AppData/Local/Programs/Python/Python310/python_file/projects/tennisvision/data/output"
    
    # ファイル存在確認
    print(f"モデルファイル確認中: {model_path}")
    if not os.path.exists(model_path):
        print(f"✗ モデルファイルが見つかりません: {model_path}")
        return
    else:
        print("✓ モデルファイル確認完了")
    
    print(f"動画ファイル確認中: {video_path}")
    if not os.path.exists(video_path):
        print(f"✗ 動画ファイルが見つかりません: {video_path}")
        return
    else:
        print("✓ 動画ファイル確認完了")
    
    # 出力ディレクトリ作成
    os.makedirs(output_dir, exist_ok=True)
    print(f"✓ 出力ディレクトリ準備完了: {output_dir}")
    
    # 初期化
    print("トラッカーとアナライザーを初期化中...")
    try:
        tracker = TennisBallTracker(model_path, imgsz=1920)
        print("✓ TennisBallTracker初期化完了")
        
        analyzer = TennisGamePhaseAnalyzer()
        print("✓ TennisGamePhaseAnalyzer初期化完了")
    except Exception as e:
        print(f"✗ 初期化エラー: {e}")
        return
    
    # ビデオを開く
    print("動画ファイルを開いています...")
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        print("✗ Error: ビデオファイルを開けませんでした")
        return
    else:
        print("✓ 動画ファイルを開きました")
    
    # 動画情報を表示
    fps = cap.get(cv2.CAP_PROP_FPS)
    frame_count_total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    
    print(f"動画情報: {width}x{height}, {fps}fps, 総フレーム数: {frame_count_total}")
    
    # アナライザーの画面サイズを更新
    analyzer.court_width = width
    analyzer.court_height = height
    
    # 最初のフレームを取得してコート座標設定
    ret, first_frame = cap.read()
    if not ret:
        print("✗ 最初のフレームを読み込めませんでした")
        return
    
    print("\n=== コート座標設定 ===")
    print("コート座標を設定しますか？")
    print("1. はい（推奨）")
    print("2. いいえ（デフォルト設定を使用）")
    
    setup_choice = input("選択 (1 または 2): ").strip()
    if setup_choice == "1":
        if not analyzer.setup_court_coordinates(first_frame):
            print("コート座標設定がキャンセルされました")
            return
        print("✓ コート座標設定完了")
    else:
        print("デフォルト設定を使用します")
    
    # ビデオを最初に戻す
    cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
    
    print("\n=== 局面分析開始 ===")
    print("局面の種類:")
    print("- serve_front/serve_back: サーブ")  
    print("- rally: ラリー中")
    print("- point_interval: ポイント間")
    print("- point_end: ポイント終了")
    print("- changeover: チェンジオーバー")
    print("\n'q'キーで終了, 'p'キーで一時停止")
    
    paused = False
    
    try:
        while True:
            if not paused:
                ret, frame = cap.read()
                if not ret:
                    print("動画の最後に到達しました")
                    break
                
                # フレーム処理（balltrackingから検出結果も取得）
                try:
                    # YOLOv8で検出（imgszを指定）
                    results = tracker.model(frame, imgsz=tracker.imgsz, verbose=False)
                    
                    # プレイヤー検出データを抽出
                    player_detections = []
                    for result in results:
                        boxes = result.boxes
                        if boxes is not None:
                            for box in boxes:
                                class_id = int(box.cls[0])
                                confidence = float(box.conf[0])
                                x1, y1, x2, y2 = map(int, box.xyxy[0])
                                
                                if (class_id in [tracker.player_front_class_id, tracker.player_back_class_id] and 
                                    confidence > tracker.player_confidence_threshold):
                                    player_detections.append((x1, y1, x2, y2, class_id, confidence))
                    
                    # ボールトラッキングを実行
                    result_frame, tracking_data = tracker.process_frame_for_analysis(frame)
                    
                    # 局面分析
                    analysis_result = analyzer.analyze_frame(tracking_data, player_detections)
                    
                    # 分析結果を描画（右上に表示）
                    final_frame = analyzer.draw_analysis_overlay(result_frame, analysis_result)
                    
                    # 追加情報を左下に表示
                    info_texts = [
                        f"Frame: {frame_count}/{frame_count_total}",
                        f"Ball missing: {analysis_result['ball_missing_count']}",
                        f"Stationary: {analysis_result['stationary_count']}",
                        f"Players moving: {analysis_result['players_moving']}",
                        f"Players detected: {len(player_detections)}"
                    ]
                    
                    for i, text in enumerate(info_texts):
                        y_pos = height - 150 + i*25
                        # 背景矩形
                        text_size = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, 0.6, 2)[0]
                        cv2.rectangle(final_frame, (5, y_pos - 20), 
                                     (15 + text_size[0], y_pos + 5), (0, 0, 0), -1)
                        # テキスト
                        cv2.putText(final_frame, text, (10, y_pos), 
                                   cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
                    
                    # 操作説明を左下に表示
                    cv2.putText(final_frame, "Press 'q' to quit, 'p' to pause", (10, height-10), 
                               cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
                    
                    frame_count += 1
                    
                except Exception as e:
                    print(f"フレーム処理エラー (frame {frame_count}): {e}")
                    continue
            
            # 表示
            if 'final_frame' in locals():
                cv2.imshow('Tennis Game Phase Analysis', final_frame)
            
            # キー入力処理
            key = cv2.waitKey(30) & 0xFF
            if key == ord('q'):
                print("ユーザーによって処理が中断されました")
                break
            elif key == ord('p'):
                paused = not paused
                print(f"{'一時停止' if paused else '再開'}しました")
            
            # 進行状況表示
            if frame_count % 100 == 0 and frame_count > 0:
                progress = (frame_count / frame_count_total) * 100 if frame_count_total > 0 else 0
                print(f"処理済みフレーム: {frame_count} ({progress:.1f}%)")
                print(f"現在の局面: {analyzer.current_phase.value}")
                current_status = analyzer.get_current_status()
                print(f"局面継続時間: {current_status['phase_duration']:.1f}秒")
                print(f"信頼度: {current_status['confidence']:.2f}")
    
    except KeyboardInterrupt:
        print("\nキーボード割り込みにより処理を中断しました")
    except Exception as e:
        print(f"処理中にエラーが発生しました: {e}")
        import traceback
        traceback.print_exc()
    
    finally:
        cap.release()
        cv2.destroyAllWindows()
        
        # データを保存
        try:
            analyzer.save_analysis_data(analysis_csv_path)
            analyzer.save_phase_history(phase_history_path)
        except Exception as e:
            print(f"データ保存エラー: {e}")
        
        print(f"\n=== 分析完了 ===")
        print(f"処理済みフレーム数: {frame_count}")
        print(f"検出された局面数: {len(analyzer.phase_history)}")
        print(f"分析データ保存先: {analysis_csv_path}")
        print(f"局面履歴保存先: {phase_history_path}")
        
        # 局面の統計を表示
        if analyzer.phase_history:
            print("\n=== 局面統計 ===")
            phase_stats = {}
            for phase_info in analyzer.phase_history:
                phase = phase_info['phase']
                duration = phase_info['duration']
                if phase not in phase_stats:
                    phase_stats[phase] = {'count': 0, 'total_duration': 0}
                phase_stats[phase]['count'] += 1
                phase_stats[phase]['total_duration'] += duration
            
            for phase, stats in phase_stats.items():
                avg_duration = stats['total_duration'] / stats['count']
                print(f"{phase}: {stats['count']}回, 平均{avg_duration:.1f}秒")
        else:
            print("局面の変更は検出されませんでした")

def run_analysis_with_options():
    """オプション付きで局面分析を実行"""
    print("=== テニス局面分析システム ===")
    print("このシステムは以下の局面を自動判定します:")
    print("1. サーブ (serve_front/serve_back)")
    print("2. ラリー (rally)")
    print("3. ポイント間 (point_interval)")
    print("4. ポイント終了 (point_end)")
    print("5. チェンジオーバー (changeover)")
    print()
    
    # balltrackingモジュールの確認
    try:
        from balltracking import TennisBallTracker
        print("✓ balltrackingモジュール確認完了")
    except ImportError as e:
        print(f"✗ balltrackingモジュールが見つかりません: {e}")
        print("同じディレクトリにballtracking.pyがあることを確認してください")
        input("Enterキーを押して終了...")
        return
    
    # 設定の選択
    print("画像サイズを選択してください:")
    print("1. 640 (高速)")
    print("2. 1280 (バランス)")
    print("3. 1920 (高精度)")
    
    while True:
        try:
            choice = int(input("選択 (1-3): "))
            if choice in [1, 2, 3]:
                imgsz = [640, 1280, 1920][choice-1]
                break
            else:
                print("1-3を選択してください")
        except ValueError:
            print("数字を入力してください")
    
    print(f"画像サイズ: {imgsz}")
    
    # 感度設定
    print("\n判定感度を選択してください:")
    print("1. 低感度 (安定した判定)")
    print("2. 標準感度")
    print("3. 高感度 (素早い反応)")
    
    while True:
        try:
            sensitivity = int(input("選択 (1-3): "))
            if sensitivity in [1, 2, 3]:
                break
            else:
                print("1-3を選択してください")
        except ValueError:
            print("数字を入力してください")
    
    # パス設定
    model_path = "C:/Users/akama/AppData/Local/Programs/Python/Python310/python_file/projects/tennisvision/models/weights/best_5_31.pt"
    video_path = "C:/Users/akama/AppData/Local/Programs/Python/Python310/python_file/projects/tennisvision/data/raw/output3.mp4"
    output_dir = "C:/Users/akama/AppData/Local/Programs/Python/Python310/python_file/projects/tennisvision/data/output"
    
    # 初期化
    tracker = TennisBallTracker(model_path, imgsz=imgsz)
    analyzer = TennisGamePhaseAnalyzer()
    
    # 感度に応じてパラメータを調整
    if sensitivity == 1:  # 低感度
        analyzer.min_phase_duration = 2.0
        analyzer.player_movement_threshold = 30
        analyzer.serve_detection_frames = 20
    elif sensitivity == 3:  # 高感度
        analyzer.min_phase_duration = 0.5
        analyzer.player_movement_threshold = 10
        analyzer.serve_detection_frames = 10
    
    print(f"感度設定: {['低感度', '標準感度', '高感度'][sensitivity-1]}")
    
    # 分析実行
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        print("Error: ビデオファイルを開けませんでした")
        return
    
    # 画面サイズを取得
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    analyzer.court_width = width
    analyzer.court_height = height
    
    # 最初のフレームでコート座標設定
    ret, first_frame = cap.read()
    if ret:
        print("\nコート座標を設定しますか？ (y/n): ", end="")
        if input().lower().startswith('y'):
            if not analyzer.setup_court_coordinates(first_frame):
                print("設定がキャンセルされました")
                return
        cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
    
    print("\n分析を開始します...")
    
    frame_count = 0
    
    try:
        while True:
            ret, frame = cap.read()
            if not ret:
                break
            
            # 検出とトラッキング
            results = tracker.model(frame, imgsz=tracker.imgsz, verbose=False)
            player_detections = []
            
            for result in results:
                boxes = result.boxes
                if boxes is not None:
                    for box in boxes:
                        class_id = int(box.cls[0])
                        confidence = float(box.conf[0])
                        x1, y1, x2, y2 = map(int, box.xyxy[0])
                        
                        if (class_id in [tracker.player_front_class_id, tracker.player_back_class_id] and 
                            confidence > tracker.player_confidence_threshold):
                            player_detections.append((x1, y1, x2, y2, class_id, confidence))
            
            result_frame, tracking_data = tracker.process_frame_for_analysis(frame)
            analysis_result = analyzer.analyze_frame(tracking_data, player_detections)
            final_frame = analyzer.draw_analysis_overlay(result_frame, analysis_result)
            
            cv2.imshow('Tennis Game Phase Analysis', final_frame)
            key = cv2.waitKey(1) & 0xFF
            if key == ord('q'):
                break
            
            frame_count += 1
            if frame_count % 100 == 0:
                print(f"処理中: {frame_count}フレーム - 現在: {analyzer.current_phase.value}")
    
    finally:
        cap.release()
        cv2.destroyAllWindows()
        analyzer.save_analysis_data(analysis_csv_path)
        analyzer.save_phase_history(phase_history_path)
        
        print(f"\n分析完了! 結果は以下に保存されました:")
        print(f"- 詳細データ: {analysis_csv_path}")
        print(f"- 局面履歴: {phase_history_path}")
        
        # 局面の統計を表示
        if analyzer.phase_history:
            print("\n=== 局面統計 ===")
            phase_stats = {}
            for phase_info in analyzer.phase_history:
                phase = phase_info['phase']
                duration = phase_info['duration']
                if phase not in phase_stats:
                    phase_stats[phase] = {'count': 0, 'total_duration': 0}
                phase_stats[phase]['count'] += 1
                phase_stats[phase]['total_duration'] += duration
            
            for phase, stats in phase_stats.items():
                avg_duration = stats['total_duration'] / stats['count']
                print(f"{phase}: {stats['count']}回, 平均{avg_duration:.1f}秒")

if __name__ == "__main__":
    print("テニス局面分析システムを開始します...")
    print("実行モードを選択してください:")
    print("1. 基本テスト実行")
    print("2. オプション付き実行")
    
    try:
        mode = input("選択 (1 または 2, デフォルト: 1): ").strip()
        if mode == "2":
            run_analysis_with_options()
        else:
            test_analyzer()
    except KeyboardInterrupt:
        print("\n処理がキャンセルされました")
    except Exception as e:
        print(f"実行エラー: {e}")
        import traceback
        traceback.print_exc()
    finally:
        input("Enterキーを押して終了...")
