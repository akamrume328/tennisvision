import json
import pandas as pd
import numpy as np
from pathlib import Path
from typing import Dict, List, Tuple, Optional, Any
from datetime import datetime
import warnings
warnings.filterwarnings('ignore')

class TennisRuleValidator:
    """
    テニスのルールに基づいて分類結果の整合性を検証・修正するクラス
    """
    
    def __init__(self, training_data_dir: str = "training_data"):
        self.training_data_dir = Path(training_data_dir)
        self.results_dir = self.training_data_dir / "validated_results"
        self.results_dir.mkdir(exist_ok=True)
        
        # 局面ラベル定義
        self.phase_labels = [
            "point_interval",           # 0: ポイント間
            "rally",                   # 1: ラリー中
            "serve_preparation",       # 2: サーブ準備
            "serve_front_deuce",      # 3: 手前デュースサイドからのサーブ
            "serve_front_ad",         # 4: 手前アドサイドからのサーブ
            "serve_back_deuce",       # 5: 奥デュースサイドからのサーブ
            "serve_back_ad",          # 6: 奥アドサイドからのサーブ
            "changeover"              # 7: チェンジコート間
        ]
        
        # テニスルールの知識ベース
        self.tennis_rules = {
            'serve_sequence': {
                'deuce_first': True,  # デュースサイドから始まる
                'alternating': True,  # サイドが交互に変わる
                'min_duration': 3.0,  # 最小サーブ時間（秒）
                'max_duration': 15.0, # 最大サーブ時間（秒）
            },
            'rally_rules': {
                'min_duration': 1.0,   # 最小ラリー時間（秒）
                'max_duration': 60.0,  # 最大ラリー時間（秒）
                'ball_movement_required': True,  # ボール移動が必要
            },
            'changeover_rules': {
                'game_intervals': [2, 4, 6],  # チェンジコートのゲーム間隔
                'min_duration': 10.0,  # 最小チェンジ時間（秒）
                'max_duration': 120.0, # 最大チェンジ時間（秒）
            },
            'phase_transitions': {
                # 有効な局面遷移パターン
                'valid_transitions': {
                    0: [1, 2, 7],      # point_interval → rally, serve_prep, changeover
                    1: [0],            # rally → point_interval
                    2: [3, 4, 5, 6],   # serve_prep → serve_*
                    3: [1, 0],         # serve_front_deuce → rally, point_interval
                    4: [1, 0],         # serve_front_ad → rally, point_interval
                    5: [1, 0],         # serve_back_deuce → rally, point_interval
                    6: [1, 0],         # serve_back_ad → rally, point_interval
                    7: [0, 2],         # changeover → point_interval, serve_prep
                }
            }
        }
        
        print(f"テニスルール検証器を初期化しました")
        print(f"結果保存先: {self.results_dir}")
    
    def load_classification_results(self, file_pattern: str = None) -> Dict[str, Any]:
        """分類結果ファイルを読み込み"""
        if file_pattern:
            result_files = list(self.training_data_dir.glob(file_pattern))
        else:
            # 各種結果ファイルを探索
            patterns = [
                "classification_results_*.json",
                "lstm_predictions_*.json", 
                "phase_predictions_*.json"
            ]
            result_files = []
            for pattern in patterns:
                result_files.extend(list(self.training_data_dir.glob(pattern)))
        
        if not result_files:
            print("❌ 分類結果ファイルが見つかりません")
            return {}
        
        all_results = {}
        for file_path in result_files:
            try:
                with open(file_path, 'r', encoding='utf-8') as f:
                    data = json.load(f)
                
                video_key = data.get('video_name', file_path.stem)
                all_results[video_key] = data
                print(f"✅ 分類結果読み込み: {file_path.name}")
                
            except Exception as e:
                print(f"❌ 読み込みエラー: {file_path.name} - {e}")
        
        return all_results
    
    def validate_phase_transitions(self, predictions: List[int], timestamps: List[float] = None) -> Dict[str, Any]:
        """局面遷移の妥当性を検証"""
        validation_result = {
            'is_valid': True,
            'invalid_transitions': [],
            'transition_violations': 0,
            'correction_suggestions': []
        }
        
        if len(predictions) < 2:
            return validation_result
        
        valid_transitions = self.tennis_rules['phase_transitions']['valid_transitions']
        
        for i in range(1, len(predictions)):
            prev_phase = predictions[i-1]
            curr_phase = predictions[i]
            
            if prev_phase == curr_phase:
                continue  # 同じ局面の継続は問題なし
            
            # 有効な遷移かチェック
            if prev_phase in valid_transitions:
                if curr_phase not in valid_transitions[prev_phase]:
                    validation_result['is_valid'] = False
                    validation_result['transition_violations'] += 1
                    
                    violation_info = {
                        'index': i,
                        'from_phase': prev_phase,
                        'to_phase': curr_phase,
                        'from_name': self.phase_labels[prev_phase],
                        'to_name': self.phase_labels[curr_phase],
                        'severity': 'high'
                    }
                    
                    if timestamps:
                        violation_info['timestamp'] = timestamps[i]
                    
                    validation_result['invalid_transitions'].append(violation_info)
                    
                    # 修正提案
                    suggested_phases = valid_transitions[prev_phase]
                    validation_result['correction_suggestions'].append({
                        'index': i,
                        'current': curr_phase,
                        'suggestions': suggested_phases
                    })
        
        return validation_result
    
    def validate_serve_sequence(self, predictions: List[int], timestamps: List[float] = None) -> Dict[str, Any]:
        """サーブシーケンスの妥当性を検証"""
        validation_result = {
            'is_valid': True,
            'serve_violations': [],
            'sequence_errors': 0
        }
        
        serve_phases = [3, 4, 5, 6]  # サーブ関連の局面
        serve_blocks = []  # サーブブロック（連続するサーブ群）
        current_block = None
        
        # サーブブロックを抽出（連続するサーブとその完了を追跡）
        for i, phase in enumerate(predictions):
            if phase in serve_phases:
                if current_block is None:
                    # 新しいサーブブロック開始
                    current_block = {
                        'start_index': i,
                        'serves': [],
                        'side': 'deuce' if phase in [3, 5] else 'ad',
                        'court': 'front' if phase in [3, 4] else 'back'
                    }
                
                serve_info = {
                    'index': i,
                    'phase': phase,
                    'side': 'deuce' if phase in [3, 5] else 'ad',
                    'court': 'front' if phase in [3, 4] else 'back'
                }
                if timestamps:
                    serve_info['timestamp'] = timestamps[i]
                current_block['serves'].append(serve_info)
                
            elif current_block is not None:
                # サーブブロック終了（ラリーやポイント間に移行）
                current_block['end_index'] = i - 1
                current_block['completed'] = True
                serve_blocks.append(current_block)
                current_block = None
        
        # 最後のサーブブロックが終了していない場合
        if current_block is not None:
            current_block['end_index'] = len(predictions) - 1
            current_block['completed'] = False
            serve_blocks.append(current_block)
        
        if len(serve_blocks) < 2:
            return validation_result
        
        # サーブ持続時間をチェック
        if timestamps:
            for block in serve_blocks:
                if len(block['serves']) > 0:
                    start_time = block['serves'][0]['timestamp']
                    end_time = block['serves'][-1]['timestamp']
                    duration = end_time - start_time
                    
                    if duration > self.tennis_rules['serve_sequence']['max_duration']:
                        validation_result['serve_violations'].append({
                            'type': 'block_too_long',
                            'duration': duration,
                            'max_allowed': self.tennis_rules['serve_sequence']['max_duration'],
                            'block_info': block,
                            'severity': 'medium'
                        })
        
        # サイド交替パターンをチェック（完了したサーブブロック間で）
        completed_blocks = [block for block in serve_blocks if block.get('completed', False)]
        
        for i in range(1, len(completed_blocks)):
            prev_block = completed_blocks[i-1]
            curr_block = completed_blocks[i]
            
            # 前のブロックと同じサイドでサーブが再開された場合は違反
            if prev_block['side'] == curr_block['side']:
                validation_result['is_valid'] = False
                validation_result['sequence_errors'] += 1
                validation_result['serve_violations'].append({
                    'type': 'same_side_after_completion',
                    'prev_block': {
                        'side': prev_block['side'],
                        'court': prev_block['court'],
                        'start_index': prev_block['start_index'],
                        'end_index': prev_block['end_index']
                    },
                    'curr_block': {
                        'side': curr_block['side'],
                        'court': curr_block['court'],
                        'start_index': curr_block['start_index'],
                        'end_index': curr_block['end_index']
                    },
                    'violation_message': f"サーブ完了後に同じ{curr_block['side']}サイドで再開"
                })
        
        return validation_result
    
    def validate_rally_characteristics(self, predictions: List[int], tracking_features: List[Dict] = None) -> Dict[str, Any]:
        """ラリー局面の特性を検証"""
        validation_result = {
            'is_valid': True,
            'rally_violations': [],
            'characteristic_errors': 0
        }
        
        if not tracking_features:
            print("⚠️  トラッキング特徴量がないため、ラリー特性の検証をスキップします")
            return validation_result
        
        rally_phases = []
        current_rally = None
        
        # ラリー区間を抽出
        for i, phase in enumerate(predictions):
            if phase == 1:  # rally
                if current_rally is None:
                    current_rally = {'start': i, 'frames': []}
                current_rally['frames'].append(i)
            else:
                if current_rally is not None:
                    current_rally['end'] = i - 1
                    rally_phases.append(current_rally)
                    current_rally = None
        
        # 最後のラリーが終了していない場合
        if current_rally is not None:
            current_rally['end'] = len(predictions) - 1
            rally_phases.append(current_rally)
        
        # 各ラリーの特性をチェック
        for rally in rally_phases:
            rally_frames = rally['frames']
            
            # ラリー持続時間チェック（フレーム数ベース）
            duration_frames = len(rally_frames)
            if duration_frames < 5:  # 5フレーム未満は短すぎる
                validation_result['is_valid'] = False
                validation_result['characteristic_errors'] += 1
                validation_result['rally_violations'].append({
                    'type': 'too_short',
                    'duration_frames': duration_frames,
                    'start_frame': rally['start'],
                    'end_frame': rally['end']
                })
            
            # ボール移動の確認
            if tracking_features:
                ball_movement_detected = False
                ball_detections = 0
                
                for frame_idx in rally_frames:
                    if frame_idx < len(tracking_features):
                        frame_data = tracking_features[frame_idx]
                        if frame_data.get('ball_detected', 0):
                            ball_detections += 1
                        if frame_data.get('ball_movement_score', 0) > 0.3:
                            ball_movement_detected = True
                
                ball_detection_rate = ball_detections / len(rally_frames) if rally_frames else 0
                
                if not ball_movement_detected or ball_detection_rate < 0.2:
                    validation_result['rally_violations'].append({
                        'type': 'insufficient_ball_movement',
                        'ball_detection_rate': ball_detection_rate,
                        'ball_movement_detected': ball_movement_detected,
                        'start_frame': rally['start'],
                        'end_frame': rally['end'],
                        'severity': 'medium'
                    })
        
        return validation_result
    
    def validate_changeover_timing(self, predictions: List[int], timestamps: List[float] = None) -> Dict[str, Any]:
        """チェンジコートのタイミングを検証"""
        validation_result = {
            'is_valid': True,
            'changeover_violations': [],
            'timing_errors': 0
        }
        
        changeover_phases = []
        
        # チェンジコート局面を抽出
        for i, phase in enumerate(predictions):
            if phase == 7:  # changeover
                changeover_info = {'index': i}
                if timestamps:
                    changeover_info['timestamp'] = timestamps[i]
                changeover_phases.append(changeover_info)
        
        if len(changeover_phases) < 2:
            return validation_result
        
        # チェンジコート間隔をチェック
        if timestamps:
            for i in range(len(changeover_phases) - 1):
                duration = changeover_phases[i+1]['timestamp'] - changeover_phases[i]['timestamp']
                
                if duration < self.tennis_rules['changeover_rules']['min_duration']:
                    validation_result['is_valid'] = False
                    validation_result['timing_errors'] += 1
                    validation_result['changeover_violations'].append({
                        'type': 'too_frequent',
                        'interval': duration,
                        'min_required': self.tennis_rules['changeover_rules']['min_duration'],
                        'changeover_info': changeover_phases[i]
                    })
        
        # チェンジコートの頻度チェック（全体に対する比率）
        total_phases = len(predictions)
        changeover_ratio = len(changeover_phases) / total_phases if total_phases > 0 else 0
        
        if changeover_ratio > 0.1:  # 10%以上はチェンジコートが多すぎる
            validation_result['changeover_violations'].append({
                'type': 'too_frequent_overall',
                'changeover_ratio': changeover_ratio,
                'changeover_count': len(changeover_phases),
                'total_phases': total_phases,
                'severity': 'medium'
            })
        
        return validation_result
    
    def correct_predictions(self, predictions: List[int], validation_results: Dict[str, Any]) -> Tuple[List[int], Dict[str, Any]]:
        """検証結果に基づいて予測を修正"""
        corrected_predictions = predictions.copy()
        corrections_made = {
            'transition_corrections': 0,
            'serve_corrections': 0,
            'rally_corrections': 0,
            'changeover_corrections': 0,
            'total_corrections': 0
        }
        
        # 遷移違反の修正
        if 'transition_validation' in validation_results:
            transition_result = validation_results['transition_validation']
            for suggestion in transition_result.get('correction_suggestions', []):
                idx = suggestion['index']
                current = suggestion['current']
                suggestions = suggestion['suggestions']
                
                if suggestions and idx < len(corrected_predictions):
                    # 最初の提案を採用
                    corrected_predictions[idx] = suggestions[0]
                    corrections_made['transition_corrections'] += 1
        
        # サーブシーケンス違反の修正
        if 'serve_validation' in validation_results:
            serve_result = validation_results['serve_validation']
            for violation in serve_result.get('serve_violations', []):
                if violation['type'] == 'same_side_after_completion':
                    # 同じサイドでの再開を修正（反対サイドに変更）
                    curr_block = violation['curr_block']
                    start_idx = curr_block['start_index']
                    end_idx = curr_block['end_index']
                    
                    # 現在のサイドから反対サイドに変更
                    for idx in range(start_idx, min(end_idx + 1, len(corrected_predictions))):
                        current_phase = corrected_predictions[idx]
                        if current_phase in [3, 4, 5, 6]:  # サーブ局面の場合
                            # サイドを反転
                            if current_phase == 3:  # serve_front_deuce → serve_front_ad
                                corrected_predictions[idx] = 4
                            elif current_phase == 4:  # serve_front_ad → serve_front_deuce
                                corrected_predictions[idx] = 3
                            elif current_phase == 5:  # serve_back_deuce → serve_back_ad
                                corrected_predictions[idx] = 6
                            elif current_phase == 6:  # serve_back_ad → serve_back_deuce
                                corrected_predictions[idx] = 5
                            corrections_made['serve_corrections'] += 1
                
                elif violation['type'] == 'block_too_long':
                    # 長すぎるサーブブロックの中間部分をサーブ準備に変更
                    block_info = violation['block_info']
                    serves = block_info['serves']
                    if len(serves) > 3:  # 3回以上のサーブがある場合、中間を調整
                        middle_start = len(serves) // 3
                        middle_end = len(serves) * 2 // 3
                        for i in range(middle_start, middle_end):
                            serve_idx = serves[i]['index']
                            if serve_idx < len(corrected_predictions):
                                corrected_predictions[serve_idx] = 2  # serve_preparation
                                corrections_made['serve_corrections'] += 1
        
        # 短すぎるラリーの修正
        if 'rally_validation' in validation_results:
            rally_result = validation_results['rally_validation']
            for violation in rally_result.get('rally_violations', []):
                if violation['type'] == 'too_short':
                    start_frame = violation['start_frame']
                    end_frame = violation['end_frame']
                    
                    # 短いラリーをpoint_intervalに変更
                    for idx in range(start_frame, end_frame + 1):
                        if idx < len(corrected_predictions):
                            corrected_predictions[idx] = 0  # point_interval
                            corrections_made['rally_corrections'] += 1
        
        # 過度に頻繁なチェンジコートの修正
        if 'changeover_validation' in validation_results:
            changeover_result = validation_results['changeover_validation']
            for violation in changeover_result.get('changeover_violations', []):
                if violation['type'] == 'too_frequent':
                    changeover_info = violation['changeover_info']
                    idx = changeover_info['index']
                    if idx < len(corrected_predictions):
                        corrected_predictions[idx] = 0  # point_intervalに変更
                        corrections_made['changeover_corrections'] += 1
        
        corrections_made['total_corrections'] = sum(corrections_made.values()) - corrections_made['total_corrections']
        
        return corrected_predictions, corrections_made
    
    def smooth_predictions(self, predictions: List[int], window_size: int = 5) -> List[int]:
        """予測結果を平滑化してノイズを除去"""
        if len(predictions) < window_size:
            return predictions
        
        smoothed = predictions.copy()
        
        for i in range(window_size // 2, len(predictions) - window_size // 2):
            # ウィンドウ内の最頻値を採用
            window = predictions[i - window_size // 2 : i + window_size // 2 + 1]
            from collections import Counter
            most_common = Counter(window).most_common(1)[0][0]
            smoothed[i] = most_common
        
        return smoothed
    
    def validate_video_predictions(self, video_name: str, predictions: List[int], 
                                 timestamps: List[float] = None, 
                                 tracking_features: List[Dict] = None) -> Dict[str, Any]:
        """単一動画の予測結果を総合的に検証"""
        print(f"\n--- {video_name} の検証開始 ---")
        
        validation_results = {
            'video_name': video_name,
            'total_frames': len(predictions),
            'validation_timestamp': datetime.now().isoformat(),
            'overall_valid': True
        }
        
        # 各種検証を実行
        print("局面遷移の検証中...")
        transition_result = self.validate_phase_transitions(predictions, timestamps)
        validation_results['transition_validation'] = transition_result
        
        print("サーブシーケンスの検証中...")
        serve_result = self.validate_serve_sequence(predictions, timestamps)
        validation_results['serve_validation'] = serve_result
        
        print("ラリー特性の検証中...")
        rally_result = self.validate_rally_characteristics(predictions, tracking_features)
        validation_results['rally_validation'] = rally_result
        
        print("チェンジコートタイミングの検証中...")
        changeover_result = self.validate_changeover_timing(predictions, timestamps)
        validation_results['changeover_validation'] = changeover_result
        
        # 全体的な妥当性判定
        validation_results['overall_valid'] = all([
            transition_result.get('is_valid', True),
            serve_result.get('is_valid', True),
            rally_result.get('is_valid', True),
            changeover_result.get('is_valid', True)
        ])
        
        # 統計サマリー
        validation_results['validation_summary'] = {
            'transition_violations': transition_result.get('transition_violations', 0),
            'serve_violations': len(serve_result.get('serve_violations', [])),
            'rally_violations': len(rally_result.get('rally_violations', [])),
            'changeover_violations': len(changeover_result.get('changeover_violations', []))
        }
        
        # 結果表示
        summary = validation_results['validation_summary']
        print(f"\n=== {video_name} 検証結果 ===")
        print(f"全体妥当性: {'✅ 妥当' if validation_results['overall_valid'] else '❌ 問題あり'}")
        print(f"遷移違反: {summary['transition_violations']}件")
        print(f"サーブ違反: {summary['serve_violations']}件")
        print(f"ラリー違反: {summary['rally_violations']}件")
        print(f"チェンジコート違反: {summary['changeover_violations']}件")
        
        return validation_results
    
    def process_and_correct_predictions(self, video_name: str, predictions: List[int],
                                      timestamps: List[float] = None,
                                      tracking_features: List[Dict] = None) -> Dict[str, Any]:
        """予測結果を検証し、修正版を作成"""
        # 検証実行
        validation_results = self.validate_video_predictions(video_name, predictions, timestamps, tracking_features)
        
        # 修正実行
        print("予測結果の修正中...")
        corrected_predictions, corrections_made = self.correct_predictions(predictions, validation_results)
        
        # 平滑化適用
        print("予測結果の平滑化中...")
        smoothed_predictions = self.smooth_predictions(corrected_predictions)
        
        # 修正後の再検証
        print("修正結果の再検証中...")
        corrected_validation = self.validate_video_predictions(f"{video_name}_corrected", smoothed_predictions, timestamps, tracking_features)
        
        # 結果統合
        processing_result = {
            'video_name': video_name,
            'original_predictions': predictions,
            'corrected_predictions': corrected_predictions,
            'smoothed_predictions': smoothed_predictions,
            'original_validation': validation_results,
            'corrected_validation': corrected_validation,
            'corrections_made': corrections_made,
            'improvement_metrics': self.calculate_improvement_metrics(validation_results, corrected_validation)
        }
        
        return processing_result
    
    def calculate_improvement_metrics(self, original_validation: Dict, corrected_validation: Dict) -> Dict[str, float]:
        """修正前後の改善度を計算"""
        original_summary = original_validation.get('validation_summary', {})
        corrected_summary = corrected_validation.get('validation_summary', {})
        
        metrics = {}
        
        for violation_type in ['transition_violations', 'serve_violations', 'rally_violations', 'changeover_violations']:
            original_count = original_summary.get(violation_type, 0)
            corrected_count = corrected_summary.get(violation_type, 0)
            
            if original_count > 0:
                improvement_rate = (original_count - corrected_count) / original_count
                metrics[f'{violation_type}_improvement'] = improvement_rate
            else:
                metrics[f'{violation_type}_improvement'] = 1.0 if corrected_count == 0 else 0.0
        
        # 全体改善度
        total_original = sum(original_summary.values())
        total_corrected = sum(corrected_summary.values())
        
        if total_original > 0:
            metrics['overall_improvement'] = (total_original - total_corrected) / total_original
        else:
            metrics['overall_improvement'] = 1.0
        
        return metrics
    
    def save_validation_results(self, processing_results: List[Dict[str, Any]]) -> str:
        """検証・修正結果を保存"""
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        output_file = self.results_dir / f"tennis_rule_validation_{timestamp}.json"
        
        # 保存用データを整理
        save_data = {
            'validation_timestamp': timestamp,
            'total_videos': len(processing_results),
            'tennis_rules_applied': self.tennis_rules,
            'phase_labels': self.phase_labels,
            'results': processing_results,
            'summary_statistics': self.calculate_summary_statistics(processing_results)
        }
        
        with open(output_file, 'w', encoding='utf-8') as f:
            json.dump(save_data, f, indent=2, ensure_ascii=False)
        
        print(f"\n=== 検証結果保存完了 ===")
        print(f"保存先: {output_file}")
        print(f"処理動画数: {len(processing_results)}")
        
        return str(output_file)
    
    def calculate_summary_statistics(self, processing_results: List[Dict[str, Any]]) -> Dict[str, Any]:
        """全体の統計サマリーを計算"""
        if not processing_results:
            return {}
        
        # 改善度統計
        improvement_metrics = []
        total_corrections = 0
        
        for result in processing_results:
            improvement = result.get('improvement_metrics', {})
            improvement_metrics.append(improvement)
            corrections = result.get('corrections_made', {})
            total_corrections += corrections.get('total_corrections', 0)
        
        # 平均改善度を計算
        avg_improvements = {}
        if improvement_metrics:
            for metric_name in improvement_metrics[0].keys():
                values = [metrics.get(metric_name, 0) for metrics in improvement_metrics]
                avg_improvements[f'avg_{metric_name}'] = np.mean(values)
        
        summary = {
            'total_videos_processed': len(processing_results),
            'total_corrections_made': total_corrections,
            'average_corrections_per_video': total_corrections / len(processing_results),
            'improvement_statistics': avg_improvements,
            'videos_with_violations': sum(1 for result in processing_results 
                                        if not result.get('original_validation', {}).get('overall_valid', True)),
            'videos_corrected_successfully': sum(1 for result in processing_results 
                                               if result.get('corrected_validation', {}).get('overall_valid', False))
        }
        
        return summary
    
    def generate_validation_report(self, processing_results: List[Dict[str, Any]]) -> str:
        """検証レポートを生成"""
        if not processing_results:
            return "処理結果がありません"
        
        report_lines = []
        report_lines.append("=" * 60)
        report_lines.append("テニスルール検証レポート")
        report_lines.append("=" * 60)
        report_lines.append(f"生成日時: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        report_lines.append(f"処理動画数: {len(processing_results)}")
        report_lines.append("")
        
        # 全体統計
        summary_stats = self.calculate_summary_statistics(processing_results)
        report_lines.append("=== 全体統計 ===")
        report_lines.append(f"総修正数: {summary_stats.get('total_corrections_made', 0)}")
        report_lines.append(f"1動画あたり平均修正数: {summary_stats.get('average_corrections_per_video', 0):.1f}")
        report_lines.append(f"違反のあった動画数: {summary_stats.get('videos_with_violations', 0)}")
        report_lines.append(f"修正により改善された動画数: {summary_stats.get('videos_corrected_successfully', 0)}")
        report_lines.append("")
        
        # 各動画の詳細
        report_lines.append("=== 動画別詳細結果 ===")
        for i, result in enumerate(processing_results, 1):
            video_name = result.get('video_name', f'Video_{i}')
            original_val = result.get('original_validation', {})
            corrected_val = result.get('corrected_validation', {})
            corrections = result.get('corrections_made', {})
            improvements = result.get('improvement_metrics', {})
            
            report_lines.append(f"{i}. {video_name}")
            report_lines.append(f"   修正前妥当性: {'✅' if original_val.get('overall_valid') else '❌'}")
            report_lines.append(f"   修正後妥当性: {'✅' if corrected_val.get('overall_valid') else '❌'}")
            report_lines.append(f"   総修正数: {corrections.get('total_corrections', 0)}")
            report_lines.append(f"   全体改善度: {improvements.get('overall_improvement', 0):.1%}")
            report_lines.append("")
        
        # 改善度統計
        if 'improvement_statistics' in summary_stats:
            report_lines.append("=== 改善度統計 ===")
            for metric, value in summary_stats['improvement_statistics'].items():
                report_lines.append(f"{metric}: {value:.1%}")
            report_lines.append("")
        
        return "\n".join(report_lines)

def main():
    """メイン関数 - テニスルール検証の実行"""
    print("=== テニスルール検証ツール ===")
    print("分類結果をテニスのルールに基づいて検証・修正します")
    
    validator = TennisRuleValidator()
    
    # 分類結果を読み込み
    classification_results = validator.load_classification_results()
    
    if not classification_results:
        print("❌ 分類結果ファイルが見つかりません")
        print("必要なファイル: classification_results_*.json または lstm_predictions_*.json")
        return
    
    processing_results = []
    
    print(f"\n=== {len(classification_results)}動画の検証開始 ===")
    
    for video_name, result_data in classification_results.items():
        try:
            predictions = result_data.get('predictions', [])
            timestamps = result_data.get('timestamps', None)
            tracking_features = result_data.get('tracking_features', None)
            
            if not predictions:
                print(f"⚠️  {video_name}: 予測データがありません")
                continue
            
            print(f"\n処理中: {video_name}")
            processing_result = validator.process_and_correct_predictions(
                video_name, predictions, timestamps, tracking_features
            )
            processing_results.append(processing_result)
            
        except Exception as e:
            print(f"❌ {video_name} 処理エラー: {e}")
    
    if not processing_results:
        print("❌ 処理可能なデータがありませんでした")
        return
    
    # 結果保存
    output_file = validator.save_validation_results(processing_results)
    
    # レポート生成
    report = validator.generate_validation_report(processing_results)
    print(f"\n{report}")
    
    # レポートファイル保存
    report_file = validator.results_dir / f"validation_report_{datetime.now().strftime('%Y%m%d_%H%M%S')}.txt"
    with open(report_file, 'w', encoding='utf-8') as f:
        f.write(report)
    
    print(f"\n🎉 テニスルール検証完了！")
    print(f"📁 結果ファイル: {output_file}")
    print(f"📄 レポートファイル: {report_file}")
    print(f"📊 処理動画数: {len(processing_results)}")

if __name__ == "__main__":
    main()
