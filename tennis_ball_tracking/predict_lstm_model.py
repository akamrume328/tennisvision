import json
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Tuple, Optional, Any
import warnings
warnings.filterwarnings('ignore')

# 機械学習ライブラリ
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score, f1_score
import joblib

# PyTorch
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, TensorDataset
import cv2

# 既存のモデルクラスをインポート（同じディレクトリ内）
from train_lstm_model import TennisLSTMModel, setup_gpu_config

# GPU設定
DEVICE = setup_gpu_config()
GPU_AVAILABLE = DEVICE.type == 'cuda'

plt.style.use('default')
sns.set_palette("husl")

class TennisLSTMPredictor:
    """
    学習済みPyTorch LSTMモデルを使用したテニス局面分類予測クラス
    """
    
    def __init__(self, model_path: str = None, metadata_path: str = None, 
                 scaler_path: str = None, training_data_dir: str = "training_data"):
        """
        初期化
        
        Args:
            model_path: モデルファイルのパス (.pth)
            metadata_path: メタデータファイルのパス (.json)
            scaler_path: スケーラーファイルのパス (.pkl)
            training_data_dir: 学習データディレクトリ
        """
        self.training_data_dir = Path(training_data_dir)
        self.models_dir = self.training_data_dir / "lstm_models"
        
        # モデル関連の変数
        self.model = None
        self.scaler = None
        self.metadata = None
        self.phase_labels = []
        self.sequence_length = 30
        self.overlap_ratio = 0.5
        
        # モデルを自動検索または指定されたパスから読み込み
        if model_path and metadata_path and scaler_path:
            self.load_model(model_path, metadata_path, scaler_path)
        else:
            self.auto_load_latest_model()
        
        print(f"PyTorch LSTM局面分類予測器を初期化しました")
        print(f"使用デバイス: {DEVICE}")
        print(f"シーケンス長: {self.sequence_length}フレーム")
    
    def auto_load_latest_model(self):
        """最新の学習済みモデルを自動検索して読み込み"""
        model_files = list(self.models_dir.glob("tennis_pytorch_model_*.pth"))
        
        if not model_files:
            print("❌ 学習済みモデルファイルが見つかりません")
            print("train_lstm_model.py を実行してモデルを学習してください")
            return False
        
        # 最新のモデルファイルを選択
        latest_model_file = max(model_files, key=lambda x: x.stat().st_mtime)
        timestamp = latest_model_file.stem.replace("tennis_pytorch_model_", "")
        
        # 対応するメタデータとスケーラーファイルを検索
        metadata_file = self.models_dir / f"tennis_pytorch_metadata_{timestamp}.json"
        scaler_file = self.models_dir / f"tennis_pytorch_scaler_{timestamp}.pkl"
        
        if not metadata_file.exists() or not scaler_file.exists():
            print("❌ 対応するメタデータまたはスケーラーファイルが見つかりません")
            return False
        
        print(f"最新モデルを読み込み: {latest_model_file.name}")
        return self.load_model(str(latest_model_file), str(metadata_file), str(scaler_file))
    
    def load_model(self, model_path: str, metadata_path: str, scaler_path: str) -> bool:
        """指定されたパスからモデルを読み込み"""
        try:
            # メタデータ読み込み
            with open(metadata_path, 'r', encoding='utf-8') as f:
                self.metadata = json.load(f)
            
            # モデル設定を取得
            self.phase_labels = self.metadata['phase_labels']
            self.sequence_length = self.metadata['sequence_length']
            self.overlap_ratio = self.metadata.get('overlap_ratio', 0.5)
            
            print(f"✅ メタデータ読み込み: {Path(metadata_path).name}")
            print(f"   モデルタイプ: {self.metadata['model_type']}")
            print(f"   精度: {self.metadata['test_accuracy']:.4f}")
            print(f"   F1スコア: {self.metadata['f1_score']:.4f}")
            print(f"   局面ラベル: {self.phase_labels}")
            
            # スケーラー読み込み
            self.scaler = joblib.load(scaler_path)
            print(f"✅ スケーラー読み込み: {Path(scaler_path).name}")
            
            # モデル読み込み
            checkpoint = torch.load(model_path, map_location=DEVICE)
            model_config = checkpoint['model_config']
            
            # モデルインスタンスを作成
            self.model = TennisLSTMModel(
                input_size=model_config['input_size'],
                hidden_sizes=model_config['hidden_sizes'],
                num_classes=model_config['num_classes'],
                dropout_rate=model_config['dropout_rate'],
                model_type=model_config['model_type']
            )
            
            # 状態辞書を読み込み
            self.model.load_state_dict(checkpoint['model_state_dict'])
            self.model = self.model.to(DEVICE)
            self.model.eval()  # 評価モードに設定
            
            print(f"✅ モデル読み込み: {Path(model_path).name}")
            print(f"   パラメータ数: {sum(p.numel() for p in self.model.parameters()):,}")
            
            return True
            
        except Exception as e:
            print(f"❌ モデル読み込みエラー: {e}")
            return False
    
    def predict_from_features(self, features_df: pd.DataFrame, 
                            video_name: str = "unknown") -> Dict[str, Any]:
        """
        特徴量データフレームから局面を予測
        
        Args:
            features_df: 特徴量データフレーム
            video_name: 動画名（オプション）
            
        Returns:
            予測結果辞書
        """
        if self.model is None:
            print("❌ モデルが読み込まれていません")
            return {}
        
        print(f"\n=== 局面予測実行 ===")
        print(f"動画: {video_name}")
        print(f"フレーム数: {len(features_df)}")
        
        # 特徴量を準備
        exclude_columns = ['label', 'video_name', 'frame_number']
        feature_columns = [col for col in features_df.columns if col not in exclude_columns]
        
        # 数値特徴量のみ使用
        numeric_features = features_df[feature_columns].select_dtypes(include=[np.number]).columns
        X_features = features_df[numeric_features].values
        
        print(f"使用特徴量数: {len(numeric_features)}")
        
        # 無限値とNaNを処理
        X_features = np.nan_to_num(X_features, nan=0.0, posinf=1e6, neginf=-1e6)
        
        # 標準化
        X_scaled = self.scaler.transform(X_features)
        
        # シーケンスを作成
        sequences = self.create_sequences_for_prediction(X_scaled)
        
        if len(sequences) == 0:
            print("❌ 予測用シーケンスを作成できませんでした")
            return {}
        
        print(f"作成されたシーケンス数: {len(sequences)}")
        
        # PyTorchテンソルに変換
        X_tensor = torch.FloatTensor(sequences).to(DEVICE)
        
        # 予測実行
        self.model.eval()
        all_predictions = []
        all_probabilities = []
        
        with torch.no_grad():
            # バッチ処理で予測
            batch_size = 64
            for i in range(0, len(X_tensor), batch_size):
                batch = X_tensor[i:i + batch_size]
                
                if GPU_AVAILABLE:
                    with torch.cuda.amp.autocast():
                        outputs = self.model(batch)
                else:
                    outputs = self.model(batch)
                
                probabilities = F.softmax(outputs, dim=1)
                _, predicted = torch.max(outputs, 1)
                
                all_predictions.extend(predicted.cpu().numpy())
                all_probabilities.extend(probabilities.cpu().numpy())
        
        # 結果をフレーム単位に展開
        frame_predictions = self.expand_predictions_to_frames(
            all_predictions, all_probabilities, len(features_df)
        )
        
        # 結果を整理
        results = {
            'video_name': video_name,
            'total_frames': len(features_df),
            'sequence_count': len(sequences),
            'frame_predictions': frame_predictions,
            'phase_distribution': self.calculate_phase_distribution(frame_predictions),
            'confidence_stats': self.calculate_confidence_stats(all_probabilities)
        }
        
        print(f"✅ 予測完了")
        print(f"   総フレーム数: {results['total_frames']}")
        print(f"   シーケンス数: {results['sequence_count']}")
        
        return results
    
    def create_sequences_for_prediction(self, features: np.ndarray) -> List[np.ndarray]:
        """予測用のシーケンスを作成"""
        sequences = []
        step_size = max(1, int(self.sequence_length * (1 - self.overlap_ratio)))
        
        for i in range(0, len(features) - self.sequence_length + 1, step_size):
            seq_features = features[i:i + self.sequence_length]
            sequences.append(seq_features)
        
        return sequences
    
    def expand_predictions_to_frames(self, predictions: List[int], 
                                   probabilities: List[np.ndarray], 
                                   total_frames: int) -> List[Dict]:
        """シーケンス予測をフレーム単位に展開"""
        frame_predictions = []
        step_size = max(1, int(self.sequence_length * (1 - self.overlap_ratio)))
        
        # 各フレームの予測を集計
        frame_votes = [[] for _ in range(total_frames)]
        frame_probs = [[] for _ in range(total_frames)]
        
        for seq_idx, (pred, prob) in enumerate(zip(predictions, probabilities)):
            # このシーケンスが対応するフレーム範囲
            start_frame = seq_idx * step_size
            end_frame = start_frame + self.sequence_length
            
            # シーケンス内の各フレームに投票
            for frame_idx in range(start_frame, min(end_frame, total_frames)):
                frame_votes[frame_idx].append(pred)
                frame_probs[frame_idx].append(prob)
        
        # 各フレームの最終予測を決定
        for frame_idx in range(total_frames):
            if frame_votes[frame_idx]:
                # 多数決で予測を決定
                unique_preds, counts = np.unique(frame_votes[frame_idx], return_counts=True)
                final_pred = unique_preds[np.argmax(counts)]
                
                # 確率の平均を計算
                avg_prob = np.mean(frame_probs[frame_idx], axis=0)
                confidence = np.max(avg_prob)
                
                frame_predictions.append({
                    'frame': frame_idx,
                    'predicted_phase_id': int(final_pred),
                    'predicted_phase_name': self.phase_labels[final_pred],
                    'confidence': float(confidence),
                    'probabilities': avg_prob.tolist(),
                    'vote_count': len(frame_votes[frame_idx])
                })
            else:
                # 予測がない場合はデフォルト値
                frame_predictions.append({
                    'frame': frame_idx,
                    'predicted_phase_id': 0,
                    'predicted_phase_name': self.phase_labels[0],
                    'confidence': 0.0,
                    'probabilities': [0.0] * len(self.phase_labels),
                    'vote_count': 0
                })
        
        return frame_predictions
    
    def calculate_phase_distribution(self, frame_predictions: List[Dict]) -> Dict:
        """局面分布を計算"""
        phase_counts = {}
        total_frames = len(frame_predictions)
        
        for pred in frame_predictions:
            phase_name = pred['predicted_phase_name']
            phase_counts[phase_name] = phase_counts.get(phase_name, 0) + 1
        
        phase_distribution = {}
        for phase_name, count in phase_counts.items():
            phase_distribution[phase_name] = {
                'count': count,
                'percentage': (count / total_frames) * 100
            }
        
        return phase_distribution
    
    def calculate_confidence_stats(self, probabilities: List[np.ndarray]) -> Dict:
        """信頼度統計を計算"""
        confidences = [np.max(prob) for prob in probabilities]
        
        return {
            'mean_confidence': float(np.mean(confidences)),
            'std_confidence': float(np.std(confidences)),
            'min_confidence': float(np.min(confidences)),
            'max_confidence': float(np.max(confidences)),
            'median_confidence': float(np.median(confidences))
        }
    
    def predict_from_csv(self, csv_path: str) -> Dict[str, Any]:
        """CSVファイルから特徴量を読み込んで予測"""
        try:
            df = pd.read_csv(csv_path)
            video_name = Path(csv_path).stem
            
            print(f"✅ CSV読み込み: {csv_path}")
            print(f"   サンプル数: {len(df)}")
            print(f"   特徴量数: {len(df.columns)}")
            
            return self.predict_from_features(df, video_name)
            
        except Exception as e:
            print(f"❌ CSV読み込みエラー: {e}")
            return {}
    
    def visualize_predictions(self, results: Dict[str, Any], save_path: str = None):
        """予測結果を可視化"""
        if not results or 'frame_predictions' not in results:
            print("可視化する予測結果がありません")
            return
        
        frame_predictions = results['frame_predictions']
        frames = [pred['frame'] for pred in frame_predictions]
        phase_ids = [pred['predicted_phase_id'] for pred in frame_predictions]
        confidences = [pred['confidence'] for pred in frame_predictions]
        
        fig, axes = plt.subplots(3, 1, figsize=(15, 12))
        
        # 局面予測の時系列表示
        axes[0].plot(frames, phase_ids, 'o-', markersize=3, linewidth=1)
        axes[0].set_title(f'Predicted Tennis Phases - {results["video_name"]}')
        axes[0].set_xlabel('Frame Number')
        axes[0].set_ylabel('Phase ID')
        axes[0].set_yticks(range(len(self.phase_labels)))
        axes[0].set_yticklabels(self.phase_labels, rotation=45)
        axes[0].grid(True, alpha=0.3)
        
        # 信頼度の時系列表示
        axes[1].plot(frames, confidences, 'g-', linewidth=1)
        axes[1].set_title('Prediction Confidence')
        axes[1].set_xlabel('Frame Number')
        axes[1].set_ylabel('Confidence')
        axes[1].set_ylim(0, 1)
        axes[1].grid(True, alpha=0.3)
        
        # 局面分布の棒グラフ
        phase_dist = results['phase_distribution']
        phase_names = list(phase_dist.keys())
        percentages = [phase_dist[name]['percentage'] for name in phase_names]
        
        bars = axes[2].bar(phase_names, percentages, color='skyblue', alpha=0.7)
        axes[2].set_title('Phase Distribution')
        axes[2].set_xlabel('Phase')
        axes[2].set_ylabel('Percentage (%)')
        axes[2].tick_params(axis='x', rotation=45)
        
        # 棒グラフに数値を表示
        for bar, pct in zip(bars, percentages):
            height = bar.get_height()
            axes[2].text(bar.get_x() + bar.get_width()/2., height + 0.1,
                        f'{pct:.1f}%', ha='center', va='bottom')
        
        plt.tight_layout()
        
        # 保存
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            print(f"可視化結果を保存: {save_path}")
        
        plt.show()
    
    def save_predictions_to_csv(self, results: Dict[str, Any], output_path: str):
        """予測結果をCSVファイルに保存"""
        if not results or 'frame_predictions' not in results:
            print("保存する予測結果がありません")
            return
        
        # データフレーム作成
        df_data = []
        for pred in results['frame_predictions']:
            row = {
                'frame': pred['frame'],
                'predicted_phase_id': pred['predicted_phase_id'],
                'predicted_phase_name': pred['predicted_phase_name'],
                'confidence': pred['confidence'],
                'vote_count': pred['vote_count']
            }
            
            # 各局面の確率を追加
            for i, prob in enumerate(pred['probabilities']):
                row[f'prob_{self.phase_labels[i]}'] = prob
            
            df_data.append(row)
        
        df = pd.DataFrame(df_data)
        df.to_csv(output_path, index=False)
        
        print(f"✅ 予測結果をCSVに保存: {output_path}")
        print(f"   総フレーム数: {len(df)}")
    
    def predict_video_pipeline(self, features_csv_path: str, 
                             output_dir: str = None) -> Dict[str, Any]:
        """動画の完全な予測パイプライン"""
        print("=== テニス動画局面予測パイプライン ===")
        
        if output_dir is None:
            output_dir = Path(features_csv_path).parent / "predictions"
        
        output_dir = Path(output_dir)
        output_dir.mkdir(exist_ok=True)
        
        # 予測実行
        results = self.predict_from_csv(features_csv_path)
        
        if not results:
            print("❌ 予測に失敗しました")
            return {}
        
        # 結果を表示
        print(f"\n=== 予測結果サマリー ===")
        print(f"動画: {results['video_name']}")
        print(f"総フレーム数: {results['total_frames']}")
        print(f"平均信頼度: {results['confidence_stats']['mean_confidence']:.3f}")
        
        print("\n局面分布:")
        for phase, stats in results['phase_distribution'].items():
            print(f"  {phase}: {stats['count']}フレーム ({stats['percentage']:.1f}%)")
        
        # 結果保存
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        video_name = results['video_name']
        
        # CSV保存
        csv_output_path = output_dir / f"predictions_{video_name}_{timestamp}.csv"
        self.save_predictions_to_csv(results, csv_output_path)
        
        # 可視化保存
        viz_output_path = output_dir / f"predictions_viz_{video_name}_{timestamp}.png"
        self.visualize_predictions(results, viz_output_path)
        
        # JSON保存（詳細結果）
        json_output_path = output_dir / f"predictions_detail_{video_name}_{timestamp}.json"
        with open(json_output_path, 'w', encoding='utf-8') as f:
            json.dump(results, f, indent=2, ensure_ascii=False)
        
        print(f"\n✅ 予測完了！結果は {output_dir} に保存されました")
        
        return results

def main():
    """メイン関数"""
    print("=== テニス動画局面分類PyTorch LSTM予測ツール ===")
    
    # 予測器を初期化
    try:
        predictor = TennisLSTMPredictor()
    except Exception as e:
        print(f"❌ 予測器の初期化に失敗: {e}")
        return
    
    # 特徴量ファイル確認
    training_data_dir = Path("training_data")
    feature_files = list(training_data_dir.glob("tennis_features_dataset_*.csv"))
    
    print(f"\n=== 特徴量ファイル確認 ===")
    print(f"利用可能な特徴量ファイル: {len(feature_files)}ファイル")
    
    if not feature_files:
        print("❌ 特徴量ファイルが見つかりません")
        print("feature_extractor.py を実行して特徴量を抽出してください")
        return
    
    for file in feature_files:
        print(f"  - {file.name}")
    
    # 最新の特徴量ファイルで予測実行
    latest_feature_file = max(feature_files, key=lambda x: x.stat().st_mtime)
    print(f"\n最新の特徴量ファイルで予測実行: {latest_feature_file.name}")
    
    try:
        # 予測パイプライン実行
        results = predictor.predict_video_pipeline(str(latest_feature_file))
        
        if results:
            print(f"\n🎉 予測処理が完了しました！")
        else:
            print(f"\n❌ 予測処理に失敗しました")
            
    except Exception as e:
        print(f"❌ 予期しないエラーが発生しました: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main()
