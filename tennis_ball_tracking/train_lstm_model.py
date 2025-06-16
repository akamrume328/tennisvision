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
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import classification_report, confusion_matrix, f1_score
import joblib

# PyTorch
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
import torch.nn.functional as F
from torch.optim.lr_scheduler import ReduceLROnPlateau
from torch.cuda.amp import GradScaler, autocast

# GPU設定
def setup_gpu_config():
    """GPU設定とCUDA最適化"""
    if torch.cuda.is_available():
        device_count = torch.cuda.device_count()
        print(f"🚀 GPU検出: {device_count}台")
        
        for i in range(device_count):
            gpu_name = torch.cuda.get_device_name(i)
            memory_total = torch.cuda.get_device_properties(i).total_memory / 1024**3
            print(f"   GPU {i}: {gpu_name} ({memory_total:.1f}GB)")
        
        print(f"✅ CUDA version: {torch.version.cuda}")
        print(f"✅ cuDNN version: {torch.backends.cudnn.version()}")
        
        torch.backends.cudnn.benchmark = True
        torch.backends.cudnn.enabled = True
        return torch.device('cuda')
    else:
        print("⚠️  GPU未検出: CPUで実行します")
        return torch.device('cpu')

DEVICE = setup_gpu_config()
GPU_AVAILABLE = DEVICE.type == 'cuda'

plt.style.use('default')
sns.set_palette("husl")

class TennisLSTMModel(nn.Module):
    """PyTorch LSTM モデル（信頼度重み付け対応）"""
    
    def __init__(self, input_size: int, hidden_sizes: List[int], num_classes: int, 
                 dropout_rate: float = 0.3, model_type: str = "bidirectional", 
                 use_batch_norm: bool = True, enable_confidence_weighting: bool = True):
        super(TennisLSTMModel, self).__init__()
        
        self.input_size = input_size
        self.hidden_sizes = hidden_sizes
        self.num_classes = num_classes
        self.dropout_rate = dropout_rate
        self.model_type = model_type
        self.use_batch_norm = use_batch_norm
        self.enable_confidence_weighting = enable_confidence_weighting
        
        # 信頼度重み付け用の注意機構
        if self.enable_confidence_weighting:
            self.confidence_attention = nn.Sequential(
                nn.Linear(input_size, hidden_sizes[0] // 4),
                nn.ReLU(),
                nn.Linear(hidden_sizes[0] // 4, 1),
                nn.Sigmoid()
            )
        
        # LSTM層を構築
        self._build_lstm_layers()
        
        # 出力層
        self.dropout = nn.Dropout(dropout_rate)
        self.fc1 = nn.Linear(self.lstm_output_size, 128)
        self.fc2 = nn.Linear(128, 64)
        self.fc_out = nn.Linear(64, num_classes)
        
        # 正規化層
        if self.use_batch_norm:
            self.bn1 = nn.BatchNorm1d(128)
            self.bn2 = nn.BatchNorm1d(64)
        else:
            self.ln1 = nn.LayerNorm(128)
            self.ln2 = nn.LayerNorm(64)
        
        self._init_weights()
    
    def _build_lstm_layers(self):
        """LSTM層の構築"""
        if self.model_type == "simple":
            self.lstm1 = nn.LSTM(self.input_size, self.hidden_sizes[0], batch_first=True, dropout=self.dropout_rate)
            self.lstm2 = nn.LSTM(self.hidden_sizes[0], self.hidden_sizes[1], batch_first=True, dropout=self.dropout_rate)
            self.lstm_output_size = self.hidden_sizes[1]
            
        elif self.model_type == "bidirectional":
            self.lstm1 = nn.LSTM(self.input_size, self.hidden_sizes[0], batch_first=True, 
                                bidirectional=True, dropout=self.dropout_rate)
            self.lstm2 = nn.LSTM(self.hidden_sizes[0] * 2, self.hidden_sizes[1], batch_first=True,
                                bidirectional=True, dropout=self.dropout_rate)
            self.lstm_output_size = self.hidden_sizes[1] * 2
            
        elif self.model_type == "stacked":
            self.lstm1 = nn.LSTM(self.input_size, self.hidden_sizes[0], batch_first=True, dropout=self.dropout_rate)
            self.lstm2 = nn.LSTM(self.hidden_sizes[0], self.hidden_sizes[1], batch_first=True, dropout=self.dropout_rate)
            self.lstm3 = nn.LSTM(self.hidden_sizes[1], 64, batch_first=True, dropout=self.dropout_rate)
            self.lstm_output_size = 64
    
    def _init_weights(self):
        """重みの初期化"""
        for module in self.modules():
            if isinstance(module, nn.LSTM):
                for name, param in module.named_parameters():
                    if 'weight_ih' in name:
                        torch.nn.init.xavier_uniform_(param.data)
                    elif 'weight_hh' in name:
                        torch.nn.init.orthogonal_(param.data)
                    elif 'bias' in name:
                        param.data.fill_(0)
            elif isinstance(module, nn.Linear):
                torch.nn.init.xavier_uniform_(module.weight)
                torch.nn.init.zeros_(module.bias)
    
    def forward(self, x, confidence_scores=None):
        """フォワードパス"""
        # 信頼度重み付け
        if self.enable_confidence_weighting and confidence_scores is not None:
            attention_weights = self.confidence_attention(x).squeeze(-1)
            combined_weights = attention_weights * confidence_scores
            x = x * combined_weights.unsqueeze(-1)
        
        # LSTM層
        lstm_out = self._forward_lstm(x)
        
        # 最後のタイムステップを使用
        lstm_out = lstm_out[:, -1, :]
        lstm_out = self.dropout(lstm_out)
        
        # Dense層
        out = F.relu(self.fc1(lstm_out))
        out = self._apply_normalization(out, self.bn1 if self.use_batch_norm else self.ln1)
        out = self.dropout(out)
        
        out = F.relu(self.fc2(out))
        out = self._apply_normalization(out, self.bn2 if self.use_batch_norm else self.ln2)
        out = self.dropout(out)
        
        return self.fc_out(out)
    
    def _forward_lstm(self, x):
        """LSTM層のフォワードパス"""
        if self.model_type == "simple":
            lstm_out, _ = self.lstm1(x)
            lstm_out = self.dropout(lstm_out)
            lstm_out, _ = self.lstm2(lstm_out)
        elif self.model_type == "bidirectional":
            lstm_out, _ = self.lstm1(x)
            lstm_out = self.dropout(lstm_out)
            lstm_out, _ = self.lstm2(lstm_out)
        elif self.model_type == "stacked":
            lstm_out, _ = self.lstm1(x)
            lstm_out = self.dropout(lstm_out)
            lstm_out, _ = self.lstm2(lstm_out)
            lstm_out = self.dropout(lstm_out)
            lstm_out, _ = self.lstm3(lstm_out)
        return lstm_out
    
    def _apply_normalization(self, x, norm_layer):
        """正規化層の適用"""
        if self.use_batch_norm and x.size(0) > 1:
            return norm_layer(x)
        elif not self.use_batch_norm:
            return norm_layer(x)
        return x

class TennisLSTMTrainer:
    """テニス局面分類LSTM学習クラス"""
    
    def __init__(self, training_data_dir: str = "training_data"):
        self.training_data_dir = Path(training_data_dir)
        self.features_dir = self.training_data_dir / "features"
        self.models_dir = self.training_data_dir / "lstm_models"
        
        self._setup_directories()
        self._setup_parameters()
        
        self.phase_labels = [
            "point_interval", "rally", "serve_preparation",
            "serve_front_deuce", "serve_front_ad", "serve_back_deuce",
            "serve_back_ad", "changeover"
        ]
        
        self.scaler = None
        self.model = None
        self.history = None
        self.label_map: Optional[Dict[int, int]] = None
        self.active_phase_labels: Optional[List[str]] = None
        
        print(f"PyTorch LSTM学習器初期化完了")
        print(f"データディレクトリ: {self.training_data_dir}")
        print(f"使用デバイス: {DEVICE}")
    
    def _setup_directories(self):
        """ディレクトリの設定"""
        try:
            self.training_data_dir.mkdir(exist_ok=True)
            self.features_dir.mkdir(exist_ok=True)
            self.models_dir.mkdir(exist_ok=True)
            
            # 書き込み権限テスト
            test_file = self.models_dir / "test_write.tmp"
            test_file.write_text("test")
            test_file.unlink()
            
        except Exception as e:
            print(f"⚠️  ディレクトリ作成エラー: {e}")
            import tempfile
            self.models_dir = Path(tempfile.gettempdir()) / "tennis_lstm_models"
            self.models_dir.mkdir(exist_ok=True)
    
    def _setup_parameters(self):
        """学習パラメータの設定"""
        if GPU_AVAILABLE:
            self.sequence_length = 30
            self.overlap_ratio = 0.5
            self.lstm_units = [256, 128]
            self.dropout_rate = 0.3
            self.learning_rate = 0.001
            self.batch_size = 64
        else:
            self.sequence_length = 20
            self.overlap_ratio = 0.5
            self.lstm_units = [128, 64]
            self.dropout_rate = 0.3
            self.learning_rate = 0.001
            self.batch_size = 32
    
    def select_dataset_file(self) -> Optional[str]:
        """データセットファイルの選択"""
        # 複数のパターンでファイル検索
        dataset_files = []
        
        # features フォルダから検索
        patterns = ["tennis_features_dataset_*.csv", "tennis_features_*.csv"]
        for pattern in patterns:
            dataset_files.extend(list(self.features_dir.glob(pattern)))
        
        # features フォルダにない場合は training_data フォルダから検索
        if not dataset_files:
            for pattern in patterns:
                dataset_files.extend(list(self.training_data_dir.glob(pattern)))
        
        # 重複を除去（同じファイルが複数パターンで検出される場合）
        dataset_files = list(set(dataset_files))
        
        if not dataset_files:
            print("❌ 特徴量データセットファイルが見つかりません")
            print("検索パターン: tennis_features_dataset_*.csv, tennis_features_*.csv")
            return None
        
        dataset_files.sort(key=lambda x: x.stat().st_mtime, reverse=True)
        
        print(f"\n=== 特徴量データセットファイル選択 ===")
        for i, file_path in enumerate(dataset_files, 1):
            stat = file_path.stat()
            size_mb = stat.st_size / (1024 * 1024)
            mtime = datetime.fromtimestamp(stat.st_mtime)
            
            print(f"{i}: {file_path.name}")
            print(f"   更新日時: {mtime.strftime('%Y-%m-%d %H:%M:%S')}")
            print(f"   サイズ: {size_mb:.2f} MB")
        
        print(f"{len(dataset_files) + 1}: 自動選択（最新ファイル）")
        
        try:
            choice = input(f"\n選択してください (1-{len(dataset_files) + 1}): ").strip()
            choice_num = int(choice)
            
            if 1 <= choice_num <= len(dataset_files):
                return str(dataset_files[choice_num - 1])
            elif choice_num == len(dataset_files) + 1:
                return str(dataset_files[0])
            else:
                return None
        except ValueError:
            return None
    
    def load_dataset(self, csv_path: str = None, sample_ratio: float = 1.0) -> Tuple[pd.DataFrame, bool]:
        """データセットの読み込み"""
        if csv_path:
            dataset_path = Path(csv_path)
        else:
            selected_path = self.select_dataset_file()
            if not selected_path:
                return pd.DataFrame(), False
            dataset_path = Path(selected_path)
        
        try:
            file_size_mb = dataset_path.stat().st_size / (1024 * 1024)
            print(f"📂 データセット読み込み: {dataset_path.name} ({file_size_mb:.1f}MB)")
            
            if sample_ratio < 1.0:
                df = self._load_sampled_data(dataset_path, sample_ratio)
            else:
                df = pd.read_csv(dataset_path)
            
            print(f"✅ 読み込み完了: {len(df):,} 行, {len(df.columns)} 特徴量")
            self.analyze_dataset(df)
            return df, True
            
        except Exception as e:
            print(f"❌ データセット読み込みエラー: {e}")
            return pd.DataFrame(), False
    
    def _load_sampled_data(self, dataset_path: Path, sample_ratio: float) -> pd.DataFrame:
        """サンプリング読み込み"""
        with open(dataset_path, 'r') as f:
            total_lines = sum(1 for line in f) - 1
        
        sample_size = int(total_lines * sample_ratio)
        skip_rows = sorted(np.random.choice(
            range(1, total_lines + 1), 
            size=total_lines - sample_size, 
            replace=False
        ))
        
        return pd.read_csv(dataset_path, skiprows=skip_rows)
    
    def analyze_dataset(self, df: pd.DataFrame):
        """データセットの基本分析"""
        print(f"\n=== データセット分析 ===")
        print(f"総データ数: {len(df):,}")
        
        # ラベル分布
        label_counts = df['label'].value_counts().sort_index()
        print("\nラベル分布:")
        for label_id, count in label_counts.items():
            if 0 <= label_id < len(self.phase_labels):
                phase_name = self.phase_labels[label_id]
                percentage = (count / len(df)) * 100
                print(f"  {label_id} ({phase_name}): {count:,} ({percentage:.1f}%)")
        
        # 動画情報
        if 'video_name' in df.columns:
            video_count = df['video_name'].nunique()
            print(f"\n動画数: {video_count}")
        
        # シーケンス予測
        if 'video_name' in df.columns:
            total_sequences = self._estimate_sequences(df)
            print(f"予想シーケンス数: {total_sequences:,}")
    
    def _estimate_sequences(self, df: pd.DataFrame) -> int:
        """シーケンス数の予測"""
        total_sequences = 0
        step_size = max(1, int(self.sequence_length * (1 - self.overlap_ratio)))
        
        for video in df['video_name'].unique():
            video_data = df[df['video_name'] == video]
            video_frames = len(video_data)
            
            if video_frames >= self.sequence_length:
                sequences = max(0, (video_frames - self.sequence_length) // step_size + 1)
                total_sequences += sequences
        
        return total_sequences
    
    def _extract_confidence_scores(self, df: pd.DataFrame) -> np.ndarray:
        """信頼度スコアの抽出"""
        if 'interpolation_confidence' in df.columns:
            return df['interpolation_confidence'].fillna(1.0).values
        elif 'interpolated' in df.columns:
            interpolated = df['interpolated'].fillna(False).astype(bool)
            return np.where(interpolated, 0.7, 1.0)
        else:
            return np.ones(len(df))

    def _create_sequences(self, X_scaled: np.ndarray, y_labels: np.ndarray, 
                         video_names: np.ndarray, confidence_scores: np.ndarray) -> Tuple[List, List, List]:
        """シーケンスの作成"""
        sequences = []
        sequence_labels = []
        confidence_sequences = []
        
        step_size = max(1, int(self.sequence_length * (1 - self.overlap_ratio)))
        
        for video in np.unique(video_names):
            video_mask = video_names == video
            video_X = X_scaled[video_mask]
            video_y = y_labels[video_mask]
            video_conf = confidence_scores[video_mask]
            
            # 動画内シーケンス作成
            for i in range(0, len(video_X) - self.sequence_length + 1, step_size):
                seq_X = video_X[i:i + self.sequence_length]
                seq_y = video_y[i:i + self.sequence_length]
                seq_conf = video_conf[i:i + self.sequence_length]
                
                target_label = seq_y[-1]
                if 0 <= target_label < len(self.phase_labels):
                    sequences.append(seq_X)
                    sequence_labels.append(target_label)
                    confidence_sequences.append(seq_conf)
        
        return sequences, sequence_labels, confidence_sequences

    def prepare_sequences(self, df: pd.DataFrame) -> Tuple[np.ndarray, np.ndarray, np.ndarray, List[str]]:
        """時系列シーケンスデータの作成（CUDAエラー対策強化）"""
        print(f"\n=== 時系列シーケンス作成（CUDAエラー対策強化） ===")
        
        # ラベル値の検証と修正（強化版）
        df_cleaned = self._validate_and_clean_labels(df)
        
        if len(df_cleaned) == 0:
            print("❌ 有効なデータがありません")
            return np.array([]), np.array([]), np.array([]), []
        
        # 特徴量抽出
        exclude_columns = ['label', 'video_name', 'frame_number', 'interpolated', 'interpolation_confidence']
        feature_columns = [col for col in df_cleaned.columns if col not in exclude_columns]
        numeric_features = df_cleaned[feature_columns].select_dtypes(include=[np.number]).columns
        
        print(f"使用特徴量数: {len(numeric_features)}")
        
        X_features = df_cleaned[numeric_features].values
        y_labels = df_cleaned['label'].values
        video_names = df_cleaned['video_name'].values
        
        # ラベル値の最終検証
        unique_labels = np.unique(y_labels)
        print(f"シーケンス作成前のラベル値検証:")
        print(f"  ユニークラベル: {sorted(unique_labels)}")
        print(f"  最小値: {np.min(unique_labels)}, 最大値: {np.max(unique_labels)}")
        print(f"  期待される最大値: {len(self.phase_labels) - 1}")
        
        # CUDA用のラベル値検証
        if np.min(unique_labels) < 0:
            print(f"❌ 負のラベル値が検出されました: {np.min(unique_labels)}")
            return np.array([]), np.array([]), np.array([]), []
        
        if np.max(unique_labels) >= len(self.phase_labels):
            print(f"❌ 範囲外のラベル値が検出されました: {np.max(unique_labels)} >= {len(self.phase_labels)}")
            return np.array([]), np.array([]), np.array([]), []
        
        # 信頼度スコア抽出
        confidence_scores = self._extract_confidence_scores(df_cleaned)
        
        # データ前処理
        X_features = np.nan_to_num(X_features, nan=0.0, posinf=1e6, neginf=-1e6)
        # self.scaler = StandardScaler() # スケーラーの初期化を削除
        # X_scaled = self.scaler.fit_transform(X_features) # ここでのスケーリングを削除
        
        # シーケンス作成
        sequences, sequence_labels, confidence_sequences = self._create_sequences(
            X_features, y_labels, video_names, confidence_scores # X_scaled の代わりに X_features を使用
        )
        
        if len(sequences) == 0:
            print("❌ シーケンスが作成されませんでした")
            return np.array([]), np.array([]), np.array([]), []
        
        # シーケンス後のラベル値検証
        final_unique_labels = np.unique(sequence_labels)
        print(f"シーケンス作成後のラベル値検証:")
        print(f"  ユニークラベル: {sorted(final_unique_labels)}")
        print(f"  最小値: {np.min(final_unique_labels)}, 最大値: {np.max(final_unique_labels)}")
        
        print(f"作成されたシーケンス数: {len(sequences)}")
        print(f"シーケンス形状: {np.array(sequences).shape}")
        print(f"✅ CUDAエラー対策：ラベル値検証完了")
        
        return np.array(sequences), np.array(sequence_labels), np.array(confidence_sequences), list(numeric_features)
    
    def _validate_and_clean_labels(self, df: pd.DataFrame) -> pd.DataFrame:
        """ラベル値の検証と修正（強化版）"""
        print(f"\n=== ラベル値検証・修正（強化版） ===")
        
        # 元のデータ統計
        original_count = len(df)
        unique_labels_before = sorted(df['label'].unique())
        print(f"修正前データ数: {original_count:,}")
        print(f"修正前ラベル値: {unique_labels_before}")
        print(f"修正前ラベル値範囲: {min(unique_labels_before)} ～ {max(unique_labels_before)}")
        
        # NaN値の処理
        nan_mask = df['label'].isna()
        if nan_mask.sum() > 0:
            print(f"⚠️  NaNラベル検出: {nan_mask.sum():,}行 - 除去")
            df = df[~nan_mask].copy()
        
        # 無限値の処理
        inf_mask = np.isinf(df['label'])
        if inf_mask.sum() > 0:
            print(f"⚠️  無限値ラベル検出: {inf_mask.sum():,}行 - 除去")
            df = df[~inf_mask].copy()
        
        # 有効なラベル値の範囲（0 ～ len(phase_labels)-1）
        valid_range = list(range(len(self.phase_labels)))
        print(f"有効ラベル範囲: {valid_range}")
        
        # 無効なラベルを特定
        invalid_mask = ~df['label'].isin(valid_range)
        invalid_count = invalid_mask.sum()
        
        if invalid_count > 0:
            invalid_labels = sorted(df[invalid_mask]['label'].unique())
            print(f"⚠️  無効ラベル検出: {invalid_count:,}行")
            print(f"無効ラベル値: {invalid_labels}")
            
            # 無効ラベルの詳細分析
            for invalid_label in invalid_labels:
                count = (df['label'] == invalid_label).sum()
                print(f"  ラベル {invalid_label}: {count:,}行")
            
            # 無効ラベルを除去
            df = df[~invalid_mask].copy()
            print(f"✅ 無効ラベル除去完了")
        
        # 最終統計
        final_count = len(df)
        if final_count > 0:
            unique_labels_after = sorted(df['label'].unique())
            print(f"修正後データ数: {final_count:,}")
            print(f"修正後ラベル値: {unique_labels_after}")
            print(f"修正後ラベル値範囲: {min(unique_labels_after)} ～ {max(unique_labels_after)}")
            
            # データ保持率
            retention_rate = (final_count / original_count) * 100
            print(f"データ保持率: {retention_rate:.1f}%")
            
            # ラベル値の型とデータ型を確認
            print(f"ラベル列データ型: {df['label'].dtype}")
            
            # 整数型に変換（CUDA用）
            df['label'] = df['label'].astype(int)
            print(f"ラベル列を整数型に変換: {df['label'].dtype}")
            
        else:
            print("❌ 有効なデータが残っていません")
        
        return df

    def build_model(self, input_size: int, num_classes: int, model_type: str = "bidirectional", 
                   enable_confidence_weighting: bool = False) -> TennisLSTMModel:
        """PyTorch LSTMモデルの構築"""
        print(f"\n=== {model_type.upper()} PyTorch LSTMモデル構築 ===")
        
        use_batch_norm = self.batch_size > 4
        
        model = TennisLSTMModel(
            input_size=input_size,
            hidden_sizes=self.lstm_units,
            num_classes=num_classes,
            dropout_rate=self.dropout_rate,
            model_type=model_type,
            use_batch_norm=use_batch_norm,
            enable_confidence_weighting=enable_confidence_weighting
        )
        
        model = model.to(DEVICE)
        
        total_params = sum(p.numel() for p in model.parameters())
        print(f"✅ モデル構築完了: {total_params:,}パラメータ")
        
        return model
    
    def train_model(self, X: np.ndarray, y: np.ndarray, confidence_scores: np.ndarray = None, 
                   model_type: str = "bidirectional") -> Dict:
        """モデル学習の実行（CUDAエラー対策強化）"""
        print(f"\n=== PyTorch LSTM学習開始（CUDAエラー対策強化） ===")
        
        # 入力データの最終検証
        print(f"学習データ検証:")
        print(f"  X shape: {X.shape}")
        print(f"  y shape: {y.shape}")
        
        # ラベル値の詳細検証と再マッピング
        original_unique_labels = sorted(np.unique(y))
        # キーをPythonのint型に変換
        self.label_map = {int(orig_label): i for i, orig_label in enumerate(original_unique_labels)}
        self.active_phase_labels = [self.phase_labels[i] for i in original_unique_labels if i < len(self.phase_labels)]
        
        y_remapped = np.array([self.label_map[label] for label in y])
        
        num_classes = len(original_unique_labels) # 再マッピング後のクラス数
        min_label_remapped = np.min(y_remapped)
        max_label_remapped = np.max(y_remapped)
        
        print(f"  オリジナルユニークラベル: {original_unique_labels}")
        print(f"  再マッピング後ユニークラベル: {sorted(np.unique(y_remapped))}")
        print(f"  アクティブな局面ラベル: {self.active_phase_labels}")
        print(f"  クラス数 (モデル用): {num_classes}")
        print(f"  再マッピング後ラベル範囲: {min_label_remapped} ～ {max_label_remapped}")
        
        # CUDAエラー対策：ラベル値の最終検証 (再マッピング後)
        if min_label_remapped < 0:
            raise ValueError(f"再マッピング後、負のラベル値が検出されました: {min_label_remapped}")
        
        # num_classes はモデルの出力層のユニット数なので、ラベルは 0 から num_classes-1 の範囲にあるべき
        if max_label_remapped >= num_classes:
            raise ValueError(
                f"再マッピング後のラベル値が範囲外です: {max_label_remapped} >= {num_classes}"
            )
        
        # ラベルデータ型の確認と修正
        y_processed = y_remapped.astype(np.int64)  # CUDA用の型に変換
        print(f"  処理後ラベルデータ型: {y_processed.dtype}")
        
        # 信頼度スコアの情報
        if confidence_scores is not None:
            print(f"信頼度重み付け: 有効")
            print(f"  信頼度スコア形状: {confidence_scores.shape}")
            print(f"  平均信頼度: {np.mean(confidence_scores):.3f}")
        else:
            print(f"信頼度重み付け: 無効")
        
        try:
            # データ分割 (再マッピングされたy_processedを使用)
            if confidence_scores is not None:
                X_train_raw, X_test_raw, y_train, y_test, conf_train, conf_test = train_test_split(
                    X, y_processed, confidence_scores, test_size=0.2, random_state=42, stratify=y_processed
                )
                X_train_raw, X_val_raw, y_train, y_val, conf_train, conf_val = train_test_split(
                    X_train_raw, y_train, conf_train, test_size=0.2, random_state=42, stratify=y_train
                )
            else:
                X_train_raw, X_test_raw, y_train, y_test = train_test_split(
                    X, y_processed, test_size=0.2, random_state=42, stratify=y_processed
                )
                X_train_raw, X_val_raw, y_train, y_val = train_test_split(
                    X_train_raw, y_train, test_size=0.2, random_state=42, stratify=y_train
                )
                conf_train = conf_val = conf_test = None

            # スケーラーの学習と適用
            self.scaler = StandardScaler()
            
            # X_train_raw の形状を (サンプル数 * シーケンス長, 特徴量数) に変形してスケーラーを学習
            nsamples_train, nx_train, ny_train = X_train_raw.shape
            X_train_reshaped = X_train_raw.reshape((nsamples_train * nx_train, ny_train))
            self.scaler.fit(X_train_reshaped)
            
            # 各データセットにスケーラーを適用
            X_train = self.scaler.transform(X_train_reshaped).reshape(nsamples_train, nx_train, ny_train)
            
            nsamples_val, nx_val, ny_val = X_val_raw.shape
            X_val_reshaped = X_val_raw.reshape((nsamples_val * nx_val, ny_val))
            X_val = self.scaler.transform(X_val_reshaped).reshape(nsamples_val, nx_val, ny_val)
            
            nsamples_test, nx_test, ny_test = X_test_raw.shape
            X_test_reshaped = X_test_raw.reshape((nsamples_test * nx_test, ny_test))
            X_test = self.scaler.transform(X_test_reshaped).reshape(nsamples_test, nx_test, ny_test)

            print(f"データ分割完了:")
            print(f"  学習データ: {X_train.shape}")
            print(f"  検証データ: {X_val.shape}")
            print(f"  テストデータ: {X_test.shape}")
            
        except Exception as e:
            print(f"❌ データ分割エラー: {e}")
            raise
        
        # モデル構築
        input_size = X.shape[2]
        enable_confidence_weighting = confidence_scores is not None
        
        # クラス数を実際のユニークラベル数に設定 (再マッピング後のクラス数)
        actual_num_classes = num_classes 
        print(f"モデル構築パラメータ:")
        print(f"  入力サイズ: {input_size}")
        print(f"  クラス数: {actual_num_classes}")
        
        self.model = self.build_model(input_size, actual_num_classes, model_type, enable_confidence_weighting)
        
        # データローダー準備
        train_loader, val_loader, test_loader = self._prepare_data_loaders(
            X_train, y_train, X_val, y_val, X_test, y_test,
            conf_train, conf_val, conf_test
        )
        
        # 学習実行
        print("🚀 学習開始...")
        history = self._train_loop(train_loader, val_loader)
        
        # テスト評価
        test_results = self._evaluate_test(test_loader)
        
        self.history = history
        
        return {
            'test_accuracy': test_results['accuracy'],
            'test_loss': test_results['loss'],
            'f1_score': test_results['f1'],
            'y_test': test_results['y_true'],
            'y_pred': test_results['y_pred'],
            'y_pred_proba': test_results['y_proba'],
            'model_type': model_type,
            'gpu_used': GPU_AVAILABLE,
            'confidence_weighting_used': enable_confidence_weighting,
            'history': history
        }
    
    def _prepare_data_loaders(self, X_train, y_train, X_val, y_val, X_test, y_test,
                             conf_train=None, conf_val=None, conf_test=None):
        """データローダーの準備"""
        # テンソル変換
        X_train_tensor = torch.FloatTensor(X_train).to(DEVICE)
        y_train_tensor = torch.LongTensor(y_train).to(DEVICE)
        X_val_tensor = torch.FloatTensor(X_val).to(DEVICE)
        y_val_tensor = torch.LongTensor(y_val).to(DEVICE)
        X_test_tensor = torch.FloatTensor(X_test).to(DEVICE)
        y_test_tensor = torch.LongTensor(y_test).to(DEVICE)
        
        # データセット作成
        if conf_train is not None:
            conf_train_tensor = torch.FloatTensor(conf_train).to(DEVICE)
            conf_val_tensor = torch.FloatTensor(conf_val).to(DEVICE)
            conf_test_tensor = torch.FloatTensor(conf_test).to(DEVICE)
            
            train_dataset = TensorDataset(X_train_tensor, y_train_tensor, conf_train_tensor)
            val_dataset = TensorDataset(X_val_tensor, y_val_tensor, conf_val_tensor)
            test_dataset = TensorDataset(X_test_tensor, y_test_tensor, conf_test_tensor)
        else:
            train_dataset = TensorDataset(X_train_tensor, y_train_tensor)
            val_dataset = TensorDataset(X_val_tensor, y_val_tensor)
            test_dataset = TensorDataset(X_test_tensor, y_test_tensor)
        
        # データローダー作成
        train_loader = DataLoader(train_dataset, batch_size=self.batch_size, shuffle=True, drop_last=True)
        val_loader = DataLoader(val_dataset, batch_size=self.batch_size, shuffle=False, drop_last=True)
        test_loader = DataLoader(test_dataset, batch_size=self.batch_size, shuffle=False, drop_last=False)
        
        return train_loader, val_loader, test_loader
    
    def _train_loop(self, train_loader, val_loader):
        """学習ループ"""
        criterion = nn.CrossEntropyLoss()
        optimizer = optim.Adam(self.model.parameters(), lr=self.learning_rate, weight_decay=1e-4)
        scheduler = ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=10)
        scaler = GradScaler() if GPU_AVAILABLE else None
        
        history = {'train_loss': [], 'train_acc': [], 'val_loss': [], 'val_acc': [], 'lr': []}
        best_val_loss = float('inf')
        best_model_state = None
        patience_counter = 0
        patience = 25 if GPU_AVAILABLE else 15
        epochs = 150 if GPU_AVAILABLE else 100
        
        for epoch in range(epochs):
            # 学習フェーズ
            train_loss, train_acc = self._train_epoch(train_loader, criterion, optimizer, scaler)
            
            # 検証フェーズ
            val_loss, val_acc = self._validate_epoch(val_loader, criterion)
            
            # スケジューラー更新
            scheduler.step(val_loss)
            
            # 履歴保存
            history['train_loss'].append(train_loss)
            history['train_acc'].append(train_acc)
            history['val_loss'].append(val_loss)
            history['val_acc'].append(val_acc)
            history['lr'].append(optimizer.param_groups[0]['lr'])
            
            # 早期停止チェック
            if val_loss < best_val_loss:
                best_val_loss = val_loss
                patience_counter = 0
                best_model_state = self.model.state_dict().copy()
            else:
                patience_counter += 1
                if patience_counter >= patience:
                    print(f"早期停止: {patience}エポック改善なし")
                    break
            
            if (epoch + 1) % 10 == 0:
                print(f'Epoch [{epoch+1}/{epochs}], Train: {train_acc:.4f}, Val: {val_acc:.4f}')
        
        # 最良モデルをロード
        if best_model_state is not None:
            self.model.load_state_dict(best_model_state)
        
        return history
    
    def _train_epoch(self, train_loader, criterion, optimizer, scaler):
        """1エポックの学習（エラーハンドリング強化）"""
        self.model.train()
        total_loss = 0.0
        correct = 0
        total = 0
        
        for batch_idx, batch_data in enumerate(train_loader):
            try:
                if len(batch_data) == 3:
                    batch_X, batch_y, batch_conf = batch_data
                else:
                    batch_X, batch_y = batch_data
                    batch_conf = None
                
                # バッチデータの検証
                if torch.any(batch_y < 0) or torch.any(batch_y >= self.model.num_classes):
                    print(f"⚠️  バッチ {batch_idx}: 無効なラベル値を検出")
                    print(f"   ラベル範囲: {torch.min(batch_y)} ～ {torch.max(batch_y)}")
                    print(f"   期待範囲: 0 ～ {self.model.num_classes - 1}")
                    continue  # このバッチをスキップ
                
                optimizer.zero_grad()
                
                if GPU_AVAILABLE and scaler:
                    with autocast():
                        outputs = self.model(batch_X, batch_conf)
                        loss = criterion(outputs, batch_y)
                    scaler.scale(loss).backward()
                    scaler.step(optimizer)
                    scaler.update()
                else:
                    outputs = self.model(batch_X, batch_conf)
                    loss = criterion(outputs, batch_y)
                    loss.backward()
                    optimizer.step()
                
                total_loss += loss.item()
                _, predicted = torch.max(outputs.data, 1)
                total += batch_y.size(0)
                correct += (predicted == batch_y).sum().item()
                
            except RuntimeError as e:
                if "device-side assert" in str(e):
                    print(f"❌ CUDAエラー (バッチ {batch_idx}): {e}")
                    print(f"   バッチラベル情報:")
                    print(f"     形状: {batch_y.shape}")
                    print(f"     データ型: {batch_y.dtype}")
                    print(f"     値: {batch_y.cpu().numpy()}")
                    print(f"     範囲: {torch.min(batch_y)} ～ {torch.max(batch_y)}")
                    raise e
                else:
                    print(f"❌ 学習エラー (バッチ {batch_idx}): {e}")
                    raise e
        
        if total == 0:
            print("⚠️  有効なバッチが存在しません")
            return 0.0, 0.0
        
        return total_loss / len(train_loader), correct / total
    
    def _validate_epoch(self, val_loader, criterion):
        """1エポックの検証"""
        self.model.eval()
        total_loss = 0.0
        correct = 0
        total = 0
        
        with torch.no_grad():
            for batch_data in val_loader:
                if len(batch_data) == 3:
                    batch_X, batch_y, batch_conf = batch_data
                else:
                    batch_X, batch_y = batch_data
                    batch_conf = None
                
                outputs = self.model(batch_X, batch_conf)
                loss = criterion(outputs, batch_y)
                
                total_loss += loss.item()
                _, predicted = torch.max(outputs.data, 1)
                total += batch_y.size(0)
                correct += (predicted == batch_y).sum().item()
        
        return total_loss / max(1, len(val_loader)), correct / total
    
    def _evaluate_test(self, test_loader):
        """テストデータの評価"""
        self.model.eval()
        criterion = nn.CrossEntropyLoss()
        total_loss = 0.0
        all_predictions = []
        all_probabilities = []
        all_true_labels = []
        
        with torch.no_grad():
            for batch_data in test_loader:
                if len(batch_data) == 3:
                    batch_X, batch_y, batch_conf = batch_data
                else:
                    batch_X, batch_y = batch_data
                    batch_conf = None
                
                outputs = self.model(batch_X, batch_conf)
                loss = criterion(outputs, batch_y)
                
                total_loss += loss.item()
                probabilities = F.softmax(outputs, dim=1)
                _, predicted = torch.max(outputs.data, 1)
                
                all_predictions.extend(predicted.cpu().numpy())
                all_probabilities.extend(probabilities.cpu().numpy())
                all_true_labels.extend(batch_y.cpu().numpy())
        
        y_true = np.array(all_true_labels)
        y_pred = np.array(all_predictions)
        
        return {
            'loss': total_loss / max(1, len(test_loader)),
            'accuracy': np.mean(y_true == y_pred),
            'f1': f1_score(y_true, y_pred, average='weighted'),
            'y_true': y_true,
            'y_pred': y_pred,
            'y_proba': np.array(all_probabilities)
        }
    
    def compare_models(self, X: np.ndarray, y: np.ndarray, confidence_scores: np.ndarray = None) -> Dict:
        """複数モデルの比較"""
        model_types = ["simple", "bidirectional", "stacked"]
        results = {}
        
        for model_type in model_types:
            print(f"\n--- {model_type.upper()} LSTM ---")
            try:
                result = self.train_model(X, y, confidence_scores, model_type)
                results[model_type] = result
                print(f"{model_type} 完了: 精度={result['test_accuracy']:.4f}")
            except Exception as e:
                print(f"❌ {model_type} 学習エラー: {e}")
                results[model_type] = {'error': str(e)}
        
        # 最良モデル選択
        valid_results = {name: result for name, result in results.items() 
                        if 'error' not in result}
        
        if valid_results:
            best_model = max(valid_results.keys(), 
                           key=lambda x: valid_results[x]['test_accuracy'])
            results['best_model'] = best_model
            print(f"\n🏆 最良モデル: {best_model.upper()}")
        
        return results
    
    def plot_training_history(self):
        """学習履歴の可視化"""
        if self.history is None:
            return
        
        fig, axes = plt.subplots(2, 2, figsize=(15, 10))
        
        # 損失
        axes[0, 0].plot(self.history['train_loss'], label='Training Loss')
        axes[0, 0].plot(self.history['val_loss'], label='Validation Loss')
        axes[0, 0].set_title('Model Loss')
        axes[0, 0].legend()
        axes[0, 0].grid(True)
        
        # 精度
        axes[0, 1].plot(self.history['train_acc'], label='Training Accuracy')
        axes[0, 1].plot(self.history['val_acc'], label='Validation Accuracy')
        axes[0, 1].set_title('Model Accuracy')
        axes[0, 1].legend()
        axes[0, 1].grid(True)
        
        # 学習率
        axes[1, 0].plot(self.history['lr'])
        axes[1, 0].set_title('Learning Rate')
        axes[1, 0].set_yscale('log')
        axes[1, 0].grid(True)
        
        # 統計情報
        final_train_acc = self.history['train_acc'][-1]
        final_val_acc = self.history['val_acc'][-1]
        axes[1, 1].text(0.1, 0.7, f'Final Train Acc: {final_train_acc:.4f}', fontsize=12)
        axes[1, 1].text(0.1, 0.5, f'Final Val Acc: {final_val_acc:.4f}', fontsize=12)
        axes[1, 1].axis('off')
        
        plt.tight_layout()
        
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        plt.savefig(self.models_dir / f'lstm_training_history_{timestamp}.png', dpi=300, bbox_inches='tight')
        plt.show()
    
    def plot_confusion_matrix(self, y_true: np.ndarray, y_pred: np.ndarray):
        """混同行列の可視化"""
        cm = confusion_matrix(y_true, y_pred)
        
        # アクティブなラベル名を使用
        display_labels = self.active_phase_labels if self.active_phase_labels else self.phase_labels
        
        plt.figure(figsize=(10, 8))
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                   xticklabels=display_labels, yticklabels=display_labels)
        plt.title('LSTM Model - Confusion Matrix')
        plt.xlabel('Predicted Phase')
        plt.ylabel('True Phase')
        plt.xticks(rotation=45)
        plt.tight_layout()
        
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        plt.savefig(self.models_dir / f'lstm_confusion_matrix_{timestamp}.png', dpi=300, bbox_inches='tight')
        plt.show()
    
    def evaluate_model(self, results: Dict):
        """詳細モデル評価"""
        print(f"\n=== 詳細評価 ===")
        
        y_test = results['y_test']  # これらは再マッピングされたラベル
        y_pred = results['y_pred']  # これらは再マッピングされたラベル
        
        # 分類レポート (アクティブなラベル名と再マッピングされたラベル範囲を使用)
        report_str = ""
        if self.active_phase_labels:
            # labels引数でレポートするクラスを指定 (0から始まる連続したインデックス)
            report_labels = np.arange(len(self.active_phase_labels))
            print("分類レポート:")
            report_str = classification_report(y_test, y_pred, target_names=self.active_phase_labels, labels=report_labels, zero_division=0)
            print(report_str)
        else:
            print("分類レポート (デフォルト):")
            report_str = classification_report(y_test, y_pred, zero_division=0)
            print(report_str)

        # 分類レポートをファイルに保存
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        report_filename = self.models_dir / f'classification_report_{timestamp}.txt'
        try:
            with open(report_filename, 'w', encoding='utf-8') as f:
                f.write(f"Classification Report ({timestamp})\n")
                f.write("=" * 30 + "\n")
                f.write(f"Model Type: {results.get('model_type', 'N/A')}\n")
                f.write(f"Test Accuracy: {results.get('test_accuracy', 0.0):.4f}\n")
                f.write(f"F1 Score (Weighted): {results.get('f1_score', 0.0):.4f}\n")
                f.write("=" * 30 + "\n\n")
                f.write(report_str)
            print(f"✅ 分類レポートを保存しました: {report_filename.name}")
        except Exception as e:
            print(f"⚠️ 分類レポートの保存に失敗しました: {e}")

        # 可視化
        self.plot_confusion_matrix(y_test, y_pred)
        self.plot_training_history()
    
    def save_model(self, results: Dict, feature_names: List[str], video_names: List[str] = None) -> Dict[str, str]:
        """学習済みモデルの保存"""
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        
        # 動画名からファイル名生成
        if video_names:
            video_part = video_names[0] if len(video_names) == 1 else "multi"
            video_part = "".join(c for c in video_part if c.isalnum() or c in "._-")
            video_part = video_part[:20] if len(video_part) > 20 else video_part
        else:
            video_part = "unknown"
        
        # ファイルパス
        model_path = self.models_dir / f"tennis_pytorch_model_{video_part}_{timestamp}.pth"
        scaler_path = self.models_dir / f"tennis_pytorch_scaler_{video_part}_{timestamp}.pkl"
        metadata_path = self.models_dir / f"tennis_pytorch_metadata_{video_part}_{timestamp}.json"
        
        # モデル保存
        torch.save({
            'model_state_dict': self.model.state_dict(),
            'model_config': {
                'input_size': self.model.input_size,
                'hidden_sizes': self.model.hidden_sizes,
                'num_classes': self.model.num_classes,
                'dropout_rate': self.model.dropout_rate,
                'model_type': self.model.model_type,
                'use_batch_norm': getattr(self.model, 'use_batch_norm', True),
                'enable_confidence_weighting': getattr(self.model, 'enable_confidence_weighting', False)
            }
        }, model_path)
        
        # スケーラー保存
        joblib.dump(self.scaler, scaler_path)
        
        # メタデータ保存
        metadata = {
            'creation_time': timestamp,
            'video_names': video_names or [],
            'framework': 'pytorch',
            'model_type': results.get('model_type', 'bidirectional'),
            'test_accuracy': float(results['test_accuracy']),
            'f1_score': float(results['f1_score']),
            'phase_labels': self.active_phase_labels if self.active_phase_labels else self.phase_labels, # アクティブなラベルを使用
            'label_map': self.label_map, # ラベルマッピング情報を追加
            'feature_names': feature_names,
            'sequence_length': int(self.sequence_length), # Pythonのint型に変換
            'device': str(DEVICE),
            'gpu_used': results.get('gpu_used', False)
        }
        
        with open(metadata_path, 'w', encoding='utf-8') as f:
            json.dump(metadata, f, indent=2, ensure_ascii=False)
        
        print(f"\n=== モデル保存完了 ===")
        print(f"モデル: {model_path.name}")
        print(f"スケーラー: {scaler_path.name}")
        print(f"メタデータ: {metadata_path.name}")
        
        return {
            'model_path': str(model_path),
            'scaler_path': str(scaler_path),
            'metadata_path': str(metadata_path)
        }
    
    def train_complete_pipeline(self, csv_path: str = None, model_comparison: bool = True, 
                               quick_mode: bool = False, sample_ratio: float = 1.0) -> bool:
        """完全学習パイプライン"""
        print("=== テニス局面分類LSTM学習開始 ===")
        
        if quick_mode:
            print("🚀 高速モード有効")
            self.sequence_length = 15
            self.lstm_units = [64, 32] if not GPU_AVAILABLE else [128, 64]
        
        # データ読み込み
        df, success = self.load_dataset(csv_path, sample_ratio)
        if not success:
            return False
        
        video_names = df['video_name'].unique().tolist() if 'video_name' in df.columns else []
        
        # シーケンス作成
        try:
            X, y, confidence_scores, feature_names = self.prepare_sequences(df)
        except Exception as e:
            print(f"❌ シーケンス作成エラー: {e}")
            return False
        
        # モデル学習
        try:
            if model_comparison and not quick_mode:
                all_results = self.compare_models(X, y, confidence_scores)
                if 'best_model' in all_results:
                    best_model_type = all_results['best_model']
                    results = self.train_model(X, y, confidence_scores, best_model_type)
                else:
                    return False
            else:
                model_type = "simple" if quick_mode else "bidirectional"
                results = self.train_model(X, y, confidence_scores, model_type)
        except Exception as e:
            print(f"❌ モデル学習エラー: {e}")
            return False
        
        # 評価と保存
        try:
            self.evaluate_model(results)
            self.save_model(results, feature_names, video_names)
            print(f"\n🎉 LSTM学習完了！")
            return True
        except Exception as e:
            print(f"❌ 評価・保存エラー: {e}")
            return False

def check_cuda_installation():
    """CUDA環境チェック"""
    print("=== CUDA環境チェック ===")
    print(f"PyTorch版: {torch.__version__}")
    print(f"CUDA利用可能: {'はい' if torch.cuda.is_available() else 'いいえ'}")
    
    if torch.cuda.is_available():
        device_count = torch.cuda.device_count()
        print(f"検出GPU数: {device_count}")
        
        for i in range(device_count):
            gpu_name = torch.cuda.get_device_name(i)
            memory_total = torch.cuda.get_device_properties(i).total_memory / 1024**3
            print(f"  GPU {i}: {gpu_name} ({memory_total:.1f}GB)")
    
    return torch.cuda.is_available()

def main():
    """メイン関数"""
    print("=== テニス動画局面分類PyTorch LSTM学習ツール ===")
    
    check_cuda_installation()
    trainer = TennisLSTMTrainer()
    
    # データファイル確認（複数パターン対応）
    dataset_files = []
    patterns = ["tennis_features_dataset_*.csv", "tennis_features_*.csv"]
    
    for pattern in patterns:
        dataset_files.extend(list(trainer.features_dir.glob(pattern)))
    
    if not dataset_files:
        for pattern in patterns:
            dataset_files.extend(list(trainer.training_data_dir.glob(pattern)))
    
    # 重複を除去
    dataset_files = list(set(dataset_files))
    
    if not dataset_files:
        print("\n❌ 特徴量データセットファイルが見つかりません")
        print("検索パターン: tennis_features_dataset_*.csv, tennis_features_*.csv")
        print(f"検索場所: {trainer.features_dir}, {trainer.training_data_dir}")
        return
    
    print(f"\n📂 データファイル確認: {len(dataset_files)}ファイル検出")
    
    # 学習モード選択
    print(f"\n学習モード選択:")
    print(f"1: 通常学習（全データ・モデル比較あり）")
    print(f"2: 高速学習（軽量モデル・サンプリング）")
    print(f"3: ファイル選択学習")
    print(f"4: 最新ファイル自動学習")
    print(f"5: 終了")
    
    try:
        choice = input("\n選択してください (1-5): ").strip()
        
        config = {
            '1': (None, False, True, 1.0),
            '2': (None, True, False, 0.3),
            '3': (None, False, True, 1.0),
            '4': (str(max(dataset_files, key=lambda x: x.stat().st_mtime)), False, True, 1.0),
            '5': None
        }
        
        if choice == '5':
            return
        
        if choice not in config:
            print("❌ 無効な選択です")
            return
        
        selected_file, quick_mode, model_comparison, sample_ratio = config[choice]
        
        # 学習実行
        success = trainer.train_complete_pipeline(
            csv_path=selected_file,
            model_comparison=model_comparison,
            quick_mode=quick_mode,
            sample_ratio=sample_ratio
        )
        
        if success:
            print(f"\n✅ 全ての処理が正常に完了しました")
        else:
            print(f"\n❌ 処理中にエラーが発生しました")
            
    except KeyboardInterrupt:
        print("\n操作がキャンセルされました")
    except Exception as e:
        print(f"❌ 予期しないエラー: {e}")

if __name__ == "__main__":
    main()
