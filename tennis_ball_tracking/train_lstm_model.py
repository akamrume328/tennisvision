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
from sklearn.model_selection import train_test_split, TimeSeriesSplit
from sklearn.preprocessing import StandardScaler, LabelEncoder, MinMaxScaler
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score, f1_score
import joblib

# PyTorch with GPU configuration
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
import torch.nn.functional as F
from torch.optim.lr_scheduler import ReduceLROnPlateau
from torch.cuda.amp import GradScaler, autocast

# GPU設定とCUDA最適化
def setup_gpu_config():
    """GPU設定とCUDA最適化"""
    try:
        if torch.cuda.is_available():
            device_count = torch.cuda.device_count()
            print(f"🚀 GPU検出: {device_count}台")
            
            for i in range(device_count):
                gpu_name = torch.cuda.get_device_name(i)
                memory_total = torch.cuda.get_device_properties(i).total_memory / 1024**3
                print(f"   GPU {i}: {gpu_name} ({memory_total:.1f}GB)")
            
            # CUDAの詳細情報
            cuda_version = torch.version.cuda
            cudnn_version = torch.backends.cudnn.version()
            print(f"✅ CUDA version: {cuda_version}")
            print(f"✅ cuDNN version: {cudnn_version}")
            
            # 混合精度学習の確認
            if torch.cuda.is_bf16_supported():
                print("✅ BF16混合精度学習対応")
            else:
                print("✅ FP16混合精度学習対応")
            
            # CuDNN最適化を有効化
            torch.backends.cudnn.benchmark = True
            torch.backends.cudnn.enabled = True
            print("✅ CuDNN最適化を有効化")
            
            device = torch.device('cuda')
            return device
        else:
            print("⚠️  GPU未検出: CPUで実行します")
            device = torch.device('cpu')
            return device
            
    except Exception as e:
        print(f"⚠️  GPU設定エラー: {e}")
        print("CPUで実行します")
        device = torch.device('cpu')
        return device

# GPU設定を最初に実行
DEVICE = setup_gpu_config()
GPU_AVAILABLE = DEVICE.type == 'cuda'

plt.style.use('default')
sns.set_palette("husl")

class TennisLSTMModel(nn.Module):
    """PyTorch LSTM モデル"""
    
    def __init__(self, input_size: int, hidden_sizes: List[int], num_classes: int, 
                 dropout_rate: float = 0.3, model_type: str = "bidirectional"):
        super(TennisLSTMModel, self).__init__()
        
        self.input_size = input_size
        self.hidden_sizes = hidden_sizes
        self.num_classes = num_classes
        self.dropout_rate = dropout_rate
        self.model_type = model_type
        
        # LSTM層を構築
        if model_type == "simple":
            self.lstm1 = nn.LSTM(input_size, hidden_sizes[0], batch_first=True, dropout=dropout_rate)
            self.lstm2 = nn.LSTM(hidden_sizes[0], hidden_sizes[1], batch_first=True, dropout=dropout_rate)
            lstm_output_size = hidden_sizes[1]
            
        elif model_type == "bidirectional":
            self.lstm1 = nn.LSTM(input_size, hidden_sizes[0], batch_first=True, 
                                bidirectional=True, dropout=dropout_rate)
            self.lstm2 = nn.LSTM(hidden_sizes[0] * 2, hidden_sizes[1], batch_first=True, 
                                bidirectional=True, dropout=dropout_rate)
            lstm_output_size = hidden_sizes[1] * 2
            
        elif model_type == "stacked":
            self.lstm1 = nn.LSTM(input_size, hidden_sizes[0], batch_first=True, dropout=dropout_rate)
            self.lstm2 = nn.LSTM(hidden_sizes[0], hidden_sizes[1], batch_first=True, dropout=dropout_rate)
            self.lstm3 = nn.LSTM(hidden_sizes[1], 64, batch_first=True, dropout=dropout_rate)
            lstm_output_size = 64
        
        # Dropout層
        self.dropout = nn.Dropout(dropout_rate)
        
        # Dense層
        self.fc1 = nn.Linear(lstm_output_size, 128)
        self.fc2 = nn.Linear(128, 64)
        self.fc_out = nn.Linear(64, num_classes)
        
        # Batch Normalization
        self.bn1 = nn.BatchNorm1d(128)
        self.bn2 = nn.BatchNorm1d(64)
        
        # 重み初期化
        self._init_weights()
    
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
    
    def forward(self, x):
        # LSTM層
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
        
        # 最後のタイムステップの出力を使用
        lstm_out = lstm_out[:, -1, :]
        lstm_out = self.dropout(lstm_out)
        
        # Dense層
        out = F.relu(self.fc1(lstm_out))
        out = self.bn1(out)
        out = self.dropout(out)
        
        out = F.relu(self.fc2(out))
        out = self.bn2(out)
        out = self.dropout(out)
        
        out = self.fc_out(out)
        
        return out

class TennisLSTMTrainer:
    """
    PyTorch LSTM時系列モデルを使用したテニス局面分類学習クラス
    """
    
    def __init__(self, training_data_dir: str = "training_data"):
        self.training_data_dir = Path(training_data_dir)
        self.models_dir = self.training_data_dir / "lstm_models"
        self.models_dir.mkdir(exist_ok=True)
        
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
        
        # LSTM設定（GPU使用時は設定を調整）
        if GPU_AVAILABLE:
            self.sequence_length = 30  # GPU使用時は長いシーケンスも処理可能
            self.overlap_ratio = 0.5
            self.lstm_units = [256, 128]  # GPU使用時はより大きなユニット数
            self.dense_units = [128, 64]
            self.dropout_rate = 0.3
            self.learning_rate = 0.001
            self.batch_size = 64  # GPU使用時はより大きなバッチサイズ
        else:
            self.sequence_length = 20  # CPU使用時は軽量化
            self.overlap_ratio = 0.5
            self.lstm_units = [128, 64]
            self.dense_units = [64, 32]
            self.dropout_rate = 0.3
            self.learning_rate = 0.001
            self.batch_size = 32
        
        self.scaler = None
        self.label_encoder = None
        self.model = None
        self.history = None
        
        print(f"PyTorch LSTM局面分類モデル学習器を初期化しました")
        print(f"データディレクトリ: {self.training_data_dir}")
        print(f"モデル保存先: {self.models_dir}")
        print(f"シーケンス長: {self.sequence_length}フレーム")
        print(f"使用デバイス: {DEVICE}")
        print(f"バッチサイズ: {self.batch_size}")
    
    def load_dataset(self, csv_path: str = None) -> Tuple[pd.DataFrame, bool]:
        """特徴量データセットを読み込み"""
        if csv_path:
            dataset_path = Path(csv_path)
        else:
            # 最新のデータセットファイルを自動検索
            dataset_files = list(self.training_data_dir.glob("tennis_features_dataset_*.csv"))
            if not dataset_files:
                print("❌ 特徴量データセットファイルが見つかりません")
                print("feature_extractor.py を実行してデータセットを作成してください")
                return pd.DataFrame(), False
            
            dataset_path = max(dataset_files, key=lambda x: x.stat().st_mtime)
        
        try:
            df = pd.read_csv(dataset_path)
            print(f"✅ データセット読み込み: {dataset_path.name}")
            print(f"   サンプル数: {len(df)}")
            print(f"   特徴量数: {len(df.columns)}")
            
            # 基本情報表示
            self.analyze_dataset(df)
            
            return df, True
            
        except Exception as e:
            print(f"❌ データセット読み込みエラー: {e}")
            return pd.DataFrame(), False
    
    def analyze_dataset(self, df: pd.DataFrame):
        """データセットの基本分析"""
        print(f"\n=== データセット分析 ===")
        
        # ラベル分布
        print("ラベル分布:")
        label_counts = df['label'].value_counts().sort_index()
        
        # ラベル値の範囲をチェック
        unique_labels = sorted(df['label'].unique())
        max_label = max(unique_labels) if unique_labels else -1
        
        print(f"ラベル値の範囲: {min(unique_labels)} ～ {max_label}")
        print(f"期待される最大ラベル値: {len(self.phase_labels) - 1}")
        
        # 無効なラベルをチェック
        invalid_labels = [label for label in unique_labels if label >= len(self.phase_labels) or label < 0]
        if invalid_labels:
            print(f"⚠️  無効なラベル値が検出されました: {invalid_labels}")
            print("これらのラベルは学習から除外されます")
        
        for label_id, count in label_counts.items():
            if label_id < len(self.phase_labels) and label_id >= 0:
                phase_name = self.phase_labels[label_id]
                percentage = (count / len(df)) * 100
                print(f"  {label_id} ({phase_name}): {count} ({percentage:.1f}%)")
            else:
                print(f"  {label_id} (無効なラベル): {count}")
        
        # 動画別分布
        print(f"\n動画数: {df['video_name'].nunique()}")
        video_counts = df['video_name'].value_counts()
        print("動画別フレーム数:")
        for video, count in video_counts.head(10).items():
            print(f"  {video}: {count}フレーム")
        
        # 特徴量タイプ分析
        feature_types = {
            'basic': [],
            'temporal': [],
            'court': [],
            'contextual': []
        }
        
        for col in df.columns:
            if col in ['label', 'video_name', 'frame_number']:
                continue
            elif any(suffix in col for suffix in ['_ma_', '_std_', '_diff', '_trend_', '_cv_']):
                feature_types['temporal'].append(col)
            elif 'court' in col or 'distance_to' in col:
                feature_types['court'].append(col)
            elif any(keyword in col for keyword in ['activity', 'interaction', 'stability', 'dynamics']):
                feature_types['contextual'].append(col)
            else:
                feature_types['basic'].append(col)
        
        print(f"\n特徴量タイプ別:")
        for ftype, features in feature_types.items():
            print(f"  {ftype}: {len(features)}特徴量")
    
    def prepare_sequences(self, df: pd.DataFrame) -> Tuple[np.ndarray, np.ndarray, List[str]]:
        """時系列シーケンスデータを作成"""
        print(f"\n=== 時系列シーケンス作成 ===")
        
        # ラベル値の範囲チェックと修正
        df_cleaned = self.validate_and_clean_labels(df)
        
        # 特徴量列を選択
        exclude_columns = ['label', 'video_name', 'frame_number']
        feature_columns = [col for col in df_cleaned.columns if col not in exclude_columns]
        
        # 数値特徴量のみ使用
        numeric_features = df_cleaned[feature_columns].select_dtypes(include=[np.number]).columns
        X_features = df_cleaned[numeric_features].values
        y_labels = df_cleaned['label'].values
        video_names = df_cleaned['video_name'].values
        
        print(f"使用特徴量数: {len(numeric_features)}")
        print(f"総フレーム数: {len(X_features)}")
        
        # ラベル値の最終チェック
        unique_labels = np.unique(y_labels)
        max_label = np.max(unique_labels)
        expected_num_classes = len(self.phase_labels)
        
        print(f"実際のラベル値範囲: {np.min(unique_labels)} ～ {max_label}")
        print(f"期待されるクラス数: {expected_num_classes}")
        
        if max_label >= expected_num_classes:
            print(f"❌ ラベル値エラー: 最大ラベル値 {max_label} がクラス数 {expected_num_classes} を超えています")
            print("データを再確認してください")
            return np.array([]), np.array([]), []
        
        # 無限値とNaNを処理
        X_features = np.nan_to_num(X_features, nan=0.0, posinf=1e6, neginf=-1e6)
        
        # 標準化
        self.scaler = StandardScaler()
        X_scaled = self.scaler.fit_transform(X_features)
        
        # 動画ごとにシーケンスを作成
        sequences = []
        sequence_labels = []
        sequence_videos = []
        
        unique_videos = df_cleaned['video_name'].unique()
        
        for video in unique_videos:
            video_mask = video_names == video
            video_features = X_scaled[video_mask]
            video_labels = y_labels[video_mask]
            
            # 動画内でシーケンスを作成
            video_sequences, video_seq_labels = self.create_video_sequences(
                video_features, video_labels
            )
            
            sequences.extend(video_sequences)
            sequence_labels.extend(video_seq_labels)
            sequence_videos.extend([video] * len(video_sequences))
            
            print(f"  {video}: {len(video_sequences)}シーケンス作成")
        
        X_sequences = np.array(sequences)
        y_sequences = np.array(sequence_labels)
        
        # 最終的なラベル値チェック
        final_unique_labels = np.unique(y_sequences)
        final_max_label = np.max(final_unique_labels) if len(final_unique_labels) > 0 else -1
        
        print(f"作成されたシーケンス数: {len(X_sequences)}")
        print(f"シーケンス形状: {X_sequences.shape}")
        print(f"最終ラベル値範囲: {np.min(final_unique_labels)} ～ {final_max_label}")
        
        return X_sequences, y_sequences, list(numeric_features)
    
    def validate_and_clean_labels(self, df: pd.DataFrame) -> pd.DataFrame:
        """ラベル値を検証し、無効なラベルを除去"""
        print("ラベル値の検証と清浄化...")
        
        df_cleaned = df.copy()
        original_count = len(df_cleaned)
        
        # 有効なラベル値の範囲
        valid_label_range = range(len(self.phase_labels))
        
        # 無効なラベルを持つ行を特定
        invalid_mask = ~df_cleaned['label'].isin(valid_label_range)
        invalid_count = invalid_mask.sum()
        
        if invalid_count > 0:
            print(f"⚠️  無効なラベルを持つ行数: {invalid_count}/{original_count}")
            
            # 無効なラベル値の詳細
            invalid_labels = df_cleaned[invalid_mask]['label'].unique()
            print(f"無効なラベル値: {sorted(invalid_labels)}")
            
            # 無効な行を除去
            df_cleaned = df_cleaned[~invalid_mask].copy()
            print(f"除去後のデータ数: {len(df_cleaned)}/{original_count}")
        
        # ラベル値を連続した整数に再マッピング（必要に応じて）
        unique_labels = sorted(df_cleaned['label'].unique())
        if unique_labels != list(range(len(unique_labels))):
            print("ラベル値を連続した整数に再マッピング...")
            
            # 元のラベルから新しいラベルへのマッピング
            label_mapping = {old_label: new_label for new_label, old_label in enumerate(unique_labels)}
            df_cleaned['label'] = df_cleaned['label'].map(label_mapping)
            
            print("ラベルマッピング:")
            for old_label, new_label in label_mapping.items():
                if old_label < len(self.phase_labels):
                    phase_name = self.phase_labels[old_label]
                    print(f"  {old_label} ({phase_name}) -> {new_label}")
            
            # phase_labelsを更新
            self.phase_labels = [self.phase_labels[old_label] for old_label in unique_labels 
                               if old_label < len(self.phase_labels)]
            print(f"使用される局面ラベル: {self.phase_labels}")
        
        # 最終検証
        final_unique_labels = sorted(df_cleaned['label'].unique())
        expected_labels = list(range(len(self.phase_labels)))
        
        if final_unique_labels != expected_labels:
            print(f"⚠️  ラベル値が期待値と一致しません")
            print(f"実際: {final_unique_labels}")
            print(f"期待: {expected_labels}")
        else:
            print(f"✅ ラベル値の検証完了: {final_unique_labels}")
        
        return df_cleaned
    
    def create_video_sequences(self, features: np.ndarray, labels: np.ndarray) -> Tuple[List, List]:
        """単一動画からシーケンスを作成"""
        sequences = []
        sequence_labels = []
        
        # ステップサイズを計算（オーバーラップを考慮）
        step_size = max(1, int(self.sequence_length * (1 - self.overlap_ratio)))
        
        for i in range(0, len(features) - self.sequence_length + 1, step_size):
            seq_features = features[i:i + self.sequence_length]
            seq_labels = labels[i:i + self.sequence_length]
            
            # シーケンスの最後のラベルを使用（予測対象）
            target_label = seq_labels[-1]
            
            # 有効なラベル値のみを使用
            if 0 <= target_label < len(self.phase_labels):
                sequences.append(seq_features)
                sequence_labels.append(target_label)
        
        return sequences, sequence_labels
    
    def build_lstm_model(self, input_size: int, num_classes: int, model_type: str = "bidirectional") -> TennisLSTMModel:
        """PyTorch LSTMモデルを構築"""
        print(f"\n=== {model_type.upper()} PyTorch LSTMモデル構築 ===")
        print(f"入力特徴量数: {input_size}")
        print(f"出力クラス数: {num_classes}")
        
        # クラス数の検証
        if num_classes <= 1:
            raise ValueError(f"無効なクラス数: {num_classes}. 2以上である必要があります")
        
        model = TennisLSTMModel(
            input_size=input_size,
            hidden_sizes=self.lstm_units,
            num_classes=num_classes,
            dropout_rate=self.dropout_rate,
            model_type=model_type
        )
        
        # モデルをGPUに移動
        model = model.to(DEVICE)
        
        # モデル情報を表示
        total_params = sum(p.numel() for p in model.parameters())
        trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
        
        print(f"✅ モデル構築完了:")
        print(f"   総パラメータ数: {total_params:,}")
        print(f"   学習可能パラメータ数: {trainable_params:,}")
        print(f"   デバイス: {DEVICE}")
        
        return model
    
    def train_model(self, X: np.ndarray, y: np.ndarray, model_type: str = "bidirectional") -> Dict:
        """PyTorchモデル学習を実行"""
        print(f"\n=== PyTorch LSTM学習開始 ===")
        
        # ラベル値の最終検証
        unique_labels = np.unique(y)
        num_classes = len(unique_labels)
        max_label = np.max(unique_labels)
        
        print(f"学習データのラベル情報:")
        print(f"  ユニークラベル: {sorted(unique_labels)}")
        print(f"  クラス数: {num_classes}")
        print(f"  最大ラベル値: {max_label}")
        
        # ラベル値の妥当性チェック
        if max_label >= num_classes:
            raise ValueError(f"ラベル値エラー: 最大ラベル値 {max_label} がクラス数 {num_classes} 以上です")
        
        if min(unique_labels) < 0:
            raise ValueError(f"ラベル値エラー: 負のラベル値が含まれています: {min(unique_labels)}")
        
        # データ分割
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.2, random_state=42, stratify=y
        )
        
        X_train, X_val, y_train, y_val = train_test_split(
            X_train, y_train, test_size=0.2, random_state=42, stratify=y_train
        )
        
        print(f"学習データ: {X_train.shape}")
        print(f"検証データ: {X_val.shape}")
        print(f"テストデータ: {X_test.shape}")
        
        # PyTorchテンソルに変換
        X_train_tensor = torch.FloatTensor(X_train).to(DEVICE)
        y_train_tensor = torch.LongTensor(y_train).to(DEVICE)
        X_val_tensor = torch.FloatTensor(X_val).to(DEVICE)
        y_val_tensor = torch.LongTensor(y_val).to(DEVICE)
        X_test_tensor = torch.FloatTensor(X_test).to(DEVICE)
        y_test_tensor = torch.LongTensor(y_test).to(DEVICE)
        
        # データローダー作成
        train_dataset = TensorDataset(X_train_tensor, y_train_tensor)
        val_dataset = TensorDataset(X_val_tensor, y_val_tensor)
        test_dataset = TensorDataset(X_test_tensor, y_test_tensor)
        
        train_loader = DataLoader(train_dataset, batch_size=self.batch_size, shuffle=True)
        val_loader = DataLoader(val_dataset, batch_size=self.batch_size, shuffle=False)
        test_loader = DataLoader(test_dataset, batch_size=self.batch_size, shuffle=False)
        
        # モデル構築
        input_size = X.shape[2]  # 特徴量数
        self.model = self.build_lstm_model(input_size, num_classes, model_type)
        
        # 損失関数と最適化関数
        criterion = nn.CrossEntropyLoss()
        optimizer = optim.Adam(self.model.parameters(), lr=self.learning_rate, weight_decay=1e-4)
        
        # 学習率スケジューラー（verboseパラメータを削除）
        scheduler = ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=10)
        
        # 混合精度学習の準備
        scaler = GradScaler() if GPU_AVAILABLE else None
        
        # 学習履歴
        history = {
            'train_loss': [], 'train_acc': [],
            'val_loss': [], 'val_acc': [],
            'lr': []
        }
        
        # 早期停止の設定
        best_val_loss = float('inf')
        patience_counter = 0
        patience = 25 if GPU_AVAILABLE else 15
        
        # エポック数
        epochs = 150 if GPU_AVAILABLE else 100
        
        print("学習開始...")
        print(f"使用デバイス: {DEVICE}")
        print(f"混合精度学習: {'有効' if GPU_AVAILABLE else '無効'}")
        
        # 学習ループ
        for epoch in range(epochs):
            # 学習フェーズ
            self.model.train()
            train_loss = 0.0
            train_correct = 0
            train_total = 0
            
            for batch_X, batch_y in train_loader:
                optimizer.zero_grad()
                
                if GPU_AVAILABLE and scaler:
                    # 混合精度学習
                    with autocast():
                        outputs = self.model(batch_X)
                        loss = criterion(outputs, batch_y)
                    
                    scaler.scale(loss).backward()
                    scaler.step(optimizer)
                    scaler.update()
                else:
                    # 通常の学習
                    outputs = self.model(batch_X)
                    loss = criterion(outputs, batch_y)
                    loss.backward()
                    optimizer.step()
                
                train_loss += loss.item()
                _, predicted = torch.max(outputs.data, 1)
                train_total += batch_y.size(0)
                train_correct += (predicted == batch_y).sum().item()
            
            train_loss = train_loss / len(train_loader)
            train_acc = train_correct / train_total
            
            # 検証フェーズ
            self.model.eval()
            val_loss = 0.0
            val_correct = 0
            val_total = 0
            
            with torch.no_grad():
                for batch_X, batch_y in val_loader:
                    if GPU_AVAILABLE:
                        with autocast():
                            outputs = self.model(batch_X)
                            loss = criterion(outputs, batch_y)
                    else:
                        outputs = self.model(batch_X)
                        loss = criterion(outputs, batch_y)
                    
                    val_loss += loss.item()
                    _, predicted = torch.max(outputs.data, 1)
                    val_total += batch_y.size(0)
                    val_correct += (predicted == batch_y).sum().item()
            
            val_loss = val_loss / len(val_loader)
            val_acc = val_correct / val_total
            
            # 学習率スケジューラー（手動でverbose出力）
            old_lr = optimizer.param_groups[0]['lr']
            scheduler.step(val_loss)
            new_lr = optimizer.param_groups[0]['lr']
            
            # 学習率変更時の出力
            if new_lr != old_lr:
                print(f"Epoch {epoch+1}: 学習率を {old_lr:.2e} から {new_lr:.2e} に削減")
            
            # 履歴を保存
            history['train_loss'].append(train_loss)
            history['train_acc'].append(train_acc)
            history['val_loss'].append(val_loss)
            history['val_acc'].append(val_acc)
            history['lr'].append(optimizer.param_groups[0]['lr'])
            
            # プログレス表示
            if (epoch + 1) % 10 == 0:
                print(f'Epoch [{epoch+1}/{epochs}], '
                      f'Train Loss: {train_loss:.4f}, Train Acc: {train_acc:.4f}, '
                      f'Val Loss: {val_loss:.4f}, Val Acc: {val_acc:.4f}, '
                      f'LR: {optimizer.param_groups[0]["lr"]:.2e}')
            
            # 早期停止チェック
            if val_loss < best_val_loss:
                best_val_loss = val_loss
                patience_counter = 0
                # 最良モデルを保存
                torch.save(self.model.state_dict(), self.models_dir / 'best_pytorch_model.pth')
            else:
                patience_counter += 1
                if patience_counter >= patience:
                    print(f"早期停止: {patience}エポック改善なし")
                    break
        
        # 最良モデルをロード
        self.model.load_state_dict(torch.load(self.models_dir / 'best_pytorch_model.pth'))
        
        # テストデータで評価
        self.model.eval()
        test_loss = 0.0
        test_correct = 0
        test_total = 0
        all_predictions = []
        all_probabilities = []
        all_true_labels = []
        
        with torch.no_grad():
            for batch_X, batch_y in test_loader:
                if GPU_AVAILABLE:
                    with autocast():
                        outputs = self.model(batch_X)
                        loss = criterion(outputs, batch_y)
                else:
                    outputs = self.model(batch_X)
                    loss = criterion(outputs, batch_y)
                
                test_loss += loss.item()
                probabilities = F.softmax(outputs, dim=1)
                _, predicted = torch.max(outputs.data, 1)
                
                test_total += batch_y.size(0)
                test_correct += (predicted == batch_y).sum().item()
                
                all_predictions.extend(predicted.cpu().numpy())
                all_probabilities.extend(probabilities.cpu().numpy())
                all_true_labels.extend(batch_y.cpu().numpy())
        
        test_loss = test_loss / len(test_loader)
        test_accuracy = test_correct / test_total
        
        # 評価指標計算
        y_test_np = np.array(all_true_labels)
        y_pred_np = np.array(all_predictions)
        y_pred_proba_np = np.array(all_probabilities)
        
        f1 = f1_score(y_test_np, y_pred_np, average='weighted')
        
        # 履歴を保存
        self.history = history
        
        results = {
            'test_accuracy': test_accuracy,
            'test_loss': test_loss,
            'f1_score': f1,
            'y_test': y_test_np,
            'y_pred': y_pred_np,
            'y_pred_proba': y_pred_proba_np,
            'model_type': model_type,
            'gpu_used': GPU_AVAILABLE,
            'history': history
        }
        
        print(f"\n=== 学習結果 ===")
        print(f"テスト精度: {test_accuracy:.4f}")
        print(f"テスト損失: {test_loss:.4f}")
        print(f"F1スコア: {f1:.4f}")
        print(f"使用デバイス: {DEVICE}")
        
        if GPU_AVAILABLE:
            print("✅ GPU加速による高速学習が完了しました")
        
        return results
    
    def compare_models(self, X: np.ndarray, y: np.ndarray) -> Dict:
        """複数のLSTMモデルを比較"""
        print(f"\n=== LSTM モデル比較 ===")
        
        model_types = ["simple", "bidirectional", "stacked"]
        results = {}
        
        for model_type in model_types:
            print(f"\n--- {model_type.upper()} LSTM ---")
            try:
                result = self.train_model(X, y, model_type)
                results[model_type] = result
                
                print(f"{model_type} 完了:")
                print(f"  精度: {result['test_accuracy']:.4f}")
                print(f"  F1: {result['f1_score']:.4f}")
                
            except Exception as e:
                print(f"❌ {model_type} 学習エラー: {e}")
                results[model_type] = {'error': str(e)}
        
        # 最良モデルを選択
        valid_results = {name: result for name, result in results.items() 
                        if 'error' not in result}
        
        if valid_results:
            best_model = max(valid_results.keys(), 
                           key=lambda x: valid_results[x]['test_accuracy'])
            
            print(f"\n🏆 最良モデル: {best_model.upper()}")
            print(f"   精度: {valid_results[best_model]['test_accuracy']:.4f}")
            
            results['best_model'] = best_model
        
        return results
    
    def plot_training_history(self):
        """学習履歴を可視化"""
        if self.history is None:
            print("学習履歴がありません")
            return
        
        fig, axes = plt.subplots(2, 2, figsize=(15, 10))
        
        # 損失
        axes[0, 0].plot(self.history['train_loss'], label='Training Loss')
        axes[0, 0].plot(self.history['val_loss'], label='Validation Loss')
        axes[0, 0].set_title('Model Loss')
        axes[0, 0].set_xlabel('Epoch')
        axes[0, 0].set_ylabel('Loss')
        axes[0, 0].legend()
        axes[0, 0].grid(True)
        
        # 精度
        axes[0, 1].plot(self.history['train_acc'], label='Training Accuracy')
        axes[0, 1].plot(self.history['val_acc'], label='Validation Accuracy')
        axes[0, 1].set_title('Model Accuracy')
        axes[0, 1].set_xlabel('Epoch')
        axes[0, 1].set_ylabel('Accuracy')
        axes[0, 1].legend()
        axes[0, 1].grid(True)
        
        # 学習率（もしあれば）
        if 'lr' in self.history:
            axes[1, 0].plot(self.history['lr'])
            axes[1, 0].set_title('Learning Rate')
            axes[1, 0].set_xlabel('Epoch')
            axes[1, 0].set_ylabel('Learning Rate')
            axes[1, 0].set_yscale('log')
            axes[1, 0].grid(True)
        
        # 最終エポックの情報
        final_epoch = len(self.history['train_loss'])
        final_train_acc = self.history['train_acc'][-1]
        final_val_acc = self.history['val_acc'][-1]
        
        axes[1, 1].text(0.1, 0.7, f'Final Epoch: {final_epoch}', fontsize=12)
        axes[1, 1].text(0.1, 0.5, f'Final Train Acc: {final_train_acc:.4f}', fontsize=12)
        axes[1, 1].text(0.1, 0.3, f'Final Val Acc: {final_val_acc:.4f}', fontsize=12)
        axes[1, 1].set_xlim(0, 1)
        axes[1, 1].set_ylim(0, 1)
        axes[1, 1].axis('off')
        
        plt.tight_layout()
        
        # 保存
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        plt.savefig(self.models_dir / f'lstm_training_history_{timestamp}.png', 
                   dpi=300, bbox_inches='tight')
        plt.show()
    
    def plot_confusion_matrix(self, y_true: np.ndarray, y_pred: np.ndarray):
        """混同行列を可視化"""
        cm = confusion_matrix(y_true, y_pred)
        
        plt.figure(figsize=(10, 8))
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                   xticklabels=self.phase_labels, yticklabels=self.phase_labels)
        plt.title('LSTM Model - Confusion Matrix')
        plt.xlabel('Predicted Phase')
        plt.ylabel('True Phase')
        plt.xticks(rotation=45)
        plt.yticks(rotation=0)
        plt.tight_layout()
        
        # 保存
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        plt.savefig(self.models_dir / f'lstm_confusion_matrix_{timestamp}.png', 
                   dpi=300, bbox_inches='tight')
        plt.show()
    
    def evaluate_model(self, results: Dict):
        """詳細なモデル評価"""
        print(f"\n=== 詳細評価 ===")
        
        y_test = results['y_test']
        y_pred = results['y_pred']
        
        # 分類レポート
        target_names = [self.phase_labels[i] for i in sorted(np.unique(y_test))]
        print("分類レポート:")
        print(classification_report(y_test, y_pred, target_names=target_names))
        
        # 混同行列を可視化
        self.plot_confusion_matrix(y_test, y_pred)
        
        # 学習履歴を可視化
        self.plot_training_history()
    
    def save_model(self, results: Dict, feature_names: List[str]) -> Dict[str, str]:
        """学習済みPyTorchモデルを保存"""
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        
        # ファイルパス
        model_path = self.models_dir / f"tennis_pytorch_model_{timestamp}.pth"
        scaler_path = self.models_dir / f"tennis_pytorch_scaler_{timestamp}.pkl"
        metadata_path = self.models_dir / f"tennis_pytorch_metadata_{timestamp}.json"
        
        # モデル保存（状態辞書とモデル構造）
        torch.save({
            'model_state_dict': self.model.state_dict(),
            'model_config': {
                'input_size': self.model.input_size,
                'hidden_sizes': self.model.hidden_sizes,
                'num_classes': self.model.num_classes,
                'dropout_rate': self.model.dropout_rate,
                'model_type': self.model.model_type
            }
        }, model_path)
        
        joblib.dump(self.scaler, scaler_path)
        
        # メタデータ作成
        metadata = {
            'creation_time': timestamp,
            'framework': 'pytorch',
            'model_type': results.get('model_type', 'bidirectional'),
            'test_accuracy': float(results['test_accuracy']),
            'test_loss': float(results['test_loss']),
            'f1_score': float(results['f1_score']),
            'phase_labels': self.phase_labels,
            'feature_names': feature_names,
            'sequence_length': self.sequence_length,
            'overlap_ratio': self.overlap_ratio,
            'lstm_units': self.lstm_units,
            'dense_units': self.dense_units,
            'dropout_rate': self.dropout_rate,
            'learning_rate': self.learning_rate,
            'batch_size': self.batch_size,
            'model_path': str(model_path),
            'scaler_path': str(scaler_path),
            'feature_count': len(feature_names),
            'device': str(DEVICE),
            'gpu_used': results.get('gpu_used', False),
            'cuda_version': torch.version.cuda if GPU_AVAILABLE else None,
            'pytorch_version': torch.__version__
        }
        
        with open(metadata_path, 'w', encoding='utf-8') as f:
            json.dump(metadata, f, indent=2, ensure_ascii=False)
        
        print(f"\n=== PyTorch モデル保存完了 ===")
        print(f"🤖 モデル: {model_path}")
        print(f"📊 スケーラー: {scaler_path}")
        print(f"📋 メタデータ: {metadata_path}")
        print(f"⚡ GPU使用: {'はい' if results.get('gpu_used', False) else 'いいえ'}")
        
        return {
            'model_path': str(model_path),
            'scaler_path': str(scaler_path),
            'metadata_path': str(metadata_path)
        }
    
    def train_complete_pipeline(self, csv_path: str = None, model_comparison: bool = True) -> bool:
        """完全な学習パイプラインを実行"""
        print("=== テニス局面分類LSTM学習開始 ===")
        
        # データ読み込み
        df, success = self.load_dataset(csv_path)
        if not success:
            return False
        
        # シーケンス作成
        try:
            X, y, feature_names = self.prepare_sequences(df)
            print(f"✅ シーケンス作成完了: {X.shape}")
        except Exception as e:
            print(f"❌ シーケンス作成エラー: {e}")
            return False
        
        # モデル学習
        try:
            if model_comparison:
                # 複数モデル比較
                all_results = self.compare_models(X, y)
                
                if 'best_model' in all_results:
                    best_model_type = all_results['best_model']
                    results = all_results[best_model_type]
                    
                    # 最良モデルで再学習
                    print(f"\n最良モデル({best_model_type})で最終学習...")
                    results = self.train_model(X, y, best_model_type)
                else:
                    print("❌ 有効なモデルがありません")
                    return False
            else:
                # 単一モデル学習
                results = self.train_model(X, y, "bidirectional")
            
        except Exception as e:
            print(f"❌ モデル学習エラー: {e}")
            return False
        
        # 詳細評価
        try:
            self.evaluate_model(results)
        except Exception as e:
            print(f"⚠️  モデル評価でエラー: {e}")
        
        # モデル保存
        try:
            save_info = self.save_model(results, feature_names)
            print(f"\n🎉 LSTM学習完了！")
            print(f"次のステップ: 保存されたモデルで予測を実行できます")
            return True
            
        except Exception as e:
            print(f"❌ モデル保存エラー: {e}")
            return False

def check_cuda_installation():
    """CUDA環境をチェック"""
    print("=== CUDA環境チェック（PyTorch） ===")
    
    # PyTorchのバージョン
    print(f"PyTorch版: {torch.__version__}")
    
    # CUDA利用可能性
    cuda_available = torch.cuda.is_available()
    print(f"CUDA利用可能: {'はい' if cuda_available else 'いいえ'}")
    
    if cuda_available:
        # GPU情報
        device_count = torch.cuda.device_count()
        print(f"検出GPU数: {device_count}")
        
        for i in range(device_count):
            gpu_name = torch.cuda.get_device_name(i)
            memory_total = torch.cuda.get_device_properties(i).total_memory / 1024**3
            print(f"  GPU {i}: {gpu_name} ({memory_total:.1f}GB)")
        
        # CUDA詳細情報
        cuda_version = torch.version.cuda
        print(f"CUDA版: {cuda_version}")
        
        # cuDNN情報
        cudnn_available = torch.backends.cudnn.is_available()
        if cudnn_available:
            cudnn_version = torch.backends.cudnn.version()
            print(f"cuDNN版: {cudnn_version}")
        
        # 混合精度サポート
        if torch.cuda.is_bf16_supported():
            print("混合精度: BF16対応")
        else:
            print("混合精度: FP16対応")
    
    return cuda_available

def main():
    """メイン関数"""
    print("=== テニス動画局面分類PyTorch LSTM学習ツール ===")
    
    # CUDA環境チェック
    cuda_ok = check_cuda_installation()
    
    # 学習器を初期化
    trainer = TennisLSTMTrainer()
    
    # データファイル確認
    dataset_files = list(trainer.training_data_dir.glob("tennis_features_dataset_*.csv"))
    print(f"\n=== データファイル確認 ===")
    print(f"特徴量データセット: {len(dataset_files)}ファイル")
    
    if not dataset_files:
        print("\n❌ 特徴量データセットファイルが見つかりません")
        print("feature_extractor.py を実行してデータセットを作成してください")
        return
    
    for file in dataset_files:
        print(f"  - {file.name}")
    
    # GPU使用状況の最終確認
    if GPU_AVAILABLE:
        print(f"\n🚀 PyTorch GPU加速学習を開始します")
        print(f"   デバイス: {DEVICE}")
        print(f"   混合精度学習: 有効")
        print(f"   CuDNN最適化: 有効")
    else:
        print(f"\n💻 PyTorch CPU学習を開始します")
        if not cuda_ok:
            print("   GPU/CUDAが利用できません。以下を確認してください:")
            print("   1. NVIDIA GPU（CUDA対応）が搭載されているか")
            print("   2. CUDAドライバーがインストールされているか")
            print("   3. PyTorch CUDA版がインストールされているか")
    
    try:
        # 完全な学習パイプラインを実行
        success = trainer.train_complete_pipeline(model_comparison=True)
        
        if success:
            print(f"\n✅ 全ての処理が正常に完了しました")
            if GPU_AVAILABLE:
                print("🚀 PyTorch GPU加速により高速学習が実現されました")
        else:
            print(f"\n❌ 処理中にエラーが発生しました")
            
    except Exception as e:
        print(f"❌ 予期しないエラーが発生しました: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main()
