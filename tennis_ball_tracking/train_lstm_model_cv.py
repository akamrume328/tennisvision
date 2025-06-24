import json
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Tuple, Optional, Any
import warnings
import argparse
import yaml

warnings.filterwarnings('ignore')

# 機械学習ライブラリ
from sklearn.model_selection import StratifiedKFold, train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import classification_report, confusion_matrix, f1_score, accuracy_score
from sklearn.utils.class_weight import compute_class_weight
import joblib

# PyTorch
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
import torch.nn.functional as F
from torch.optim.lr_scheduler import ReduceLROnPlateau
from torch.cuda.amp import GradScaler, autocast

from model import TennisLSTMModel  # モデル定義をインポート

# ===== 設定読み込み =====
def load_config(path: str) -> Dict:
    """YAML設定ファイルを読み込む"""
    try:
        with open(path, 'r', encoding='utf-8') as f:
            config = yaml.safe_load(f)
        print(f"設定ファイルを読み込みました: {path}")

        if not torch.cuda.is_available() and 'cpu_settings' in config:
            print("警告: GPUが利用できません。CPU用の設定で上書きします。")
            config.update(config['cpu_settings'])
            
        return config
    except FileNotFoundError:
        print(f"エラー: 設定ファイルが見つかりません: {path}")
        raise
    except Exception as e:
        print(f"エラー: 設定ファイルの読み込み中にエラーが発生しました: {e}")
        raise


# ===== GPU設定 =====
def setup_gpu_config():
    """GPU設定とCUDA最適化"""
    if torch.cuda.is_available():
        device_count = torch.cuda.device_count()
        print(f"GPU検出: {device_count}台")
        
        for i in range(device_count):
            gpu_name = torch.cuda.get_device_name(i)
            memory_total = torch.cuda.get_device_properties(i).total_memory / 1024**3
            print(f"   GPU {i}: {gpu_name} ({memory_total:.1f}GB)")
        
        print(f"CUDA version: {torch.version.cuda}")
        print(f"cuDNN version: {torch.backends.cudnn.version()}")
        
        torch.backends.cudnn.benchmark = True
        torch.backends.cudnn.enabled = True
        return torch.device('cuda')
    else:
        print("警告: GPU未検出: CPUで実行します")
        return torch.device('cpu')

DEVICE = setup_gpu_config()
GPU_AVAILABLE = DEVICE.type == 'cuda'

plt.style.use('default')
sns.set_palette("husl")

class TennisLSTMTrainer:
    """テニス局面分類LSTM学習クラス（交差検証対応）"""
    
    def __init__(self, config: Dict):
        self.config = config
        self.training_data_dir = Path(self.config['training_data_dir'])
        self.features_dir = self.training_data_dir / "features"
        self.models_dir = self.training_data_dir / "lstm_models"
        
        self._setup_directories()
        
        self.phase_labels = [
            "point_interval", "rally",
            "serve_front_deuce", "serve_front_ad", "serve_back_deuce",
            "serve_back_ad", "changeover"
        ]
        
        self.model: Optional[nn.Module] = None
        self.scaler: Optional[StandardScaler] = None
        self.history: Optional[Dict] = None
        self.label_map: Optional[Dict[int, int]] = None
        self.active_phase_labels: Optional[List[str]] = None
        
        print("PyTorch LSTM学習器（交差検証対応）初期化完了")
        print(f"データディレクトリ: {self.training_data_dir}")
        print(f"使用デバイス: {DEVICE}")

    def _setup_directories(self):
        """ディレクトリの設定"""
        self.training_data_dir.mkdir(exist_ok=True)
        self.features_dir.mkdir(exist_ok=True)
        self.models_dir.mkdir(exist_ok=True)

    def load_dataset(self, csv_path: str = None) -> Tuple[pd.DataFrame, bool]:
        """データセットの読み込み"""
        if not csv_path:
            all_files = list(self.features_dir.glob("*.csv")) + list(self.training_data_dir.glob("*.csv"))
            if not all_files:
                print("エラー: 特徴量データセットファイルが見つかりません。")
                return pd.DataFrame(), False
            csv_path = str(max(all_files, key=lambda p: p.stat().st_mtime))

        dataset_path = Path(csv_path)
        if not dataset_path.exists():
            print(f"エラー: 指定されたファイルが見つかりません: {csv_path}")
            return pd.DataFrame(), False
            
        try:
            file_size_mb = dataset_path.stat().st_size / (1024 * 1024)
            print(f"データセット読み込み: {dataset_path.name} ({file_size_mb:.1f}MB)")
            
            df = pd.read_csv(dataset_path)
            
            if self.config['sample_ratio'] < 1.0:
                print(f"データを {self.config['sample_ratio']*100:.0f}%にサンプリングします。")
                df = df.sample(frac=self.config['sample_ratio'], random_state=42).reset_index(drop=True)

            print(f"読み込み完了: {len(df):,} 行, {len(df.columns)} 特徴量")
            return df, True
        except Exception as e:
            print(f"データセット読み込みエラー: {e}")
            return pd.DataFrame(), False

    def _validate_and_clean_labels(self, df: pd.DataFrame) -> pd.DataFrame:
        """ラベル値の検証と修正"""
        original_count = len(df)
        df.dropna(subset=['label'], inplace=True)
        valid_range = list(range(len(self.phase_labels)))
        df = df[df['label'].isin(valid_range)].copy()
        df['label'] = df['label'].astype(int)
        
        final_count = len(df)
        print(f"ラベル検証・修正完了。データ保持率: {(final_count/original_count)*100:.1f}%")
        return df

    def prepare_sequences(self, df: pd.DataFrame) -> Tuple[np.ndarray, np.ndarray, np.ndarray, List[str]]:
        """時系列シーケンスデータの作成"""
        print("時系列シーケンス作成開始...")
        df_cleaned = self._validate_and_clean_labels(df)
        
        exclude_cols = ['label', 'video_name', 'frame_number', 'interpolated', 'interpolation_confidence']
        feature_columns = [c for c in df_cleaned.columns if c not in exclude_cols and df_cleaned[c].dtype in [np.int64, np.float64]]
        
        X_features = df_cleaned[feature_columns].values
        y_labels = df_cleaned['label'].values
        video_names = df_cleaned['video_name'].values
        
        confidence_scores = df_cleaned['interpolation_confidence'].fillna(1.0).values if 'interpolation_confidence' in df_cleaned.columns else np.ones(len(df_cleaned))
        
        sequences, sequence_labels, confidence_sequences = [], [], []
        step_size = max(1, int(self.config['sequence_length'] * (1 - self.config['overlap_ratio'])))

        for video in pd.unique(video_names):
            mask = video_names == video
            video_X, video_y, video_conf = X_features[mask], y_labels[mask], confidence_scores[mask]
            
            for i in range(0, len(video_X) - self.config['sequence_length'] + 1, step_size):
                sequences.append(video_X[i:i + self.config['sequence_length']])
                sequence_labels.append(video_y[i + self.config['sequence_length'] - 1])
                confidence_sequences.append(video_conf[i:i + self.config['sequence_length']])
        
        print(f"作成されたシーケンス数: {len(sequences)}")
        return np.array(sequences), np.array(sequence_labels), np.array(confidence_sequences), feature_columns

    def build_model(self, input_size: int, num_classes: int) -> TennisLSTMModel:
        """PyTorch LSTMモデルの構築"""
        model_config = {
            'input_size': input_size,
            'num_classes': num_classes,
            'hidden_sizes': self.config['lstm_units'],
            'dropout_rate': self.config['dropout_rate'],
            'model_type': self.config['model_type'],
            'use_batch_norm': self.config['batch_size'] > 1,
            'enable_confidence_weighting': self.config['enable_confidence_weighting']
        }
        model = TennisLSTMModel(**model_config).to(DEVICE)
        total_params = sum(p.numel() for p in model.parameters())
        print(f"{self.config['model_type']}モデル構築完了: {total_params:,} パラメータ")
        return model

    def _train_epoch(self, model, train_loader, criterion, optimizer, scaler):
            model.train()
            total_loss, total_correct, total_samples = 0, 0, 0
            for batch in train_loader:
                # --- ここから修正 ---
                # モデルに渡す前に、すべてのテンソルをGPUに移動させる
                X_batch = batch[0].to(DEVICE)
                y_batch = batch[1].to(DEVICE)
                conf_data = batch[2].to(DEVICE) if len(batch) == 3 else None
                # --- ここまで修正 ---

                optimizer.zero_grad()
                if GPU_AVAILABLE and scaler:
                    with autocast():
                        outputs = model(X_batch, conf_data)
                        loss = criterion(outputs, y_batch)
                    scaler.scale(loss).backward()
                    scaler.step(optimizer)
                    scaler.update()
                else:
                    outputs = model(X_batch, conf_data)
                    loss = criterion(outputs, y_batch)
                    loss.backward()
                    optimizer.step()
                
                total_loss += loss.item() * X_batch.size(0)
                _, predicted = torch.max(outputs.data, 1)
                total_correct += (predicted == y_batch).sum().item()
                total_samples += y_batch.size(0)
            return total_loss / total_samples, total_correct / total_samples

    def _validate_epoch(self, model, val_loader, criterion):
            model.eval()
            total_loss, total_correct, total_samples = 0, 0, 0
            with torch.no_grad():
                for batch in val_loader:
                    # --- ここから修正 ---
                    # モデルに渡す前に、すべてのテンソルをGPUに移動させる
                    X_batch = batch[0].to(DEVICE)
                    y_batch = batch[1].to(DEVICE)
                    conf_data = batch[2].to(DEVICE) if len(batch) == 3 else None
                    # --- ここまで修正 ---

                    outputs = model(X_batch, conf_data)
                    loss = criterion(outputs, y_batch)
                    total_loss += loss.item() * X_batch.size(0)
                    _, predicted = torch.max(outputs.data, 1)
                    total_correct += (predicted == y_batch).sum().item()
                    total_samples += y_batch.size(0)
            return total_loss / total_samples, total_correct / total_samples

    def _train_and_evaluate_fold(self, model, X_train, y_train, X_val, y_val, class_weights, conf_train=None, conf_val=None):
        """1つのfoldを学習・評価し、履歴を返す"""
        train_ds = TensorDataset(torch.FloatTensor(X_train), torch.LongTensor(y_train), *( (torch.FloatTensor(conf_train),) if conf_train is not None else () ))
        val_ds = TensorDataset(torch.FloatTensor(X_val), torch.LongTensor(y_val), *( (torch.FloatTensor(conf_val),) if conf_val is not None else () ))
        train_loader = DataLoader(train_ds, batch_size=self.config['batch_size'], shuffle=True, drop_last=True)
        val_loader = DataLoader(val_ds, batch_size=self.config['batch_size'], shuffle=False, drop_last=True)
        
        criterion = nn.CrossEntropyLoss(weight=class_weights)
        optimizer = optim.Adam(model.parameters(), lr=self.config['learning_rate'])
        scheduler = ReduceLROnPlateau(optimizer, 'min', factor=0.5, patience=10)
        scaler = GradScaler() if GPU_AVAILABLE else None
        
        best_val_loss = float('inf')
        patience_counter = 0
        history = {'train_loss': [], 'val_loss': [], 'train_acc': [], 'val_acc': []}
        best_model_state = None

        for epoch in range(self.config['epochs']):
            train_loss, train_acc = self._train_epoch(model, train_loader, criterion, optimizer, scaler)
            val_loss, val_acc = self._validate_epoch(model, val_loader, criterion)
            
            history['train_loss'].append(train_loss)
            history['train_acc'].append(train_acc)
            history['val_loss'].append(val_loss)
            history['val_acc'].append(val_acc)
            
            scheduler.step(val_loss)
            
            if val_loss < best_val_loss:
                best_val_loss = val_loss
                patience_counter = 0
                best_model_state = model.state_dict()
            else:
                patience_counter += 1
                if patience_counter >= self.config['patience']:
                    print(f"Epoch {epoch+1}: 早期停止。")
                    break
            
            if (epoch + 1) % 10 == 0:
                print(f"Epoch [{epoch+1}/{self.config['epochs']}] Train Loss: {train_loss:.4f}, Val Loss: {val_loss:.4f}, Val Acc: {val_acc:.4f}")

        if best_model_state:
            model.load_state_dict(best_model_state)
        
        self.history = history
        return model

    def _evaluate_model_on_test(self, model, X_test, y_test, conf_test=None):
        """テストデータでモデルを評価"""
        model.eval()
        test_ds = TensorDataset(torch.FloatTensor(X_test), torch.LongTensor(y_test), *( (torch.FloatTensor(conf_test),) if conf_test is not None else () ))
        test_loader = DataLoader(test_ds, batch_size=self.config['batch_size'], shuffle=False)
        
        all_preds, all_true = [], []
        with torch.no_grad():
            for batch in test_loader:
                conf_data = batch[2] if len(batch) == 3 else None
                outputs = model(batch[0].to(DEVICE), conf_data.to(DEVICE) if conf_data is not None else None)
                _, predicted = torch.max(outputs.data, 1)
                all_preds.extend(predicted.cpu().numpy())
                all_true.extend(batch[1].cpu().numpy())

        return np.array(all_true), np.array(all_preds)

    def run_training_pipeline(self, csv_path: str = None):
        """交差検証を含む完全な学習パイプライン"""
        # 1. データ準備
        df, success = self.load_dataset(csv_path)
        if not success: return
        
        X, y, confidence, feature_names = self.prepare_sequences(df)
        if len(X) == 0:
            print("エラー: シーケンスが作成されませんでした。処理を終了します。")
            return
            
        original_labels = sorted(np.unique(y))
        self.label_map = {label: i for i, label in enumerate(original_labels)}
        self.active_phase_labels = [self.phase_labels[i] for i in original_labels]
        y_mapped = np.array([self.label_map[label] for label in y])
        num_classes = len(original_labels)
        print(f"クラス数: {num_classes}, アクティブラベル: {self.active_phase_labels}")

        # --- ★★★★★ 修正ポイント１：ここでクラスの重みを計算・定義します ★★★★★ ---
        class_weights = compute_class_weight(
            class_weight='balanced',
            classes=np.unique(y_mapped),
            y=y_mapped
        )
        class_weights_tensor = torch.tensor(class_weights, dtype=torch.float).to(DEVICE)
        print(f"クラスの重みを計算しました: {class_weights}")
        # --- ここまで ---

        n_splits = self.config.get('n_splits', 5)
        if n_splits > 1:
            # 2. 交差検証
            print(f"\n===== {n_splits}分割交差検証を開始します =====")
            skf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=42)
            
            all_fold_preds, all_fold_true = [], []
            fold_scores = []

            for fold, (train_val_idx, test_idx) in enumerate(skf.split(X, y_mapped)):
                print(f"--- FOLD {fold + 1}/{n_splits} ---")
                
                X_train_val, X_test = X[train_val_idx], X[test_idx]
                y_train_val, y_test = y_mapped[train_val_idx], y_mapped[test_idx]
                conf_train_val, conf_test = (confidence[train_val_idx], confidence[test_idx]) if self.config['enable_confidence_weighting'] else (None, None)
                
                X_train, X_val, y_train, y_val, conf_train, conf_val = train_test_split(
                    X_train_val, y_train_val, *( (conf_train_val,) if conf_train_val is not None else () ), 
                    test_size=0.2, stratify=y_train_val, random_state=42
                )
                
                scaler = StandardScaler()
                X_train_reshaped = X_train.reshape(-1, X_train.shape[-1])
                scaler.fit(X_train_reshaped)
                X_train = scaler.transform(X_train_reshaped).reshape(X_train.shape)
                X_val = scaler.transform(X_val.reshape(-1, X_val.shape[-1])).reshape(X_val.shape)
                X_test = scaler.transform(X_test.reshape(-1, X_test.shape[-1])).reshape(X_test.shape)

                model = self.build_model(X.shape[2], num_classes)
                
                # --- ★★★★★ 修正ポイント２：計算した`class_weights_tensor`を引数として渡します ★★★★★ ---
                model = self._train_and_evaluate_fold(model, X_train, y_train, X_val, y_val, class_weights_tensor, conf_train, conf_val)
                
                true_labels, pred_labels = self._evaluate_model_on_test(model, X_test, y_test, conf_test)
                all_fold_true.extend(true_labels)
                all_fold_preds.extend(pred_labels)
                
                acc = accuracy_score(true_labels, pred_labels)
                f1 = f1_score(true_labels, pred_labels, average='weighted')
                fold_scores.append({'accuracy': acc, 'f1_score': f1})
                print(f"FOLD {fold + 1} 結果: Accuracy={acc:.4f}, F1-score={f1:.4f}")

            self.evaluate_cv_results(all_fold_true, all_fold_preds, fold_scores)

        # 4. 最終モデルの学習と保存
        if self.config['execution_mode'] == 'full':
            print("\n===== 全データを使用して最終モデルの学習を開始します =====")
            X_train, X_val, y_train, y_val, conf_train, conf_val = train_test_split(
                X, y_mapped, *( (confidence,) if self.config['enable_confidence_weighting'] else () ),
                test_size=0.2, stratify=y_mapped, random_state=42
            )
            self.scaler = StandardScaler()
            X_train_reshaped = X_train.reshape(-1, X_train.shape[-1])
            self.scaler.fit(X_train_reshaped)
            X_train = self.scaler.transform(X_train_reshaped).reshape(X_train.shape)
            X_val = self.scaler.transform(X_val.reshape(-1, X_val.shape[-1])).reshape(X_val.shape)
            
            self.model = self.build_model(X.shape[2], num_classes)
            
            # --- ★★★★★ 修正ポイント３：最終学習でも`class_weights_tensor`を引数として渡します ★★★★★ ---
            self.model = self._train_and_evaluate_fold(self.model, X_train, y_train, X_val, y_val, class_weights_tensor, conf_train, conf_val)
            
            video_names = df['video_name'].unique().tolist() if 'video_name' in df.columns else []
            self.save_model(feature_names, video_names)
            self.plot_training_history()
        
        print("\n全ての処理が正常に完了しました。")

    def evaluate_cv_results(self, y_true, y_pred, fold_scores):
        """交差検証の結果を評価・表示"""
        print("\n===== 交差検証 最終結果 =====")
        
        avg_acc = np.mean([s['accuracy'] for s in fold_scores])
        std_acc = np.std([s['accuracy'] for s in fold_scores])
        avg_f1 = np.mean([s['f1_score'] for s in fold_scores])
        std_f1 = np.std([s['f1_score'] for s in fold_scores])
        
        print(f"平均精度: {avg_acc:.4f} (+/- {std_acc:.4f})")
        print(f"平均F1スコア: {avg_f1:.4f} (+/- {std_f1:.4f})")
        
        print("\n総合分類レポート (全Fold):")
        report = classification_report(y_true, y_pred, target_names=self.active_phase_labels, zero_division=0)
        print(report)
        
        self.plot_confusion_matrix(y_true, y_pred, title_suffix="(Cross-Validation Overall)")

    def save_model(self, feature_names: List[str], video_names: List[str] = None):
        """学習済みモデル、スケーラー、メタデータを保存"""
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        base_name = f"tennis_pytorch_{self.config['model_type']}_{timestamp}"
        
        model_path = self.models_dir / f"{base_name}_model.pth"
        scaler_path = self.models_dir / f"{base_name}_scaler.pkl"
        metadata_path = self.models_dir / f"{base_name}_metadata.json"
        
        torch.save(self.model.state_dict(), model_path)
        joblib.dump(self.scaler, scaler_path)

        json_compatible_label_map = {int(k): int(v) for k, v in self.label_map.items()}
        
        metadata = {
            'creation_time': timestamp,
            'model_config': self.config,
            'video_names': video_names or [],
            'phase_labels': self.active_phase_labels,
            'label_map': json_compatible_label_map,
            'feature_names': feature_names,
            'device': str(DEVICE),
        }
        with open(metadata_path, 'w', encoding='utf-8') as f:
            json.dump(metadata, f, indent=2, ensure_ascii=False)
        
        print(f"\nモデルを保存しました: {model_path.name}")
        print(f"スケーラーを保存しました: {scaler_path.name}")
        print(f"メタデータを保存しました: {metadata_path.name}")

    def plot_confusion_matrix(self, y_true: np.ndarray, y_pred: np.ndarray, title_suffix: str = ""):
        """混同行列の可視化"""
        cm = confusion_matrix(y_true, y_pred)
        plt.figure(figsize=(12, 10))
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
                    xticklabels=self.active_phase_labels, yticklabels=self.active_phase_labels)
        plt.title(f'Confusion Matrix {title_suffix}')
        plt.xlabel('Predicted Phase')
        plt.ylabel('True Phase')
        plt.xticks(rotation=45, ha='right')
        plt.yticks(rotation=0)
        plt.tight_layout()
        save_path = self.models_dir / f'confusion_matrix_{datetime.now().strftime("%Y%m%d_%H%M%S")}.png'
        plt.savefig(save_path, dpi=300)
        plt.show()
        print(f"混同行列を保存しました: {save_path}")

    def plot_training_history(self):
        """学習履歴の可視化"""
        if not self.history: return
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 6))
        
        ax1.plot(self.history['train_loss'], label='Training Loss')
        ax1.plot(self.history['val_loss'], label='Validation Loss')
        ax1.set_title('Model Loss')
        ax1.set_xlabel('Epoch')
        ax1.set_ylabel('Loss')
        ax1.legend()
        ax1.grid(True)
        
        ax2.plot(self.history['train_acc'], label='Training Accuracy')
        ax2.plot(self.history['val_acc'], label='Validation Accuracy')
        ax2.set_title('Model Accuracy')
        ax2.set_xlabel('Epoch')
        ax2.set_ylabel('Accuracy')
        ax2.legend()
        ax2.grid(True)
        
        plt.tight_layout()
        save_path = self.models_dir / f'training_history_{datetime.now().strftime("%Y%m%d_%H%M%S")}.png'
        plt.savefig(save_path, dpi=300)
        plt.show()
        print(f"学習履歴グラフを保存しました: {save_path}")


def main():
    """メイン関数"""
    parser = argparse.ArgumentParser(description="テニス局面分類PyTorch LSTM学習ツール（交差検証対応）")
    parser.add_argument('--config', type=str, default='config.yaml', help='設定ファイルのパス')
    parser.add_argument('--csv_path', type=str, default=None, help='(オプション) 使用するデータセットCSVファイルのパス')
    
    args = parser.parse_args()

    try:
        config = load_config(args.config)
        trainer = TennisLSTMTrainer(config)
        trainer.run_training_pipeline(csv_path=args.csv_path)
    except Exception as e:
        print(f"パイプラインの実行中に予期せぬエラーが発生しました: {e}")
        # 詳細なエラー情報を表示したい場合
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main()