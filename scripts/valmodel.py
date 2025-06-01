import torch
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
import yaml
from ultralytics import YOLO

class YOLOv8Evaluator:
    def __init__(self, model_path, device='cuda' if torch.cuda.is_available() else 'cpu'):
        """
        YOLOv8モデル評価クラスの初期化
        
        Args:
            model_path (str): 学習済みモデルのパス
            device (str): 使用デバイス ('cuda' or 'cpu')
        """
        self.model_path = model_path
        self.device = device
        self.model = YOLO(model_path)
        print(f"YOLOv8モデルを読み込みました: {model_path}")
        
    def evaluate(self, data_yaml_path, save_dir='./evaluation_results', imgsz=1920):
        """
        YOLOv8の組み込み評価機能を使用してモデルを評価
        
        Args:
            data_yaml_path (str): データセット設定YAMLファイルのパス
            save_dir (str): 結果保存ディレクトリ
            imgsz (int): 画像サイズ (デフォルト: 640)
            
        Returns:
            dict: 評価結果
        """
        print("モデル評価を開始します...")
        
        # YOLOv8の評価実行
        results = self.model.val(
            data=data_yaml_path,
            save_dir=save_dir,
            plots=True,
            verbose=True,
            imgsz=imgsz
        )
        
        return results
    
    def generate_report(self, results, save_path=None):
        """
        評価レポートを生成
        
        Args:
            results: YOLOv8の評価結果
            save_path (str): 保存パス
        """
        print("\n=== YOLOv8 評価レポート ===")
        
        # メトリクス取得
        metrics = results.results_dict
        
        print(f"mAP50: {metrics.get('metrics/mAP50(B)', 'N/A'):.4f}")
        print(f"mAP50-95: {metrics.get('metrics/mAP50-95(B)', 'N/A'):.4f}")
        print(f"Precision: {metrics.get('metrics/precision(B)', 'N/A'):.4f}")
        print(f"Recall: {metrics.get('metrics/recall(B)', 'N/A'):.4f}")
        
        if save_path:
            with open(save_path, 'w', encoding='utf-8') as f:
                f.write("=== YOLOv8 モデル評価レポート ===\n\n")
                f.write(f"モデルパス: {self.model_path}\n")
                f.write(f"mAP50: {metrics.get('metrics/mAP50(B)', 'N/A'):.4f}\n")
                f.write(f"mAP50-95: {metrics.get('metrics/mAP50-95(B)', 'N/A'):.4f}\n")
                f.write(f"Precision: {metrics.get('metrics/precision(B)', 'N/A'):.4f}\n")
                f.write(f"Recall: {metrics.get('metrics/recall(B)', 'N/A'):.4f}\n")
                f.write(f"\n詳細結果は {save_path.parent} の画像ファイルを確認してください。\n")
            print(f"レポートを保存しました: {save_path}")

def create_data_yaml(data_path, yaml_path, class_names):
    """
    YOLOv8用のデータ設定YAMLファイルを作成
    
    Args:
        data_path (str): データセットのルートパス
        yaml_path (str): 作成するYAMLファイルのパス
        class_names (list): クラス名のリスト
    """
    data_config = {
        'path': str(Path(data_path).absolute()),
        'train': 'train/images',
        'val': 'val/images',
        'test': 'test/images',
        'nc': len(class_names),
        'names': {i: name for i, name in enumerate(class_names)}
    }
    
    with open(yaml_path, 'w', encoding='utf-8') as f:
        yaml.dump(data_config, f, default_flow_style=False, allow_unicode=True)
    
    print(f"データ設定YAMLファイルを作成しました: {yaml_path}")

def evaluate_yolo_model(model_path, data_path=None, data_yaml_path=None, 
                       class_names=None, output_dir='./evaluation_results', imgsz=1920):
    """
    YOLOv8モデル評価のメイン関数
    
    Args:
        model_path (str): 学習済みYOLOv8モデルのパス
        data_path (str): データセットのパス
        data_yaml_path (str): データ設定YAMLファイルのパス
        class_names (list): クラス名のリスト
        output_dir (str): 結果保存ディレクトリ
        imgsz (int): 画像サイズ (デフォルト: 640)
    """
    # 出力ディレクトリ作成
    output_dir = Path(output_dir)
    output_dir.mkdir(exist_ok=True)
    
    # 評価器の初期化
    evaluator = YOLOv8Evaluator(model_path)
    
    # データ設定YAMLファイルの準備
    if data_yaml_path is None and data_path is not None:
        if class_names is None:
            class_names = ['player_front', 'player_back', 'tennis_ball']
        
        yaml_path = output_dir / 'data_config.yaml'
        create_data_yaml(data_path, yaml_path, class_names)
        data_yaml_path = yaml_path
    
    if data_yaml_path is None:
        raise ValueError("data_yaml_pathまたはdata_pathとclass_namesが必要です")
    
    # 評価実行
    results = evaluator.evaluate(data_yaml_path, output_dir, imgsz=imgsz)
    
    # レポート生成
    evaluator.generate_report(
        results,
        save_path=output_dir / 'evaluation_report.txt'
    )
    
    print(f"\n評価完了！結果は {output_dir} に保存されました。")
    print("混同行列、PR曲線、F1曲線などの詳細な可視化結果も自動生成されました。")
    
    return results

if __name__ == "__main__":
    # 使用例
    model_path = "C:/Users/akama/AppData/Local/Programs/Python/Python310/python_file/projects/tennisvision/models/weights/best_5_31.pt"
    data_path = "C:/Users/akama/AppData/Local/Programs/Python/Python310/python_file/projects/tennisvision/data/processed/datasets/final_merged_dataset"
    
    # 評価実行
    results = evaluate_yolo_model(
        model_path=model_path,
        data_path=data_path,
        class_names=['player_front', 'player_back', 'tennis_ball'],
        output_dir="./evaluation_results",
        imgsz=1920  # 画像サイズを指定 (例: 640, 1280など)
    )
