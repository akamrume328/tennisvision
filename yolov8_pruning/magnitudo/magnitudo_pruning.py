import time
import torch
from ultralytics import YOLO
import torch.nn.utils.prune as prune
import os

class GradualPruner:
    def __init__(self, model, target_sparsity=0.01, num_steps=5):  # より保守的に1%に変更
        self.model = model
        self.target_sparsity = target_sparsity
        self.num_steps = num_steps
        self.step_size = target_sparsity / num_steps

    def apply_gradual_pruning(self, data_yaml, epochs_per_step=3):
        for step in range(self.num_steps):
            current_sparsity = (step + 1) * self.step_size
            print(f"\n=== プルーニングステップ {step+1}/{self.num_steps} (sparsity: {current_sparsity*100:.2f}%) ===")
            
            # プルーニング適用（検出ヘッドは除外）
            for name, module in self.model.model.named_modules():
                if hasattr(module, 'weight') and len(module.weight.shape) > 1:
                    # 検出ヘッド（Detectレイヤー）はプルーニングしない
                    if 'detect' not in name.lower() and 'head' not in name.lower():
                        prune.l1_unstructured(module, name='weight', amount=self.step_size)
            
            # プルーニングマスクを永続化
            self._make_pruning_permanent()
            
            # 短時間のファインチューニング
            print(f"ファインチューニング中...")
            self.model.train(data=data_yaml, epochs=epochs_per_step, batch=4, imgsz=1920, verbose=False)
            
            # 精度チェック（テニスボール専用）
            results = self.model.val(data=data_yaml, verbose=False)
            tennis_ball_map50 = self._get_tennis_ball_map50(results)
            print(f"全体 mAP50: {results.box.map50:.3f}, テニスボール mAP50: {tennis_ball_map50:.3f}")

    def _make_pruning_permanent(self):
        """プルーニングマスクを永続化して、実際に重みを0にする"""
        for module in self.model.model.modules():
            if hasattr(module, 'weight') and hasattr(module, 'weight_mask'):
                prune.remove(module, 'weight')

    def _get_tennis_ball_map50(self, results):
        """テニスボールのmAP50を取得"""
        try:
            # results.box.maps50はクラス別のmAP50のリスト
            class_results = results.box.maps50
            print(f"デバッグ: クラス数 = {len(class_results)}, 各クラスのmAP50 = {class_results}")
            
            # data.yamlのクラス順序を確認（通常: player_front=0, player_back=1, tennis_ball=2）
            if len(class_results) >= 3:
                return class_results[2]  # tennis_ballのインデックス
            elif len(class_results) == 1:
                # もしテニスボールのみの場合
                return class_results[0]
            else:
                print(f"警告: 期待されるクラス数と異なります。実際のクラス数: {len(class_results)}")
                return 0.0
        except Exception as e:
            print(f"エラー: テニスボールmAP50の取得に失敗 - {e}")
            # フォールバック: results.box.ap50を直接確認
            try:
                if hasattr(results.box, 'ap50') and len(results.box.ap50) >= 3:
                    return results.box.ap50[2]
            except:
                pass
            return 0.0

def count_parameters(model):
    """モデルのパラメータ数をカウント"""
    total_params = sum(p.numel() for p in model.parameters())
    non_zero_params = sum((p != 0).sum().item() for p in model.parameters())
    sparsity = (total_params - non_zero_params) / total_params * 100
    return total_params, non_zero_params, sparsity

def debug_results(results):
    """デバッグ情報を出力"""
    try:
        print(f"デバッグ情報: {results.box}")
    except Exception as e:
        print(f"デバッグ情報の取得に失敗: {e}")

def main():
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = YOLO('C:/Users/akama/AppData/Local/Programs/Python/Python310/python_file/projects/tennisvision/models/weights/best_5_25.pt').to(device)
    
    data_yaml = 'C:/Users/akama/AppData/Local/Programs/Python/Python310/python_file/projects/tennisvision/data/processed/datasets/final_merged_dataset/data.yaml'
    
    # ベースライン精度を測定
    print("=== ベースライン精度 ===")
    baseline_results = model.val(data=data_yaml, verbose=False)
    baseline_map50 = baseline_results.box.map50
    baseline_map = baseline_results.box.map

    # デバッグ情報を出力
    debug_results(baseline_results)

    # テニスボールの元の精度を記録
    pruner = GradualPruner(model=model)
    baseline_tennis_ball_map50 = pruner._get_tennis_ball_map50(baseline_results)
    
    print(f"元モデル - 全体 mAP50: {baseline_map50:.3f}, mAP50-95: {baseline_map:.3f}")
    print(f"元モデル - テニスボール mAP50: {baseline_tennis_ball_map50:.3f}")
    
    # 元のパラメータ数
    total_params_before, non_zero_before, sparsity_before = count_parameters(model.model)
    print(f"元パラメータ数: {total_params_before:,} (非ゼロ: {non_zero_before:,}, スパース性: {sparsity_before:.1f}%)")
    
    # 非常に保守的な段階的プルーニング（1%のみ）
    pruner = GradualPruner(model=model, target_sparsity=0.01, num_steps=5)
    pruner.apply_gradual_pruning(data_yaml, epochs_per_step=3)
    
    # プルーニング後のパラメータ数
    total_params_after, non_zero_after, sparsity_after = count_parameters(model.model)
    print(f"\nプルーニング後パラメータ数: {total_params_after:,} (非ゼロ: {non_zero_after:,}, スパース性: {sparsity_after:.1f}%)")
    
    # 最終的なファインチューニング（より長時間）
    print("\n=== 最終ファインチューニング ===")
    model.train(data=data_yaml, epochs=10, batch=4, imgsz=1920)
    
    # 最終評価
    final_results = model.val(data=data_yaml, verbose=False)
    final_map50 = final_results.box.map50
    final_map = final_results.box.map
    final_tennis_ball_map50 = pruner._get_tennis_ball_map50(final_results)
    
    print(f"\n=== 最終結果 ===")
    print(f"元モデル - 全体 mAP50: {baseline_map50:.3f}, テニスボール mAP50: {baseline_tennis_ball_map50:.3f}")
    print(f"プルーニング後 - 全体 mAP50: {final_map50:.3f}, テニスボール mAP50: {final_tennis_ball_map50:.3f}")
    print(f"精度低下 - 全体 mAP50: {(baseline_map50-final_map50)*100:.1f}%, テニスボール mAP50: {(baseline_tennis_ball_map50-final_tennis_ball_map50)*100:.1f}%")
    print(f"スパース性: {sparsity_after:.1f}% (削減されたパラメータ: {total_params_before-non_zero_after:,})")
    
    # テニスボールの精度を最優先で判定
    tennis_ball_threshold = 0.95  # テニスボールは95%以上の精度を維持
    overall_threshold = 0.90      # 全体は90%以上でOK
    
    tennis_ball_ok = final_tennis_ball_map50 >= baseline_tennis_ball_map50 * tennis_ball_threshold
    overall_ok = final_map50 >= baseline_map50 * overall_threshold
    
    if tennis_ball_ok and overall_ok:
        os.makedirs('results/models', exist_ok=True)
        model.save('results/models/yolov8_tennis_ball_optimized.pt')
        print("✅ テニスボール精度基準をクリア！モデルを保存しました")
        print(f"   テニスボール精度維持率: {final_tennis_ball_map50/baseline_tennis_ball_map50*100:.1f}%")
    else:
        print("❌ テニスボール精度が基準を下回りました")
        if not tennis_ball_ok:
            print(f"   テニスボール精度: {final_tennis_ball_map50:.3f} < 要求: {baseline_tennis_ball_map50 * tennis_ball_threshold:.3f}")
        if not overall_ok:
            print(f"   全体精度: {final_map50:.3f} < 要求: {baseline_map50 * overall_threshold:.3f}")

if __name__ == '__main__':
    main()
