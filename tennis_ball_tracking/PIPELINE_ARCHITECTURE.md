# Tennis Ball Tracking Learning Pipeline - Independent Architecture

## 概要

テニスボール追跡学習パイプラインを3つの独立したコンポーネントに再構築しました。各コンポーネントは特定の責任に特化し、独立して動作できます。

## アーキテクチャ

### 1. balltracking.py - ボール・プレイヤー検出と追跡
**責任**: 純粋な検出・追跡データの出力
**入力**: ビデオファイル、YOLOモデル
**出力**: `tracking_features_*.json`

**特徴量**:
- ボール位置座標 (`ball_position`)
- ボール速度 (`ball_velocity`) 
- ボール検出状態 (`ball_detected`)
- プレイヤー位置 (`player1_position`, `player2_position`)
- プレイヤー検出状態 (`player1_detected`, `player2_detected`)
- タイムスタンプ (`timestamp`)

### 2. data_collector.py - コート座標設定とゲーム位相ラベル付け
**責任**: 手動アノテーションと空間的コンテキスト設定
**入力**: ビデオファイル、ユーザー入力
**出力**: `phase_annotations_*.json`, `court_coords_*.json`

**位相種類** (8種類):
- `point_interval` - ポイント間
- `rally` - ラリー中
- `serve_preparation` - サーブ準備
- `serve_front_deuce` - 手前デュースサイドからのサーブ
- `serve_front_ad` - 手前アドサイドからのサーブ
- `serve_back_deuce` - 奥デュースサイドからのサーブ
- `serve_back_ad` - 奥アドサイドからのサーブ
- `changeover` - チェンジコート間

**操作方法**:
- `スペース`: 再生/一時停止
- `1`: ポイント間
- `2`: ラリー中
- `3`: サーブ準備
- `4`: 手前デュースサイドからのサーブ
- `5`: 手前アドサイドからのサーブ
- `6`: 奥デュースサイドからのサーブ
- `7`: 奥アドサイドからのサーブ
- `8`: チェンジコート間
- `A/D`: フレーム移動
- `S/W`: 10フレーム移動
- `Z/X`: 100フレーム移動
- `Q`: 終了

### 3. train_phase_model.py - データ統合と機械学習モデル訓練
**責任**: 全データソースの統合と位相分類モデル訓練
**入力**: JSONデータファイル (`tracking_features_*.json`, `phase_annotations_*.json`, `court_coords_*.json`)
**出力**: `tennis_phase_model.pth`

**統合機能**:
- トラッキングデータと位相アノテーションのマージ
- コート座標を使った追加特徴量計算
- LSTMベースのシーケンス分類モデル訓練

## データフロー

```
ビデオファイル
    ↓
balltracking.py → tracking_features_*.json
    ↓
data_collector.py → phase_annotations_*.json
                 → court_coords_*.json
    ↓
train_phase_model.py → tennis_phase_model.pth
```

## 使用方法

### 1. トラッキングデータ生成
```bash
python balltracking.py
```
- ビデオファイルを選択
- YOLOモデルによる自動検出・追跡
- `training_data/tracking_features_*.json`を出力

### 2. 位相アノテーション
```bash
python data_collector.py
```
- メニューから「Phase annotation」を選択
- ビデオを再生しながら手動でゲーム位相をラベル付け
- `training_data/phase_annotations_*.json`を出力

### 3. コート座標設定
```bash
python data_collector.py
```
- メニューから「Court coordinates」を選択
- マウスクリックでコート四隅を設定
- `training_data/court_coords_*.json`を出力

### 4. モデル訓練
```bash
python train_phase_model.py
```
- 全JSONファイルを自動的に統合
- LSTMモデルで位相分類を学習
- `tennis_phase_model.pth`を出力

## ファイル構造

```
tennis_ball_tracking/
├── balltracking.py              # 検出・追跡コンポーネント
├── data_collector.py            # アノテーションコンポーネント  
├── train_phase_model.py         # ML訓練コンポーネント
├── test_pipeline.py             # パイプライン独立性テスト
└── training_data/               # データ交換ディレクトリ
    ├── tracking_features_*.json      # ボール・プレイヤー追跡データ
    ├── phase_annotations_*.json      # ゲーム位相アノテーション
    ├── court_coords_*.json           # コート座標データ
    └── tennis_phase_model.pth        # 訓練済みモデル
```

## データ形式例

### tracking_features_*.json
```json
[
  {
    "ball_position": [320.5, 240.8],
    "ball_velocity": [15.2, -8.7], 
    "ball_detected": true,
    "player1_position": [150.0, 400.0],
    "player2_position": [480.0, 200.0],
    "player1_detected": true,
    "player2_detected": true,
    "timestamp": 16.67
  }
]
```

### phase_annotations_*.json
```json
{
  "video_info": {
    "filename": "tennis_video.mp4",
    "total_frames": 1000,
    "fps": 30.0
  },
  "annotations": [
    {"frame": 0, "phase": "serve_preparation", "timestamp": "2025-06-01T12:00:00"},
    {"frame": 30, "phase": "serve_motion", "timestamp": "2025-06-01T12:00:01"}
  ]
}
```

### court_coords_*.json
```json
{
  "court_corners": [[100, 500], [700, 500], [700, 100], [100, 100]],
  "net_position": [400, 300],
  "service_boxes": {
    "left": [[100, 300], [400, 500]],
    "right": [[400, 100], [700, 300]]
  }
}
```

## 利点

1. **関心の分離**: 各コンポーネントが特定の責任に集中
2. **独立性**: コンポーネントが互いに依存せず独立して動作
3. **再利用性**: 各コンポーネントを他のプロジェクトで再利用可能
4. **保守性**: 機能追加・修正が他のコンポーネントに影響しない
5. **並行開発**: 複数の開発者が異なるコンポーネントを同時に開発可能

## 検証

パイプラインの独立性は`test_pipeline.py`で検証できます：

```bash
python test_pipeline.py
```

このテストは以下を確認します：
- ファイル構造の整合性
- コンポーネント間の独立性
- データフォーマットの一貫性
- データ統合の可能性
