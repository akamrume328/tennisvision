# 特徴量辞書 (Feature Dictionary)

このドキュメントは、`feature_extractor_unified.py` によって生成されるすべての特徴量とその説明を記載します。

## 1. 基本トラッキング特徴量 (Basic Tracking Features)
トラッキング処理から直接得られる、または基本的な計算によって得られる最も基礎的な情報です。

| 特徴量名 | 説明 |
| :--- | :--- |
| `frame_number` | 処理上のフレーム番号（0から始まる通し番号）。 |
| `original_frame_number` | 元の動画ファイルにおけるフレーム番号。 |
| `interpolated` | このフレームが補間によって生成されたものかを示すフラグ (1=True, 0=False)。 |
| `ball_detected` | そのフレームでボールが検出されたかを示すフラグ (1=True, 0=False)。 |
| `ball_x`, `ball_y` | 画像上でのボールの中心のXY座標（ピクセル単位）。 |
| `ball_x_normalized`, `ball_y_normalized` | 画像サイズで正規化された（0〜1の範囲になる）ボールのXY座標。 |
| `ball_velocity_x`, `ball_velocity_y` | ボールのX方向およびY方向の速度。 |
| `ball_speed` | ボールの速度の大きさ（スカラー）。 |
| `ball_tracking_confidence` | ボールの追跡に対するモデルの信頼度。 |
| `ball_movement_score` | ボールの動きの大きさを評価したスコア。 |
| `player_front_count`, `player_back_count` | 検出された手前側プレイヤーと奥側プレイヤーの数。 |
| `player_front_x`, `player_front_y` | 手前側プレイヤーの代表XY座標。 |
| `player_back_x`, `player_back_y` | 奥側プレイヤーの代表XY座標。 |
| `player_distance` | 2人のプレイヤー間の画面上の距離。 |
| `trajectory_length` | ボールが連続して追跡されている軌跡の長さ（フレーム数）。 |
| `disappeared_count` | ボールが連続して見失われているフレーム数。 |

## 2. コート座標系特徴量 (Court Coordinate Features)
コートキャリブレーションに基づき、カメラの画角に依存しないコート基準の座標系で計算された特徴量です。

| 特徴量名 | 説明 |
| :--- | :--- |
| `ball_court_x`, `ball_court_y` | 正規化されたコート座標系でのボールの位置。Xは左端が0, 右端が1。Yは奥のベースラインが0, 手前が1。ネットはY=0.5。 |
| `ball_in_court` | ボールがコートの内側にあるかを示すフラグ。 |
| `ball_in_front_court`, `ball_in_back_court` | ボールがネットより手前側か奥側かを示すフラグ。 |
| `ball_in_left_court`, `ball_in_right_court` | ボールがコートの左半分か右半分かを示すフラグ。 |
| `ball_distance_to_net` | ボールとネットとのコート座標系での距離。 |
| `ball_distance_to_sideline` | ボールと最も近いサイドラインとの距離。 |
| `ball_distance_to_baseline` | ボールと最も近いベースラインとの距離。 |
| `player_front_court_x`, `player_front_court_y` | 手前側プレイヤーのコート座標。 |
| `player_back_court_x`, `player_back_court_y` | 奥側プレイヤーのコート座標。 |
| `players_court_distance` | 2人のプレイヤー間のコート座標系での距離。 |
| `ball_to_front_court_distance` | ボールと手前側プレイヤーとのコート座標系での距離。 |
| `ball_to_back_court_distance` | ボールと奥側プレイヤーとのコート座標系での距離。 |

## 3. 時間的特徴量（移動統計）
カテゴリ1および2の数値特徴量に対し、短期的な傾向やばらつきを捉えるために、移動窓（スライディングウィンドウ）を使って計算された統計値です。

* **命名規則**: `[元の特徴量名]_[統計の種類]_[ウィンドウサイズ]`
* **統計の種類**:
    * `_ma`: 移動平均 (Moving Average)
    * `_std`: 移動標準偏差 (Standard Deviation)
    * `_max`: 移動最大値 (Maximum)
    * `_min`: 移動最小値 (Minimum)
* **ウィンドウサイズ**: 3, 5, 10, 15 (フレーム数)

**(例)**
* `ball_speed_ma_5`: ボール速度の過去5フレームの移動平均。
* `player_distance_std_10`: プレイヤー間距離の過去10フレームの移動標準偏差。

## 4. 文脈的特徴量 (Contextual Features)
複数の特徴量を組み合わせて、より高度なシーンの「文脈」を表現する特徴量です。

| 特徴量名 | 説明 |
| :--- | :--- |
| `ball_activity` | ボールの検出状態や動きを総合した活動スコア。 |
| `players_interaction` | プレイヤーの検出数を基にした相互作用の指標。 |
| `players_confidence_avg` | 両プレイヤーの検出信頼度の平均値。 |
| `ball_closer_to_front` | ボールが手前側プレイヤーと奥側プレイヤーのどちらに近いかを示すフラグ。 |
| `tracking_quality` | ボールのトラッキング品質を総合的に評価したスコア。 |
| `ball_detection_stability_{window}` | 指定したウィンドウ内のボール検出率（安定性）。 |
| `ball_movement_distance` | 前のフレームからのボールの移動距離。 |
| `player_..._movement_distance` | 各プレイヤーの前のフレームからの移動距離。 |
| `player_..._activity_{window}` | 各プレイヤーの移動距離の移動平均（活動度）。 |
| `scene_dynamics` | シーン全体の動きの激しさを表す指標（ボールとプレイヤーの動きの平均）。 |
| `ball_movement_spike` | ボールの動きが急激に変化した（スパイクした）かを示すフラグ。 |
| `ball_events_frequency_{window}`| 指定したウィンドウ内でのボールのスパイクイベントの発生頻度。 |

## 5. 高度な特徴量（追加実装分）
ボール検出が不完全な状況などを考慮して、より頑健な判断を行うために追加された特徴量です。

| 特徴量名 | 説明 |
| :--- | :--- |
| `frames_since_last_ball_detection` | 最後にボールが検出されてからの経過フレーム数。`rally`と`point_interval`の区別に極めて有効。 |
| `ball_vertical_velocity` | ボールの垂直方向の速度。サーブトスの検出に有効。 |
| `ball_acceleration_magnitude` | ボールの加速度の大きさ。ボールの打撃イベントの検出に有効。 |
| `ball_trajectory_angle_change` | ボールの進行方向の変化量。打撃による方向転換を捉える。 |
| `player_front_acceleration`, `player_back_acceleration` | 各プレイヤーの加速度。選手の急な動きを捉え、ボールが見えなくてもラリー中であることを推測するのに役立つ。 |
| `player_is_moving_towards_ball` | ボールに最も近い選手が、ボールに向かって動いているかを示すフラグ。ラリー継続の意図を判断する。 |
| `players_y_separation` | コート座標系でのプレイヤー間のY軸方向の距離。ラリー中とチェンジオーバーの区別などに有効。 |
| `last_known_ball_y_position` | ボールが見えない間、最後に観測されたコート上のY座標を保持し続ける。ボールがどこで消えたかの文脈を維持する。 |
| `far_side_ball_disappearance_rate` | 「最後に観測されたボールの位置がコート奥側」である状況で、ボールが見えなくなったフレームの割合（移動窓）。奥側でのラリー中のボールロストを捉える。 |