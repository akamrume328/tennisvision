"""
テニス局面分類モデル学習用設定ファイル
"""

class TrainingConfig:
    """学習設定クラス"""
    
    # 局面ラベル定義
    PHASE_LABELS = [
        "point_interval",           # 0: ポイント間
        "rally",                   # 1: ラリー中
        "serve_preparation",       # 2: サーブ準備
        "serve_front_deuce",      # 3: 手前デュースサイドからのサーブ
        "serve_front_ad",         # 4: 手前アドサイドからのサーブ
        "serve_back_deuce",       # 5: 奥デュースサイドからのサーブ
        "serve_back_ad",          # 6: 奥アドサイドからのサーブ
        "changeover"              # 7: チェンジコート間
    ]
    
    # GPU使用時の設定
    GPU_CONFIG = {
        'sequence_length': 30,
        'overlap_ratio': 0.5,
        'lstm_units': [256, 128],
        'dense_units': [128, 64],
        'dropout_rate': 0.3,
        'learning_rate': 0.001,
        'batch_size': 64,
        'epochs': 150,
        'patience': 25
    }
    
    # CPU使用時の軽量設定
    CPU_CONFIG = {
        'sequence_length': 20,
        'overlap_ratio': 0.5,
        'lstm_units': [128, 64],
        'dense_units': [64, 32],
        'dropout_rate': 0.3,
        'learning_rate': 0.001,
        'batch_size': 32,
        'epochs': 100,
        'patience': 15
    }
    
    # ファイルパス設定
    PATHS = {
        'training_data_dir': 'training_data',
        'features_dir': 'training_data/features',
        'models_dir': 'training_data/lstm_models'
    }
