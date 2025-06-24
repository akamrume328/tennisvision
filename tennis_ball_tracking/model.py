import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import List

class TennisLSTMModel(nn.Module):
    """PyTorch LSTM モデル（信頼度重み付け対応）"""
    
    def __init__(self, input_size: int, hidden_sizes: List[int], num_classes: int, 
                 dropout_rate: float, model_type: str, use_batch_norm: bool, 
                 enable_confidence_weighting: bool):
        super(TennisLSTMModel, self).__init__()
        
        self.input_size = input_size
        self.hidden_sizes = hidden_sizes
        self.num_classes = num_classes
        self.dropout_rate = dropout_rate
        self.model_type = model_type
        self.use_batch_norm = use_batch_norm
        self.enable_confidence_weighting = enable_confidence_weighting
        
        if self.enable_confidence_weighting:
            self.confidence_attention = nn.Sequential(
                nn.Linear(input_size, hidden_sizes[0] // 4),
                nn.ReLU(),
                nn.Linear(hidden_sizes[0] // 4, 1),
                nn.Sigmoid()
            )
        
        self._build_lstm_layers()
        
        self.dropout = nn.Dropout(dropout_rate)
        self.fc1 = nn.Linear(self.lstm_output_size, 128)
        self.fc2 = nn.Linear(128, 64)
        self.fc_out = nn.Linear(64, num_classes)
        
        if self.use_batch_norm:
            self.bn1 = nn.BatchNorm1d(128)
            self.bn2 = nn.BatchNorm1d(64)
        else:
            self.ln1 = nn.LayerNorm(128)
            self.ln2 = nn.LayerNorm(64)
        
        self._init_weights()
    
    def _build_lstm_layers(self):
        if self.model_type == "simple" or self.model_type == "stacked":
            num_layers = 2 if self.model_type == "simple" else len(self.hidden_sizes)
            self.lstm = nn.LSTM(
                self.input_size, self.hidden_sizes[0], num_layers=num_layers,
                batch_first=True, dropout=self.dropout_rate if num_layers > 1 else 0
            )
            self.lstm_output_size = self.hidden_sizes[0]
        elif self.model_type == "bidirectional":
            self.lstm = nn.LSTM(
                self.input_size, self.hidden_sizes[0], num_layers=len(self.hidden_sizes),
                batch_first=True, dropout=self.dropout_rate if len(self.hidden_sizes) > 1 else 0,
                bidirectional=True
            )
            self.lstm_output_size = self.hidden_sizes[0] * 2
    
    def _init_weights(self):
            """重みの初期化"""
            for name, param in self.named_parameters():
                # LSTMの重みを初期化
                if 'lstm' in name:
                    if 'weight_ih' in name:
                        # .data を使わず直接 param を渡すのがモダンな書き方です
                        nn.init.xavier_uniform_(param)
                    elif 'weight_hh' in name:
                        nn.init.orthogonal_(param)
                    elif 'bias' in name:
                        nn.init.zeros_(param)
                # 全結合層（fc）とアテンション層の重みを初期化
                elif 'fc' in name or 'confidence_attention' in name:
                    if 'weight' in name:
                        # param自体が重みテンソルなので、paramを直接使用
                        nn.init.xavier_uniform_(param)
                    elif 'bias' in name:
                        # param自体がバイアスなので、paramを直接使用
                        nn.init.zeros_(param)

    def forward(self, x, confidence_scores=None):
        if self.enable_confidence_weighting and confidence_scores is not None:
            attention_weights = self.confidence_attention(x).squeeze(-1)
            combined_weights = attention_weights * confidence_scores
            x = x * combined_weights.unsqueeze(-1)
        
        lstm_out, _ = self.lstm(x)
        lstm_out = lstm_out[:, -1, :]
        lstm_out = self.dropout(lstm_out)
        
        out = F.relu(self.fc1(lstm_out))
        out = self._apply_normalization(out, self.bn1 if self.use_batch_norm else self.ln1)
        out = self.dropout(out)
        
        out = F.relu(self.fc2(out))
        out = self._apply_normalization(out, self.bn2 if self.use_batch_norm else self.ln2)
        out = self.dropout(out)
        
        return self.fc_out(out)

    def _apply_normalization(self, x, norm_layer):
        if self.use_batch_norm:
            return norm_layer(x) if x.size(0) > 1 else x
        return norm_layer(x)
