import math
import numpy as np
import torch
from torch import nn, optim
import torch.nn.functional as F
from rtdl_revisiting_models import MLP, ResNet, FTTransformer

# =====================
# FT-Transformer for for NCT Clinical Tabular Data
# =====================
class FTTransformer_o(nn.Module):
    def __init__(self, n_cont_features, cat_cardinalities=None, d_out=1):
        super().__init__()
        default_kwargs = FTTransformer.get_default_kwargs()
        self.model = FTTransformer(
            n_cont_features=n_cont_features,
            cat_cardinalities=cat_cardinalities,
            d_out=d_out,
            **default_kwargs,
        )
    
    def forward(self, x):
        x_cont, x_cat = x, None
        out = self.model(x_cont, x_cat).squeeze(-1)
        return out

class FTTransformer_r(nn.Module):
    def __init__(self,
                 num_features,
                 d_token=32,
                 n_layers=4,
                 n_heads=8,
                 dim_feedforward=128,
                 dropout=0.1):
        super().__init__()
        # Embed each numerical feature to a token
        self.token_emb = nn.Linear(1, d_token)
        # Class token
        self.cls_token = nn.Parameter(torch.randn(1, 1, d_token))
        # Positional embeddings for features + cls
        self.pos_emb = nn.Parameter(torch.randn(1, num_features + 1, d_token))
        # Transformer encoder
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_token,
            nhead=n_heads,
            dim_feedforward=dim_feedforward,
            dropout=dropout,
            batch_first=True
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=n_layers)
        # Prediction head
        self.head = nn.Sequential(
            nn.LayerNorm(d_token),
            nn.Linear(d_token, 1)
        )

    def forward(self, x):
        # x: [batch_size, num_features]
        b, f = x.size()
        # Tokenize each feature into a vector
        tokens = x.unsqueeze(-1)  # [b, f, 1]
        tokens = self.token_emb(tokens)  # [b, f, d_token]
        # Prepend cls token
        cls = self.cls_token.expand(b, -1, -1)  # [b, 1, d_token]
        tokens = torch.cat([cls, tokens], dim=1)  # [b, f+1, d_token]
        # Add positional embeddings
        tokens = tokens + self.pos_emb
        # Transformer
        encoded = self.transformer(tokens)  # [b, f+1, d_token]
        cls_out = encoded[:, 0, :]  # [b, d_token]
        return self.head(cls_out).squeeze(-1)


# =====================
# Time Series Transformer for NCT Clinical Data
# =====================
import torch
import torch.nn as nn

class NumericNormalizer(nn.Module):
    def __init__(self, mode="zscore", learnable=True, eps=1e-6):
        """
        mode: 'zscore', 'minmax', 'log1p'
        learnable: 是否让均值/方差作为可学习参数（仅 zscore/minmax）
        eps: 防止除0
        """
        super().__init__()
        self.mode = mode
        self.eps = eps

        if mode in ("zscore", "minmax"):
            mean_init = 0.0
            scale_init = 1.0 if mode == "zscore" else 1.0  # minmax 下scale≈max-min
            if learnable:
                self.mean = nn.Parameter(torch.tensor(mean_init, dtype=torch.float32))
                self.scale = nn.Parameter(torch.tensor(scale_init, dtype=torch.float32))
            else:
                self.register_buffer("mean", torch.tensor(mean_init, dtype=torch.float32))
                self.register_buffer("scale", torch.tensor(scale_init, dtype=torch.float32))

    def forward(self, x):
        """
        x: [B, S, 1] 连续特征
        """
        if self.mode == "zscore":
            return (x - self.mean) / (self.scale + self.eps)
        elif self.mode == "minmax":
            return (x - self.mean) / (self.scale + self.eps)
        elif self.mode == "log1p":
            return torch.log1p(torch.clamp(x, min=0.0))
        else:
            raise ValueError(f"Unknown mode: {self.mode}")

class Time2Vec(nn.Module):
    """简单版 Time2Vec：连续标量 -> 向量"""
    def __init__(self, k: int, d_out: int):
        super().__init__()
        self.k = k
        # 线性项
        self.w0 = nn.Parameter(torch.randn(1))
        self.b0 = nn.Parameter(torch.randn(1))
        # 周期项
        self.w = nn.Parameter(torch.randn(k))
        self.b = nn.Parameter(torch.randn(k))
        # 投影到目标维度
        self.proj = nn.Linear(k + 1, d_out)

    def forward(self, x):  # x: [B, S, 1]
        # [B,S,1] -> [B,S,k] with broadcasting
        periodic = torch.sin(x * self.w + self.b)   # [B,S,k]
        linear = x * self.w0 + self.b0              # [B,S,1]
        t2v = torch.cat([linear, periodic], dim=-1) # [B,S,k+1]
        return self.proj(t2v)                       # [B,S,d_out]

class NCTTimeSeriesTransformer(nn.Module):
    def __init__(self, 
                 d_model=128,
                 n_heads=8, 
                 n_layers=2,
                 dropout=0.1,
                 max_seq_len=256,
                 indicator_vocab_size=300,
                 d_out=1,
                 use_time2vec=True,
                 time2vec_k=8):
        super().__init__()
        assert d_model % n_heads == 0

        self.max_seq_len = max_seq_len
        self.d_model = d_model

        # 标准化模块
        self.age_norm  = NumericNormalizer(mode="zscore", learnable=True)
        self.value_norm = NumericNormalizer(mode="zscore", learnable=True)
        self.time_norm = NumericNormalizer(mode="log1p", learnable=False)  # 时间用 log1p

        # 嵌入模块
        if use_time2vec:
            self.time_embedding = Time2Vec(k=time2vec_k, d_out=d_model)
        else:
            self.time_embedding = nn.Linear(1, d_model)

        self.age_embedding  = nn.Linear(1, d_model)
        self.value_embedding = nn.Linear(1, d_model)
        self.indicator_embedding = nn.Embedding(indicator_vocab_size, d_model)

        self.fusion_norm = nn.LayerNorm(d_model)
        self.pos_encoding = nn.Parameter(torch.zeros(1, max_seq_len, d_model))

        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=n_heads,
            dim_feedforward=4*d_model,
            dropout=dropout,
            batch_first=True,
            activation='gelu'
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=n_layers)

        self.classifier = nn.Sequential(
            nn.LayerNorm(d_model),
            nn.Linear(d_model, d_model // 2),
            nn.BatchNorm1d(d_model // 2),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(d_model // 2, d_out)
        )

        self.dropout = nn.Dropout(dropout)

    def forward(self, batch):
        x = batch['features']  # [B, S, 4]
        seq_lengths = batch['seq_len']
        B, S, _ = x.shape

        if S > self.max_seq_len:
            x = x[:, :self.max_seq_len, :]
            S = self.max_seq_len

        age  = self.age_norm(x[:, :, 0:1])
        ind  = x[:, :, 1]
        val  = self.value_norm(x[:, :, 2:3])
        time = self.time_norm(x[:, :, 3:4])

        ind = torch.clamp(ind, 0, self.indicator_embedding.num_embeddings - 1).long()

        age_e  = self.age_embedding(age)
        ind_e  = self.indicator_embedding(ind)
        val_e  = self.value_embedding(val)
        time_e = self.time_embedding(time)

        combined = (age_e + ind_e + val_e + time_e) / math.sqrt(4.0)
        combined = self.fusion_norm(combined)

        combined = combined + self.pos_encoding[:, :S, :]
        combined = self.dropout(combined)

        padding_mask = self._create_padding_mask(seq_lengths, S, combined.device)
        out = self.transformer(combined, src_key_padding_mask=padding_mask)

        pooled = self._masked_mean(out, padding_mask)
        logits = self.classifier(pooled)

        return logits.squeeze(-1) if logits.shape[-1] == 1 else logits

    @staticmethod
    def _create_padding_mask(seq_lengths, max_len, device):
        rng = torch.arange(max_len, device=device).unsqueeze(0)
        lens = seq_lengths.to(device).unsqueeze(1)
        return rng >= lens

    @staticmethod
    def _masked_mean(x, pad_mask):
        valid = (~pad_mask).unsqueeze(-1)
        x = x * valid
        denom = valid.sum(dim=1).clamp_min(1)
        return x.sum(dim=1) / denom



# =====================
# LSTM for NCT Clinical Time Series Data (支持双向开关)
# =====================
class NCTTimeSeriesLSTM(nn.Module):
    def __init__(self, 
                 hidden_size=128,
                 num_layers=2,
                 dropout=0.1,
                 max_seq_len=256,
                 indicator_vocab_size=300,
                 d_out=1,
                 use_time2vec=True,
                 time2vec_k=8,
                 bidirectional=True):
        super().__init__()
        
        self.max_seq_len = max_seq_len
        self.hidden_size = hidden_size
        self.bidirectional = bidirectional
        
        # 标准化模块（复用Transformer的设计）
        self.age_norm = NumericNormalizer(mode="zscore", learnable=True)
        self.value_norm = NumericNormalizer(mode="zscore", learnable=True)
        self.time_norm = NumericNormalizer(mode="log1p", learnable=False)
        
        # 嵌入模块
        if use_time2vec:
            self.time_embedding = Time2Vec(k=time2vec_k, d_out=hidden_size)
        else:
            self.time_embedding = nn.Linear(1, hidden_size)
            
        self.age_embedding = nn.Linear(1, hidden_size)
        self.value_embedding = nn.Linear(1, hidden_size)
        self.indicator_embedding = nn.Embedding(indicator_vocab_size, hidden_size)
        
        # 特征融合
        self.fusion_norm = nn.LayerNorm(hidden_size)
        
        # LSTM（支持双向开关）
        self.lstm = nn.LSTM(
            input_size=hidden_size,
            hidden_size=hidden_size,
            num_layers=num_layers,
            dropout=dropout if num_layers > 1 else 0,
            bidirectional=bidirectional,
            batch_first=True
        )
        
        # 分类器（根据是否双向调整输入维度）
        lstm_output_size = 2 * hidden_size if bidirectional else hidden_size
        self.classifier = nn.Sequential(
            nn.LayerNorm(lstm_output_size),
            nn.Linear(lstm_output_size, hidden_size),
            nn.BatchNorm1d(hidden_size),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_size, d_out)
        )
        
        self.dropout = nn.Dropout(dropout)
    
    def forward(self, batch):
        x = batch['features']  # [B, S, 4]
        seq_lengths = batch['seq_len']
        B, S, _ = x.shape
        
        if S > self.max_seq_len:
            x = x[:, :self.max_seq_len, :]
            S = self.max_seq_len
            seq_lengths = torch.clamp(seq_lengths, max=self.max_seq_len)
        
        # 特征标准化和嵌入
        age = self.age_norm(x[:, :, 0:1])
        ind = x[:, :, 1]
        val = self.value_norm(x[:, :, 2:3])
        time = self.time_norm(x[:, :, 3:4])
        
        ind = torch.clamp(ind, 0, self.indicator_embedding.num_embeddings - 1).long()
        
        age_e = self.age_embedding(age)
        ind_e = self.indicator_embedding(ind)
        val_e = self.value_embedding(val)
        time_e = self.time_embedding(time)
        
        # 特征融合
        combined = (age_e + ind_e + val_e + time_e) / math.sqrt(4.0)
        combined = self.fusion_norm(combined)
        combined = self.dropout(combined)
        
        # 打包序列以处理变长输入
        packed_input = nn.utils.rnn.pack_padded_sequence(
            combined, seq_lengths.cpu(), batch_first=True, enforce_sorted=False
        )
        
        # LSTM前向传播
        packed_output, (h_n, c_n) = self.lstm(packed_input)
        
        # 解包序列
        lstm_out, _ = nn.utils.rnn.pad_packed_sequence(
            packed_output, batch_first=True, total_length=S
        )
        
        # 使用最后一个有效时间步的输出
        # h_n: [num_layers*directions, batch, hidden_size]
        if self.bidirectional:
            # 双向：取最后一层的前向和后向隐状态
            forward_hidden = h_n[-2, :, :]  # 前向LSTM的最后隐状态
            backward_hidden = h_n[-1, :, :]  # 后向LSTM的最后隐状态
            final_hidden = torch.cat([forward_hidden, backward_hidden], dim=1)
        else:
            # 单向：取最后一层的隐状态
            final_hidden = h_n[-1, :, :]
        
        # 分类
        logits = self.classifier(final_hidden)
        
        return logits.squeeze(-1) if logits.shape[-1] == 1 else logits


# =====================
# Bi-GRU for NCT Clinical Time Series Data  
# =====================
class NCTTimeSeriesGRU(nn.Module):
    def __init__(self, 
                 hidden_size=128,
                 num_layers=2,
                 dropout=0.1,
                 max_seq_len=256,
                 indicator_vocab_size=300,
                 d_out=1,
                 use_time2vec=True,
                 time2vec_k=8,
                 bidirectional=True):
        super().__init__()
        
        self.max_seq_len = max_seq_len
        self.hidden_size = hidden_size
        self.bidirectional = bidirectional
        
        # 标准化模块（复用Transformer的设计）
        self.age_norm = NumericNormalizer(mode="zscore", learnable=True)
        self.value_norm = NumericNormalizer(mode="zscore", learnable=True)
        self.time_norm = NumericNormalizer(mode="log1p", learnable=False)
        
        # 嵌入模块
        if use_time2vec:
            self.time_embedding = Time2Vec(k=time2vec_k, d_out=hidden_size)
        else:
            self.time_embedding = nn.Linear(1, hidden_size)
            
        self.age_embedding = nn.Linear(1, hidden_size)
        self.value_embedding = nn.Linear(1, hidden_size)
        self.indicator_embedding = nn.Embedding(indicator_vocab_size, hidden_size)
        
        # 特征融合
        self.fusion_norm = nn.LayerNorm(hidden_size)
        
        # GRU
        self.gru = nn.GRU(
            input_size=hidden_size,
            hidden_size=hidden_size,
            num_layers=num_layers,
            dropout=dropout if num_layers > 1 else 0,
            bidirectional=self.bidirectional,
            batch_first=True
        )
        
        # 分类器（根据bidirectional参数确定输入维度）
        classifier_input_size = 2 * hidden_size if self.bidirectional else hidden_size
        self.classifier = nn.Sequential(
            nn.LayerNorm(classifier_input_size),
            nn.Linear(classifier_input_size, hidden_size),
            nn.BatchNorm1d(hidden_size),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_size, d_out)
        )
        
        self.dropout = nn.Dropout(dropout)
    
    def forward(self, batch):
        x = batch['features']  # [B, S, 4]
        seq_lengths = batch['seq_len']
        B, S, _ = x.shape
        
        if S > self.max_seq_len:
            x = x[:, :self.max_seq_len, :]
            S = self.max_seq_len
            seq_lengths = torch.clamp(seq_lengths, max=self.max_seq_len)
        
        # 特征标准化和嵌入
        age = self.age_norm(x[:, :, 0:1])
        ind = x[:, :, 1]
        val = self.value_norm(x[:, :, 2:3])
        time = self.time_norm(x[:, :, 3:4])
        
        ind = torch.clamp(ind, 0, self.indicator_embedding.num_embeddings - 1).long()
        
        age_e = self.age_embedding(age)
        ind_e = self.indicator_embedding(ind)
        val_e = self.value_embedding(val)
        time_e = self.time_embedding(time)
        
        # 特征融合
        combined = (age_e + ind_e + val_e + time_e) / math.sqrt(4.0)
        combined = self.fusion_norm(combined)
        combined = self.dropout(combined)
        
        # 打包序列以处理变长输入
        packed_input = nn.utils.rnn.pack_padded_sequence(
            combined, seq_lengths.cpu(), batch_first=True, enforce_sorted=False
        )
        
        # GRU前向传播
        packed_output, h_n = self.gru(packed_input)
        
        # 解包序列
        gru_out, _ = nn.utils.rnn.pad_packed_sequence(
            packed_output, batch_first=True, total_length=S
        )
        
        # 使用最后一个有效时间步的输出
        # h_n: [num_layers*directions, batch, hidden_size]
        if self.bidirectional:
            # 双向：取最后一层的前向和后向隐状态
            forward_hidden = h_n[-2, :, :]  # 前向GRU的最后隐状态
            backward_hidden = h_n[-1, :, :]  # 后向GRU的最后隐状态
            final_hidden = torch.cat([forward_hidden, backward_hidden], dim=1)
        else:
            # 单向：取最后一层的隐状态
            final_hidden = h_n[-1, :, :]
        
        # 分类
        logits = self.classifier(final_hidden)
        
        return logits.squeeze(-1) if logits.shape[-1] == 1 else logits


class LSTMPredictor(nn.Module):
    def __init__(self, hidden_size, num_embeddings, num_layers=1, time2vec_k=8, bi_directional=False):
        super().__init__()

        self.num_embeddings = num_embeddings
        self.hidden_size = hidden_size
        self.num_layers = num_layers

        self.age_embedding = nn.Linear(1, hidden_size)
        self.type_embedding = nn.Embedding(num_embeddings+1, hidden_size, padding_idx=0)
        self.value_embedding = nn.Linear(1, hidden_size)
        self.time_embedding = Time2Vec(k=time2vec_k, d_out=hidden_size)

        self.lstm = nn.LSTM(input_size=hidden_size+1, hidden_size=hidden_size, num_layers=num_layers, batch_first=True, bidirectional=bi_directional)
        self.predictor = nn.Sequential(nn.Linear(hidden_size, 2))

    def forward(self, batch):
        x = batch['features']  # [B, S, 4]
        seq_lengths = batch['seq_len']

        age = x[:, :, 0:1]
        types = x[:, :, 1].long()
        value = x[:, :, 2:3]
        times = x[:, :, 3]

        age_embeds = self.age_embedding(age)
        type_embeds = self.type_embedding(types)
        value_embeds = self.value_embedding(value)
        # print('type_embeds', type_embeds.shape, times.shape)
        outputs, (hidden, cell) = self.lstm(torch.cat([times.unsqueeze(-1), type_embeds], dim=2))
        pred = self.predictor(outputs.mean(dim=1))
        return pred
    

if __name__ == "__main__":
    pass