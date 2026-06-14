"""
PatchTST Official Implementation.

Точная реализация на основе официального репозитория:
https://github.com/yuqinie98/PatchTST

Ключевые отличия от упрощённой версии:
- Residual Attention (передача scores между слоями)
- BatchNorm вместо LayerNorm
- Официальная архитектура TSTEncoder
- Корректная инициализация PE
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import math
from typing import Optional


# ============================================================
# Вспомогательные классы (из PatchTST_layers.py)
# ============================================================

class Transpose(nn.Module):
    """Транспозиция для BatchNorm."""
    def __init__(self, *dims, contiguous=False):
        super().__init__()
        self.dims = dims
        self.contiguous = contiguous

    def forward(self, x):
        if self.contiguous:
            return x.transpose(*self.dims).contiguous()
        return x.transpose(*self.dims)


def get_activation_fn(activation):
    """Получение функции активации."""
    if callable(activation):
        return activation()
    elif activation.lower() == "relu":
        return nn.ReLU()
    elif activation.lower() == "gelu":
        return nn.GELU()
    raise ValueError(f'{activation} is not available')


def positional_encoding(pe, learn_pe, q_len, d_model):
    """
    Позиционное кодирование (официальная реализация).

    pe: 'zeros', 'sincos', или None
    learn_pe: обучаемое или нет
    """
    if pe is None:
        W_pos = torch.empty((q_len, d_model))
        nn.init.uniform_(W_pos, -0.02, 0.02)
        learn_pe = False
    elif pe == 'zeros':
        W_pos = torch.empty((q_len, d_model))
        nn.init.uniform_(W_pos, -0.02, 0.02)
    elif pe == 'sincos':
        W_pos = torch.zeros(q_len, d_model)
        position = torch.arange(0, q_len).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2) * -(math.log(10000.0) / d_model))
        W_pos[:, 0::2] = torch.sin(position * div_term)
        W_pos[:, 1::2] = torch.cos(position * div_term)
        # Нормализация
        W_pos = W_pos - W_pos.mean()
        W_pos = W_pos / (W_pos.std() * 10)
    else:
        W_pos = torch.empty((q_len, d_model))
        nn.init.uniform_(W_pos, -0.02, 0.02)

    return nn.Parameter(W_pos, requires_grad=learn_pe)


# ============================================================
# Multi-Head Attention с Residual Attention
# ============================================================

class MultiheadAttention(nn.Module):
    """
    Multi-Head Attention с опциональным Residual Attention.

    Residual Attention: scores передаются между слоями для улучшения
    моделирования длинных последовательностей.
    """
    def __init__(self, d_model, n_heads, d_k=None, d_v=None,
                 attn_dropout=0., proj_dropout=0., res_attention=False):
        super().__init__()

        d_k = d_k or d_model // n_heads
        d_v = d_v or d_model // n_heads

        self.n_heads = n_heads
        self.d_k = d_k
        self.d_v = d_v
        self.res_attention = res_attention

        self.W_Q = nn.Linear(d_model, d_k * n_heads, bias=False)
        self.W_K = nn.Linear(d_model, d_k * n_heads, bias=False)
        self.W_V = nn.Linear(d_model, d_v * n_heads, bias=False)
        self.W_O = nn.Linear(d_v * n_heads, d_model, bias=False)

        self.attn_dropout = nn.Dropout(attn_dropout)
        self.proj_dropout = nn.Dropout(proj_dropout)

        self.scale = d_k ** -0.5

    def forward(self, Q, K, V, prev=None):
        """
        Args:
            Q, K, V: (batch, seq_len, d_model)
            prev: предыдущие attention scores (для residual attention)

        Returns:
            output: (batch, seq_len, d_model)
            attn: attention weights
            scores: attention scores (если res_attention=True)
        """
        bs, q_len, _ = Q.shape
        _, k_len, _ = K.shape

        # Linear projections
        Q = self.W_Q(Q).view(bs, q_len, self.n_heads, self.d_k).transpose(1, 2)
        K = self.W_K(K).view(bs, k_len, self.n_heads, self.d_k).transpose(1, 2)
        V = self.W_V(V).view(bs, k_len, self.n_heads, self.d_v).transpose(1, 2)

        # Attention scores
        scores = torch.matmul(Q, K.transpose(-2, -1)) * self.scale

        # Residual Attention: добавляем предыдущие scores
        if prev is not None:
            scores = scores + prev

        # Softmax и dropout
        attn = F.softmax(scores, dim=-1)
        attn = self.attn_dropout(attn)

        # Output
        context = torch.matmul(attn, V)
        context = context.transpose(1, 2).contiguous().view(bs, q_len, -1)
        output = self.W_O(context)
        output = self.proj_dropout(output)

        if self.res_attention:
            return output, attn, scores
        return output, attn


# ============================================================
# TSTEncoder Layer (официальная архитектура)
# ============================================================

class TSTEncoderLayer(nn.Module):
    """
    Transformer Encoder Layer с официальной архитектурой:
    - BatchNorm вместо LayerNorm (по умолчанию)
    - Residual Attention
    - Pre/Post norm опция
    """
    def __init__(self, d_model, n_heads, d_ff=256,
                 norm='BatchNorm', attn_dropout=0., dropout=0.,
                 activation='gelu', res_attention=False, pre_norm=False):
        super().__init__()

        assert d_model % n_heads == 0
        d_k = d_model // n_heads
        d_v = d_model // n_heads

        self.res_attention = res_attention
        self.pre_norm = pre_norm

        # Multi-Head Attention
        self.self_attn = MultiheadAttention(
            d_model, n_heads, d_k, d_v,
            attn_dropout=attn_dropout, proj_dropout=dropout,
            res_attention=res_attention
        )

        # Dropout
        self.dropout_attn = nn.Dropout(dropout)

        # Normalization после attention
        if 'batch' in norm.lower():
            self.norm_attn = nn.Sequential(
                Transpose(1, 2),
                nn.BatchNorm1d(d_model),
                Transpose(1, 2)
            )
        else:
            self.norm_attn = nn.LayerNorm(d_model)

        # Feed-Forward Network
        self.ff = nn.Sequential(
            nn.Linear(d_model, d_ff),
            get_activation_fn(activation),
            nn.Dropout(dropout),
            nn.Linear(d_ff, d_model)
        )

        # Dropout после FFN
        self.dropout_ffn = nn.Dropout(dropout)

        # Normalization после FFN
        if 'batch' in norm.lower():
            self.norm_ffn = nn.Sequential(
                Transpose(1, 2),
                nn.BatchNorm1d(d_model),
                Transpose(1, 2)
            )
        else:
            self.norm_ffn = nn.LayerNorm(d_model)

    def forward(self, src, prev=None):
        """
        Args:
            src: (batch, seq_len, d_model)
            prev: предыдущие attention scores

        Returns:
            output: (batch, seq_len, d_model)
            scores: attention scores (если res_attention)
        """
        # Multi-Head Attention sublayer
        if self.pre_norm:
            src2 = self.norm_attn(src)
        else:
            src2 = src

        if self.res_attention:
            src2, attn, scores = self.self_attn(src2, src2, src2, prev)
        else:
            src2, attn = self.self_attn(src2, src2, src2)
            scores = None

        # Residual connection
        src = src + self.dropout_attn(src2)
        if not self.pre_norm:
            src = self.norm_attn(src)

        # Feed-Forward sublayer
        if self.pre_norm:
            src2 = self.ff(self.norm_ffn(src))
        else:
            src2 = self.ff(src)

        # Residual connection
        src = src + self.dropout_ffn(src2)
        if not self.pre_norm:
            src = self.norm_ffn(src)

        if self.res_attention:
            return src, scores
        return src


class TSTEncoder(nn.Module):
    """Стек TSTEncoderLayer."""
    def __init__(self, d_model, n_heads, d_ff=256,
                 norm='BatchNorm', attn_dropout=0., dropout=0.,
                 activation='gelu', res_attention=False, pre_norm=False,
                 n_layers=3):
        super().__init__()

        self.layers = nn.ModuleList([
            TSTEncoderLayer(d_model, n_heads, d_ff, norm, attn_dropout, dropout,
                           activation, res_attention, pre_norm)
            for _ in range(n_layers)
        ])
        self.res_attention = res_attention

    def forward(self, src):
        """
        Args:
            src: (batch, seq_len, d_model)

        Returns:
            output: (batch, seq_len, d_model)
        """
        output = src
        scores = None

        if self.res_attention:
            for layer in self.layers:
                output, scores = layer(output, prev=scores)
        else:
            for layer in self.layers:
                output = layer(output)

        return output


# ============================================================
# RevIN (Reversible Instance Normalization)
# ============================================================

class RevIN(nn.Module):
    """Reversible Instance Normalization."""
    def __init__(self, num_features, eps=1e-5, affine=True):
        super().__init__()
        self.num_features = num_features
        self.eps = eps
        self.affine = affine

        if self.affine:
            self.affine_weight = nn.Parameter(torch.ones(num_features))
            self.affine_bias = nn.Parameter(torch.zeros(num_features))

    def forward(self, x, mode):
        """
        x: (batch, seq_len, num_features)
        mode: 'norm' или 'denorm'
        """
        if mode == 'norm':
            self.mean = x.mean(dim=1, keepdim=True)
            self.std = x.std(dim=1, keepdim=True) + self.eps
            x = (x - self.mean) / self.std
            if self.affine:
                x = x * self.affine_weight + self.affine_bias
        elif mode == 'denorm':
            if self.affine:
                x = (x - self.affine_bias) / self.affine_weight
            x = x * self.std + self.mean
        return x


# ============================================================
# Heads (Pretrain и Prediction)
# ============================================================

class PretrainHead(nn.Module):
    """Head для self-supervised pretraining."""
    def __init__(self, d_model, patch_len, dropout=0.):
        super().__init__()
        self.dropout = nn.Dropout(dropout)
        self.linear = nn.Linear(d_model, patch_len)

    def forward(self, x):
        """
        x: (batch, num_patches, d_model)
        output: (batch, num_patches, patch_len)
        """
        x = self.linear(self.dropout(x))
        return x


class PredictionHead(nn.Module):
    """
    Head для прогнозирования (официальная архитектура).

    Один linear слой: d_model * num_patches → pred_len
    """
    def __init__(self, d_model, num_patches, pred_len, dropout=0.):
        super().__init__()
        self.flatten = nn.Flatten(start_dim=1)
        self.dropout = nn.Dropout(dropout)
        self.linear = nn.Linear(d_model * num_patches, pred_len)

    def forward(self, x):
        """
        x: (batch, num_patches, d_model)
        output: (batch, pred_len)
        """
        x = self.flatten(x)  # (batch, d_model * num_patches)
        x = self.dropout(x)
        x = self.linear(x)   # (batch, pred_len)
        return x


# ============================================================
# PatchTST Official (главный класс)
# ============================================================

class PatchTST_Official(nn.Module):
    """
    PatchTST с официальной архитектурой.

    Ключевые особенности:
    - Residual Attention
    - BatchNorm
    - Официальная структура heads
    - Self-Supervised pretraining
    """

    def __init__(
        self,
        input_len=252,       # 1 год (официальный подход: input << train_window)
        pred_len=21,         # 1 месяц
        patch_len=16,
        stride=8,
        d_model=128,
        n_heads=16,          # Официальное значение
        n_layers=3,
        d_ff=256,            # Официальное значение (было 512)
        dropout=0.2,
        attn_dropout=0.,
        mask_ratio=0.4,
        use_revin=True,
        # Официальные параметры
        norm='BatchNorm',    # BatchNorm по умолчанию
        res_attention=True,  # Residual Attention
        pre_norm=False,      # Post-norm
        pe='zeros',          # Positional encoding type
        learn_pe=True,       # Learnable PE
        padding_patch=True,  # ReplicationPad в конце окна (официальная реализация)
    ):
        super().__init__()

        self.input_len = input_len
        self.pred_len = pred_len
        self.patch_len = patch_len
        self.stride = stride
        self.mask_ratio = mask_ratio
        self.use_revin = use_revin
        self.d_model = d_model
        self.padding_patch = padding_patch

        # Число патчей
        self.num_patches = (input_len - patch_len) // stride + 1
        if padding_patch:
            # Паддинг повторением последнего значения: без него хвост окна
            # (input_len - последний патч) не попадает в модель
            self.padding_patch_layer = nn.ReplicationPad1d((0, stride))
            self.num_patches += 1

        # RevIN
        if use_revin:
            self.revin = RevIN(1, affine=True)

        # Patch embedding (shared)
        self.patch_embedding = nn.Linear(patch_len, d_model)

        # Positional encoding (официальная)
        self.W_pos = positional_encoding(pe, learn_pe, self.num_patches, d_model)

        # Dropout
        self.dropout = nn.Dropout(dropout)

        # Transformer Encoder (официальная архитектура)
        self.encoder = TSTEncoder(
            d_model=d_model,
            n_heads=n_heads,
            d_ff=d_ff,
            norm=norm,
            attn_dropout=attn_dropout,
            dropout=dropout,
            activation='gelu',
            res_attention=res_attention,
            pre_norm=pre_norm,
            n_layers=n_layers
        )

        # Pretrain Head
        self.pretrain_head = PretrainHead(d_model, patch_len, dropout)

        # Prediction Head
        self.prediction_head = PredictionHead(d_model, self.num_patches, pred_len, dropout)

        # Mask token
        self.mask_token = nn.Parameter(torch.zeros(1, 1, d_model))
        nn.init.normal_(self.mask_token, std=0.02)

    def create_patches(self, x):
        """
        Разбиение временного ряда на патчи.

        Args:
            x: (batch, input_len)
        Returns:
            patches: (batch, num_patches, patch_len)
        """
        batch_size = x.shape[0]
        patches = []

        if self.padding_patch:
            x = self.padding_patch_layer(x.unsqueeze(1)).squeeze(1)

        for i in range(self.num_patches):
            start = i * self.stride
            end = start + self.patch_len
            patches.append(x[:, start:end])

        return torch.stack(patches, dim=1)

    def random_masking(self, patches):
        """
        Случайное маскирование патчей.

        Returns:
            mask: булева маска (True = замаскирован)
        """
        batch_size, num_patches, _ = patches.shape
        num_mask = int(num_patches * self.mask_ratio)

        noise = torch.rand(batch_size, num_patches, device=patches.device)
        ids_shuffle = torch.argsort(noise, dim=1)
        ids_restore = torch.argsort(ids_shuffle, dim=1)

        mask = torch.zeros(batch_size, num_patches, device=patches.device)
        mask[:, :num_mask] = 1
        mask = torch.gather(mask, dim=1, index=ids_restore).bool()

        return mask

    def forward_encoder(self, x, mask=None):
        """
        Encoder forward pass.

        Args:
            x: (batch, input_len)
            mask: маска патчей
        Returns:
            encoded: (batch, num_patches, d_model)
            patches: оригинальные патчи
        """
        batch_size = x.shape[0]

        # RevIN нормализация
        if self.use_revin:
            x = x.unsqueeze(-1)  # (batch, input_len, 1)
            x = self.revin(x, 'norm')
            x = x.squeeze(-1)   # (batch, input_len)

        # Создаём патчи
        patches = self.create_patches(x)  # (batch, num_patches, patch_len)

        # Embedding
        x = self.patch_embedding(patches)  # (batch, num_patches, d_model)

        # Применяем маску
        if mask is not None:
            mask_tokens = self.mask_token.expand(batch_size, self.num_patches, -1)
            x = torch.where(mask.unsqueeze(-1), mask_tokens, x)

        # Positional encoding
        x = self.dropout(x + self.W_pos)

        # Transformer encoder
        x = self.encoder(x)

        return x, patches

    def forward_pretrain(self, x):
        """
        Self-supervised pre-training.

        Returns:
            loss: MSE loss на замаскированных патчах
        """
        # Создаём маску
        with torch.no_grad():
            patches_for_mask = self.create_patches(x)
            mask = self.random_masking(patches_for_mask)

        # Encode с маской
        encoded, original_patches = self.forward_encoder(x, mask)

        # Восстанавливаем патчи
        pred_patches = self.pretrain_head(encoded)

        # Loss только на замаскированных патчах
        loss = F.mse_loss(pred_patches[mask], original_patches[mask])

        return loss, pred_patches, mask

    def forward_predict(self, x):
        """
        Прогнозирование.

        Returns:
            prediction: (batch, pred_len)
        """
        # Encode без маски
        encoded, _ = self.forward_encoder(x, mask=None)

        # Prediction head
        prediction = self.prediction_head(encoded)

        # RevIN денормализация (инвертирует affine, как в официальной реализации)
        if self.use_revin and hasattr(self.revin, 'std'):
            prediction = self.revin(prediction.unsqueeze(-1), 'denorm').squeeze(-1)

        return prediction

    def forward(self, x, mode='predict'):
        """
        Args:
            x: (batch, input_len)
            mode: 'pretrain' или 'predict'
        """
        if mode == 'pretrain':
            return self.forward_pretrain(x)
        return self.forward_predict(x)


# ============================================================
# Функции обучения (те же что в patchtst.py)
# ============================================================

def create_sequences(data, input_len, pred_len):
    """Создание пар (вход, выход) для fine-tuning."""
    X, y = [], []
    for i in range(len(data) - input_len - pred_len + 1):
        X.append(data[i:i + input_len])
        y.append(data[i + input_len:i + input_len + pred_len])
    return np.array(X), np.array(y)


def pretrain_patchtst(model, data, epochs=10, lr=0.0001, batch_size=64, verbose=False):
    """Self-Supervised pre-training."""
    device = next(model.parameters()).device
    model.train()

    input_len = model.input_len
    step = 5
    X_train = []
    for i in range(0, len(data) - input_len + 1, step):
        X_train.append(data[i:i + input_len])

    if len(X_train) == 0:
        X_train = [data[-input_len:]]

    X_train = np.array(X_train)
    X_tensor = torch.FloatTensor(X_train).to(device)

    dataset = torch.utils.data.TensorDataset(X_tensor)
    loader = torch.utils.data.DataLoader(dataset, batch_size=batch_size, shuffle=True)

    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=0.01)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs)

    for epoch in range(epochs):
        total_loss = 0
        for (batch,) in loader:
            optimizer.zero_grad()
            loss, _, _ = model(batch, mode='pretrain')
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            total_loss += loss.item()

        scheduler.step()

        if verbose and (epoch + 1) % 5 == 0:
            print(f"  Pretrain Epoch {epoch+1}/{epochs}, Loss: {total_loss/len(loader):.6f}")

    return model


def finetune_patchtst(model, X_train, y_train, epochs=5, lr=0.00005, batch_size=64, verbose=False):
    """Fine-tuning prediction head."""
    device = next(model.parameters()).device
    model.train()

    X_tensor = torch.FloatTensor(X_train).to(device)
    y_tensor = torch.FloatTensor(y_train).to(device)

    dataset = torch.utils.data.TensorDataset(X_tensor, y_tensor)
    loader = torch.utils.data.DataLoader(dataset, batch_size=batch_size, shuffle=True)

    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=0.01)
    criterion = nn.MSELoss()

    for epoch in range(epochs):
        total_loss = 0
        for X_batch, y_batch in loader:
            optimizer.zero_grad()
            pred = model(X_batch, mode='predict')
            loss = criterion(pred, y_batch)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            total_loss += loss.item()

        if verbose and (epoch + 1) % 5 == 0:
            print(f"  Finetune Epoch {epoch+1}/{epochs}, Loss: {total_loss/len(loader):.6f}")

    return model


def forecast_patchtst(model, last_input):
    """Прогноз."""
    device = next(model.parameters()).device
    model.eval()

    with torch.no_grad():
        x = torch.FloatTensor(last_input).unsqueeze(0).to(device)
        pred = model(x, mode='predict')

    return pred.cpu().numpy().flatten()


# ============================================================
# PatchTST MultiChannel (для компонент декомпозиции ICEEMDAN)
# ============================================================

class PatchTST_MultiChannel(nn.Module):
    """
    Многоканальный PatchTST с channel-independence и общими весами
    (как в официальной реализации: каналы проходят через общий энкодер
    независимо, батч расширяется до B*C).

    Используется для прогноза компонент декомпозиции (noise/cycle/trend):
    вход (batch, n_channels, input_len), выход (batch, n_channels, pred_len).

    RevIN — per-channel (num_features=n_channels), денормализация корректно
    инвертирует affine-преобразование через RevIN(mode='denorm').
    """

    def __init__(
        self,
        n_channels=3,
        input_len=252,
        pred_len=21,
        patch_len=16,
        stride=8,
        d_model=128,
        n_heads=16,
        n_layers=3,
        d_ff=256,
        dropout=0.2,
        attn_dropout=0.,
        mask_ratio=0.4,
        use_revin=True,
        norm='BatchNorm',
        res_attention=True,
        pre_norm=False,
        pe='zeros',
        learn_pe=True,
        padding_patch=True,
    ):
        super().__init__()

        self.n_channels = n_channels
        self.input_len = input_len
        self.pred_len = pred_len
        self.patch_len = patch_len
        self.stride = stride
        self.mask_ratio = mask_ratio
        self.use_revin = use_revin
        self.d_model = d_model
        self.padding_patch = padding_patch

        self.num_patches = (input_len - patch_len) // stride + 1
        if padding_patch:
            self.padding_patch_layer = nn.ReplicationPad1d((0, stride))
            self.num_patches += 1

        if use_revin:
            self.revin = RevIN(n_channels, affine=True)

        self.patch_embedding = nn.Linear(patch_len, d_model)
        self.W_pos = positional_encoding(pe, learn_pe, self.num_patches, d_model)
        self.dropout = nn.Dropout(dropout)

        self.encoder = TSTEncoder(
            d_model=d_model,
            n_heads=n_heads,
            d_ff=d_ff,
            norm=norm,
            attn_dropout=attn_dropout,
            dropout=dropout,
            activation='gelu',
            res_attention=res_attention,
            pre_norm=pre_norm,
            n_layers=n_layers
        )

        self.pretrain_head = PretrainHead(d_model, patch_len, dropout)
        self.prediction_head = PredictionHead(d_model, self.num_patches, pred_len, dropout)

        self.mask_token = nn.Parameter(torch.zeros(1, 1, d_model))
        nn.init.normal_(self.mask_token, std=0.02)

    def create_patches(self, x):
        """
        x: (batch*n_channels, input_len)
        Returns: (batch*n_channels, num_patches, patch_len)
        """
        patches = []

        if self.padding_patch:
            x = self.padding_patch_layer(x.unsqueeze(1)).squeeze(1)

        for i in range(self.num_patches):
            start = i * self.stride
            end = start + self.patch_len
            patches.append(x[:, start:end])

        return torch.stack(patches, dim=1)

    def random_masking(self, batch_size, device):
        """Случайная маска патчей: (batch*n_channels, num_patches), True = замаскирован."""
        num_mask = int(self.num_patches * self.mask_ratio)

        noise = torch.rand(batch_size, self.num_patches, device=device)
        ids_shuffle = torch.argsort(noise, dim=1)
        ids_restore = torch.argsort(ids_shuffle, dim=1)

        mask = torch.zeros(batch_size, self.num_patches, device=device)
        mask[:, :num_mask] = 1
        return torch.gather(mask, dim=1, index=ids_restore).bool()

    def forward_encoder(self, x, mask=None):
        """
        x: (batch, n_channels, input_len)
        Returns:
            encoded: (batch*n_channels, num_patches, d_model)
            patches: нормализованные патчи (batch*n_channels, num_patches, patch_len)
        """
        batch_size = x.shape[0]

        # RevIN per-channel: (B, C, L) -> (B, L, C) -> norm -> обратно
        if self.use_revin:
            x = x.permute(0, 2, 1)
            x = self.revin(x, 'norm')
            x = x.permute(0, 2, 1)

        # Channel-independence: каналы как независимые элементы батча
        x = x.reshape(batch_size * self.n_channels, self.input_len)

        patches = self.create_patches(x)          # (B*C, P, patch_len)
        x = self.patch_embedding(patches)          # (B*C, P, d_model)

        if mask is not None:
            mask_tokens = self.mask_token.expand(x.shape[0], self.num_patches, -1)
            x = torch.where(mask.unsqueeze(-1), mask_tokens, x)

        x = self.dropout(x + self.W_pos)
        x = self.encoder(x)

        return x, patches

    def forward_pretrain(self, x):
        """Self-supervised pre-training: MSE на замаскированных патчах."""
        mask = self.random_masking(x.shape[0] * self.n_channels, x.device)

        encoded, original_patches = self.forward_encoder(x, mask)
        pred_patches = self.pretrain_head(encoded)

        loss = F.mse_loss(pred_patches[mask], original_patches[mask])
        return loss, pred_patches, mask

    def forward_predict(self, x):
        """
        x: (batch, n_channels, input_len)
        Returns: (batch, n_channels, pred_len)
        """
        batch_size = x.shape[0]

        encoded, _ = self.forward_encoder(x, mask=None)
        prediction = self.prediction_head(encoded)             # (B*C, pred_len)
        prediction = prediction.reshape(batch_size, self.n_channels, self.pred_len)

        # RevIN денормализация с инверсией affine: (B, C, T) -> (B, T, C) -> обратно
        if self.use_revin:
            prediction = prediction.permute(0, 2, 1)
            prediction = self.revin(prediction, 'denorm')
            prediction = prediction.permute(0, 2, 1)

        return prediction

    def forward(self, x, mode='predict'):
        """
        x: (batch, n_channels, input_len)
        mode: 'pretrain' или 'predict'
        """
        if mode == 'pretrain':
            return self.forward_pretrain(x)
        return self.forward_predict(x)


def create_sequences_mc(components, input_len, pred_len):
    """
    Скользящие supervised-пары для многоканального ряда.

    Args:
        components: (n_channels, T)
    Returns:
        X: (N, n_channels, input_len), y: (N, n_channels, pred_len)
    """
    T = components.shape[1]
    X, y = [], []
    for i in range(T - input_len - pred_len + 1):
        X.append(components[:, i:i + input_len])
        y.append(components[:, i + input_len:i + input_len + pred_len])
    return np.array(X), np.array(y)


def pretrain_patchtst_mc(model, components, epochs=10, lr=0.0001, batch_size=64, verbose=False):
    """Self-Supervised pre-training многоканальной модели (components: (n_channels, T))."""
    device = next(model.parameters()).device
    model.train()

    input_len = model.input_len
    T = components.shape[1]
    step = 5
    X_train = []
    for i in range(0, T - input_len + 1, step):
        X_train.append(components[:, i:i + input_len])

    if len(X_train) == 0:
        X_train = [components[:, -input_len:]]

    X_tensor = torch.FloatTensor(np.array(X_train)).to(device)

    dataset = torch.utils.data.TensorDataset(X_tensor)
    loader = torch.utils.data.DataLoader(dataset, batch_size=batch_size, shuffle=True)

    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=0.01)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs)

    for epoch in range(epochs):
        total_loss = 0
        for (batch,) in loader:
            optimizer.zero_grad()
            loss, _, _ = model(batch, mode='pretrain')
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            total_loss += loss.item()

        scheduler.step()

        if verbose and (epoch + 1) % 5 == 0:
            print(f"  Pretrain Epoch {epoch+1}/{epochs}, Loss: {total_loss/len(loader):.6f}")

    return model


def finetune_patchtst_mc(model, X_train, y_train, epochs=5, lr=0.00005, batch_size=64, verbose=False):
    """Fine-tuning end-to-end: MSE по всем каналам."""
    device = next(model.parameters()).device
    model.train()

    X_tensor = torch.FloatTensor(X_train).to(device)
    y_tensor = torch.FloatTensor(y_train).to(device)

    dataset = torch.utils.data.TensorDataset(X_tensor, y_tensor)
    loader = torch.utils.data.DataLoader(dataset, batch_size=batch_size, shuffle=True)

    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=0.01)
    criterion = nn.MSELoss()

    for epoch in range(epochs):
        total_loss = 0
        for X_batch, y_batch in loader:
            optimizer.zero_grad()
            pred = model(X_batch, mode='predict')
            loss = criterion(pred, y_batch)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            total_loss += loss.item()

        if verbose and (epoch + 1) % 5 == 0:
            print(f"  Finetune Epoch {epoch+1}/{epochs}, Loss: {total_loss/len(loader):.6f}")

    return model


def forecast_patchtst_mc(model, last_input):
    """
    Прогноз многоканальной модели.

    Args:
        last_input: (n_channels, input_len)
    Returns:
        (n_channels, pred_len)
    """
    device = next(model.parameters()).device
    model.eval()

    with torch.no_grad():
        x = torch.FloatTensor(last_input).unsqueeze(0).to(device)
        pred = model(x, mode='predict')

    return pred.cpu().numpy()[0]


# Алиас для совместимости
PatchTST_SelfSupervised = PatchTST_Official
