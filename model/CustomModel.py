import torch
import torch.nn as nn
import torch.nn.functional as F
import math
from math import sqrt, ceil, log


# ─────────────────────────────────────────────────────────────────────
# 1.  CONVOLUTION PRIMITIVES
# ─────────────────────────────────────────────────────────────────────

class CausalConv(nn.Module):
    """Causal 1-D convolution (no dilation).
    Input/Output: (B, C_in, T) → (B, C_out, T)
    """

    def __init__(self, in_channels: int, out_channels: int = None,
                 kernel_size: int = 2, groups: int = None, bias: bool = False):
        super().__init__()
        out_channels = out_channels or in_channels
        if groups is None:
            groups = in_channels if in_channels == out_channels else 1
        self.pad = kernel_size - 1
        self.conv = nn.Conv1d(
            in_channels, out_channels, kernel_size,
            padding=0, groups=groups, bias=bias,
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.conv(F.pad(x, (self.pad, 0)))


class ConvBlock(nn.Module):
    """Single causal convolution + GELU + dropout with residual.
    Input/Output: (B, C, T)
    """

    def __init__(self, n_channels: int, kernel_size: int = 12,
                 dropout: float = 0.1):
        super().__init__()
        self.net = nn.Sequential(
            CausalConv(n_channels, kernel_size=kernel_size),
            nn.GELU(),
            nn.Dropout(dropout),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x) + x


# ─────────────────────────────────────────────────────────────────────
# 2.  TIME EMBEDDING
# ─────────────────────────────────────────────────────────────────────

class TimeEmbedding(nn.Module):
    """Per-variate feature extraction then causal patch unfolding.

    1 → d_model (causal conv) → n_heads (1×1 conv) → refine (ConvBlock) → patches
    Output: (B, n_heads, n_patches * N, patch_len)
    """

    def __init__(self, seq_len: int, patch_len: int = 64,
                 stride: int = 32, n_heads: int = 8,
                 d_model: int = 32, dropout: float = 0.1, enc_in: int = 1):
        super().__init__()
        self.patch_len = patch_len
        self.stride = stride
        self.n_heads = n_heads
        self.n_patches = (seq_len - patch_len) // stride + 1

        self.project = CausalConv(1, d_model, kernel_size=3)
        self.channel_reduce = CausalConv(d_model, n_heads, kernel_size=1)
        self.refine = ConvBlock(n_heads, kernel_size=12, dropout=dropout)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor, x_mark=None):
        """x: (B, T, N) → (B, n_heads, n_patches*N, patch_len)"""
        B, T, N = x.shape
        x = x.permute(0, 2, 1).reshape(B * N, 1, T)

        x = self.project(x)
        x = self.channel_reduce(x)
        x = self.refine(x)

        x = torch.flip(x, dims=[2])
        patches = x.unfold(dimension=2, size=self.patch_len, step=self.stride)
        patches = patches.flip(dims=[3]).flip(dims=[2])

        patches = patches.view(B, N, self.n_heads, self.n_patches, self.patch_len)
        patches = patches.permute(0, 2, 3, 1, 4)
        patches = patches.reshape(B, self.n_heads, self.n_patches * N, self.patch_len)

        return self.dropout(patches)


# ─────────────────────────────────────────────────────────────────────
# 3.  SIGNED ATTENTION
# ─────────────────────────────────────────────────────────────────────

class SignedAttention(nn.Module):
    """A = softmax(+s) - softmax(-s), row-normalised.

    Causal mask: query patch t attends only to key patches t' where
    0 < (t - t') < attention_window (excludes self).
    """

    def __init__(self, n_patches: int, patch_len: int, n_heads: int,
                 attention_dropout: float = 0.1,
                 output_attention: bool = False,
                 attention_window: int = 10):
        super().__init__()
        self.n_patches = n_patches
        self.dropout = nn.Dropout(attention_dropout)
        self.output_attention = output_attention
        self.attention_window = attention_window
        self._cached_mask = None
        self._cached_key = None
        self.log_scale = nn.Parameter(torch.tensor(log(2.0)))

    def _get_causal_mask(self, channels: int, device: torch.device):
        key = (self.n_patches, channels, device)
        if self._cached_key != key or self._cached_mask is None:
            pid = torch.arange(self.n_patches, device=device).repeat_interleave(channels)
            diff = pid.unsqueeze(0) - pid.unsqueeze(1)
            mask = (diff <= 0) | (diff >= self.attention_window)
            self._cached_mask = mask.unsqueeze(0).unsqueeze(0)
            self._cached_key = key
        return self._cached_mask

    def forward(self, queries, keys, values, attn_mask=None):
        B, H, L, D = queries.shape
        channels = L // self.n_patches

        scores = torch.matmul(queries, keys.transpose(-1, -2))
        scale = self.log_scale.exp().clamp(1, 30.0) / sqrt(D)

        cmask = self._get_causal_mask(channels, queries.device)
        pos = scores.masked_fill(cmask, -1e4)
        neg = (-scores).masked_fill(cmask, -1e4)

        A = torch.softmax(scale * pos, dim=-1) - torch.softmax(scale * neg, dim=-1)
        A = self.dropout(A)

        V = torch.matmul(A, values)
        return V.contiguous(), (A if self.output_attention else None)


# ─────────────────────────────────────────────────────────────────────
# 4.  ATTENTION LAYER + STACK
# ─────────────────────────────────────────────────────────────────────

class ConvFFN(nn.Module):
    """Depthwise causal expand-then-reduce FFN.
    Input/Output: (B, H, L, D)
    """
    def __init__(self, n_heads: int, patch_len: int, expand: int = 2,
                 kernel_size: int = 3, dropout: float = 0.1):
        super().__init__()
        self.conv_up = CausalConv(n_heads, n_heads * expand,
                                  kernel_size=kernel_size, groups=n_heads)
        self.conv_down = CausalConv(n_heads * expand, n_heads,
                                    kernel_size=1, groups=n_heads)
        self.act = nn.GELU()
        self.drop = nn.Dropout(dropout)
        self.norm = nn.RMSNorm(patch_len)

    def forward(self, x):
        B, H, L, D = x.shape
        r = x.permute(0, 2, 1, 3).reshape(B * L, H, D)
        r = self.act(self.conv_up(r))
        r = self.conv_down(self.drop(r))
        r = r.reshape(B, L, H, D).permute(0, 2, 1, 3)
        return self.norm(x + r)


class SignedAttentionLayer(nn.Module):
    """SignedAttention + residual dropout + norm + ConvFFN."""

    def __init__(self, n_patches: int, patch_len: int, n_heads: int,
                 dropout: float = 0.1, output_attention: bool = False,
                 attention_window: int = 10):
        super().__init__()
        self.attn = SignedAttention(n_patches, patch_len, n_heads,
                                    dropout, output_attention, attention_window)
        self.drop = nn.Dropout(dropout)
        self.norm = nn.RMSNorm(patch_len)
        self.ffn = ConvFFN(n_heads, patch_len, expand=2, kernel_size=3,
                           dropout=dropout)

    def forward(self, h, attn_mask=None):
        out, att = self.attn(h, h, h, attn_mask)
        h = self.norm(h + self.drop(out))
        h = self.ffn(h)
        return h, att


class Stack(nn.Module):
    """One N-BEATS-style stack:
    e_layers × SignedAttentionLayer → aggregate heads → forecast
    """

    def __init__(self, n_heads: int, patch_len: int, seq_len: int, pred_len: int,
                 stride: int, e_layers: int = 2, dropout: float = 0.1,
                 output_attention: bool = False, attention_window: int = 10):
        super().__init__()
        n_patches = (seq_len - patch_len) // stride + 1
        self.n_patches = n_patches
        self.patch_len = patch_len
        self.pred_len = pred_len
        self.n_fore = ceil(pred_len / patch_len)

        self.layers = nn.ModuleList([
            SignedAttentionLayer(n_patches, patch_len, n_heads, dropout,
                                output_attention, attention_window)
            for _ in range(e_layers)
        ])

        self.channel_mixing = nn.Linear(n_heads, 1)
        self.fore_projection = nn.Linear(n_patches, self.n_fore)

    def forward(self, tokens, x_original, attn_mask=None):
        B, H, L, D = tokens.shape
        N = L // self.n_patches

        h = tokens
        attns = []
        for layer in self.layers:
            h, att = layer(h, attn_mask)
            attns.append(att)

        h = h.reshape(B, H, self.n_patches, N, D)
        h = h.permute(0, 3, 4, 2, 1).contiguous()
        h = self.channel_mixing(h).squeeze(-1)          # (B, N, D, P)

        forecast = self.fore_projection(h)              # (B, N, D, n_fore)
        forecast = forecast.flatten(start_dim=2)[:, :, -self.pred_len:]
        forecast = forecast.permute(0, 2, 1)            # (B, pred_len, N)

        return forecast, x_original, attns


# ─────────────────────────────────────────────────────────────────────
# 5.  STACKED ENCODER
# ─────────────────────────────────────────────────────────────────────

class StackedEncoder(nn.Module):

    def __init__(self, n_stacks: int, n_heads: int, patch_len: int,
                 seq_len: int, pred_len: int, d_model: int = 32, stride: int = 32,
                 e_layers: int = 2, dropout: float = 0.1,
                 output_attention: bool = False, enc_in: int = 1,
                 attention_window: int = 10):
        super().__init__()
        self.embeddings = nn.ModuleList([
            TimeEmbedding(seq_len, patch_len // (2 ** i), stride // (2 ** i),
                          n_heads, d_model, dropout, enc_in)
            for i in range(n_stacks)
        ])
        self.stacks = nn.ModuleList([
            Stack(n_heads, patch_len // (2 ** i), seq_len, pred_len,
                  stride=stride // (2 ** i), e_layers=e_layers, dropout=dropout,
                  output_attention=output_attention, attention_window=attention_window)
            for i in range(n_stacks)
        ])
        self.forecast_weights = nn.Parameter(torch.ones(n_stacks))

    def forward(self, x, attn_mask=None):
        """x: (B, seq_len, N) → (B, pred_len, N), attns"""
        residual = x
        forecasts = []
        all_attns = []

        for emb, stack in zip(self.embeddings, self.stacks):
            tokens = emb(residual)
            forecast_i, residual, attns = stack(tokens, residual, attn_mask)
            forecasts.append(forecast_i)
            all_attns.append(attns)

        w = torch.softmax(self.forecast_weights, dim=0)
        final = (torch.stack(forecasts, dim=-1) * w).sum(dim=-1)

        return final, all_attns


# ─────────────────────────────────────────────────────────────────────
# 6.  FOURIER SEASONAL MODEL
# ─────────────────────────────────────────────────────────────────────

class FourierSeasonalModel(nn.Module):
    """Learnable Fourier decomposition on temporal marks.

    For each temporal feature (hour, weekday, day, month) and each variate,
    learns Fourier coefficients to capture systematic periodic patterns.
    The output is interpretable: each harmonic k on each temporal feature
    corresponds to a specific periodicity (e.g. k=1 on hour-of-day = daily
    cycle, k=2 = twice-daily cycle).

    Input:  x_mark (B, T, n_time_features)  — values in [-0.5, 0.5]
    Output: seasonal component (B, T, n_variates)
    """

    def __init__(self, n_time_features: int, n_variates: int,
                 n_harmonics: int = 4):
        super().__init__()
        self.n_harmonics = n_harmonics
        self.n_time_features = n_time_features
        self.n_variates = n_variates

        self.cos_coeffs = nn.Parameter(
            torch.randn(n_time_features, n_variates, n_harmonics) * 0.02)
        self.sin_coeffs = nn.Parameter(
            torch.randn(n_time_features, n_variates, n_harmonics) * 0.02)
        self.bias = nn.Parameter(torch.zeros(n_variates))

    def forward(self, x_mark: torch.Tensor) -> torch.Tensor:
        """x_mark: (B, T, F) → (B, T, N_variates)"""
        k = torch.arange(1, self.n_harmonics + 1,
                         device=x_mark.device, dtype=x_mark.dtype)
        phases = (x_mark + 0.5).unsqueeze(-1) * (2.0 * math.pi) * k

        seasonal = (torch.einsum('fnh,btfh->btn', self.cos_coeffs,
                                  torch.cos(phases))
                    + torch.einsum('fnh,btfh->btn', self.sin_coeffs,
                                   torch.sin(phases)))
        return seasonal + self.bias


# ─────────────────────────────────────────────────────────────────────
# 7.  FULL MODEL
# ─────────────────────────────────────────────────────────────────────

class Model(nn.Module):
    """
    FourierSeasonal → RevIN(residual) → n_stacks × [embed → SignedAttention → forecast] → RevIN⁻¹ → + seasonal_forecast
    """

    def __init__(self, configs):
        super().__init__()
        self.seq_len = configs.seq_len + configs.pred_len
        self.pred_len = configs.pred_len
        self.output_attention = configs.output_attention
        self.use_norm = configs.use_norm

        self.encoder = StackedEncoder(
            n_stacks=getattr(configs, 'n_stacks', 3),
            n_heads=configs.n_heads,
            patch_len=configs.patch_len,
            seq_len=self.seq_len,
            pred_len=self.pred_len,
            d_model=getattr(configs, 'd_model', 32),
            stride=configs.stride,
            e_layers=configs.e_layers,
            dropout=configs.dropout,
            output_attention=configs.output_attention,
            enc_in=configs.enc_in,
            attention_window=configs.attention_window,
        )

        # Explicit --n_time_features > 0 wins; otherwise infer from --freq (timeenc=1 required)
        _freq_to_nfeats = {'h': 4, 't': 5, 's': 6, 'd': 3, 'b': 3, 'w': 2, 'm': 1, 'a': 1}
        freq = getattr(configs, 'freq', 'h')
        explicit = getattr(configs, 'n_time_features', 0)
        n_time_features = explicit if explicit > 0 else _freq_to_nfeats.get(freq, 4)
        n_harmonics = getattr(configs, 'n_harmonics', 4)
        self.seasonal = FourierSeasonalModel(
            n_time_features=n_time_features,
            n_variates=configs.enc_in,
            n_harmonics=n_harmonics,
        )
        self.flag = 0
        n_params = sum(p.numel() for p in self.parameters())
        print(f"Model parameters: {n_params:,}")

    def forecast(self, x_enc, x_mark_enc=None, x_dec=None, x_mark_dec=None):
        has_marks = x_mark_enc is not None and x_mark_dec is not None

        if self.use_norm:
            means = x_enc.mean(dim=1, keepdim=True).detach()
            x_enc = x_enc - means
            stdev = (x_enc.var(dim=1, keepdim=True, correction=0) + 1e-5).sqrt()
            x_enc = x_enc / stdev
        if has_marks:
            print(x_mark_enc.shape)
            seasonal_enc = self.seasonal(x_mark_enc)
            seasonal_dec = self.seasonal(x_mark_dec[:, -self.pred_len:, :])
            x_enc = x_enc - seasonal_enc

        if self.flag > 4:
            x_enc = F.pad(x_enc, (0, 0, 0, self.pred_len))
            
            dec_out, attns = self.encoder(x_enc)
            
            if has_marks:
                dec_out = dec_out + seasonal_dec
        else:
            dec_out = seasonal_dec
            attns = None
        self.flag = self.flag + 1

        if self.use_norm:
            dec_out = dec_out * stdev.squeeze(1).unsqueeze(1)
            dec_out = dec_out + means.squeeze(1).unsqueeze(1)

        return dec_out, attns

    def forward(self, x_enc, x_mark_enc=None, x_dec=None, x_mark_dec=None, mask=None):
        dec_out, attns = self.forecast(x_enc, x_mark_enc, x_dec, x_mark_dec)
        if self.output_attention:
            return dec_out[:, -self.pred_len:, :], attns
        return dec_out[:, -self.pred_len:, :]
