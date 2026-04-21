import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from math import sqrt, ceil

# ─────────────────────────────────────────────────────────────────────
# 1.  CONVOLUTION PRIMITIVES
# ─────────────────────────────────────────────────────────────────────

class CausalConv(nn.Module):
    """Causal 1-D convolution. Supports both depthwise (in==out, grouped)
    and channel-changing (1→d_model, etc.).
    Input/Output: (B, C_in, T) → (B, C_out, T)
    """

    def __init__(self, in_channels: int, out_channels: int = None,
                 kernel_size: int = 2, dilation: int = 1, bias: bool = False):
        super().__init__()
        out_channels = out_channels or in_channels
        self.kernel_size = kernel_size
        self.dilation = dilation
        groups = in_channels if in_channels == out_channels else 1
        self.conv = nn.Conv1d(
            in_channels, out_channels, kernel_size,
            padding=0, dilation=dilation, groups=groups, bias=bias,
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        left = self.dilation * (self.kernel_size - 1)
        return self.conv(F.pad(x, (left, 0)))


class ChannelMix(nn.Module):
    """1×1 conv mixing channels at each time step.
    (B, C_in, T) → (B, C_out, T)
    """

    def __init__(self, in_channels: int, out_channels: int, bias: bool = False):
        super().__init__()
        self.conv = nn.Conv1d(in_channels, out_channels, kernel_size=1, bias=bias)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.conv(x)


class TCNBlock(nn.Module):
    """Depthwise causal dilated convolutions + GELU + dropout.
    No normalisation. Residual connection around the block.
    Input/Output: (B, C, T)
    """

    def __init__(self, n_channels: int, dilations: tuple = (1, 2, 4),
                 kernel_size: int = 2, dropout: float = 0.1):
        super().__init__()
        layers = []
        for d in dilations:
            layers.extend([
                CausalConv(n_channels, kernel_size=kernel_size, dilation=d),
                nn.GELU(),
                nn.Dropout(dropout),
            ])
        self.net = nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x) + x


# ─────────────────────────────────────────────────────────────────────
# 2.  TIME EMBEDDING  (depthwise → channel-mix → depthwise)
# ─────────────────────────────────────────────────────────────────────

class TimeEmbedding(nn.Module):
    """Three-stage feature extraction per variate:

    Stage 1 — Causal conv: 1 → d_model channels
    Stage 2 — Channel-mix 1×1 conv: d_model → n_heads
    Stage 3 — TCNBlock: refines each head independently

    Then unfold into causal patches.
    Output: (B, n_heads, n_patches * N, patch_len)
    """

    def __init__(self, seq_len: int, patch_len: int = 64,
                 stride: int = 32, n_heads: int = 8,
                 d_model: int = 32, dropout: float = 0.1, enc_in: int = 1):
        super().__init__()
        self.patch_len = patch_len
        self.stride = stride
        self.n_heads = n_heads
        self.d_model = d_model
        self.n_patches = (seq_len - patch_len) // stride + 1

        self.stage1 = nn.Sequential(
            CausalConv(1, d_model, kernel_size=24, dilation=1),
        )
        self.stage2 = nn.Sequential(
            ChannelMix(d_model, n_heads),
        )
        self.stage3 = TCNBlock(
            n_channels=n_heads,
            dilations=(1,),
            kernel_size=8,
            dropout=dropout,
        )
        self.dropout = nn.Dropout(dropout)

    @staticmethod
    def _haar_2level(x):
        """2-level Haar wavelet on last dim (must be length 32). x: (..., D)"""
        a1 = (x[..., 0::2] + x[..., 1::2]) * 0.5   # (..., 16)
        d1 = (x[..., 0::2] - x[..., 1::2]) * 0.5   # (..., 16)
        a2 = (a1[..., 0::2] + a1[..., 1::2]) * 0.5  # (..., 8)
        d2 = (a1[..., 0::2] - a1[..., 1::2]) * 0.5  # (..., 8)
        return torch.cat([a2, d2, d1], dim=-1)        # (..., 32)

    def forward(self, x: torch.Tensor, x_mark=None):
        """x: (B, T, N) → (B, n_heads, n_patches*N, patch_len)"""
        x = x.permute(0, 2, 1)                   # (B, N, T)
        B, N, T = x.shape
        x = x.reshape(B * N, 1, T)               # (B*N, 1, T)

        x = self.stage1(x)                        # (B*N, d_model, T)
        x = self.stage2(x)                        # (B*N, n_heads, T)
        x = self.stage3(x)                        # (B*N, n_heads, T)

        # Causal patch extraction
        x = torch.flip(x, dims=[2])
        patches = x.unfold(dimension=2, size=self.patch_len, step=self.stride)
        patches = patches.flip(dims=[3]).flip(dims=[2])

        patches = patches.view(B, N, self.n_heads, self.n_patches, self.patch_len)
        patches = patches.permute(0, 2, 3, 1, 4)         # (B, H, P, N, D)
        patches = patches.reshape(B, self.n_heads, self.n_patches * N, self.patch_len)

        patches = self._haar_2level(patches)              # multi-scale wavelet representation
        return self.dropout(patches)


# ─────────────────────────────────────────────────────────────────────
# 3.  SIGNED ATTENTION
# ─────────────────────────────────────────────────────────────────────

class SignedAttention(nn.Module):
    """A = softmax(+s) - softmax(-s), row-normalised.

    Q/K/V projections via causal 1D convs along the patch time dim.
    Causal mask enforces that query patch t can only attend to key patch t' <= t.
    attention_window limits how far back each query looks (in patch steps).
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
        self.log_scale = nn.Parameter(torch.tensor(np.log(2.0)))

    def _get_causal_mask(self, channels: int, attention_window: int, device: torch.device):
        key = (self.n_patches, channels, device)
        if self._cached_key != key or self._cached_mask is None:
            L = self.n_patches * channels
            pid = torch.arange(self.n_patches, device=device).repeat_interleave(channels)
            diff = pid.unsqueeze(0) - pid.unsqueeze(1)  # q_pid - k_pid
            mask = (diff <= 0) | (diff >= attention_window)
            self._cached_mask = mask.unsqueeze(0).unsqueeze(0)  # (1, 1, L, L)
            self._cached_key = key
        return self._cached_mask

    @staticmethod
    def _standardize(t):
        t = t - t.mean(dim=-1, keepdim=True)
        return t / (t.std(dim=-1, keepdim=True, correction=0) + 1e-4)

    def forward(self, queries, keys, values, attn_mask=None):
        B, H, L, D = queries.shape
        channels = L // self.n_patches

        q, k, v = queries, keys, values

        scores = torch.matmul(q, k.transpose(-1, -2))
        scale = self.log_scale.exp().clamp(1, 30.0) / sqrt(D)

        cmask = self._get_causal_mask(channels, self.attention_window, queries.device)
        pos = scores.masked_fill(cmask, -1e4)
        neg = (-scores).masked_fill(cmask, -1e4)

        A = torch.softmax(scale * pos, dim=-1) - torch.softmax(scale * neg, dim=-1)
        A = self.dropout(A)

        V = torch.matmul(A, v)
        return V.contiguous(), (A if self.output_attention else None)


# ─────────────────────────────────────────────────────────────────────
# 4.  STACK
# ─────────────────────────────────────────────────────────────────────

class ConvFFN(nn.Module):
    """Per-token FFN using depthwise causal convs instead of Linear.
    Preserves the time-series interpretation of each patch token.
    Input/Output: (B, H, L, D)
    """
    def __init__(self, n_heads: int, patch_len: int, expand: int = 2,
                 kernel_size: int = 3, dropout: float = 0.1):
        super().__init__()
        self.pad = kernel_size - 1
        self.conv_up = nn.Conv1d(n_heads, n_heads * expand, kernel_size,
                                 padding=0, groups=n_heads, bias=False)
        self.conv_down = nn.Conv1d(n_heads * expand, n_heads, 1,
                                   groups=n_heads, bias=False)
        self.act = nn.GELU()
        self.drop = nn.Dropout(dropout)
        self.norm = nn.RMSNorm(patch_len)

    def forward(self, x):
        B, H, L, D = x.shape
        r = x.permute(0, 2, 1, 3).reshape(B * L, H, D)
        r = F.pad(r, (self.pad, 0))
        r = self.conv_up(r)
        r = self.act(r)
        r = self.drop(r)
        r = self.conv_down(r)
        r = r.reshape(B, L, H, D).permute(0, 2, 1, 3)
        return self.norm(x + r)


class Stack(nn.Module):
    """One N-BEATS-style stack:
    1. e_layers × SignedAttention (residual + dropout + norm)
    2. ConvFFN per layer
    3. Aggregate heads → forecast (B, pred_len, N) and pass residual forward
    """

    def __init__(self, n_heads: int, patch_len: int, seq_len: int, pred_len: int,
                 stride: int, e_layers: int = 2, dropout: float = 0.1,
                 output_attention: bool = False, attention_window: int = 10):
        super().__init__()
        self.n_heads = n_heads
        self.n_patches = (seq_len - patch_len) // stride + 1
        self.patch_len = patch_len
        self.pred_len = pred_len
        self.n_fore = ceil(pred_len / patch_len)

        self.attn_layers = nn.ModuleList([
            SignedAttention(self.n_patches, patch_len, n_heads, dropout,
                            output_attention, attention_window=attention_window)
            for _ in range(e_layers)
        ])
        self.attn_drops = nn.ModuleList([nn.Dropout(dropout) for _ in range(e_layers)])
        self.attn_norms = nn.ModuleList([nn.RMSNorm(patch_len) for _ in range(e_layers)])
        self.ffns = nn.ModuleList([
            ConvFFN(n_heads, patch_len, expand=2, kernel_size=3, dropout=dropout)
            for _ in range(e_layers)
        ])

        self.channel_mixing = nn.Linear(n_heads, 1)
        self.fore_projection = nn.Linear(self.n_patches, self.n_fore)

    def forward(self, tokens, x_original, attn_mask=None):
        """
        tokens     : (B, H, P*N, D)
        x_original : (B, seq_len, N)

        Returns:
            forecast : (B, pred_len, N)
            residual : (B, seq_len, N)
            attns    : list
        """
        B, H_heads, L, D = tokens.shape
        N = L // self.n_patches

        h = tokens
        attns = []
        for attn, drop, norm, ffn in zip(self.attn_layers, self.attn_drops,
                                          self.attn_norms, self.ffns):
            out, att = attn(h, h, h, attn_mask)
            h = norm(h + drop(out))
            h = ffn(h)
            attns.append(att)

        h = h.reshape(B, H_heads, self.n_patches, N, D)
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
        self.n_stacks = n_stacks

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
# 6.  FULL MODEL
# ─────────────────────────────────────────────────────────────────────

class MovingAvg(nn.Module):
    """Moving average for trend decomposition."""
    def __init__(self, kernel_size: int):
        super().__init__()
        self.kernel_size = kernel_size
        self.avg = nn.AvgPool1d(kernel_size=kernel_size, stride=1,
                                padding=(kernel_size - 1) // 2)

    def forward(self, x):
        """x: (B, N, T) → (B, N, T)"""
        return self.avg(x)[:, :, :x.shape[2]]


class DLinearForecast(nn.Module):
    """DLinear: decompose into trend + remainder, project trend separately."""
    def __init__(self, seq_len: int, pred_len: int, kernel_size: int = 13):
        super().__init__()
        self.seq_len = seq_len
        self.decomp = MovingAvg(kernel_size)
        self.linear_trend = nn.Linear(seq_len, pred_len)

    def forward(self, x):
        """x: (B, N, seq_len) → (B, N, pred_len), residual (B, N, seq_len)"""
        trend = self.decomp(x)
        resid = x - trend
        return self.linear_trend(trend[:, :, -self.seq_len:]), resid


class Model(nn.Module):
    """
    RevIN → trend decomp → n_stacks × [embed → SignedAttention × e_layers → forecast residual] → trend + residual → RevIN⁻¹
    """

    def __init__(self, configs):
        super().__init__()
        self.seq_len = configs.seq_len + configs.pred_len
        self.pred_len = configs.pred_len
        self.output_attention = configs.output_attention
        self.use_norm = configs.use_norm
        self.patch_len = configs.patch_len
        self.n_heads = configs.n_heads
        self.enc_in = configs.enc_in
        self.n_stacks = getattr(configs, 'n_stacks', 3)

        self.trend_decomp = DLinearForecast(configs.seq_len, configs.pred_len, kernel_size=25)

        self.encoder = StackedEncoder(
            n_stacks=self.n_stacks,
            n_heads=self.n_heads,
            patch_len=self.patch_len,
            seq_len=self.seq_len,
            pred_len=self.pred_len,
            d_model=getattr(configs, 'd_model', 32),
            stride=configs.stride,
            e_layers=configs.e_layers,
            dropout=configs.dropout,
            output_attention=configs.output_attention,
            enc_in=self.enc_in,
            attention_window=configs.attention_window,
        )

        n_params = sum(p.numel() for p in self.parameters())
        print(f"Model parameters: {n_params:,}")

    def forecast(self, x_enc, x_mark_enc=None, x_dec=None, x_mark_dec=None):
        if self.use_norm:
            means = x_enc.mean(dim=1, keepdim=True).detach()
            x_enc = x_enc - means
            stdev = (x_enc.var(dim=1, keepdim=True, correction=0) + 1e-5).sqrt()
            x_enc = x_enc / stdev

        # Decompose into trend forecast + residual; encoder handles residual
        x_t = x_enc.permute(0, 2, 1)                         # (B, N, T)
        trend_forecast, resid = self.trend_decomp(x_t)        # (B, N, pred_len), (B, N, T)
        resid = resid.permute(0, 2, 1)                        # (B, T, N)
        trend_forecast = trend_forecast.permute(0, 2, 1)      # (B, pred_len, N)

        resid = F.pad(resid, (0, 0, 0, self.pred_len))

        resid_out, attns = self.encoder(resid)

        dec_out = resid_out + trend_forecast

        if self.use_norm:
            dec_out = dec_out * stdev.squeeze(1).unsqueeze(1)
            dec_out = dec_out + means.squeeze(1).unsqueeze(1)

        return dec_out, attns

    def forward(self, x_enc, x_mark_enc=None, x_dec=None, x_mark_dec=None, mask=None):
        dec_out, attns = self.forecast(x_enc, x_mark_enc, x_dec, x_mark_dec)
        if self.output_attention:
            return dec_out[:, -self.pred_len:, :], attns
        return dec_out[:, -self.pred_len:, :]
