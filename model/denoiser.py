import torch
import torch.nn as nn
import math


class TimestepEmbedder(nn.Module):
    def __init__(self, hidden_dim, frequency_embedding_size=256, max_period=10000):
        super().__init__()
        assert frequency_embedding_size % 2 == 0
        self.frequency_embedding_size = frequency_embedding_size
        self.mlp = nn.Sequential(
            nn.Linear(frequency_embedding_size, hidden_dim, bias=True),
            nn.SiLU(),
            nn.Linear(hidden_dim, hidden_dim, bias=True)
        )  # FIXME: this is too big! Why this is such necessary?
        half = frequency_embedding_size // 2
        freqs = torch.exp(-math.log(max_period) * torch.arange(start=0, end=half) / half)
        self.register_buffer("freqs", freqs)

    def forward(self, t):
        args = t[:, None].float() * self.freqs
        t_freq = torch.cat([torch.cos(args), torch.sin(args)], dim=-1)
        t_emb = self.mlp(t_freq)
        return t_emb


class OneDimCNN(nn.Module):
    def __init__(self, layer_channels: list, model_dim: int, kernel_size: int):
        super().__init__()
        self.time_embedder = TimestepEmbedder(hidden_dim=model_dim)
        self.encoder_list = nn.ModuleList([])
        for i in range(len(layer_channels) // 2 + 1):
            self.encoder_list.append(nn.ModuleList([
                nn.Conv1d(layer_channels[i], layer_channels[i+1], kernel_size, 1, kernel_size // 2),
                nn.Sequential(nn.BatchNorm1d(layer_channels[i+1]), nn.ELU())
            ]))
        self.decoder_list = nn.ModuleList([])
        for i in range(len(layer_channels) // 2 + 1, len(layer_channels) - 1):
            self.decoder_list.append(nn.ModuleList([
                nn.Conv1d(layer_channels[i], layer_channels[i+1], kernel_size, 1, kernel_size // 2),
                nn.Sequential(nn.BatchNorm1d(layer_channels[i+1]), nn.ELU())
                    if layer_channels[i+1] != 1 else nn.Identity(),
            ]))

    def forward(self, x, t, c=0.):
        x = x[:, None, :]
        t = self.time_embedder(t)[:, None, :]
        x_list = []
        for i, (module, activation) in enumerate(self.encoder_list):
            x = module(x + t)
            x = activation(x)
            if i < len(self.encoder_list) - 2:
                x_list.append(x)
        for i, (module, activation) in enumerate(self.decoder_list):
            x = x + x_list[-i-1]
            x = module(x + t)
            x = activation(x)
        return x[:, 0, :]


class SinusoidalPositionalEncoding(nn.Module):
    def __init__(self, sequence_length: int, embedding_dim: int):
        super().__init__()
        assert embedding_dim % 2 == 0, "embedding_dim must be even for sinusoidal PE"
        position = torch.arange(sequence_length, dtype=torch.float32).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, embedding_dim, 2, dtype=torch.float32) *
                             (-math.log(10000.0) / embedding_dim))
        pe = torch.zeros(sequence_length, embedding_dim, dtype=torch.float32)
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        self.register_buffer('pe', pe, persistent=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: [batch, seq_len, embed_dim]
        return x + self.pe.unsqueeze(0).to(dtype=x.dtype, device=x.device)


class TransformerBlock(nn.Module):
    def __init__(self, d_model: int, n_heads: int, dim_feedforward: int, dropout: float):
        super().__init__()
        self.attn = nn.MultiheadAttention(embed_dim=d_model, num_heads=n_heads, dropout=dropout, batch_first=True)
        self.attn_norm = nn.LayerNorm(d_model)
        self.ff = nn.Sequential(
            nn.Linear(d_model, dim_feedforward),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(dim_feedforward, d_model),
        )
        self.ff_norm = nn.LayerNorm(d_model)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Self-attention block with pre-norm
        attn_input = self.attn_norm(x)
        attn_out, _ = self.attn(attn_input, attn_input, attn_input, need_weights=False)
        x = x + self.dropout(attn_out)
        # Feed-forward block with pre-norm
        ff_input = self.ff_norm(x)
        ff_out = self.ff(ff_input)
        x = x + self.dropout(ff_out)
        return x


class TransformerDenoiser(nn.Module):
    def __init__(
        self,
        sequence_length: int,
        time_embedding_dim: int,
        d_model: int,
        n_layers: int = 4,
        n_heads: int = 8,
        dim_feedforward: int = 1024,
        dropout: float = 0.1,
        use_positional_encoding: bool = True,
    ):
        super().__init__()
        self.sequence_length = sequence_length
        self.time_embedder = TimestepEmbedder(hidden_dim=time_embedding_dim)
        self.time_to_model = nn.Linear(time_embedding_dim, d_model)
        self.input_proj = nn.Linear(1, d_model)
        self.use_positional_encoding = use_positional_encoding
        if use_positional_encoding:
            self.positional_encoding = SinusoidalPositionalEncoding(sequence_length, d_model)
        self.blocks = nn.ModuleList([
            TransformerBlock(d_model=d_model, n_heads=n_heads, dim_feedforward=dim_feedforward, dropout=dropout)
            for _ in range(n_layers)
        ])
        self.output_norm = nn.LayerNorm(d_model)
        self.output_proj = nn.Linear(d_model, 1)

    def forward(self, x: torch.Tensor, t: torch.Tensor, c: torch.Tensor | float = 0.0) -> torch.Tensor:
        # x: [batch, seq_len], t: [batch]
        assert x.dim() == 2, f"Expected x shape [B, L], got {x.shape}"
        assert x.size(1) == self.sequence_length, (
            f"sequence_length mismatch: expected {self.sequence_length}, got {x.size(1)}"
        )
        b, l = x.size(0), x.size(1)
        x = x.unsqueeze(-1)  # [B, L, 1]
        x = self.input_proj(x)  # [B, L, d_model]
        if self.use_positional_encoding:
            x = self.positional_encoding(x)
        # time conditioning
        t_emb = self.time_embedder(t)  # [B, time_embedding_dim]
        t_emb = self.time_to_model(t_emb).unsqueeze(1)  # [B, 1, d_model]
        x = x + t_emb
        # transformer blocks
        for block in self.blocks:
            x = block(x)
        x = self.output_norm(x)
        x = self.output_proj(x).squeeze(-1)  # [B, L]
        return x