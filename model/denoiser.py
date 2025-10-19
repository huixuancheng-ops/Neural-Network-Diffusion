import math
import torch
import torch.nn as nn


class TimestepEmbedder(nn.Module):
    """Maps discrete diffusion timestep to a learnable embedding.

    Args:
        hidden_dim: Output embedding dimension used to condition the denoiser.
        frequency_embedding_size: Size of the sinusoidal input features.
        max_period: Controls the frequency range of the sinusoidal features.
    """

    def __init__(self, hidden_dim: int, frequency_embedding_size: int = 256, max_period: int = 10000) -> None:
        super().__init__()
        assert frequency_embedding_size % 2 == 0
        self.frequency_embedding_size = frequency_embedding_size
        self.mlp = nn.Sequential(
            nn.Linear(frequency_embedding_size, hidden_dim, bias=True),
            nn.SiLU(),
            nn.Linear(hidden_dim, hidden_dim, bias=True),
        )
        half = frequency_embedding_size // 2
        freqs = torch.exp(-math.log(max_period) * torch.arange(start=0, end=half) / half)
        self.register_buffer("freqs", freqs)

    def forward(self, t: torch.Tensor) -> torch.Tensor:
        """Return timestep embedding of shape [batch_size, hidden_dim]."""
        args = t[:, None].float() * self.freqs
        t_freq = torch.cat([torch.cos(args), torch.sin(args)], dim=-1)
        t_emb = self.mlp(t_freq)
        return t_emb


class SinusoidalPositionalEmbedding(nn.Module):
    """Standard 1D sinusoidal positional embedding.

    Produces a tensor of shape [1, seq_len, dim] that can be added to token features.
    """

    def __init__(self, dim: int) -> None:
        super().__init__()
        self.dim = dim

    def forward(self, seq_len: int, device: torch.device, dtype: torch.dtype = torch.float32) -> torch.Tensor:
        position = torch.arange(0, seq_len, device=device, dtype=dtype).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, self.dim, 2, device=device, dtype=dtype) * (-math.log(10000.0) / self.dim))
        pe = torch.zeros((seq_len, self.dim), device=device, dtype=dtype)
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        return pe.unsqueeze(0)  # [1, seq_len, dim]


class TransformerDenoiser(nn.Module):
    """Transformer-based denoiser for 1D vector signals (epsilon prediction).

    This module takes a vector x_t of shape [batch, length], embeds per-position scalars
    to a feature space, adds timestep conditioning, processes with a Transformer encoder,
    and projects back to a scalar per position to predict noise.

    Args:
        embed_dim: Token feature dimension inside the Transformer.
        num_layers: Number of Transformer encoder layers.
        num_heads: Number of attention heads.
        mlp_ratio: Expansion ratio of the feed-forward network.
        dropout: Dropout probability used in attention and MLP.
        time_embed_dim: Hidden dimension of the timestep embedding.
    """

    def __init__(
        self,
        embed_dim: int = 128,
        num_layers: int = 4,
        num_heads: int = 8,
        mlp_ratio: float = 4.0,
        dropout: float = 0.0,
        time_embed_dim: int | None = None,
    ) -> None:
        super().__init__()
        self.embed_dim = embed_dim
        self.time_embed_dim = embed_dim if time_embed_dim is None else time_embed_dim

        self.time_embedder = TimestepEmbedder(hidden_dim=self.time_embed_dim)
        self.time_to_token = nn.Linear(self.time_embed_dim, embed_dim, bias=True)

        # Project scalar per position -> token features
        self.input_proj = nn.Linear(1, embed_dim, bias=True)
        self.pos_embed = SinusoidalPositionalEmbedding(embed_dim)

        encoder_layer = nn.TransformerEncoderLayer(
            d_model=embed_dim,
            nhead=num_heads,
            dim_feedforward=int(mlp_ratio * embed_dim),
            dropout=dropout,
            activation="gelu",
            batch_first=True,
            norm_first=True,
        )
        self.encoder = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)

        # Project token features -> scalar per position
        self.output_proj = nn.Linear(embed_dim, 1, bias=True)

    def forward(self, x: torch.Tensor, t: torch.Tensor, c: torch.Tensor | float = 0.0) -> torch.Tensor:
        """Predict epsilon given noised sample x and timestep t.

        Args:
            x: Noised input of shape [batch, length].
            t: Integer timesteps of shape [batch].
            c: Optional condition (unused placeholder for interface compatibility).

        Returns:
            Tensor of shape [batch, length], predicted noise per position.
        """
        batch_size, seq_len = x.shape
        x_tokens = self.input_proj(x.unsqueeze(-1))  # [B, L, C]

        # positional embedding
        pos = self.pos_embed(seq_len=seq_len, device=x_tokens.device, dtype=x_tokens.dtype)  # [1, L, C]
        x_tokens = x_tokens + pos

        # timestep conditioning (broadcast add)
        t_emb = self.time_to_token(self.time_embedder(t)).unsqueeze(1)  # [B, 1, C]
        x_tokens = x_tokens + t_emb

        # transformer encoder
        x_tokens = self.encoder(x_tokens)  # [B, L, C]

        # project back to scalar per token
        out = self.output_proj(x_tokens).squeeze(-1)  # [B, L]
        return out