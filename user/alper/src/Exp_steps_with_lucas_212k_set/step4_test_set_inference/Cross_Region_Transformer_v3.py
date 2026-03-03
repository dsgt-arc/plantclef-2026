import torch
import torch.nn as nn
import torch.nn.functional as F


class CrossRegionTransformer(nn.Module):
    """
    Cross Region Transformer for multi label classification from cached tile embeddings.

    Safe upgrades added
      1) Learnable tile positional embedding when pos_encodings is not provided
      2) Gated metadata injection into CLS token to reduce early domination by metadata
      3) Pre norm TransformerEncoderLayer when supported by the installed torch version

    Inputs
        region_embeddings: [B, N, input_dim]
        metadata optional: [B, metadata_dim]
        pos_encodings optional: [N, hidden_dim] must match tile ordering

    Output
        logits: [B, num_classes]
    """

    def __init__(
        self,
        input_dim: int = 768,
        hidden_dim: int = 512,
        num_heads: int = 8,
        num_layers: int = 4,
        num_classes: int = 7806,
        metadata_dim: int = 4,
        dropout: float = 0.1,
        use_attention_pooling: bool = True,
    ):
        super().__init__()

        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.num_classes = num_classes
        self.metadata_dim = metadata_dim
        self.use_attention_pooling = use_attention_pooling

        # 1) Linear projection to transformer hidden size
        self.embedding_projection = nn.Linear(input_dim, hidden_dim)

        # 2) Metadata projection for CLS injection
        self.metadata_fc = nn.Sequential(
            nn.Linear(metadata_dim, hidden_dim // 2),
            nn.ReLU(),
            nn.Linear(hidden_dim // 2, hidden_dim),
        )

        # Metadata gate, starts near zero after sigmoid
        # sigmoid(0.0) = 0.5, so we bias it negative for a smaller initial effect
        self.meta_gate = nn.Parameter(torch.tensor(-2.0))

        # 3) Dedicated global token
        self.cls_token = nn.Parameter(torch.zeros(1, 1, hidden_dim))
        nn.init.trunc_normal_(self.cls_token, std=0.02)

        # 4) Learnable tile positional embedding fallback
        # Must cover your tile counts, 16 for 4x4 and 25 for 5x5, plus headroom
        self.max_tiles = 64
        self.tile_pos_emb = nn.Parameter(torch.zeros(1, self.max_tiles, hidden_dim))
        nn.init.trunc_normal_(self.tile_pos_emb, std=0.02)

        # 5) Transformer encoder
        # Use pre norm when supported, else fall back
        encoder_layer = None
        try:
            encoder_layer = nn.TransformerEncoderLayer(
                d_model=hidden_dim,
                nhead=num_heads,
                dim_feedforward=hidden_dim * 4,
                dropout=dropout,
                activation="gelu",
                batch_first=True,
                norm_first=True,
            )
        except TypeError:
            encoder_layer = nn.TransformerEncoderLayer(
                d_model=hidden_dim,
                nhead=num_heads,
                dim_feedforward=hidden_dim * 4,
                dropout=dropout,
                activation="gelu",
                batch_first=True,
            )

        self.transformer_encoder = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)

        # 6) Attention pooling head optional
        self.pooling_attention = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.Tanh(),
            nn.Linear(hidden_dim // 2, 1),
        )

        self.pre_classifier_norm = nn.LayerNorm(hidden_dim)

        # 7) Multi label classifier head
        self.classifier = nn.Linear(hidden_dim, num_classes)

        self.dropout = nn.Dropout(dropout)

    def forward(
        self,
        region_embeddings: torch.Tensor,
        metadata: torch.Tensor | None = None,
        pos_encodings: torch.Tensor | None = None,
    ) -> torch.Tensor:
        if region_embeddings.dim() != 3:
            raise ValueError(f"region_embeddings must be [B, N, D], got {tuple(region_embeddings.shape)}")

        B, N, D = region_embeddings.shape
        if D != self.input_dim:
            raise ValueError(f"Expected input_dim={self.input_dim}, got {D}")

        if N <= 0:
            raise ValueError("Num tiles must be positive")

        # Step 1: Project tile embeddings to hidden dim
        x = self.embedding_projection(region_embeddings)  # [B, N, H]

        # Step 2: Add positional encodings to tiles
        if pos_encodings is not None:
            if pos_encodings.dim() != 2:
                raise ValueError(f"pos_encodings must be [N, H], got {tuple(pos_encodings.shape)}")
            if pos_encodings.shape != (N, self.hidden_dim):
                raise ValueError(
                    f"pos_encodings must be [N, H]=[{N}, {self.hidden_dim}], got {tuple(pos_encodings.shape)}"
                )
            x = x + pos_encodings.unsqueeze(0)
        else:
            if N > self.max_tiles:
                raise ValueError(f"Num tiles N={N} exceeds max_tiles={self.max_tiles}")
            x = x + self.tile_pos_emb[:, :N, :]

        # Step 3: Build CLS token and inject metadata into CLS only, with gate
        cls = self.cls_token.expand(B, 1, self.hidden_dim)  # [B, 1, H]
        if metadata is not None:
            if metadata.dim() != 2:
                raise ValueError(f"metadata must be [B, metadata_dim], got {tuple(metadata.shape)}")
            if metadata.shape[0] != B or metadata.shape[1] != self.metadata_dim:
                raise ValueError(
                    f"metadata must be [B, metadata_dim]=[{B}, {self.metadata_dim}], got {tuple(metadata.shape)}"
                )
            g = torch.sigmoid(self.meta_gate).to(dtype=cls.dtype)
            cls = cls + (g * self.metadata_fc(metadata).unsqueeze(1))

        # Step 4: Prepend CLS to token sequence
        x = torch.cat([cls, x], dim=1)  # [B, 1+N, H]

        # Step 5: Self attention
        x = self.dropout(x)
        x = self.transformer_encoder(x)  # [B, 1+N, H]

        # Step 6: Pooling
        if self.use_attention_pooling:
            attn = self.pooling_attention(x)      # [B, 1+N, 1]
            attn = F.softmax(attn, dim=1)         # [B, 1+N, 1]
            pooled = torch.sum(x * attn, dim=1)   # [B, H]
        else:
            pooled = x[:, 0, :]                   # [B, H]

        pooled = self.pre_classifier_norm(pooled)

        # Step 7: Classifier head
        logits = self.classifier(pooled)
        return logits
