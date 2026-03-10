"""CrossRegionTransformer v4 — set-function aggregator for tile embeddings.

Changes from v3:
  - Removed positional embeddings (top-down vegetation has no spatial prior;
    consistency loss assumes position-invariance)
  - Removed metadata injection (test dataset has no metadata)
  - Added attention masking for variable-length tile input
  - Added embedding dropout on raw 768-dim input before projection
  - Added optional return of attention pooling weights for visualization
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class CrossRegionTransformer(nn.Module):
    """Set-function transformer for multi-label classification from tile embeddings.

    Operates as an order-invariant aggregator: no positional embeddings, no
    metadata. Tile order does not affect output.

    Inputs
        tile_embeddings : [B, N, input_dim]   (N varies per image, padded in batch)
        attention_mask  : [B, N] bool          (True = valid tile, False = padding)

    Output
        logits          : [B, num_classes]
        attn_weights    : [B, 1+N] (optional, when return_attention=True)
    """

    def __init__(
        self,
        input_dim: int = 768,
        hidden_dim: int = 512,
        num_heads: int = 8,
        num_layers: int = 4,
        num_classes: int = 7806,
        dropout: float = 0.1,
        embed_dropout: float = 0.1,
        use_attention_pooling: bool = True,
    ):
        super().__init__()

        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.num_classes = num_classes
        self.use_attention_pooling = use_attention_pooling

        # Dropout on raw input embeddings before projection
        self.embed_dropout = nn.Dropout(embed_dropout)

        # Linear projection to transformer hidden size
        self.embedding_projection = nn.Linear(input_dim, hidden_dim)

        # Learnable CLS token for global aggregation
        self.cls_token = nn.Parameter(torch.zeros(1, 1, hidden_dim))
        nn.init.trunc_normal_(self.cls_token, std=0.02)

        # Transformer encoder (pre-norm)
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=hidden_dim,
            nhead=num_heads,
            dim_feedforward=hidden_dim * 4,
            dropout=dropout,
            activation="gelu",
            batch_first=True,
            norm_first=True,
        )
        self.transformer_encoder = nn.TransformerEncoder(
            encoder_layer, num_layers=num_layers
        )

        # Attention pooling head
        self.pooling_attention = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.Tanh(),
            nn.Linear(hidden_dim // 2, 1),
        )

        self.pre_classifier_norm = nn.LayerNorm(hidden_dim)

        # Multi-label classifier
        self.classifier = nn.Linear(hidden_dim, num_classes)

        self.dropout = nn.Dropout(dropout)

    def forward(
        self,
        tile_embeddings: torch.Tensor,
        attention_mask: torch.Tensor | None = None,
        return_attention: bool = False,
    ) -> torch.Tensor | tuple[torch.Tensor, torch.Tensor]:
        """
        Args:
            tile_embeddings: [B, N, input_dim] tile embeddings (768-dim DINOv2 CLS)
            attention_mask:  [B, N] boolean mask. True = real tile, False = padding.
                             If None, all tiles are treated as valid.
            return_attention: if True, return (logits, attn_weights) tuple

        Returns:
            logits: [B, num_classes]
            attn_weights: [B, 1+N] attention pooling weights (only if return_attention=True)
        """
        if tile_embeddings.dim() != 3:
            raise ValueError(
                f"tile_embeddings must be [B, N, D], got {tuple(tile_embeddings.shape)}"
            )

        B, N, D = tile_embeddings.shape
        if D != self.input_dim:
            raise ValueError(f"Expected input_dim={self.input_dim}, got {D}")

        # Step 1: Embedding dropout on raw input
        x = self.embed_dropout(tile_embeddings)

        # Step 2: Project to hidden dim
        x = self.embedding_projection(x)  # [B, N, H]

        # Step 3: Prepend CLS token
        cls = self.cls_token.expand(B, 1, self.hidden_dim)  # [B, 1, H]
        x = torch.cat([cls, x], dim=1)  # [B, 1+N, H]

        # Step 4: Build transformer attention mask
        # nn.TransformerEncoder expects src_key_padding_mask: [B, S]
        # where True = IGNORE (padding position)
        src_key_padding_mask = None
        if attention_mask is not None:
            # Prepend True for CLS token (always valid)
            cls_mask = torch.ones(B, 1, dtype=torch.bool, device=attention_mask.device)
            full_mask = torch.cat([cls_mask, attention_mask], dim=1)  # [B, 1+N]
            # Invert: transformer expects True = ignore
            src_key_padding_mask = ~full_mask  # [B, 1+N]

        # Step 5: Transformer self-attention
        x = self.dropout(x)
        x = self.transformer_encoder(
            x, src_key_padding_mask=src_key_padding_mask
        )  # [B, 1+N, H]

        # Step 6: Pooling
        if self.use_attention_pooling:
            attn_logits = self.pooling_attention(x)  # [B, 1+N, 1]

            # Mask padding positions before softmax
            if src_key_padding_mask is not None:
                attn_logits = attn_logits.masked_fill(
                    src_key_padding_mask.unsqueeze(-1), float("-inf")
                )

            attn_weights = F.softmax(attn_logits, dim=1)  # [B, 1+N, 1]
            pooled = torch.sum(x * attn_weights, dim=1)  # [B, H]
        else:
            pooled = x[:, 0, :]  # CLS token [B, H]
            attn_weights = None

        pooled = self.pre_classifier_norm(pooled)

        # Step 7: Classifier
        logits = self.classifier(pooled)  # [B, num_classes]

        if return_attention and attn_weights is not None:
            return logits, attn_weights.squeeze(-1)  # [B, 1+N]
        return logits
