"""Small, original column-to-row-to-dataset in-context classifier.

Inspired by TabICL (ICML 2025), not a reproduction of its architecture or weights.
Rows are unordered. Feature slots have shared-across-rows identity embeddings
so that the model can match the same column between different examples.
"""

from __future__ import annotations

import torch
from torch import nn
from torch.nn import functional as F


class MiniTabICL(nn.Module):
    def __init__(self, width: int = 32, heads: int = 4, dataset_layers: int = 2):
        super().__init__()
        self.width = width
        self.cell = nn.Sequential(nn.Linear(1, width), nn.GELU(), nn.Linear(width, width))
        self.feature_id = nn.Embedding(64, width)
        self.column = nn.TransformerEncoder(
            nn.TransformerEncoderLayer(width, heads, width * 2, dropout=0.0,
                                       batch_first=True, norm_first=True), 1,
            enable_nested_tensor=False)
        self.cls = nn.Parameter(torch.zeros(1, 1, width))
        self.row = nn.TransformerEncoder(
            nn.TransformerEncoderLayer(width, heads, width * 2, dropout=0.0,
                                       batch_first=True, norm_first=True), 1,
            enable_nested_tensor=False)
        # Three tokens: observed class 0, observed class 1, hidden query label.
        self.label = nn.Embedding(3, width)
        self.dataset = nn.TransformerEncoder(
            nn.TransformerEncoderLayer(width, heads, width * 4, dropout=0.0,
                                       batch_first=True, norm_first=True),
            dataset_layers, enable_nested_tensor=False)
        self.head = nn.Sequential(nn.LayerNorm(width), nn.Linear(width, 1))
        # A learnable local-attention path provides a useful inductive bias at
        # this tiny CPU budget; the row transformer learns a global correction.
        self.log_kernel_precision = nn.Parameter(torch.tensor(0.0))
        self.kernel_weight = nn.Parameter(torch.tensor(0.5))

    def forward(self, x: torch.Tensor, context_y: torch.Tensor,
                n_features: torch.Tensor) -> torch.Tensor:
        """Return logits for queries; context is the first C rows.

        x: [batch, C+Q, max_features]; context_y: [batch, C].
        n_features: number of valid columns for each dataset in the batch.
        Query labels are never passed in or encoded.
        """
        if x.ndim != 3 or context_y.ndim != 2:
            raise ValueError("Expected x=[B,N,F] and context_y=[B,C]")
        batch, n_rows, max_features = x.shape
        if max_features > self.feature_id.num_embeddings:
            raise ValueError("At most 64 columns are supported")
        if context_y.shape[0] != batch or not 0 < context_y.shape[1] < n_rows:
            raise ValueError("Context labels must match x and leave query rows")
        if n_features.shape != (batch,) or torch.any(n_features < 1) or torch.any(n_features > max_features):
            raise ValueError("n_features must be in [1,max_features] for each task")
        n_context = context_y.shape[1]
        if torch.any((context_y != 0) & (context_y != 1)):
            raise ValueError("This compact implementation supports binary labels only")

        # The first attention stage compares values within each column.
        cells = self.cell(x.unsqueeze(-1))
        cells = cells + self.feature_id(torch.arange(max_features, device=x.device))[None, None]
        columns = cells.permute(0, 2, 1, 3).reshape(batch * max_features, n_rows, self.width)
        columns = self.column(columns)
        cells = columns.reshape(batch, max_features, n_rows, self.width).permute(0, 2, 1, 3)

        # A shared CLS token pools the unordered columns within each row.
        row_tokens = torch.cat((self.cls.expand(batch, n_rows, 1, self.width), cells), dim=2)
        row_tokens = row_tokens.reshape(batch * n_rows, max_features + 1, self.width)
        feature_mask = torch.arange(max_features, device=x.device)[None, :] >= n_features[:, None]
        row_mask = torch.cat((torch.zeros(batch, 1, device=x.device, dtype=torch.bool),
                              feature_mask), dim=1)
        row_mask = row_mask[:, None, :].expand(batch, n_rows, -1).reshape(
            batch * n_rows, max_features + 1)
        rows = self.row(row_tokens, src_key_padding_mask=row_mask)[:, 0]
        rows = rows.reshape(batch, n_rows, self.width)

        # Context rows carry observed labels. Query rows carry only the MASK token.
        label_ids = torch.cat((context_y.long(),
                               torch.full((batch, n_rows - n_context), 2,
                                          device=x.device, dtype=torch.long)), dim=1)
        rows = rows + self.label(label_ids)
        # Context cannot attend to queries. Each query can see context and itself,
        # but not other queries. No test-label information is present.
        forbidden = torch.zeros(n_rows, n_rows, device=x.device, dtype=torch.bool)
        forbidden[:n_context, n_context:] = True
        forbidden[n_context:, n_context:] = True
        query_idx = torch.arange(n_context, n_rows, device=x.device)
        forbidden[query_idx, query_idx] = False
        encoded = self.dataset(rows, mask=forbidden)
        transformer_logit = self.head(encoded[:, n_context:]).squeeze(-1)
        valid = (~feature_mask).float()
        distance = ((x[:, n_context:, None, :] - x[:, None, :n_context, :]) ** 2
                    * valid[:, None, None, :]).sum(-1) / n_features[:, None, None]
        weights = torch.softmax(-distance * self.log_kernel_precision.exp(), dim=-1)
        local_probability = (weights * context_y[:, None, :].float()).sum(-1)
        local_probability = (local_probability * 0.98 + 0.01).clamp(1e-4, 1 - 1e-4)
        local_logit = torch.logit(local_probability)
        return transformer_logit + F.softplus(self.kernel_weight) * local_logit
