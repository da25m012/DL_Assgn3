import math
import copy
import os
import gdown
from collections import Counter
from typing import Optional, Tuple

import spacy
import torch
import torch.nn as nn
import torch.nn.functional as F


def scaled_dot_product_attention(
    Q: torch.Tensor,
    K: torch.Tensor,
    V: torch.Tensor,
    mask: Optional[torch.Tensor] = None,
) -> Tuple[torch.Tensor, torch.Tensor]:
    d_k = Q.size(-1)
    scores = torch.matmul(Q, K.transpose(-2, -1)) / math.sqrt(d_k)
    if mask is not None:
        scores = scores.masked_fill(mask, float("-inf"))
    attn_w = torch.softmax(scores, dim=-1)
    output = torch.matmul(attn_w, V)
    return output, attn_w


def make_src_mask(
    src: torch.Tensor,
    pad_idx: int = 1,
) -> torch.Tensor:
    # True where padding
    mask = (src == pad_idx).unsqueeze(1).unsqueeze(2)  # [batch, 1, 1, src_len]
    return mask


def make_tgt_mask(
    tgt: torch.Tensor,
    pad_idx: int = 1,
) -> torch.Tensor:
    tgt_len = tgt.size(1)
    pad_mask = (tgt == pad_idx).unsqueeze(1).unsqueeze(2)  # [batch, 1, 1, tgt_len]
    causal_mask = torch.triu(
        torch.ones(tgt_len, tgt_len, dtype=torch.bool, device=tgt.device), diagonal=1
    )  # [tgt_len, tgt_len]
    mask = pad_mask | causal_mask.unsqueeze(0).unsqueeze(0)  # [batch, 1, tgt_len, tgt_len]
    return mask


class MultiHeadAttention(nn.Module):
    def __init__(self, d_model: int, num_heads: int, dropout: float = 0.1) -> None:
        super().__init__()
        assert d_model % num_heads == 0, "d_model must be divisible by num_heads"

        self.d_model   = d_model
        self.num_heads = num_heads
        self.d_k       = d_model // num_heads

        self.W_q = nn.Linear(d_model, d_model)
        self.W_k = nn.Linear(d_model, d_model)
        self.W_v = nn.Linear(d_model, d_model)
        self.W_o = nn.Linear(d_model, d_model)
        self.dropout = nn.Dropout(p=dropout)

    def forward(
        self,
        query: torch.Tensor,
        key:   torch.Tensor,
        value: torch.Tensor,
        mask:  Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        batch_size = query.size(0)

        Q = self.W_q(query).view(batch_size, -1, self.num_heads, self.d_k).transpose(1, 2)
        K = self.W_k(key).view(batch_size, -1, self.num_heads, self.d_k).transpose(1, 2)
        V = self.W_v(value).view(batch_size, -1, self.num_heads, self.d_k).transpose(1, 2)

        attn_out, _ = scaled_dot_product_attention(Q, K, V, mask)
        attn_out = self.dropout(attn_out)

        attn_out = attn_out.transpose(1, 2).contiguous().view(batch_size, -1, self.d_model)
        return self.W_o(attn_out)


class PositionalEncoding(nn.Module):
    def __init__(self, d_model: int, dropout: float = 0.1, max_len: int = 5000) -> None:
        super().__init__()
        self.dropout = nn.Dropout(p=dropout)

        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2, dtype=torch.float) * (-math.log(10000.0) / d_model))

        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0)  # [1, max_len, d_model]

        self.register_buffer("pe", pe)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = x + self.pe[:, :x.size(1), :]
        return self.dropout(x)


class PositionwiseFeedForward(nn.Module):
    def __init__(self, d_model: int, d_ff: int, dropout: float = 0.1) -> None:
        super().__init__()
        self.linear1 = nn.Linear(d_model, d_ff)
        self.linear2 = nn.Linear(d_ff, d_model)
        self.dropout = nn.Dropout(p=dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.linear2(self.dropout(F.relu(self.linear1(x))))


class EncoderLayer(nn.Module):
    def __init__(self, d_model: int, num_heads: int, d_ff: int, dropout: float = 0.1) -> None:
        super().__init__()
        self.self_attn = MultiHeadAttention(d_model, num_heads, dropout)
        self.ffn       = PositionwiseFeedForward(d_model, d_ff, dropout)
        self.norm1     = nn.LayerNorm(d_model)
        self.norm2     = nn.LayerNorm(d_model)
        self.dropout   = nn.Dropout(p=dropout)

    def forward(self, x: torch.Tensor, src_mask: torch.Tensor) -> torch.Tensor:
        # pre-layernorm variant (stable training)
        x = x + self.dropout(self.self_attn(self.norm1(x), self.norm1(x), self.norm1(x), src_mask))
        x = x + self.dropout(self.ffn(self.norm2(x)))
        return x


class DecoderLayer(nn.Module):
    def __init__(self, d_model: int, num_heads: int, d_ff: int, dropout: float = 0.1) -> None:
        super().__init__()
        self.self_attn  = MultiHeadAttention(d_model, num_heads, dropout)
        self.cross_attn = MultiHeadAttention(d_model, num_heads, dropout)
        self.ffn        = PositionwiseFeedForward(d_model, d_ff, dropout)
        self.norm1      = nn.LayerNorm(d_model)
        self.norm2      = nn.LayerNorm(d_model)
        self.norm3      = nn.LayerNorm(d_model)
        self.dropout    = nn.Dropout(p=dropout)

    def forward(
        self,
        x:        torch.Tensor,
        memory:   torch.Tensor,
        src_mask: torch.Tensor,
        tgt_mask: torch.Tensor,
    ) -> torch.Tensor:
        x = x + self.dropout(self.self_attn(self.norm1(x), self.norm1(x), self.norm1(x), tgt_mask))
        x = x + self.dropout(self.cross_attn(self.norm2(x), memory, memory, src_mask))
        x = x + self.dropout(self.ffn(self.norm3(x)))
        return x


class Encoder(nn.Module):
    def __init__(self, layer: EncoderLayer, N: int) -> None:
        super().__init__()
        self.layers = nn.ModuleList([copy.deepcopy(layer) for _ in range(N)])
        self.norm   = nn.LayerNorm(layer.self_attn.d_model)

    def forward(self, x: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
        for layer in self.layers:
            x = layer(x, mask)
        return self.norm(x)


class Decoder(nn.Module):
    def __init__(self, layer: DecoderLayer, N: int) -> None:
        super().__init__()
        self.layers = nn.ModuleList([copy.deepcopy(layer) for _ in range(N)])
        self.norm   = nn.LayerNorm(layer.self_attn.d_model)

    def forward(
        self,
        x:        torch.Tensor,
        memory:   torch.Tensor,
        src_mask: torch.Tensor,
        tgt_mask: torch.Tensor,
    ) -> torch.Tensor:
        for layer in self.layers:
            x = layer(x, memory, src_mask, tgt_mask)
        return self.norm(x)


class Transformer(nn.Module):
    """
    Full Encoder-Decoder Transformer for German to English translation.
    """

    def __init__(
        self,
        d_model:   int   = 512,
        N:         int   = 6,
        num_heads: int   = 8,
        d_ff:      int   = 2048,
        dropout:   float = 0.1,
        max_len:   int   = 256,
        gdrive_file_id: str = "1a5UPWOyneoleXRce694iGx86E6bJX54H",
        checkpoint_path: str = "transformer_best.pt",
    ) -> None:
        super().__init__()

        self.d_model   = d_model
        self.max_len   = max_len

        try:
            self.de_nlp = spacy.load("de_core_news_sm")
        except OSError:
            spacy.cli.download("de_core_news_sm")
            self.de_nlp = spacy.load("de_core_news_sm")

        try:
            self.en_nlp = spacy.load("en_core_web_sm")
        except OSError:
            spacy.cli.download("en_core_web_sm")
            self.en_nlp = spacy.load("en_core_web_sm")

        # build vocabularies from Multi30k training set
        self.src_vocab, self.tgt_vocab = self._build_vocabs()
        self.src_vocab_size = len(self.src_vocab)
        self.tgt_vocab_size = len(self.tgt_vocab)

        # reverse lookup for decoding
        self.tgt_itos = {v: k for k, v in self.tgt_vocab.items()}

        self.pad_idx = self.src_vocab["<pad>"]
        self.sos_idx = self.tgt_vocab["<sos>"]
        self.eos_idx = self.tgt_vocab["<eos>"]

        # build model components
        enc_layer = EncoderLayer(d_model, num_heads, d_ff, dropout)
        dec_layer = DecoderLayer(d_model, num_heads, d_ff, dropout)

        self.src_embed = nn.Embedding(self.src_vocab_size, d_model, padding_idx=self.pad_idx)
        self.tgt_embed = nn.Embedding(self.tgt_vocab_size, d_model, padding_idx=self.pad_idx)
        self.pos_enc   = PositionalEncoding(d_model, dropout, max_len=5000)
        self.encoder   = Encoder(enc_layer, N)
        self.decoder   = Decoder(dec_layer, N)
        self.proj      = nn.Linear(d_model, self.tgt_vocab_size)

        self._init_parameters()

        # download and load weights
        if gdrive_file_id != "YOUR_GDRIVE_FILE_ID" and not os.path.exists(checkpoint_path):
            gdown.download(id=gdrive_file_id, output=checkpoint_path, quiet=False)

        if os.path.exists(checkpoint_path):
            ckpt = torch.load(checkpoint_path, map_location="cpu")
            state = ckpt.get("model_state_dict", ckpt)
            self.load_state_dict(state, strict=False)

    def _build_vocabs(self):
        from datasets import load_dataset
        raw = load_dataset("bentrevett/multi30k")
        train_data = raw["train"]

        specials = ["<unk>", "<pad>", "<sos>", "<eos>"]

        de_counter = Counter()
        en_counter = Counter()
        for item in train_data:
            de_counter.update(t.text.lower() for t in self.de_nlp.tokenizer(item["de"]))
            en_counter.update(t.text.lower() for t in self.en_nlp.tokenizer(item["en"]))

        src_vocab = {tok: idx for idx, tok in enumerate(specials)}
        for tok, _ in de_counter.most_common():
            if tok not in src_vocab:
                src_vocab[tok] = len(src_vocab)

        tgt_vocab = {tok: idx for idx, tok in enumerate(specials)}
        for tok, _ in en_counter.most_common():
            if tok not in tgt_vocab:
                tgt_vocab[tok] = len(tgt_vocab)

        return src_vocab, tgt_vocab

    def _init_parameters(self):
        for p in self.parameters():
            if p.dim() > 1:
                nn.init.xavier_uniform_(p)

    def encode(self, src: torch.Tensor, src_mask: torch.Tensor) -> torch.Tensor:
        x = self.pos_enc(self.src_embed(src) * math.sqrt(self.d_model))
        return self.encoder(x, src_mask)

    def decode(
        self,
        memory:   torch.Tensor,
        src_mask: torch.Tensor,
        tgt:      torch.Tensor,
        tgt_mask: torch.Tensor,
    ) -> torch.Tensor:
        x = self.pos_enc(self.tgt_embed(tgt) * math.sqrt(self.d_model))
        x = self.decoder(x, memory, src_mask, tgt_mask)
        return self.proj(x)

    def forward(
        self,
        src:      torch.Tensor,
        tgt:      torch.Tensor,
        src_mask: torch.Tensor,
        tgt_mask: torch.Tensor,
    ) -> torch.Tensor:
        memory = self.encode(src, src_mask)
        return self.decode(memory, src_mask, tgt, tgt_mask)

    def infer(self, src_sentence: str, max_len: int = 100) -> str:
        """
        Translates a German sentence to English using greedy autoregressive decoding.
        """
        self.eval()
        device = next(self.parameters()).device

        # tokenize german
        tokens = [t.text.lower() for t in self.de_nlp.tokenizer(src_sentence)]
        unk_idx = self.src_vocab["<unk>"]
        src_ids = (
            [self.src_vocab["<sos>"]]
            + [self.src_vocab.get(t, unk_idx) for t in tokens]
            + [self.src_vocab["<eos>"]]
        )
        src = torch.tensor(src_ids, dtype=torch.long).unsqueeze(0).to(device)
        src_mask = make_src_mask(src, pad_idx=self.pad_idx)

        with torch.no_grad():
            memory = self.encode(src, src_mask)

            ys = torch.tensor([[self.sos_idx]], dtype=torch.long, device=device)

            for _ in range(max_len):
                tgt_mask = make_tgt_mask(ys, pad_idx=self.pad_idx)
                logits = self.decode(memory, src_mask, ys, tgt_mask)
                next_token = logits[:, -1, :].argmax(dim=-1, keepdim=True)
                ys = torch.cat([ys, next_token], dim=1)
                if next_token.item() == self.eos_idx:
                    break

        # detokenize (strip <sos> and <eos>)
        generated = ys[0, 1:].tolist()
        words = []
        for idx in generated:
            if idx == self.eos_idx:
                break
            token = self.tgt_itos.get(idx, "<unk>")
            words.append(token)

        return " ".join(words)
