import math
import collections
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from functools import partial
from typing import Optional

import wandb


def corpus_bleu(list_of_references, hypotheses, max_n=4):
    clipped_counts = collections.Counter()
    total_counts   = collections.Counter()
    hyp_len = 0
    ref_len = 0

    for refs, hyp in zip(list_of_references, hypotheses):
        hyp_len += len(hyp)
        ref_len += min((abs(len(r) - len(hyp)), len(r)) for r in refs)[1]

        for n in range(1, max_n + 1):
            hyp_ngrams = collections.Counter(
                tuple(hyp[i:i + n]) for i in range(len(hyp) - n + 1)
            )
            max_ref_counts = collections.Counter()
            for ref in refs:
                ref_ngrams = collections.Counter(
                    tuple(ref[i:i + n]) for i in range(len(ref) - n + 1)
                )
                for ng, cnt in ref_ngrams.items():
                    max_ref_counts[ng] = max(max_ref_counts[ng], cnt)

            for ng, cnt in hyp_ngrams.items():
                clipped_counts[n] += min(cnt, max_ref_counts.get(ng, 0))
            total_counts[n] += max(len(hyp) - n + 1, 0)

    precisions = []
    for n in range(1, max_n + 1):
        if total_counts[n] == 0:
            precisions.append(0.0)
        else:
            precisions.append(clipped_counts[n] / total_counts[n])

    if min(precisions) == 0.0:
        return 0.0

    log_avg = sum(math.log(p) for p in precisions) / max_n
    bp = 1.0 if hyp_len >= ref_len else math.exp(1 - ref_len / hyp_len)
    return bp * math.exp(log_avg)

from model import Transformer, make_src_mask, make_tgt_mask
from dataset import Multi30kDataset, collate_fn
from lr_scheduler import NoamScheduler


class LabelSmoothingLoss(nn.Module):
    def __init__(self, vocab_size: int, pad_idx: int, smoothing: float = 0.1) -> None:
        super().__init__()
        self.vocab_size = vocab_size
        self.pad_idx    = pad_idx
        self.smoothing  = smoothing
        self.confidence = 1.0 - smoothing

    def forward(self, logits: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        log_probs = torch.log_softmax(logits, dim=-1)

        with torch.no_grad():
            smooth_dist = torch.full_like(log_probs, self.smoothing / (self.vocab_size - 2))
            smooth_dist.scatter_(1, target.unsqueeze(1), self.confidence)
            smooth_dist[:, self.pad_idx] = 0.0
            mask = (target == self.pad_idx)
            smooth_dist[mask] = 0.0

        loss = -(smooth_dist * log_probs).sum(dim=-1)
        non_pad = (~mask).sum()
        return loss.sum() / non_pad.clamp(min=1)


def run_epoch(
    data_iter,
    model: Transformer,
    loss_fn: nn.Module,
    optimizer: Optional[torch.optim.Optimizer],
    scheduler=None,
    epoch_num: int = 0,
    is_train: bool = True,
    device: str = "cpu",
) -> float:
    model.train() if is_train else model.eval()

    total_loss = 0.0
    total_tokens = 0

    ctx = torch.enable_grad() if is_train else torch.no_grad()
    with ctx:
        for src, tgt in data_iter:
            src = src.to(device)
            tgt = tgt.to(device)

            tgt_in  = tgt[:, :-1]
            tgt_out = tgt[:, 1:]

            src_mask = make_src_mask(src, pad_idx=1)
            tgt_mask = make_tgt_mask(tgt_in, pad_idx=1)

            logits = model(src, tgt_in, src_mask, tgt_mask)

            batch, seq_len, vocab = logits.shape
            loss = loss_fn(logits.reshape(-1, vocab), tgt_out.reshape(-1))

            if is_train:
                optimizer.zero_grad()
                loss.backward()
                nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
                optimizer.step()
                if scheduler is not None:
                    scheduler.step()

            non_pad = (tgt_out != 1).sum().item()
            total_loss   += loss.item() * non_pad
            total_tokens += non_pad

    return total_loss / max(total_tokens, 1)


def greedy_decode(
    model: Transformer,
    src: torch.Tensor,
    src_mask: torch.Tensor,
    max_len: int,
    start_symbol: int,
    end_symbol: int,
    device: str = "cpu",
) -> torch.Tensor:
    model.eval()
    with torch.no_grad():
        memory = model.encode(src, src_mask)
        ys = torch.tensor([[start_symbol]], dtype=torch.long, device=device)

        for _ in range(max_len - 1):
            tgt_mask = make_tgt_mask(ys, pad_idx=1)
            logits = model.decode(memory, src_mask, ys, tgt_mask)
            next_token = logits[:, -1, :].argmax(dim=-1, keepdim=True)
            ys = torch.cat([ys, next_token], dim=1)
            if next_token.item() == end_symbol:
                break

    return ys


def evaluate_bleu(
    model: Transformer,
    test_dataloader: DataLoader,
    tgt_vocab,
    device: str = "cpu",
    max_len: int = 100,
) -> float:
    model.eval()

    # build reverse vocab
    if isinstance(tgt_vocab, dict):
        itos = {v: k for k, v in tgt_vocab.items()}
        sos_idx = tgt_vocab.get("<sos>", 2)
        eos_idx = tgt_vocab.get("<eos>", 3)
        pad_idx = tgt_vocab.get("<pad>", 1)
    else:
        itos = {i: tgt_vocab.lookup_token(i) for i in range(len(tgt_vocab))}
        sos_idx = tgt_vocab.lookup_indices(["<sos>"])[0]
        eos_idx = tgt_vocab.lookup_indices(["<eos>"])[0]
        pad_idx = tgt_vocab.lookup_indices(["<pad>"])[0]

    hypotheses = []
    references = []

    with torch.no_grad():
        for src, tgt in test_dataloader:
            src = src.to(device)
            tgt = tgt.to(device)

            for i in range(src.size(0)):
                src_i    = src[i].unsqueeze(0)
                src_mask = make_src_mask(src_i, pad_idx=1)
                ys = greedy_decode(model, src_i, src_mask, max_len, sos_idx, eos_idx, device)

                pred_tokens = []
                for idx in ys[0, 1:].tolist():
                    if idx == eos_idx:
                        break
                    tok = itos.get(idx, "<unk>")
                    if tok not in ("<sos>", "<eos>", "<pad>"):
                        pred_tokens.append(tok)

                ref_tokens = []
                for idx in tgt[i, 1:].tolist():
                    if idx == eos_idx:
                        break
                    tok = itos.get(idx, "<unk>")
                    if tok not in ("<sos>", "<eos>", "<pad>"):
                        ref_tokens.append(tok)

                hypotheses.append(pred_tokens)
                references.append([ref_tokens])

    score = corpus_bleu(references, hypotheses)
    return score * 100.0


def save_checkpoint(
    model: Transformer,
    optimizer: torch.optim.Optimizer,
    scheduler,
    epoch: int,
    path: str = "checkpoint.pt",
) -> None:
    torch.save({
        "epoch": epoch,
        "model_state_dict": model.state_dict(),
        "optimizer_state_dict": optimizer.state_dict(),
        "scheduler_state_dict": scheduler.state_dict(),
        "model_config": {
            "d_model":        model.d_model,
            "src_vocab_size": model.src_vocab_size,
            "tgt_vocab_size": model.tgt_vocab_size,
        },
    }, path)


def load_checkpoint(
    path: str,
    model: Transformer,
    optimizer: Optional[torch.optim.Optimizer] = None,
    scheduler=None,
) -> int:
    ckpt = torch.load(path, map_location="cpu")
    model.load_state_dict(ckpt["model_state_dict"])
    if optimizer is not None and "optimizer_state_dict" in ckpt:
        optimizer.load_state_dict(ckpt["optimizer_state_dict"])
    if scheduler is not None and "scheduler_state_dict" in ckpt:
        scheduler.load_state_dict(ckpt["scheduler_state_dict"])
    return ckpt.get("epoch", 0)


def run_training_experiment() -> None:
    config = {
        "d_model":       512,
        "N":             6,
        "num_heads":     8,
        "d_ff":          2048,
        "dropout":       0.1,
        "batch_size":    128,
        "num_epochs":    20,
        "warmup_steps":  4000,
        "smoothing":     0.1,
        "max_len":       256,
    }

    wandb.init(project="da6401-a3", config=config)
    cfg = wandb.config

    device = "cuda" if torch.cuda.is_available() else "cpu"

    train_ds = Multi30kDataset(split="train", max_len=cfg.max_len)
    val_ds   = Multi30kDataset(split="validation", src_vocab=train_ds.src_vocab,
                               tgt_vocab=train_ds.tgt_vocab, max_len=cfg.max_len)
    test_ds  = Multi30kDataset(split="test", src_vocab=train_ds.src_vocab,
                               tgt_vocab=train_ds.tgt_vocab, max_len=cfg.max_len)

    pad_fn = partial(collate_fn, src_pad_idx=train_ds.src_vocab["<pad>"],
                     tgt_pad_idx=train_ds.tgt_vocab["<pad>"])

    train_loader = DataLoader(train_ds, batch_size=cfg.batch_size, shuffle=True,
                              collate_fn=pad_fn, num_workers=2)
    val_loader   = DataLoader(val_ds, batch_size=cfg.batch_size, shuffle=False,
                              collate_fn=pad_fn, num_workers=2)
    test_loader  = DataLoader(test_ds, batch_size=cfg.batch_size, shuffle=False,
                              collate_fn=pad_fn, num_workers=2)

    model = Transformer(
        d_model=cfg.d_model,
        N=cfg.N,
        num_heads=cfg.num_heads,
        d_ff=cfg.d_ff,
        dropout=cfg.dropout,
        max_len=cfg.max_len,
    ).to(device)

    optimizer = torch.optim.Adam(model.parameters(), lr=1.0, betas=(0.9, 0.98), eps=1e-9)
    scheduler = NoamScheduler(optimizer, d_model=cfg.d_model, warmup_steps=cfg.warmup_steps)

    pad_idx = train_ds.src_vocab["<pad>"]
    loss_fn = LabelSmoothingLoss(
        vocab_size=model.tgt_vocab_size,
        pad_idx=pad_idx,
        smoothing=cfg.smoothing,
    )

    best_val_loss = float("inf")
    for epoch in range(cfg.num_epochs):
        train_loss = run_epoch(train_loader, model, loss_fn, optimizer, scheduler,
                               epoch_num=epoch, is_train=True, device=device)
        val_loss   = run_epoch(val_loader, model, loss_fn, None, None,
                               epoch_num=epoch, is_train=False, device=device)

        wandb.log({"train_loss": train_loss, "val_loss": val_loss,
                   "lr": optimizer.param_groups[0]["lr"], "epoch": epoch})

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            save_checkpoint(model, optimizer, scheduler, epoch, "best_checkpoint.pt")

    bleu = evaluate_bleu(model, test_loader, train_ds.tgt_vocab, device=device)
    wandb.log({"test_bleu": bleu})
    print(f"Test BLEU: {bleu:.2f}")
    wandb.finish()


if __name__ == "__main__":
    run_training_experiment()
