from __future__ import annotations

import argparse
import json
import math
import random
import sys
from pathlib import Path
from typing import Any

import torch
from torch import nn
from torch.utils.data import DataLoader, Dataset


SPECIALS = ["<pad>", "<bos>", "<eos>", "<unk>"]
PAD, BOS, EOS, UNK = range(4)


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")


def jsonable_args(args: argparse.Namespace) -> dict[str, Any]:
    result: dict[str, Any] = {}
    for key, value in vars(args).items():
        if callable(value):
            continue
        result[key] = str(value) if isinstance(value, Path) else value
    return result


def make_pairs(args: argparse.Namespace) -> None:
    rows = []
    for pred_path in sorted(args.pred_lmx_dir.glob("*.lmx")):
        sample = pred_path.stem
        gt_path = args.gt_root / f"{sample}.lmx.txt"
        if not gt_path.exists():
            continue
        src = pred_path.read_text(encoding="utf-8").strip()
        tgt = gt_path.read_text(encoding="utf-8").strip()
        rows.append(
            {
                "sample": sample,
                "src": src,
                "tgt": tgt,
                "src_tokens": len(src.split()),
                "tgt_tokens": len(tgt.split()),
                "pred_lmx": str(pred_path),
                "gt_lmx": str(gt_path),
            }
        )
    args.out_jsonl.parent.mkdir(parents=True, exist_ok=True)
    args.out_jsonl.write_text("\n".join(json.dumps(row, ensure_ascii=False) for row in rows) + "\n", encoding="utf-8")
    print(json.dumps({"pairs": len(rows), "out_jsonl": str(args.out_jsonl)}, indent=2))


def make_inputs(args: argparse.Namespace) -> None:
    """Build inference rows without reading target-domain ground truth."""
    rows = []
    for pred_path in sorted(args.pred_lmx_dir.glob("*.lmx")):
        sample = pred_path.stem
        src = pred_path.read_text(encoding="utf-8").strip()
        rows.append(
            {
                "sample": sample,
                "src": src,
                # PairDataset pads a target tensor, but predict() never consumes it.
                # Reusing src keeps the inference artifact strictly label-free.
                "tgt": src,
                "src_tokens": len(src.split()),
                "pred_lmx": str(pred_path),
            }
        )
    args.out_jsonl.parent.mkdir(parents=True, exist_ok=True)
    args.out_jsonl.write_text("\n".join(json.dumps(row, ensure_ascii=False) for row in rows) + "\n", encoding="utf-8")
    print(json.dumps({"inputs": len(rows), "out_jsonl": str(args.out_jsonl), "ground_truth_read": False}, indent=2))


def build_vocab(rows: list[dict[str, Any]], min_freq: int = 1) -> dict[str, int]:
    counts: dict[str, int] = {}
    for row in rows:
        for field in ("src", "tgt"):
            for token in row[field].split():
                counts[token] = counts.get(token, 0) + 1
    vocab = {token: idx for idx, token in enumerate(SPECIALS)}
    for token, count in sorted(counts.items(), key=lambda item: (-item[1], item[0])):
        if count >= min_freq and token not in vocab:
            vocab[token] = len(vocab)
    return vocab


def encode(text: str, vocab: dict[str, int], max_len: int) -> list[int]:
    ids = [vocab.get(token, UNK) for token in text.split()]
    return ids[:max_len]


class PairDataset(Dataset):
    def __init__(self, rows: list[dict[str, Any]], vocab: dict[str, int], max_src_len: int, max_tgt_len: int):
        self.rows = rows
        self.vocab = vocab
        self.max_src_len = max_src_len
        self.max_tgt_len = max_tgt_len

    def __len__(self) -> int:
        return len(self.rows)

    def __getitem__(self, index: int) -> dict[str, Any]:
        row = self.rows[index]
        src = [BOS, *encode(row["src"], self.vocab, self.max_src_len - 2), EOS]
        tgt = [BOS, *encode(row["tgt"], self.vocab, self.max_tgt_len - 2), EOS]
        return {"sample": row["sample"], "src": torch.tensor(src), "tgt": torch.tensor(tgt)}


def pad_batch(items: list[dict[str, Any]]) -> dict[str, Any]:
    src_len = max(item["src"].numel() for item in items)
    tgt_len = max(item["tgt"].numel() for item in items)
    src = torch.full((len(items), src_len), PAD, dtype=torch.long)
    tgt = torch.full((len(items), tgt_len), PAD, dtype=torch.long)
    for idx, item in enumerate(items):
        src[idx, : item["src"].numel()] = item["src"]
        tgt[idx, : item["tgt"].numel()] = item["tgt"]
    return {"sample": [item["sample"] for item in items], "src": src, "tgt": tgt}


class PositionalEncoding(nn.Module):
    def __init__(self, dim: int, max_len: int = 2048):
        super().__init__()
        positions = torch.arange(max_len).unsqueeze(1)
        div_terms = torch.exp(torch.arange(0, dim, 2) * (-math.log(10000.0) / dim))
        pe = torch.zeros(max_len, dim)
        pe[:, 0::2] = torch.sin(positions * div_terms)
        pe[:, 1::2] = torch.cos(positions * div_terms)
        self.register_buffer("pe", pe.unsqueeze(0))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x + self.pe[:, : x.size(1)]


class TokenCorrector(nn.Module):
    def __init__(self, vocab_size: int, dim: int, layers: int, heads: int, dropout: float):
        super().__init__()
        self.embed = nn.Embedding(vocab_size, dim, padding_idx=PAD)
        self.pos = PositionalEncoding(dim)
        self.transformer = nn.Transformer(
            d_model=dim,
            nhead=heads,
            num_encoder_layers=layers,
            num_decoder_layers=layers,
            dim_feedforward=dim * 4,
            dropout=dropout,
            batch_first=True,
        )
        self.out = nn.Linear(dim, vocab_size)
        self.scale = math.sqrt(dim)

    def forward(self, src: torch.Tensor, tgt_in: torch.Tensor) -> torch.Tensor:
        src_pad = src.eq(PAD)
        tgt_pad = tgt_in.eq(PAD)
        tgt_mask = nn.Transformer.generate_square_subsequent_mask(tgt_in.size(1), device=tgt_in.device)
        hidden = self.transformer(
            self.pos(self.embed(src) * self.scale),
            self.pos(self.embed(tgt_in) * self.scale),
            tgt_mask=tgt_mask,
            src_key_padding_mask=src_pad,
            tgt_key_padding_mask=tgt_pad,
            memory_key_padding_mask=src_pad,
        )
        return self.out(hidden)


@torch.no_grad()
def greedy_decode(model: TokenCorrector, src: torch.Tensor, max_len: int, repeat_patience: int = 32) -> torch.Tensor:
    model.eval()
    generated = torch.full((src.size(0), 1), BOS, dtype=torch.long, device=src.device)
    finished = torch.zeros(src.size(0), dtype=torch.bool, device=src.device)
    repeat_counts = torch.zeros(src.size(0), dtype=torch.long, device=src.device)
    previous = torch.full((src.size(0),), -1, dtype=torch.long, device=src.device)
    for _ in range(max_len - 1):
        logits = model(src, generated)
        nxt = logits[:, -1].argmax(dim=-1)
        nxt = torch.where(finished, torch.full_like(nxt, EOS), nxt)
        repeat_counts = torch.where(nxt.eq(previous), repeat_counts + 1, torch.ones_like(repeat_counts))
        repeated_out = repeat_counts.ge(repeat_patience)
        nxt = torch.where(repeated_out, torch.full_like(nxt, EOS), nxt)
        generated = torch.cat([generated, nxt[:, None]], dim=1)
        finished |= nxt.eq(EOS)
        previous = nxt
        if bool(finished.all()):
            break
    return generated


def load_ser_metric(olimpic_root: Path):
    sys.path.insert(0, str((olimpic_root / "zeus").resolve()))
    from ser_metric import ser_metric

    return ser_metric


def ids_to_text(ids: list[int], inv_vocab: list[str]) -> str:
    tokens = []
    for idx in ids:
        if idx == EOS:
            break
        if idx in (PAD, BOS):
            continue
        tokens.append(inv_vocab[idx] if 0 <= idx < len(inv_vocab) else "<unk>")
    return " ".join(tokens)


@torch.no_grad()
def evaluate(
    model: TokenCorrector,
    loader: DataLoader,
    inv_vocab: list[str],
    device: torch.device,
    max_len: int,
    ser_metric,
    max_length_ratio: float = 1.5,
    max_extra_tokens: int = 32,
    repeat_patience: int = 32,
) -> dict[str, float]:
    gold: list[str] = []
    pred: list[str] = []
    for batch in loader:
        src = batch["src"].to(device)
        src_max = int(src.ne(PAD).sum(dim=1).max().item())
        decode_limit = min(max_len, max(4, int(math.ceil(src_max * max_length_ratio)) + max_extra_tokens))
        out = greedy_decode(model, src, max_len=decode_limit, repeat_patience=repeat_patience).cpu().tolist()
        pred.extend(ids_to_text(ids, inv_vocab) for ids in out)
        gold.extend(ids_to_text(ids.tolist(), inv_vocab) for ids in batch["tgt"])
    return ser_metric(gold, pred)


def split_rows(rows: list[dict[str, Any]], dev_ratio: float, seed: int) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    rows = list(rows)
    random.Random(seed).shuffle(rows)
    dev_n = int(round(len(rows) * dev_ratio))
    return rows[dev_n:], rows[:dev_n]


def train(args: argparse.Namespace) -> None:
    random.seed(args.seed)
    torch.manual_seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(args.seed)
    rows = read_jsonl(args.train_jsonl)
    train_rows, dev_rows = split_rows(rows, args.dev_ratio, args.seed) if args.dev_jsonl is None else (rows, read_jsonl(args.dev_jsonl))
    vocab = build_vocab(train_rows, min_freq=args.min_freq)
    inv_vocab = [None] * len(vocab)
    for token, idx in vocab.items():
        inv_vocab[idx] = token

    device = torch.device(args.device if args.device else ("cuda" if torch.cuda.is_available() else "cpu"))
    model = TokenCorrector(len(vocab), args.dim, args.layers, args.heads, args.dropout).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    criterion = nn.CrossEntropyLoss(ignore_index=PAD)
    train_loader = DataLoader(
        PairDataset(train_rows, vocab, args.max_src_len, args.max_tgt_len),
        batch_size=args.batch_size,
        shuffle=True,
        collate_fn=pad_batch,
    )
    dev_loader = DataLoader(
        PairDataset(dev_rows, vocab, args.max_src_len, args.max_tgt_len),
        batch_size=args.batch_size,
        shuffle=False,
        collate_fn=pad_batch,
    )
    ser_metric = load_ser_metric(args.olimpic_root)

    args.out_dir.mkdir(parents=True, exist_ok=True)
    write_json(
        args.out_dir / "config.json",
        {
            **jsonable_args(args),
            "train_rows": len(train_rows),
            "dev_rows": len(dev_rows),
            "vocab_size": len(vocab),
            "device": str(device),
        },
    )
    write_json(args.out_dir / "vocab.json", vocab)
    train_samples = [str(row.get("sample")) for row in train_rows]
    dev_samples = [str(row.get("sample")) for row in dev_rows]
    overlap = sorted(set(train_samples) & set(dev_samples))
    if overlap:
        raise ValueError(f"Training/dev sample overlap detected: {overlap[:5]}")
    write_json(
        args.out_dir / "split_manifest.json",
        {
            "seed": args.seed,
            "dev_ratio": args.dev_ratio,
            "train_count": len(train_samples),
            "dev_count": len(dev_samples),
            "overlap": overlap,
            "train_samples": train_samples,
            "dev_samples": dev_samples,
        },
    )

    best_ser = float("inf")
    history = []
    for epoch in range(1, args.epochs + 1):
        model.train()
        total_loss = 0.0
        total_tokens = 0
        for batch in train_loader:
            src = batch["src"].to(device)
            tgt = batch["tgt"].to(device)
            logits = model(src, tgt[:, :-1])
            loss = criterion(logits.reshape(-1, logits.size(-1)), tgt[:, 1:].reshape(-1))
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), args.clip_grad)
            optimizer.step()
            tokens = int(tgt[:, 1:].ne(PAD).sum().item())
            total_loss += float(loss.item()) * tokens
            total_tokens += tokens
        row = {"epoch": epoch, "train_loss": total_loss / max(1, total_tokens)}
        if dev_rows and (epoch == 1 or epoch % args.eval_each == 0 or epoch == args.epochs):
            metrics = evaluate(model, dev_loader, inv_vocab, device, args.max_tgt_len, ser_metric)
            row.update({f"dev_{key}": value for key, value in metrics.items()})
            if metrics["SER"] < best_ser:
                best_ser = metrics["SER"]
                torch.save(model.state_dict(), args.out_dir / "best.pt")
                write_json(args.out_dir / "best.json", {"epoch": epoch, **metrics, "selection_split": "training-domain dev"})
        elif not dev_rows:
            torch.save(model.state_dict(), args.out_dir / "best.pt")
            write_json(args.out_dir / "best.json", {"epoch": epoch, "selection_split": "no dev; latest epoch"})
        history.append(row)
        write_json(args.out_dir / "history.json", history)
        print(json.dumps(row, ensure_ascii=False))
    if not (args.out_dir / "best.pt").exists():
        torch.save(model.state_dict(), args.out_dir / "best.pt")
        write_json(args.out_dir / "best.json", {"epoch": args.epochs, "selection_split": "fallback final epoch"})


def predict(args: argparse.Namespace) -> None:
    config = json.loads((args.model_dir / "config.json").read_text(encoding="utf-8"))
    vocab = json.loads((args.model_dir / "vocab.json").read_text(encoding="utf-8"))
    inv_vocab = [None] * len(vocab)
    for token, idx in vocab.items():
        inv_vocab[idx] = token
    rows = read_jsonl(args.input_jsonl)
    device = torch.device(args.device if args.device else ("cuda" if torch.cuda.is_available() else "cpu"))
    model = TokenCorrector(len(vocab), config["dim"], config["layers"], config["heads"], config["dropout"]).to(device)
    model.load_state_dict(torch.load(args.model_dir / "best.pt", map_location=device))
    loader = DataLoader(
        PairDataset(rows, vocab, config["max_src_len"], config["max_tgt_len"]),
        batch_size=args.batch_size,
        shuffle=False,
        collate_fn=pad_batch,
    )
    pred_rows = []
    predictions = []
    for batch in loader:
        src = batch["src"].to(device)
        src_max = int(src.ne(PAD).sum(dim=1).max().item())
        decode_limit = min(config["max_tgt_len"], max(4, int(math.ceil(src_max * args.max_length_ratio)) + args.max_extra_tokens))
        out = greedy_decode(model, src, max_len=decode_limit, repeat_patience=args.repeat_patience).cpu().tolist()
        for sample, ids in zip(batch["sample"], out):
            text = ids_to_text(ids, inv_vocab)
            predictions.append(text)
            pred_rows.append({"sample": sample, "prediction": text})
    args.out_lmx.parent.mkdir(parents=True, exist_ok=True)
    args.out_lmx.write_text("\n".join(predictions) + "\n", encoding="utf-8")
    if args.out_json:
        write_json(
            args.out_json,
            {
                "rows": pred_rows,
                "decode": {
                    "max_length_ratio": args.max_length_ratio,
                    "max_extra_tokens": args.max_extra_tokens,
                    "repeat_patience": args.repeat_patience,
                    "absolute_max_tgt_len": config["max_tgt_len"],
                },
            },
        )
    print(json.dumps({"predictions": len(predictions), "out_lmx": str(args.out_lmx)}, indent=2))


def main() -> None:
    parser = argparse.ArgumentParser(description="Train/apply a token-level LMX corrector for StaffOMR outputs.")
    sub = parser.add_subparsers(dest="cmd", required=True)

    p = sub.add_parser("make-pairs")
    p.add_argument("--pred-lmx-dir", type=Path, required=True)
    p.add_argument("--gt-root", type=Path, required=True)
    p.add_argument("--out-jsonl", type=Path, required=True)
    p.set_defaults(func=make_pairs)

    p = sub.add_parser("make-inputs")
    p.add_argument("--pred-lmx-dir", type=Path, required=True)
    p.add_argument("--out-jsonl", type=Path, required=True)
    p.set_defaults(func=make_inputs)

    p = sub.add_parser("train")
    p.add_argument("--train-jsonl", type=Path, required=True)
    p.add_argument("--dev-jsonl", type=Path)
    p.add_argument("--out-dir", type=Path, required=True)
    p.add_argument("--olimpic-root", type=Path, default=Path(".local-tools/olimpic-icdar24"))
    p.add_argument("--device", default=None)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--dev-ratio", type=float, default=0.2)
    p.add_argument("--min-freq", type=int, default=1)
    p.add_argument("--max-src-len", type=int, default=700)
    p.add_argument("--max-tgt-len", type=int, default=700)
    p.add_argument("--dim", type=int, default=256)
    p.add_argument("--layers", type=int, default=3)
    p.add_argument("--heads", type=int, default=4)
    p.add_argument("--dropout", type=float, default=0.1)
    p.add_argument("--batch-size", type=int, default=8)
    p.add_argument("--epochs", type=int, default=80)
    p.add_argument("--eval-each", type=int, default=5)
    p.add_argument("--lr", type=float, default=3e-4)
    p.add_argument("--weight-decay", type=float, default=1e-4)
    p.add_argument("--clip-grad", type=float, default=1.0)
    p.set_defaults(func=train)

    p = sub.add_parser("predict")
    p.add_argument("--model-dir", type=Path, required=True)
    p.add_argument("--input-jsonl", type=Path, required=True)
    p.add_argument("--out-lmx", type=Path, required=True)
    p.add_argument("--out-json", type=Path)
    p.add_argument("--device", default=None)
    p.add_argument("--batch-size", type=int, default=8)
    p.add_argument("--max-length-ratio", type=float, default=1.5)
    p.add_argument("--max-extra-tokens", type=int, default=32)
    p.add_argument("--repeat-patience", type=int, default=32)
    p.set_defaults(func=predict)

    args = parser.parse_args()
    args.func(args)


if __name__ == "__main__":
    main()
