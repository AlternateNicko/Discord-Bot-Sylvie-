#!/usr/bin/env python3
"""
ByteBPE-LLM: a small trainable GPT-style language model with a byte-level BPE
tokenizer, written in PyTorch. Single file, menu driven.

Train data: .txt, .csv, .jsonl files (or folders / glob patterns containing them).
Requires: pip install torch
Run:      python bytebpe_llm.py
"""
import os, sys, re, csv, json, glob, math, time, random
from collections import Counter, defaultdict

try:
    import torch
    import torch.nn as nn
    import torch.nn.functional as F
except ImportError:
    print("PyTorch is required:  pip install torch")
    sys.exit(1)

csv.field_size_limit(min(sys.maxsize, 2**31 - 1))
EOS_TEXT = "<|endoftext|>"

# ----------------------------------------------------------------------------
# Byte-level BPE tokenizer
# ----------------------------------------------------------------------------
PRETOKEN = re.compile(r"'s|'t|'re|'ve|'m|'ll|'d| ?[A-Za-z]+| ?[0-9]+| ?[^\sA-Za-z0-9]+|\s+(?!\S)|\s+")


class ByteBPE:
    """Ids 0-255 are raw bytes, then one id per learned merge, then <|endoftext|>."""

    def __init__(self, merges=None):
        self.merges = [tuple(m) for m in (merges or [])]
        self._build()

    def _build(self):
        self.ranks = {m: i for i, m in enumerate(self.merges)}
        self.vocab = {i: bytes([i]) for i in range(256)}
        for i, (a, b) in enumerate(self.merges):
            self.vocab[256 + i] = self.vocab[a] + self.vocab[b]
        self.eos_id = 256 + len(self.merges)
        self.vocab_size = self.eos_id + 1
        self._cache = {}

    # -- training -----------------------------------------------------------
    def train(self, text, vocab_size, verbose=True):
        n_merges = max(0, vocab_size - 257)
        counts = Counter(PRETOKEN.findall(text))
        words = [list(w.encode("utf-8")) for w in counts]
        freqs = [counts[w] for w in counts]
        pair_counts = defaultdict(int)
        where = defaultdict(set)
        for wi, w in enumerate(words):
            for p in zip(w, w[1:]):
                pair_counts[p] += freqs[wi]
                where[p].add(wi)
        merges = []
        t0 = time.time()
        for m in range(n_merges):
            if not pair_counts:
                break
            best = max(pair_counts, key=pair_counts.get)
            if pair_counts[best] < 2:
                break
            new_id = 256 + len(merges)
            merges.append(best)
            for wi in list(where[best]):
                w, f = words[wi], freqs[wi]
                for p in zip(w, w[1:]):  # remove old pairs
                    pair_counts[p] -= f
                    if pair_counts[p] <= 0:
                        del pair_counts[p]
                out, i = [], 0
                while i < len(w):
                    if i < len(w) - 1 and w[i] == best[0] and w[i + 1] == best[1]:
                        out.append(new_id); i += 2
                    else:
                        out.append(w[i]); i += 1
                words[wi] = out
                for p in zip(out, out[1:]):  # add new pairs
                    pair_counts[p] += f
                    where[p].add(wi)
            pair_counts.pop(best, None)
            where.pop(best, None)
            if verbose and (m + 1) % 250 == 0:
                print(f"  merges {m + 1}/{n_merges}  ({time.time() - t0:.1f}s)")
        self.merges = merges
        self._build()

    # -- encode / decode ----------------------------------------------------
    def _encode_chunk(self, chunk):
        hit = self._cache.get(chunk)
        if hit is not None:
            return hit
        ids = list(chunk.encode("utf-8"))
        while len(ids) >= 2:
            best, best_rank = None, None
            for p in zip(ids, ids[1:]):
                r = self.ranks.get(p)
                if r is not None and (best_rank is None or r < best_rank):
                    best, best_rank = p, r
            if best is None:
                break
            new_id, out, i = 256 + best_rank, [], 0
            while i < len(ids):
                if i < len(ids) - 1 and ids[i] == best[0] and ids[i + 1] == best[1]:
                    out.append(new_id); i += 2
                else:
                    out.append(ids[i]); i += 1
            ids = out
        if len(self._cache) < 200000:
            self._cache[chunk] = ids
        return ids

    def encode(self, text, allow_special=True):
        out = []
        parts = text.split(EOS_TEXT) if allow_special else [text]
        for k, part in enumerate(parts):
            for chunk in PRETOKEN.findall(part):
                out.extend(self._encode_chunk(chunk))
            if k < len(parts) - 1:
                out.append(self.eos_id)
        return out

    def decode(self, ids):
        b = b"".join(self.vocab[i] if i != self.eos_id else EOS_TEXT.encode() for i in ids)
        return b.decode("utf-8", errors="replace")


# ----------------------------------------------------------------------------
# Data loading (.txt / .csv / .jsonl)
# ----------------------------------------------------------------------------
SUPPORTED = (".txt", ".csv", ".jsonl")


def expand_paths(spec):
    found = []
    for item in spec.replace(",", " ").split() if '"' not in spec else re.findall(r'"([^"]+)"|(\S+)', spec):
        if isinstance(item, tuple):
            item = item[0] or item[1]
        item = os.path.expanduser(item)
        if os.path.isdir(item):
            for root, _, files in os.walk(item):
                for f in sorted(files):
                    if f.lower().endswith(SUPPORTED):
                        found.append(os.path.join(root, f))
        else:
            matches = glob.glob(item) or [item]
            found.extend(m for m in matches if os.path.isfile(m))
    return [f for f in found if f.lower().endswith(SUPPORTED)]


def fmt_pair(S, prompt, response):
    return f"{S['user_tag']} {str(prompt).strip()}\n{S['bot_tag']} {str(response).strip()}"


def _split_list(spec):
    return [s.strip() for s in str(spec or "").split(",") if s.strip()]


def _resolve_cols(spec, header):
    """Column names (case-insensitive) or 0-based indices -> list of indices."""
    out, low = [], [h.strip().lower() for h in header]
    for s in _split_list(spec):
        if s.lower() in low:
            out.append(low.index(s.lower()))
        elif s.isdigit() and int(s) < len(header):
            out.append(int(s))
        else:
            raise ValueError(f"column '{s}' not found (available: {', '.join(header)})")
    return out


def open_csv(path):
    """Return (file, reader) with delimiter auto-detected. Caller closes file."""
    f = open(path, "r", encoding="utf-8-sig", errors="replace", newline="")
    sample = f.read(65536)
    f.seek(0)
    try:
        dialect = csv.Sniffer().sniff(sample, delimiters=",;\t|")
    except csv.Error:
        dialect = csv.excel
    return f, csv.reader(f, dialect)


def csv_header(path):
    f, rd = open_csv(path)
    try:
        return next(rd, [])
    finally:
        f.close()


def _dig(obj, key):
    """Fetch key from dict; supports dotted paths like 'data.question'."""
    for part in key.split("."):
        if isinstance(obj, dict) and part in obj:
            obj = obj[part]
        else:
            return None
    return obj


def _join_keys(obj, spec):
    vals = [_dig(obj, k) for k in _split_list(spec)]
    vals = [str(v) for v in vals if v not in (None, "")]
    return "\n".join(vals)


def _format_json_record(obj, S):
    if isinstance(obj, str):
        return obj
    if not isinstance(obj, dict):
        return str(obj)
    if S["json_prompt_key"] and S["json_response_key"]:
        p, r = _join_keys(obj, S["json_prompt_key"]), _join_keys(obj, S["json_response_key"])
        return fmt_pair(S, p, r) if p and r else ""
    if S["json_key"]:
        return _join_keys(obj, S["json_key"])
    if "text" in obj:
        return str(obj["text"])
    for pk, rk in (("prompt", "response"), ("prompt", "completion"), ("instruction", "output"),
                   ("question", "answer"), ("input", "output"), ("user", "assistant")):
        if pk in obj and rk in obj:
            return fmt_pair(S, obj[pk], obj[rk])
    if isinstance(obj.get("messages"), list):
        lines = []
        for m in obj["messages"]:
            if isinstance(m, dict) and "content" in m:
                bot = m.get("role") in ("assistant", "model", "bot")
                lines.append(f"{S['bot_tag'] if bot else S['user_tag']} {m['content']}")
        if lines:
            return "\n".join(lines)
    return "\n".join(str(v) for v in obj.values() if isinstance(v, str))


def read_documents(path, S):
    """Yield documents (strings) from one file, using settings dict S."""
    ext = os.path.splitext(path)[1].lower()
    if ext == ".txt":
        with open(path, "r", encoding="utf-8", errors="replace") as f:
            yield f.read()
    elif ext == ".csv":
        f, rd = open_csv(path)
        try:
            header = next(rd, [])
            pc = _resolve_cols(S["csv_prompt_col"], header)
            rc = _resolve_cols(S["csv_response_col"], header)
            tc = _resolve_cols(S["csv_text_col"], header)
            if bool(pc) != bool(rc):
                raise ValueError("set BOTH csv_prompt_col and csv_response_col (or neither)")
            pick = lambda row, idx: "\n".join(row[i].strip() for i in idx if i < len(row) and row[i].strip())
            for row in rd:
                if pc and rc:
                    p, r = pick(row, pc), pick(row, rc)
                    if p and r:
                        yield fmt_pair(S, p, r)
                elif tc:
                    t = pick(row, tc)
                    if t:
                        yield t
                else:
                    yield " ".join(c for c in row if c)
        finally:
            f.close()
    elif ext == ".jsonl":
        with open(path, "r", encoding="utf-8", errors="replace") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                try:
                    yield _format_json_record(json.loads(line), S)
                except json.JSONDecodeError:
                    continue


def load_corpus(files, S):
    docs = []
    for p in files:
        n = 0
        try:
            for d in read_documents(p, S):
                if d and d.strip():
                    docs.append(d); n += 1
        except Exception as e:
            print(f"  ! could not read {p}: {e}")
            continue
        print(f"  {p}: {n} document(s)")
    return docs


# ----------------------------------------------------------------------------
# GPT model
# ----------------------------------------------------------------------------
class Attention(nn.Module):
    def __init__(self, c):
        super().__init__()
        self.nh = c["n_head"]
        self.qkv = nn.Linear(c["n_embd"], 3 * c["n_embd"], bias=False)
        self.proj = nn.Linear(c["n_embd"], c["n_embd"], bias=False)
        self.drop = c["dropout"]

    def forward(self, x):
        B, T, C = x.shape
        q, k, v = self.qkv(x).split(C, dim=2)
        q, k, v = (t.view(B, T, self.nh, C // self.nh).transpose(1, 2) for t in (q, k, v))
        y = F.scaled_dot_product_attention(q, k, v, is_causal=True,
                                           dropout_p=self.drop if self.training else 0.0)
        return self.proj(y.transpose(1, 2).contiguous().view(B, T, C))


class Block(nn.Module):
    def __init__(self, c):
        super().__init__()
        self.ln1, self.ln2 = nn.LayerNorm(c["n_embd"]), nn.LayerNorm(c["n_embd"])
        self.attn = Attention(c)
        self.mlp = nn.Sequential(nn.Linear(c["n_embd"], 4 * c["n_embd"], bias=False), nn.GELU(),
                                 nn.Linear(4 * c["n_embd"], c["n_embd"], bias=False))
        self.drop = nn.Dropout(c["dropout"])

    def forward(self, x):
        x = x + self.drop(self.attn(self.ln1(x)))
        return x + self.drop(self.mlp(self.ln2(x)))


class GPT(nn.Module):
    def __init__(self, c):
        super().__init__()
        assert c["n_embd"] % c["n_head"] == 0, "n_embd must be divisible by n_head"
        self.c = c
        self.tok = nn.Embedding(c["vocab_size"], c["n_embd"])
        self.pos = nn.Embedding(c["block_size"], c["n_embd"])
        self.drop = nn.Dropout(c["dropout"])
        self.blocks = nn.ModuleList(Block(c) for _ in range(c["n_layer"]))
        self.lnf = nn.LayerNorm(c["n_embd"])
        self.head = nn.Linear(c["n_embd"], c["vocab_size"], bias=False)
        self.head.weight = self.tok.weight  # weight tying
        self.apply(self._init)
        for n, p in self.named_parameters():
            if n.endswith("proj.weight") or n.endswith("mlp.2.weight"):
                nn.init.normal_(p, 0.0, 0.02 / math.sqrt(2 * c["n_layer"]))

    @staticmethod
    def _init(m):
        if isinstance(m, (nn.Linear, nn.Embedding)):
            nn.init.normal_(m.weight, 0.0, 0.02)

    def forward(self, idx, targets=None):
        B, T = idx.shape
        x = self.drop(self.tok(idx) + self.pos(torch.arange(T, device=idx.device)))
        for b in self.blocks:
            x = b(x)
        logits = self.head(self.lnf(x))
        loss = None if targets is None else F.cross_entropy(logits.view(-1, logits.size(-1)), targets.view(-1))
        return logits, loss

    def n_params(self):
        return sum(p.numel() for p in self.parameters())

    @torch.no_grad()
    def generate(self, ids, max_new, temperature=0.8, top_k=50, top_p=0.95, rep_penalty=1.1, eos_id=None):
        """Generator yielding one new token id at a time."""
        self.eval()
        dev = next(self.parameters()).device
        idx = torch.tensor([ids], dtype=torch.long, device=dev)
        for _ in range(max_new):
            logits = self(idx[:, -self.c["block_size"]:])[0][0, -1].float()
            if rep_penalty and rep_penalty != 1.0:
                seen = torch.unique(idx[0, -256:])
                l = logits[seen]
                logits[seen] = torch.where(l > 0, l / rep_penalty, l * rep_penalty)
            if temperature <= 0:
                nxt = torch.argmax(logits).view(1)
            else:
                logits = logits / temperature
                if top_k and top_k > 0:
                    kth = torch.topk(logits, min(top_k, logits.numel()))[0][-1]
                    logits[logits < kth] = -float("inf")
                if top_p and top_p < 1.0:
                    sl, si = torch.sort(logits, descending=True)
                    cp = torch.cumsum(F.softmax(sl, -1), -1)
                    drop = cp - F.softmax(sl, -1) > top_p
                    sl[drop] = -float("inf")
                    logits = torch.full_like(logits, -float("inf")).scatter(0, si, sl)
                nxt = torch.multinomial(F.softmax(logits, -1), 1)
            idx = torch.cat([idx, nxt.view(1, 1)], 1)
            t = int(nxt)
            yield t
            if eos_id is not None and t == eos_id:
                break


# ----------------------------------------------------------------------------
# Application state + defaults
# ----------------------------------------------------------------------------
DEFAULT_SETTINGS = {
    # model architecture (only used when creating a NEW model)
    "n_layer": 6, "n_head": 6, "n_embd": 384, "block_size": 256, "dropout": 0.1,
    # tokenizer
    "vocab_size": 4000, "tok_train_chars": 5_000_000,
    # training
    "steps": 2000, "batch_size": 16, "grad_accum": 1, "lr": 3e-4, "min_lr": 3e-5,
    "warmup": 100, "weight_decay": 0.1, "grad_clip": 1.0, "eval_every": 250,
    "eval_iters": 20, "val_split": 0.05, "mixed_precision": True,
    # data parsing
    "csv_prompt_col": "", "csv_response_col": "", "csv_text_col": "",
    "json_prompt_key": "", "json_response_key": "", "json_key": "",
    # generation / chat
    "temperature": 0.8, "top_k": 50, "top_p": 0.95, "rep_penalty": 1.1, "max_new_tokens": 200,
    "user_tag": "User:", "bot_tag": "Assistant:", "system_prompt": "",
}
ARCH_KEYS = ("n_layer", "n_head", "n_embd", "block_size", "dropout")


class App:
    def __init__(self):
        self.S = dict(DEFAULT_SETTINGS)
        self.files = []
        self.tok = None
        self.model = None
        self.total_steps = 0
        self.history = []  # (step, train_loss, val_loss)
        self.device = "cuda" if torch.cuda.is_available() else (
            "mps" if getattr(torch.backends, "mps", None) and torch.backends.mps.is_available() else "cpu")
        self.save_path = "model.pt"

    # ---- data ---------------------------------------------------------------
    def add_files(self):
        spec = input("File(s), folder(s) or glob (e.g. data/*.txt) separated by spaces: ").strip()
        if not spec:
            return
        new = [f for f in expand_paths(spec) if f not in self.files]
        self.files += new
        print(f"Added {len(new)} file(s). Total: {len(self.files)}")

    def list_files(self):
        if not self.files:
            print("No data files loaded.")
        for i, f in enumerate(self.files):
            print(f"  [{i}] {f}  ({os.path.getsize(f) / 1e6:.2f} MB)")

    def clear_files(self):
        self.files = []
        print("Cleared data file list.")

    # ---- column setup / preview ----------------------------------------------
    def column_setup(self):
        S = self.S
        csvs = [f for f in self.files if f.lower().endswith(".csv")]
        header = []
        if csvs:
            print("CSV columns found:")
            for f in csvs:
                try:
                    h = csv_header(f)
                    header = header or h
                    print(f"  {f}: " + ", ".join(f"[{i}] {c}" for i, c in enumerate(h)))
                except Exception as e:
                    print(f"  {f}: could not read ({e})")
        else:
            print("(No CSV files loaded yet - you can still type column names.)")
        print("\nEnter column NAME(s) or 0-based NUMBER(s), comma-separated to join several.")
        print("Leave blank to keep the current value, type '-' to clear it.")

        def ask(label, key):
            v = input(f"{label} [{S[key] or 'unset'}]: ").strip()
            if v == "-":
                S[key] = ""
            elif v:
                S[key] = v

        print("\n-- CSV: chat pairs (prompt column -> AI response column)")
        ask("Prompt column(s)", "csv_prompt_col")
        ask("Response column(s)", "csv_response_col")
        if not (S["csv_prompt_col"] and S["csv_response_col"]):
            print("-- CSV: plain text column(s) (used when no prompt/response pair is set)")
            ask("Text column(s)", "csv_text_col")
        print("\n-- JSONL: chat pairs (dotted keys allowed, e.g. data.question)")
        ask("Prompt key(s)", "json_prompt_key")
        ask("Response key(s)", "json_response_key")
        ask("Plain text key(s) (fallback)", "json_key")
        print(f"\nRows will be formatted as:  {S['user_tag']} <prompt>\\n{S['bot_tag']} <response>")
        if input("Preview formatted samples? (Y/n) ").lower() != "n":
            self.preview()

    def preview(self, n=3):
        if not self.files:
            print("No data files loaded."); return
        for p in self.files:
            print(f"--- {p}")
            try:
                for i, d in enumerate(read_documents(p, self.S)):
                    if i >= n:
                        break
                    print(d[:400] + ("..." if len(d) > 400 else ""))
                    print("  ~")
            except Exception as e:
                print(f"  ! {e}")

    # ---- tokenizer ----------------------------------------------------------
    def train_tokenizer(self):
        if not self.files:
            print("Load data files first."); return
        if self.model is not None:
            if input("A model exists; a new tokenizer makes it unusable. Continue? (y/N) ").lower() != "y":
                return
            self.model = None
        print("Reading data...")
        docs = load_corpus(self.files, self.S)
        if not docs:
            print("No text found."); return
        text = EOS_TEXT.join(docs)[: int(self.S["tok_train_chars"])]
        print(f"Training byte-level BPE on {len(text):,} characters, target vocab {self.S['vocab_size']}...")
        self.tok = ByteBPE()
        self.tok.train(text, int(self.S["vocab_size"]))
        sample = "Hello, world! héllo 你好 🙂"
        ok = self.tok.decode(self.tok.encode(sample)) == sample
        print(f"Done. Vocab size: {self.tok.vocab_size}. Round-trip test: {'OK' if ok else 'FAILED'}")

    # ---- model --------------------------------------------------------------
    def build_model(self):
        if self.tok is None:
            print("Train (or load) a tokenizer first."); return False
        cfg = {k: self.S[k] for k in ARCH_KEYS}
        cfg["vocab_size"] = self.tok.vocab_size
        for k in ("n_layer", "n_head", "n_embd", "block_size"):
            cfg[k] = int(cfg[k])
        try:
            self.model = GPT(cfg).to(self.device)
        except AssertionError as e:
            print("Error:", e); return False
        self.total_steps, self.history = 0, []
        print(f"Created new model: {self.model.n_params() / 1e6:.2f}M parameters on {self.device}")
        return True

    def reset_model(self):
        self.model = None
        print("Model discarded (tokenizer kept). Use Train to create a fresh one.")

    # ---- training -----------------------------------------------------------
    def _tokenize_data(self):
        print("Reading and tokenizing data...")
        docs = load_corpus(self.files, self.S)
        if not docs:
            return None
        ids = []
        for d in docs:
            ids.extend(self.tok.encode(d, allow_special=False))
            ids.append(self.tok.eos_id)
        data = torch.tensor(ids, dtype=torch.long)
        print(f"{len(data):,} tokens.")
        return data

    def train(self):
        if not self.files:
            print("Load data files first."); return
        if self.tok is None:
            print("No tokenizer yet - training one first.")
            self.train_tokenizer()
            if self.tok is None:
                return
        if self.model is None and not self.build_model():
            return
        S, m = self.S, self.model
        block = m.c["block_size"]
        data = self._tokenize_data()
        if data is None:
            print("No data."); return
        if len(data) < block * 4:
            print(f"Not enough data ({len(data)} tokens) for block size {block}. Add more text or shrink block_size.")
            return
        n_val = max(block + 1, int(len(data) * float(S["val_split"])))
        train_d, val_d = data[:-n_val], data[-n_val:]
        if len(train_d) <= block + 1:
            train_d = data; val_d = data

        def batch(split):
            d = train_d if split == "train" else val_d
            ix = torch.randint(len(d) - block - 1, (int(S["batch_size"]),))
            x = torch.stack([d[i:i + block] for i in ix])
            y = torch.stack([d[i + 1:i + block + 1] for i in ix])
            return x.to(self.device), y.to(self.device)

        use_amp = bool(S["mixed_precision"]) and self.device == "cuda"
        dtype = torch.bfloat16 if (use_amp and torch.cuda.is_bf16_supported()) else torch.float16
        scaler = torch.amp.GradScaler("cuda", enabled=use_amp and dtype == torch.float16)

        decay = [p for p in m.parameters() if p.dim() >= 2]
        nodecay = [p for p in m.parameters() if p.dim() < 2]
        opt = torch.optim.AdamW([{"params": decay, "weight_decay": float(S["weight_decay"])},
                                 {"params": nodecay, "weight_decay": 0.0}],
                                lr=float(S["lr"]), betas=(0.9, 0.95))
        steps, warm = int(S["steps"]), int(S["warmup"])
        lr, min_lr = float(S["lr"]), float(S["min_lr"])

        def lr_at(it):
            if it < warm:
                return lr * (it + 1) / max(1, warm)
            p = (it - warm) / max(1, steps - warm)
            return min_lr + 0.5 * (1 + math.cos(math.pi * min(1.0, p))) * (lr - min_lr)

        @torch.no_grad()
        def evaluate():
            m.eval()
            out = {}
            for sp in ("train", "val"):
                tot = 0.0
                for _ in range(int(S["eval_iters"])):
                    x, y = batch(sp)
                    with torch.autocast(self.device if self.device != "mps" else "cpu", dtype=dtype, enabled=use_amp):
                        tot += m(x, y)[1].item()
                out[sp] = tot / int(S["eval_iters"])
            m.train()
            return out

        print(f"Training {steps} steps on {self.device} (Ctrl+C to stop early)...")
        m.train()
        t0 = time.time()
        ga = int(S["grad_accum"])
        try:
            for it in range(steps):
                for g in opt.param_groups:
                    g["lr"] = lr_at(it)
                loss_acc = 0.0
                for _ in range(ga):
                    x, y = batch("train")
                    with torch.autocast("cuda", dtype=dtype, enabled=use_amp):
                        loss = m(x, y)[1] / ga
                    scaler.scale(loss).backward()
                    loss_acc += loss.item()
                scaler.unscale_(opt)
                if float(S["grad_clip"]) > 0:
                    nn.utils.clip_grad_norm_(m.parameters(), float(S["grad_clip"]))
                scaler.step(opt); scaler.update()
                opt.zero_grad(set_to_none=True)
                self.total_steps += 1
                if (it + 1) % 10 == 0:
                    el = time.time() - t0
                    print(f"\rstep {it + 1}/{steps}  loss {loss_acc:.4f}  lr {lr_at(it):.2e}  "
                          f"{(it + 1) / el:.2f} it/s", end="", flush=True)
                if (it + 1) % int(S["eval_every"]) == 0 or it + 1 == steps:
                    ev = evaluate()
                    self.history.append((self.total_steps, ev["train"], ev["val"]))
                    print(f"\n  [eval] step {self.total_steps}: train {ev['train']:.4f}  val {ev['val']:.4f}")
        except KeyboardInterrupt:
            print("\nTraining interrupted.")
        print(f"\nFinished. Total steps trained: {self.total_steps}")
        m.eval()
        if input(f"Save to '{self.save_path}' now? (Y/n) ").lower() != "n":
            self.save()

    # ---- save / load --------------------------------------------------------
    def save(self, path=None):
        if self.model is None or self.tok is None:
            print("Nothing to save (need tokenizer and model)."); return
        path = path or input(f"Save path [{self.save_path}]: ").strip() or self.save_path
        self.save_path = path
        torch.save({"config": self.model.c, "model": self.model.state_dict(),
                    "merges": [list(m) for m in self.tok.merges], "settings": self.S,
                    "total_steps": self.total_steps, "history": self.history,
                    "files": self.files}, path)
        print(f"Saved to {path} ({os.path.getsize(path) / 1e6:.1f} MB)")

    def load(self, path=None):
        path = path or input(f"Load path [{self.save_path}]: ").strip() or self.save_path
        if not os.path.isfile(path):
            print("File not found."); return
        try:
            ck = torch.load(path, map_location=self.device, weights_only=False)
            self.tok = ByteBPE(ck["merges"])
            self.model = GPT(ck["config"]).to(self.device)
            self.model.load_state_dict(ck["model"])
            self.model.eval()
            self.S.update({k: v for k, v in ck.get("settings", {}).items() if k in self.S})
            for k in ARCH_KEYS:
                self.S[k] = ck["config"][k]
            self.total_steps = ck.get("total_steps", 0)
            self.history = ck.get("history", [])
            self.save_path = path
            print(f"Loaded {path}: {self.model.n_params() / 1e6:.2f}M params, vocab {self.tok.vocab_size}, "
                  f"{self.total_steps} steps trained.")
        except Exception as e:
            print(f"Failed to load: {e}")

    # ---- generation ---------------------------------------------------------
    def _stream(self, prompt_ids, stop_strs=()):
        S = self.S
        out_ids, printed, text = [], "", ""
        gen = self.model.generate(prompt_ids, int(S["max_new_tokens"]), float(S["temperature"]),
                                  int(S["top_k"]), float(S["top_p"]), float(S["rep_penalty"]), self.tok.eos_id)
        for t in gen:
            if t == self.tok.eos_id:
                break
            out_ids.append(t)
            text = self.tok.decode(out_ids)
            stopped = False
            for s in stop_strs:
                if s in text:
                    text = text[: text.index(s)]; stopped = True
            safe = text[:-1] if text.endswith("\ufffd") else text
            if len(safe) > len(printed):
                print(safe[len(printed):], end="", flush=True)
                printed = safe
            if stopped:
                break
        if len(text) > len(printed):
            print(text[len(printed):], end="", flush=True)
        print()
        return text.strip()

    def _ready(self):
        if self.model is None or self.tok is None:
            print("Train or load a model first."); return False
        return True

    def complete(self):
        if not self._ready():
            return
        prompt = input("Prompt: ")
        ids = self.tok.encode(prompt)[-self.model.c["block_size"]:] or [self.tok.eos_id]
        print(prompt, end="")
        try:
            self._stream(ids)
        except KeyboardInterrupt:
            print("\n[stopped]")

    def chat(self):
        if not self._ready():
            return
        S = self.S
        ut, bt = S["user_tag"], S["bot_tag"]
        hist = (S["system_prompt"] + "\n") if S["system_prompt"] else ""
        print("Chat started. Commands: /reset  /quit   (Ctrl+C stops a reply)")
        print("Note: the model chats well only if trained on data like "
              f"'{ut} ...\\n{bt} ...' (jsonl prompt/response pairs are auto-formatted this way).")
        while True:
            try:
                u = input("\nYou: ").strip()
            except (EOFError, KeyboardInterrupt):
                print(); break
            if not u:
                continue
            if u in ("/quit", "/exit"):
                break
            if u == "/reset":
                hist = (S["system_prompt"] + "\n") if S["system_prompt"] else ""
                print("[history cleared]"); continue
            hist += f"{ut} {u}\n{bt}"
            ids = self.tok.encode(hist, allow_special=False)[-(self.model.c["block_size"] - 1):]
            print("Bot:", end=" ", flush=True)
            try:
                reply = self._stream(ids, stop_strs=(f"\n{ut}", f"{ut} "))
            except KeyboardInterrupt:
                print("\n[stopped]"); reply = ""
            hist += f" {reply}\n"

    # ---- info / settings ----------------------------------------------------
    def info(self):
        print("\n=== AI INFO ===")
        print(f"Device          : {self.device}" + (f" ({torch.cuda.get_device_name(0)})" if self.device == "cuda" else ""))
        print(f"PyTorch         : {torch.__version__}")
        print(f"Tokenizer       : " + (f"byte-level BPE, vocab {self.tok.vocab_size} "
                                       f"({len(self.tok.merges)} merges)" if self.tok else "not trained"))
        if self.model:
            c = self.model.c
            print(f"Model           : GPT, {self.model.n_params() / 1e6:.2f}M params")
            print(f"  layers/heads  : {c['n_layer']} / {c['n_head']}   embd {c['n_embd']}   "
                  f"context {c['block_size']}   dropout {c['dropout']}")
            print(f"Steps trained   : {self.total_steps}")
            if self.history:
                s, tl, vl = self.history[-1]
                print(f"Last eval       : step {s}  train {tl:.4f}  val {vl:.4f}  (ppl {math.exp(min(vl, 20)):.1f})")
        else:
            print("Model           : none")
        print(f"Data files      : {len(self.files)}")
        print(f"Save path       : {self.save_path}")

    def settings_menu(self):
        groups = [("Model architecture (new models only)", ("n_layer", "n_head", "n_embd", "block_size", "dropout")),
                  ("Tokenizer", ("vocab_size", "tok_train_chars")),
                  ("Training", ("steps", "batch_size", "grad_accum", "lr", "min_lr", "warmup", "weight_decay",
                                "grad_clip", "eval_every", "eval_iters", "val_split", "mixed_precision")),
                  ("Data parsing (comma-separate several; CSV accepts names or 0-based numbers)",
                   ("csv_prompt_col", "csv_response_col", "csv_text_col", "json_prompt_key", "json_response_key", "json_key")),
                  ("Generation / chat", ("temperature", "top_k", "top_p", "rep_penalty", "max_new_tokens",
                                         "user_tag", "bot_tag", "system_prompt"))]
        while True:
            print("\n=== SETTINGS ===")
            keys = []
            for title, ks in groups:
                print(f"-- {title}")
                for k in ks:
                    keys.append(k)
                    print(f"  {len(keys):2d}. {k:16s} = {self.S[k]!r}")
            print("  d. reset to defaults    b. back")
            c = input("Number to change: ").strip().lower()
            if c in ("b", ""):
                return
            if c == "d":
                self.S = dict(DEFAULT_SETTINGS); print("Defaults restored."); continue
            if not c.isdigit() or not 1 <= int(c) <= len(keys):
                continue
            k = keys[int(c) - 1]
            raw = input(f"New value for {k} (current {self.S[k]!r}): ")
            try:
                cur = DEFAULT_SETTINGS[k]
                if isinstance(cur, bool):
                    val = raw.strip().lower() in ("1", "true", "y", "yes", "on")
                elif isinstance(cur, int):
                    val = int(float(raw))
                elif isinstance(cur, float):
                    val = float(raw)
                else:
                    val = raw.replace("\\n", "\n")
                self.S[k] = val
                if k in ARCH_KEYS and self.model is not None:
                    print("(Architecture changes apply to the next NEW model; use 'Discard model' to rebuild.)")
            except ValueError:
                print("Invalid value.")


def main():
    app = App()
    if len(sys.argv) > 1 and os.path.isfile(sys.argv[1]) and sys.argv[1].endswith(".pt"):
        app.load(sys.argv[1])
    items = [
        ("Add training data (.txt / .csv / .jsonl, folders, globs)", app.add_files),
        ("List data files", app.list_files),
        ("Clear data files", app.clear_files),
        ("Set up CSV / JSONL prompt & response columns", app.column_setup),
        ("Preview how data will be formatted", app.preview),
        ("Train / retrain tokenizer (byte-level BPE)", app.train_tokenizer),
        ("Train model (new, or continue current)", app.train),
        ("Chat", app.chat),
        ("Text completion (raw prompt)", app.complete),
        ("Training & generation options", app.settings_menu),
        ("Save model", app.save),
        ("Load model", app.load),
        ("AI info", app.info),
        ("Discard model (keep tokenizer)", app.reset_model),
    ]
    while True:
        print("\n==================== ByteBPE-LLM ====================")
        for i, (name, _) in enumerate(items, 1):
            print(f" {i:2d}. {name}")
        print("  0. Exit")
        c = input("> ").strip()
        if c == "0":
            if app.model is not None and input("Exit without saving? (y/N) ").lower() != "y":
                continue
            break
        if c.isdigit() and 1 <= int(c) <= len(items):
            try:
                items[int(c) - 1][1]()
            except KeyboardInterrupt:
                print("\n[cancelled]")
        else:
            print("Invalid choice.")


if __name__ == "__main__":
    main()
