#!/usr/bin/env python3
"""
Sylvie LLM: a small trainable GPT-style language model with a byte-level BPE
tokenizer, written in PyTorch. Single file, menu driven.

Training is split into two separate stages:
  Stage 1  PRETRAINING   raw text (.txt / .csv / .jsonl)  -> next-token loss on everything
  Stage 2  INSTRUCTION   prompt/response or chat data      -> loss on assistant replies ONLY
           FINE-TUNING   (.csv / .jsonl)                      (prompt tokens are masked out)

Conversations use dedicated special tokens, not text tags:
  <|system|> <|user|> <|assistant|> <|end|> (end of turn)  and  <|endoftext|> (document break)

A built-in evaluation measures instruction following with programmatically
checkable constraints (IFEval-style), plus held-out response loss.

Requires: pip install torch
Run:      python sylvie_llm.py [model.pt]
"""
import os, sys, re, csv, json, glob, math, time, random, contextlib
from collections import Counter, defaultdict

try:
    import torch
    import torch.nn as nn
    import torch.nn.functional as F
except ImportError:
    print("PyTorch is required:  pip install torch")
    sys.exit(1)

csv.field_size_limit(min(sys.maxsize, 2**31 - 1))

# ----------------------------------------------------------------------------
# Special tokens
# ----------------------------------------------------------------------------
EOS_TEXT = "<|endoftext|>"      # separates documents (pretraining)
SYSTEM_TEXT = "<|system|>"
USER_TEXT = "<|user|>"
ASSISTANT_TEXT = "<|assistant|>"
END_TEXT = "<|end|>"            # end of one conversation turn
# Order matters: ids are assigned right after the BPE merges. <|endoftext|> keeps
# the same id it had in older checkpoints, so those can be migrated.
SPECIALS = [EOS_TEXT, SYSTEM_TEXT, USER_TEXT, ASSISTANT_TEXT, END_TEXT]
SPECIAL_RE = re.compile("(" + "|".join(re.escape(s) for s in SPECIALS) + ")")
IGNORE = -100                   # label value excluded from the loss


# ----------------------------------------------------------------------------
# Byte-level BPE tokenizer
# ----------------------------------------------------------------------------
PRETOKEN = re.compile(r"'s|'t|'re|'ve|'m|'ll|'d| ?[A-Za-z]+| ?[0-9]+| ?[^\sA-Za-z0-9]+|\s+(?!\S)|\s+")


class ByteBPE:
    """Ids 0-255 are raw bytes, then one id per learned merge, then the special tokens."""

    def __init__(self, merges=None):
        self.merges = [tuple(m) for m in (merges or [])]
        self._build()

    def _build(self):
        self.ranks = {m: i for i, m in enumerate(self.merges)}
        self.vocab = {i: bytes([i]) for i in range(256)}
        for i, (a, b) in enumerate(self.merges):
            self.vocab[256 + i] = self.vocab[a] + self.vocab[b]
        base = 256 + len(self.merges)
        self.special_ids = {s: base + i for i, s in enumerate(SPECIALS)}
        self.id_to_special = {v: k for k, v in self.special_ids.items()}
        self.eos_id = self.special_ids[EOS_TEXT]
        self.system_id = self.special_ids[SYSTEM_TEXT]
        self.user_id = self.special_ids[USER_TEXT]
        self.assistant_id = self.special_ids[ASSISTANT_TEXT]
        self.end_id = self.special_ids[END_TEXT]
        self.role_ids = {"system": self.system_id, "user": self.user_id, "assistant": self.assistant_id}
        self.vocab_size = base + len(SPECIALS)
        self._cache = {}

    # -- training -----------------------------------------------------------
    def train(self, texts, vocab_size, verbose=True):
        """texts: iterable of strings. Special tokens are never merged."""
        n_merges = max(0, vocab_size - 256 - len(SPECIALS))
        counts = Counter()
        for t in texts:
            counts.update(PRETOKEN.findall(t))
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
        """allow_special=False treats '<|user|>' etc. in the text as ordinary characters,
        so training data and user messages can never inject control tokens."""
        out = []
        parts = SPECIAL_RE.split(text) if allow_special else [text]
        for k, part in enumerate(parts):
            if allow_special and k % 2 == 1:
                out.append(self.special_ids[part])
                continue
            for chunk in PRETOKEN.findall(part):
                out.extend(self._encode_chunk(chunk))
        return out

    def decode(self, ids, skip_special=False):
        buf = []
        for i in ids:
            if i in self.id_to_special:
                if not skip_special:
                    buf.append(self.id_to_special[i].encode())
            elif i in self.vocab:
                buf.append(self.vocab[i])
        return b"".join(buf).decode("utf-8", errors="replace")


# ----------------------------------------------------------------------------
# Conversation encoding with response-only loss masks
# ----------------------------------------------------------------------------
ROLE_ALIASES = {"user": "user", "human": "user", "prompt": "user", "assistant": "assistant",
                "model": "assistant", "bot": "assistant", "gpt": "assistant", "ai": "assistant",
                "system": "system"}


def with_system(messages, system_prompt):
    """Prepend the configured system prompt unless the conversation already has one."""
    if system_prompt and system_prompt.strip() and (not messages or messages[0]["role"] != "system"):
        return [{"role": "system", "content": system_prompt.strip()}] + list(messages)
    return list(messages)


def encode_conversation(tok, messages, response_only=True):
    """
    Turn [{'role','content'}, ...] into (ids, labels).
    Layout per turn:  <|role|> content-tokens <|end|>
    labels[i] is the token id the model should learn to produce at position i,
    or IGNORE. With response_only=True only assistant content and its closing
    <|end|> are supervised; system/user turns and every role token are masked.
    The <|end|> after a reply IS supervised so the model learns when to stop.
    """
    ids, labels = [], []
    for m in messages:
        head = [tok.role_ids[m["role"]]]
        body = tok.encode(str(m["content"]).strip(), allow_special=False) + [tok.end_id]
        ids += head + body
        if not response_only:
            labels += head + body
        elif m["role"] == "assistant":
            labels += [IGNORE] * len(head) + body
        else:
            labels += [IGNORE] * (len(head) + len(body))
    return ids, labels


def chat_prompt_ids(tok, messages, budget):
    """Encode a conversation ending with an open <|assistant|> turn, dropping the oldest
    non-system turns (in whole user/assistant pairs) until it fits in `budget` tokens."""
    msgs = list(messages)
    while True:
        ids = encode_conversation(tok, msgs, False)[0] + [tok.assistant_id]
        if len(ids) <= budget:
            return ids
        first = 1 if msgs and msgs[0]["role"] == "system" else 0
        if len(msgs) - first <= 1:
            break
        del msgs[first]
        while first < len(msgs) and msgs[first]["role"] == "assistant":
            del msgs[first]
    return [tok.user_id] + ids[-(budget - 1):]


ROLE_TEXT = {"system": SYSTEM_TEXT, "user": USER_TEXT, "assistant": ASSISTANT_TEXT}


def render_conversation(messages):
    return "".join(ROLE_TEXT[m["role"]] + m["content"] + END_TEXT for m in messages)


# ----------------------------------------------------------------------------
# Data loading (.txt / .csv / .jsonl)
# ----------------------------------------------------------------------------
SUPPORTED = (".txt", ".csv", ".jsonl")
SFT_SUPPORTED = (".csv", ".jsonl")


def expand_paths(spec, exts=SUPPORTED):
    found = []
    for item in spec.replace(",", " ").split() if '"' not in spec else re.findall(r'"([^"]+)"|(\S+)', spec):
        if isinstance(item, tuple):
            item = item[0] or item[1]
        item = os.path.expanduser(item)
        if os.path.isdir(item):
            for root, _, files in os.walk(item):
                for f in sorted(files):
                    if f.lower().endswith(exts):
                        found.append(os.path.join(root, f))
        else:
            matches = glob.glob(item) or [item]
            found.extend(m for m in matches if os.path.isfile(m))
    return [f for f in found if f.lower().endswith(exts)]


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


def _content_text(c):
    """Message content may be a string or a list of OpenAI-style parts."""
    if isinstance(c, str):
        return c.strip()
    if isinstance(c, list):
        return "\n".join(_content_text(p.get("text", "") if isinstance(p, dict) else p) for p in c).strip()
    return "" if c is None else str(c).strip()


PAIR_KEYS = (("prompt", "response"), ("prompt", "completion"), ("instruction", "output"),
             ("instruction", "response"), ("question", "answer"), ("input", "output"), ("user", "assistant"))


def _pair_from_obj(obj, S):
    """Return (prompt, response) from a dict record, or None."""
    if S["json_prompt_key"] and S["json_response_key"]:
        p, r = _join_keys(obj, S["json_prompt_key"]), _join_keys(obj, S["json_response_key"])
        return (p, r) if p and r else None
    for pk, rk in PAIR_KEYS:
        if pk in obj and rk in obj:
            p, r = _content_text(obj[pk]), _content_text(obj[rk])
            if pk == "instruction" and _content_text(obj.get("input")):   # Alpaca style
                p = p + "\n\n" + _content_text(obj["input"])
            return (p, r) if p and r else None
    return None


def _record_messages(obj, S):
    """Dict record -> list of {'role','content'} (or None if it is not a conversation)."""
    if not isinstance(obj, dict):
        return None
    msgs = []
    sysm = _content_text(obj.get("system")) if isinstance(obj.get("system"), (str, list)) else ""
    if sysm:
        msgs.append({"role": "system", "content": sysm})
    if not (S["json_prompt_key"] and S["json_response_key"]) and isinstance(obj.get("messages"), list):
        raw = [(m.get("role"), m.get("content")) for m in obj["messages"] if isinstance(m, dict)]
    elif not (S["json_prompt_key"] and S["json_response_key"]) and isinstance(obj.get("conversations"), list):
        raw = [(m.get("from") or m.get("role"), m.get("value", m.get("content")))
               for m in obj["conversations"] if isinstance(m, dict)]
    else:
        pair = _pair_from_obj(obj, S)
        if not pair:
            return None
        raw = [("user", pair[0]), ("assistant", pair[1])]
    for role, content in raw:
        role = ROLE_ALIASES.get(str(role).lower())
        text = _content_text(content)
        if role and text:
            msgs.append({"role": role, "content": text})
    return msgs


def valid_conversation(msgs):
    return bool(msgs) and any(m["role"] == "assistant" for m in msgs) and any(m["role"] == "user" for m in msgs)


def read_text_documents(path, S):
    """PRETRAINING reader: yield plain-text documents. Pair-style records are flattened to
    'prompt\\nresponse' text (no chat tokens are ever used during pretraining)."""
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
            pick = lambda row, idx: "\n".join(row[i].strip() for i in idx if i < len(row) and row[i].strip())
            for row in rd:
                if tc:
                    t = pick(row, tc)
                elif pc and rc:
                    t = "\n".join(x for x in (pick(row, pc), pick(row, rc)) if x)
                else:
                    t = " ".join(c for c in row if c)
                if t:
                    yield t
        finally:
            f.close()
    elif ext == ".jsonl":
        with open(path, "r", encoding="utf-8", errors="replace") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                try:
                    obj = json.loads(line)
                except json.JSONDecodeError:
                    continue
                t = _record_text(obj, S)
                if t and t.strip():
                    yield t


def _record_text(obj, S):
    if isinstance(obj, str):
        return obj
    if not isinstance(obj, dict):
        return str(obj)
    if S["json_key"]:
        return _join_keys(obj, S["json_key"])
    if "text" in obj and isinstance(obj["text"], str):
        return obj["text"]
    msgs = _record_messages(obj, S)
    if msgs:
        return "\n".join(m["content"] for m in msgs)
    return "\n".join(str(v) for v in obj.values() if isinstance(v, str))


def read_conversations(path, S):
    """INSTRUCTION-TUNING reader: yield conversations as lists of {'role','content'}."""
    ext = os.path.splitext(path)[1].lower()
    if ext == ".csv":
        f, rd = open_csv(path)
        try:
            header = next(rd, [])
            pc = _resolve_cols(S["csv_prompt_col"], header)
            rc = _resolve_cols(S["csv_response_col"], header)
            if not (pc and rc):
                raise ValueError("instruction CSVs need BOTH csv_prompt_col and csv_response_col "
                                 "(menu: 'Set up CSV / JSONL columns')")
            pick = lambda row, idx: "\n".join(row[i].strip() for i in idx if i < len(row) and row[i].strip())
            for row in rd:
                p, r = pick(row, pc), pick(row, rc)
                if p and r:
                    yield [{"role": "user", "content": p}, {"role": "assistant", "content": r}]
        finally:
            f.close()
    elif ext == ".jsonl":
        with open(path, "r", encoding="utf-8", errors="replace") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                try:
                    msgs = _record_messages(json.loads(line), S)
                except json.JSONDecodeError:
                    continue
                if valid_conversation(msgs):
                    yield msgs
    else:
        raise ValueError("instruction data must be .csv or .jsonl (use pretraining for .txt)")


def load_corpus(files, S, reader):
    items = []
    for p in files:
        n = 0
        try:
            for d in reader(p, S):
                if d:
                    items.append(d); n += 1
        except Exception as e:
            print(f"  ! could not read {p}: {e}")
            continue
        print(f"  {p}: {n} item(s)")
    return items


# ----------------------------------------------------------------------------
# Instruction-following evaluation (verifiable constraints, IFEval-style)
# ----------------------------------------------------------------------------
def _has_letters(t):
    return any(c.isalpha() for c in t)


def _kw(r, w):
    return re.search(r"\b" + re.escape(str(w)) + r"\b", r, re.I) is not None


def _is_json(r):
    s = r.strip()
    s = re.sub(r"^```(?:json)?\s*|\s*```$", "", s)
    try:
        json.loads(s)
        return True
    except ValueError:
        return False


def _bullets(r):
    return len([l for l in r.split("\n") if re.match(r"^\s*([-*\u2022])\s+\S", l)])


def _sentences(r):
    return len([s for s in re.split(r"(?<=[.!?])\s+", r.strip()) if s.strip()])


def _paragraphs(r):
    return len([p for p in re.split(r"\n\s*\n", r.strip()) if p.strip()])


# name -> (check(response, arg) -> bool, instruction text(arg))
CHECKS = {
    "lowercase": (lambda r, a: _has_letters(r) and r == r.lower(),
                  lambda a: "Your entire response must be in lowercase letters."),
    "uppercase": (lambda r, a: _has_letters(r) and r == r.upper(),
                  lambda a: "Your entire response must be in ALL CAPITAL LETTERS."),
    "max_words": (lambda r, a: 0 < len(r.split()) <= int(a), lambda a: f"Answer with at most {a} words."),
    "min_words": (lambda r, a: len(r.split()) >= int(a), lambda a: f"Answer with at least {a} words."),
    "min_sentences": (lambda r, a: _sentences(r) >= int(a), lambda a: f"Use at least {a} sentences."),
    "include_keyword": (lambda r, a: _kw(r, a), lambda a: f'Include the word "{a}" in your response.'),
    "forbidden_word": (lambda r, a: not _kw(r, a), lambda a: f'Do not use the word "{a}".'),
    "start_with": (lambda r, a: r.strip().lower().startswith(str(a).lower()),
                   lambda a: f'Start your response with "{a}".'),
    "end_with": (lambda r, a: r.strip().lower().endswith(str(a).lower()),
                 lambda a: f'End your response with the exact phrase "{a}".'),
    "no_commas": (lambda r, a: "," not in r, lambda a: "Do not use any commas."),
    "n_bullets": (lambda r, a: _bullets(r) == int(a),
                  lambda a: f"Your answer must contain exactly {a} bullet points, each starting with '- '."),
    "n_paragraphs": (lambda r, a: _paragraphs(r) == int(a),
                     lambda a: f"Write exactly {a} paragraphs separated by a blank line."),
    "quoted": (lambda r, a: len(r.strip()) > 1 and r.strip().startswith('"') and r.strip().endswith('"'),
               lambda a: "Wrap your entire response in double quotation marks."),
    "title": (lambda r, a: re.search(r"<<[^\n<>]+>>", r) is not None,
              lambda a: "Include a title wrapped in double angular brackets, like <<title>>."),
    "json": (lambda r, a: _is_json(r), lambda a: "Format your entire response as valid JSON."),
    "repeat_prompt": (lambda r, a: r.strip().lower().startswith(str(a).strip().lower()),
                      lambda a: "First repeat the request word for word without change, then give your answer."),
    "postscript": (lambda r, a: re.search(r"^\s*P\.S\.", r, re.M | re.I) is not None,
                   lambda a: "At the end, add a postscript starting with P.S."),
    "ends_question": (lambda r, a: r.strip().endswith("?"), lambda a: "End your response with a question."),
}


def run_check(name, response, arg):
    if not response.strip():
        return False                      # an empty reply never satisfies an instruction
    try:
        return bool(CHECKS[name][0](response, arg))
    except Exception:
        return False


def _loose_variants(r):
    lines = r.split("\n")
    base = [r, "\n".join(lines[1:]), "\n".join(lines[:-1]), "\n".join(lines[1:-1])]
    return base + [v.replace("*", "") for v in base]


def run_check_loose(name, response, arg):
    return any(run_check(name, v, arg) for v in _loose_variants(response))


_TOPICS = ["the ocean", "a rainy day", "learning to cook", "your favorite season", "public libraries",
           "space travel", "a good night's sleep", "riding a bike", "a small village", "the history of tea",
           "healthy habits", "a long train trip"]
_SINGLE = [("lowercase", None), ("uppercase", None), ("max_words", 25), ("max_words", 50), ("min_words", 40),
           ("min_sentences", 4), ("include_keyword", "important"), ("include_keyword", "simple"),
           ("forbidden_word", "very"), ("start_with", "Today"), ("end_with", "Thank you."), ("no_commas", None),
           ("n_bullets", 3), ("n_bullets", 2), ("n_paragraphs", 2), ("quoted", None), ("title", None),
           ("repeat_prompt", "<TASK>"), ("postscript", None), ("ends_question", None), ("json", None)]
_PAIRS = [[("lowercase", None), ("no_commas", None)], [("max_words", 40), ("include_keyword", "important")],
          [("n_bullets", 3), ("forbidden_word", "very")], [("title", None), ("min_words", 40)],
          [("uppercase", None), ("max_words", 30)], [("start_with", "Today"), ("end_with", "Thank you.")],
          [("quoted", None), ("max_words", 25)], [("n_paragraphs", 2), ("no_commas", None)]]


def builtin_suite():
    """~29 deterministic prompts, each with one or two machine-checkable instructions."""
    items = []
    combos = [[c] for c in _SINGLE] + _PAIRS
    for i, combo in enumerate(combos):
        task = f"Write a short note about {_TOPICS[i % len(_TOPICS)]}."
        checks = [(n, task if a == "<TASK>" else a) for n, a in combo]
        prompt = task + " " + " ".join(CHECKS[n][1](a) for n, a in checks)
        items.append({"prompt": prompt, "checks": checks, "reference": None})
    return items


def _parse_checks(raw):
    out = []
    for c in raw if isinstance(raw, list) else []:
        name, arg = None, None
        if isinstance(c, dict):
            name, arg = c.get("type"), c.get("value")
        elif isinstance(c, (list, tuple)) and c:
            name, arg = c[0], (c[1] if len(c) > 1 else None)
        elif isinstance(c, str):
            name, _, a = c.partition(":")
            arg = a or None
        if name in CHECKS:
            out.append((name, arg))
        else:
            print(f"  ! unknown check type {name!r} skipped (available: {', '.join(CHECKS)})")
    return out


def load_eval_file(path):
    """JSONL, one per line: {"prompt": "...", "checks": [{"type": "max_words", "value": 30}], "reference": "..."}
    'checks' and 'reference' are both optional (prompt/response files work as a reference-only eval)."""
    items = []
    with open(path, "r", encoding="utf-8", errors="replace") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                obj = json.loads(line)
            except json.JSONDecodeError:
                continue
            if not isinstance(obj, dict):
                continue
            prompt = next((_content_text(obj[k]) for k in ("prompt", "instruction", "question", "input")
                           if _content_text(obj.get(k))), "")
            if not prompt:
                continue
            ref = next((_content_text(obj[k]) for k in ("reference", "response", "output", "answer", "completion")
                        if _content_text(obj.get(k))), None)
            items.append({"prompt": prompt, "checks": _parse_checks(obj.get("checks") or obj.get("instructions")),
                          "reference": ref})
    return items


def _norm_text(s):
    return re.sub(r"[^\w\s]", "", s.lower()).split()


def token_f1(pred, ref):
    p, r = _norm_text(pred), _norm_text(ref)
    if not p or not r:
        return float(p == r)
    n = sum((Counter(p) & Counter(r)).values())
    if n == 0:
        return 0.0
    pr, rc = n / len(p), n / len(r)
    return 2 * pr * rc / (pr + rc)


def score_instruction_following(items, responses, finished):
    ps = pl = n_cp = ins = inl = n_ci = 0
    by = defaultdict(lambda: [0, 0, 0])      # check type -> [strict ok, loose ok, total]
    f1s, ems, details = [], [], []
    for it, resp, fin in zip(items, responses, finished):
        rows = []
        for name, arg in it["checks"]:
            s = run_check(name, resp, arg)
            l = s or run_check_loose(name, resp, arg)
            rows.append({"type": name, "arg": arg, "strict": s, "loose": l})
            by[name][0] += s; by[name][1] += l; by[name][2] += 1
        if rows:
            n_cp += 1; n_ci += len(rows)
            ps += all(r["strict"] for r in rows); pl += all(r["loose"] for r in rows)
            ins += sum(r["strict"] for r in rows); inl += sum(r["loose"] for r in rows)
        d = {"prompt": it["prompt"], "response": resp, "finished": fin, "checks": rows}
        if it.get("reference"):
            d["ref_f1"] = token_f1(resp, it["reference"])
            f1s.append(d["ref_f1"]); ems.append(float(_norm_text(resp) == _norm_text(it["reference"])))
        details.append(d)
    n = len(items)
    return {"n_prompts": n, "n_checked_prompts": n_cp, "n_checks": n_ci,
            "prompt_strict": ps / n_cp if n_cp else None, "prompt_loose": pl / n_cp if n_cp else None,
            "inst_strict": ins / n_ci if n_ci else None, "inst_loose": inl / n_ci if n_ci else None,
            "by_type": {k: v for k, v in sorted(by.items())},
            "stop_rate": sum(finished) / n if n else 0.0,
            "empty_rate": sum(1 for r in responses if not r.strip()) / n if n else 0.0,
            "avg_words": sum(len(r.split()) for r in responses) / n if n else 0.0,
            "ref_f1": sum(f1s) / len(f1s) if f1s else None, "ref_em": sum(ems) / len(ems) if ems else None,
            "details": details}


def _pc(x):
    return "n/a" if x is None else f"{100 * x:.1f}%"


def print_eval_report(res, show=3):
    print("\n=== INSTRUCTION-FOLLOWING EVAL ===")
    print(f"prompts {res['n_prompts']}  (with checks: {res['n_checked_prompts']}, individual checks: {res['n_checks']})")
    if res["n_checks"]:
        print(f"  prompt-level       strict {_pc(res['prompt_strict'])}   loose {_pc(res['prompt_loose'])}")
        print(f"  instruction-level  strict {_pc(res['inst_strict'])}   loose {_pc(res['inst_loose'])}")
        print("  by instruction type (strict):")
        for k, (s, l, t) in res["by_type"].items():
            print(f"    {k:16s} {s}/{t}  {_pc(s / t)}")
    if res["ref_f1"] is not None:
        print(f"  vs reference answers: token-F1 {res['ref_f1']:.3f}   exact match {_pc(res['ref_em'])}")
    print(f"  ended with <|end|>: {_pc(res['stop_rate'])}   empty replies: {_pc(res['empty_rate'])}   "
          f"avg length: {res['avg_words']:.1f} words")
    if res.get("heldout_loss") is not None:
        print(f"  held-out response loss: {res['heldout_loss']:.4f}  (ppl {math.exp(min(res['heldout_loss'], 20)):.1f}, "
              f"{res['heldout_tokens']} response tokens)")
    for d in res["details"][:max(0, show)]:
        print("  ---")
        print("  prompt  :", d["prompt"][:200])
        print("  response:", d["response"][:300].replace("\n", " / "))
        if d["checks"]:
            print("  checks  :", ", ".join(f"{c['type']}={'ok' if c['strict'] else 'FAIL'}" for c in d["checks"]))


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

    def forward(self, idx, targets=None, last_only=False):
        """targets may contain IGNORE (-100); those positions add nothing to the loss.
        The loss is the mean over supervised tokens only (0 if a batch has none)."""
        B, T = idx.shape
        x = self.drop(self.tok(idx) + self.pos(torch.arange(T, device=idx.device)))
        for b in self.blocks:
            x = b(x)
        x = self.lnf(x)
        if targets is None:
            return self.head(x[:, -1:, :] if last_only else x), None
        logits = self.head(x)
        flat = targets.reshape(-1)
        n = (flat != IGNORE).sum().clamp(min=1)
        loss = F.cross_entropy(logits.reshape(-1, logits.size(-1)).float(), flat,
                               ignore_index=IGNORE, reduction="sum") / n
        return logits, loss

    def n_params(self):
        return sum(p.numel() for p in self.parameters())

    @torch.no_grad()
    def generate(self, ids, max_new, temperature=0.8, top_k=50, top_p=0.95, rep_penalty=1.1,
                 stop_ids=(), ban_ids=()):
        """Generator yielding one new token id at a time. Stops after yielding a stop id.
        ban_ids can never be sampled; stop/ban ids are exempt from the repetition penalty
        (otherwise <|end|> gets harder to produce every turn)."""
        self.eval()
        dev = next(self.parameters()).device
        idx = torch.tensor([ids], dtype=torch.long, device=dev)
        ban = torch.tensor(list(ban_ids), dtype=torch.long, device=dev) if ban_ids else None
        exempt = torch.tensor(list(stop_ids) + list(ban_ids), dtype=torch.long, device=dev) \
            if (stop_ids or ban_ids) else None
        stop = set(stop_ids)
        for _ in range(max_new):
            logits = self(idx[:, -self.c["block_size"]:], last_only=True)[0][0, -1].float()
            if rep_penalty and rep_penalty != 1.0:
                seen = torch.unique(idx[0, -256:])
                if exempt is not None:
                    seen = seen[~torch.isin(seen, exempt)]
                if seen.numel():
                    l = logits[seen]
                    logits[seen] = torch.where(l > 0, l / rep_penalty, l * rep_penalty)
            if ban is not None:
                logits[ban] = -float("inf")
            if temperature <= 0:
                nxt = torch.argmax(logits).view(1)
            else:
                logits = logits / temperature
                if top_k and top_k > 0:
                    kth = torch.topk(logits, min(top_k, logits.numel()))[0][-1]
                    logits[logits < kth] = -float("inf")
                if top_p and top_p < 1.0:
                    sl, si = torch.sort(logits, descending=True)
                    probs = F.softmax(sl, -1)
                    drop = torch.cumsum(probs, -1) - probs > top_p
                    sl[drop] = -float("inf")
                    logits = torch.full_like(logits, -float("inf")).scatter(0, si, sl)
                nxt = torch.multinomial(F.softmax(logits, -1), 1)
            idx = torch.cat([idx, nxt.view(1, 1)], 1)
            t = int(nxt)
            yield t
            if t in stop:
                break


# ----------------------------------------------------------------------------
# Batching helpers
# ----------------------------------------------------------------------------
class Sampler:
    """Epoch-style shuffled cycling over a list of examples."""

    def __init__(self, examples, seed=0):
        self.ex, self.rng, self.order = examples, random.Random(seed), []

    def next(self, n):
        out = []
        while len(out) < n:
            if not self.order:
                self.order = list(range(len(self.ex)))
                self.rng.shuffle(self.order)
            out.append(self.ex[self.order.pop()])
        return out


def collate(examples, pad_id, device):
    """Right-pad to the longest example. Pads are causally invisible to real tokens and
    their labels are IGNORE, so no attention mask is needed."""
    L = max(len(i) for i, _ in examples) - 1
    x = torch.full((len(examples), L), pad_id, dtype=torch.long)
    y = torch.full((len(examples), L), IGNORE, dtype=torch.long)
    for r, (ids, lab) in enumerate(examples):
        n = len(ids) - 1
        x[r, :n] = torch.tensor(ids[:-1], dtype=torch.long)
        y[r, :n] = torch.tensor(lab[1:], dtype=torch.long)
    return x.to(device), y.to(device)


def _make_scaler(enabled):
    try:
        return torch.amp.GradScaler("cuda", enabled=enabled)
    except (AttributeError, TypeError):
        return torch.cuda.amp.GradScaler(enabled=enabled)


# ----------------------------------------------------------------------------
# Application state + defaults
# ----------------------------------------------------------------------------
DEFAULT_SETTINGS = {
    # model architecture (only used when creating a NEW model)
    "n_layer": 6, "n_head": 6, "n_embd": 384, "block_size": 256, "dropout": 0.1,
    # tokenizer
    "vocab_size": 4000, "tok_train_chars": 5_000_000,
    # stage 1: pretraining (+ settings shared by both stages)
    "steps": 2000, "batch_size": 16, "grad_accum": 1, "lr": 3e-4, "min_lr": 3e-5,
    "warmup": 100, "weight_decay": 0.1, "grad_clip": 1.0, "eval_every": 250,
    "eval_iters": 20, "val_split": 0.05, "mixed_precision": True,
    # stage 2: instruction fine-tuning
    "sft_steps": 600, "sft_batch_size": 16, "sft_grad_accum": 1, "sft_lr": 1e-4, "sft_min_lr": 1e-5,
    "sft_warmup": 30, "sft_eval_every": 100, "sft_val_split": 0.05, "sft_response_only": True,
    # data parsing
    "csv_prompt_col": "", "csv_response_col": "", "csv_text_col": "",
    "json_prompt_key": "", "json_response_key": "", "json_key": "",
    # generation / chat
    "temperature": 0.8, "top_k": 50, "top_p": 0.95, "rep_penalty": 1.1, "max_new_tokens": 200,
    "system_prompt": "",
    # evaluation
    "eval_file": "", "eval_max_new_tokens": 160, "eval_temperature": 0.0, "eval_show": 3,
    "eval_report": "eval_report.json",
}
ARCH_KEYS = ("n_layer", "n_head", "n_embd", "block_size", "dropout")


class App:
    def __init__(self):
        self.S = dict(DEFAULT_SETTINGS)
        self.pt_files, self.sft_files = [], []
        self.tok = None
        self.model = None
        self.stage = "new"                        # new -> pretrained -> sft
        self.steps = {"pretrain": 0, "sft": 0}
        self.hist = {"pretrain": [], "sft": []}   # (step, train_loss, val_loss)
        self.eval_log = []
        self.device = "cuda" if torch.cuda.is_available() else (
            "mps" if getattr(torch.backends, "mps", None) and torch.backends.mps.is_available() else "cpu")
        self.save_path = "model.pt"

    # ---- data ---------------------------------------------------------------
    def _add(self, label, store, exts):
        spec = input(f"{label} file(s), folder(s) or glob (e.g. data/*{exts[-1]}) separated by spaces: ").strip()
        if not spec:
            return
        new = [f for f in expand_paths(spec, exts) if f not in store]
        store += new
        print(f"Added {len(new)} file(s). Total: {len(store)}")

    def add_pretrain_files(self):
        self._add("PRETRAINING", self.pt_files, SUPPORTED)

    def add_sft_files(self):
        print("Instruction data: .csv (prompt+response columns) or .jsonl (prompt/response, messages, conversations).")
        self._add("INSTRUCTION", self.sft_files, SFT_SUPPORTED)

    def list_files(self):
        for title, files in (("PRETRAINING", self.pt_files), ("INSTRUCTION (SFT)", self.sft_files)):
            print(f"-- {title}: {len(files)} file(s)")
            for i, f in enumerate(files):
                print(f"  [{i}] {f}  ({os.path.getsize(f) / 1e6:.2f} MB)")

    def clear_files(self):
        c = input("Clear which list? (p)retraining / (i)nstruction / (b)oth: ").strip().lower()
        if c in ("p", "b"):
            self.pt_files = []
        if c in ("i", "b"):
            self.sft_files = []
        print("Cleared." if c in ("p", "i", "b") else "Nothing cleared.")

    # ---- column setup / preview ---------------------------------------------
    def column_setup(self):
        S = self.S
        csvs = [f for f in self.pt_files + self.sft_files if f.lower().endswith(".csv")]
        if csvs:
            print("CSV columns found:")
            for f in csvs:
                try:
                    print(f"  {f}: " + ", ".join(f"[{i}] {c}" for i, c in enumerate(csv_header(f))))
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

        print("\n-- INSTRUCTION data, CSV (required for .csv in stage 2)")
        ask("Prompt column(s)", "csv_prompt_col")
        ask("Response column(s)", "csv_response_col")
        print("-- PRETRAINING data, CSV: plain text column(s) (blank = join all cells)")
        ask("Text column(s)", "csv_text_col")
        print("\n-- JSONL (dotted keys allowed, e.g. data.question). Blank = auto-detect common layouts.")
        ask("Prompt key(s)", "json_prompt_key")
        ask("Response key(s)", "json_response_key")
        ask("Plain text key(s) for PRETRAINING", "json_key")
        if input("Preview formatted samples? (Y/n) ").lower() != "n":
            self.preview()

    def preview(self, n=3):
        if not self.pt_files and not self.sft_files:
            print("No data files loaded."); return
        for p in self.pt_files:
            print(f"--- PRETRAIN {p}")
            try:
                for i, d in enumerate(read_text_documents(p, self.S)):
                    if i >= n:
                        break
                    print(d[:400] + ("..." if len(d) > 400 else ""))
                    print("  ~")
            except Exception as e:
                print(f"  ! {e}")
        ro = bool(self.S["sft_response_only"])
        for p in self.sft_files:
            print(f"--- INSTRUCTION {p}")
            try:
                for i, msgs in enumerate(read_conversations(p, self.S)):
                    if i >= n:
                        break
                    msgs = with_system(msgs, self.S["system_prompt"])
                    if self.tok is None:
                        print(render_conversation(msgs)[:500])
                    else:
                        ids, lab = encode_conversation(self.tok, msgs, ro)
                        print(self._masked_view(ids, lab)[:700])
                        sup = sum(1 for l in lab[1:] if l != IGNORE)
                        print(f"  [loss on {sup} of {len(ids) - 1} tokens]")
                    print("  ~")
            except Exception as e:
                print(f"  ! {e}")
        if self.tok is not None and self.sft_files:
            print("(Text inside [[ ]] is trained on; everything else is masked out of the loss.)")

    def _masked_view(self, ids, labels):
        out, run, cur = [], [], None
        for i, l in zip(ids, labels):
            on = l != IGNORE
            if cur is not None and on != cur:
                out.append(("[[" if cur else "") + self.tok.decode(run) + ("]]" if cur else ""))
                run = []
            cur = on
            run.append(i)
        if run:
            out.append(("[[" if cur else "") + self.tok.decode(run) + ("]]" if cur else ""))
        return "".join(out)

    # ---- tokenizer ----------------------------------------------------------
    def train_tokenizer(self):
        if not self.pt_files and not self.sft_files:
            print("Load data files first."); return
        if self.model is not None:
            if input("A model exists; a new tokenizer makes it unusable. Continue? (y/N) ").lower() != "y":
                return
            self.model, self.stage = None, "new"
            self.steps, self.hist = {"pretrain": 0, "sft": 0}, {"pretrain": [], "sft": []}
        print("Reading data...")
        texts = load_corpus(self.pt_files, self.S, read_text_documents)
        for conv in load_corpus(self.sft_files, self.S, read_conversations):
            texts.extend(m["content"] for m in conv)
        if not texts:
            print("No text found."); return
        random.Random(1337).shuffle(texts)
        cap, total, used = int(self.S["tok_train_chars"]), 0, []
        for t in texts:
            used.append(t); total += len(t)
            if total >= cap:
                break
        print(f"Training byte-level BPE on {total:,} characters, target vocab {self.S['vocab_size']} "
              f"(+{len(SPECIALS)} special tokens included)...")
        self.tok = ByteBPE()
        self.tok.train(used, int(self.S["vocab_size"]))
        sample = "Hello, world! héllo 你好 🙂"
        ok = self.tok.decode(self.tok.encode(sample)) == sample
        spec = f"{USER_TEXT}hi{END_TEXT}"
        ok2 = self.tok.encode(spec, allow_special=False) != self.tok.encode(spec) and \
            self.tok.decode(self.tok.encode(spec)) == spec
        print(f"Done. Vocab size: {self.tok.vocab_size}. Round-trip: {'OK' if ok else 'FAILED'}. "
              f"Special tokens: {'OK' if ok2 else 'FAILED'}")

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
        self.stage = "new"
        self.steps, self.hist = {"pretrain": 0, "sft": 0}, {"pretrain": [], "sft": []}
        print(f"Created new model: {self.model.n_params() / 1e6:.2f}M parameters on {self.device}")
        return True

    def reset_model(self):
        self.model, self.stage = None, "new"
        self.steps, self.hist = {"pretrain": 0, "sft": 0}, {"pretrain": [], "sft": []}
        print("Model discarded (tokenizer kept). Use a training stage to create a fresh one.")

    def _ensure_tok_and_model(self, have_data):
        if not have_data:
            print("Load data files first."); return False
        if self.tok is None:
            print("No tokenizer yet - training one first.")
            self.train_tokenizer()
            if self.tok is None:
                return False
        if self.model is None and not self.build_model():
            return False
        return True

    # ---- shared training loop -------------------------------------------------
    def _amp(self):
        use = bool(self.S["mixed_precision"]) and self.device == "cuda"
        dtype = torch.bfloat16 if (use and torch.cuda.is_bf16_supported()) else torch.float16
        ctx = (lambda: torch.autocast("cuda", dtype=dtype)) if use else contextlib.nullcontext
        return ctx, _make_scaler(use and dtype == torch.float16)

    def _fit(self, stage, get_batch, steps, lr, min_lr, warm, ga, eval_every):
        """get_batch(split) -> (x, y) on device; y may hold IGNORE. Used by BOTH stages."""
        S, m = self.S, self.model
        amp, scaler = self._amp()
        decay = [p for p in m.parameters() if p.dim() >= 2]
        nodecay = [p for p in m.parameters() if p.dim() < 2]
        opt = torch.optim.AdamW([{"params": decay, "weight_decay": float(S["weight_decay"])},
                                 {"params": nodecay, "weight_decay": 0.0}], lr=lr, betas=(0.9, 0.95))

        def lr_at(it):
            if it < warm:
                return lr * (it + 1) / max(1, warm)
            p = (it - warm) / max(1, steps - warm)
            return min_lr + 0.5 * (1 + math.cos(math.pi * min(1.0, p))) * (lr - min_lr)

        @torch.no_grad()
        def evaluate():
            m.eval()
            out, n = {}, max(1, int(S["eval_iters"]))
            for sp in ("train", "val"):
                tot = 0.0
                for _ in range(n):
                    x, y = get_batch(sp)
                    with amp():
                        tot += m(x, y)[1].item()
                out[sp] = tot / n
            m.train()
            return out

        print(f"[{stage}] {steps} steps on {self.device} (Ctrl+C to stop early)...")
        m.train()
        t0 = time.time()
        try:
            for it in range(steps):
                for g in opt.param_groups:
                    g["lr"] = lr_at(it)
                loss_acc = 0.0
                for _ in range(ga):
                    x, y = get_batch("train")
                    with amp():
                        loss = m(x, y)[1] / ga
                    scaler.scale(loss).backward()
                    loss_acc += loss.item()
                scaler.unscale_(opt)
                if float(S["grad_clip"]) > 0:
                    nn.utils.clip_grad_norm_(m.parameters(), float(S["grad_clip"]))
                scaler.step(opt); scaler.update()
                opt.zero_grad(set_to_none=True)
                self.steps[stage] += 1
                if (it + 1) % 10 == 0:
                    el = time.time() - t0
                    print(f"\rstep {it + 1}/{steps}  loss {loss_acc:.4f}  lr {lr_at(it):.2e}  "
                          f"{(it + 1) / el:.2f} it/s", end="", flush=True)
                if (it + 1) % max(1, eval_every) == 0 or it + 1 == steps:
                    ev = evaluate()
                    self.hist[stage].append((self.steps[stage], ev["train"], ev["val"]))
                    print(f"\n  [eval] {stage} step {self.steps[stage]}: train {ev['train']:.4f}  val {ev['val']:.4f}")
        except KeyboardInterrupt:
            print("\nTraining interrupted.")
        print(f"\nFinished. {stage} steps so far: {self.steps[stage]}")
        m.eval()

    # ---- stage 1: pretraining -------------------------------------------------
    def _pretrain_tokens(self):
        print("Reading and tokenizing pretraining data...")
        docs = load_corpus(self.pt_files, self.S, read_text_documents)
        if not docs:
            return None
        random.Random(1337).shuffle(docs)    # so the validation tail is not just the last file
        ids = []
        for d in docs:
            ids.extend(self.tok.encode(d, allow_special=False))
            ids.append(self.tok.eos_id)
        print(f"{len(ids):,} tokens.")
        return torch.tensor(ids, dtype=torch.long)

    def pretrain(self):
        if not self._ensure_tok_and_model(self.pt_files):
            return
        if self.stage == "sft" and input("Model is already instruction-tuned; more pretraining can erase "
                                         "that. Continue? (y/N) ").lower() != "y":
            return
        S, m = self.S, self.model
        block = m.c["block_size"]
        data = self._pretrain_tokens()
        if data is None:
            print("No data."); return
        if len(data) < block * 4:
            print(f"Not enough data ({len(data)} tokens) for block size {block}. Add more text or shrink block_size.")
            return
        n_val = max(block + 1, int(len(data) * float(S["val_split"])))
        train_d, val_d = data[:-n_val], data[-n_val:]
        if len(train_d) <= block + 1:
            train_d = val_d = data
        bs = int(S["batch_size"])

        def get_batch(split):
            d = train_d if split == "train" else val_d
            ix = torch.randint(len(d) - block - 1, (bs,))
            x = torch.stack([d[i:i + block] for i in ix])
            y = torch.stack([d[i + 1:i + block + 1] for i in ix])
            return x.to(self.device), y.to(self.device)

        self._fit("pretrain", get_batch, int(S["steps"]), float(S["lr"]), float(S["min_lr"]),
                  int(S["warmup"]), int(S["grad_accum"]), int(S["eval_every"]))
        if self.stage == "new":
            self.stage = "pretrained"
        if input(f"Save to '{self.save_path}' now? (Y/n) ").lower() != "n":
            self.save()

    # ---- stage 2: instruction fine-tuning ---------------------------------------
    def _sft_examples(self, quiet=False):
        """Build masked (ids, labels) examples; deterministic train/val split by conversation."""
        S, block = self.S, self.model.c["block_size"]
        if not quiet:
            print("Reading and tokenizing instruction data...")
        convs = load_corpus(self.sft_files, S, read_conversations) if not quiet else \
            [c for p in self.sft_files for c in self._silent_read(p)]
        resp_only = bool(S["sft_response_only"])
        ex, trunc, dropped = [], 0, 0
        for msgs in convs:
            ids, lab = encode_conversation(self.tok, with_system(msgs, S["system_prompt"]), resp_only)
            if len(ids) > block + 1:
                trunc += 1
                ids, lab = ids[:block + 1], lab[:block + 1]
            if len(ids) < 2 or not any(l != IGNORE for l in lab[1:]):
                dropped += 1       # prompt filled the whole window: nothing left to learn from
                continue
            ex.append((ids, lab))
        random.Random(1337).shuffle(ex)
        n_val = max(1, int(len(ex) * float(S["sft_val_split"]))) if len(ex) >= 20 else 0
        val, train = ex[:n_val], ex[n_val:]
        if not val:
            val = train[:min(len(train), 8)]
        stats = {"total": len(ex), "truncated": trunc, "dropped": dropped,
                 "supervised": sum(sum(1 for l in lab[1:] if l != IGNORE) for _, lab in ex),
                 "tokens": sum(len(i) - 1 for i, _ in ex)}
        return train, val, stats

    def _silent_read(self, p):
        try:
            return list(read_conversations(p, self.S))
        except Exception:
            return []

    def finetune(self):
        if not self._ensure_tok_and_model(self.sft_files):
            return
        if self.stage == "new" and self.steps["pretrain"] == 0:
            print("Note: this model has NOT been pretrained. Fine-tuning from scratch works only "
                  "for narrow tasks; run stage 1 first for general text ability.")
            if input("Continue anyway? (y/N) ").lower() != "y":
                return
        S, m = self.S, self.model
        train_ex, val_ex, st = self._sft_examples()
        if not train_ex:
            print("No usable instruction examples."); return
        print(f"{st['total']} examples ({len(train_ex)} train / {len(val_ex)} val), "
              f"{st['truncated']} truncated to {m.c['block_size']} tokens, {st['dropped']} dropped.")
        if bool(S["sft_response_only"]):
            print(f"Response-only loss: {st['supervised']:,} of {st['tokens']:,} tokens "
                  f"({100 * st['supervised'] / max(1, st['tokens']):.1f}%) contribute to the loss.")
        else:
            print("Loss on ALL tokens (sft_response_only = False).")
        bs, pad = int(S["sft_batch_size"]), self.tok.eos_id
        sampler, vrng = Sampler(train_ex, seed=1), random.Random(2)
        steps, ga = int(S["sft_steps"]), int(S["sft_grad_accum"])
        print(f"~{steps * bs * ga / max(1, len(train_ex)):.1f} epochs over the training set.")

        def get_batch(split):
            batch = sampler.next(bs) if split == "train" else \
                [val_ex[i] for i in (vrng.sample(range(len(val_ex)), min(bs, len(val_ex))))]
            return collate(batch, pad, self.device)

        self._fit("sft", get_batch, steps, float(S["sft_lr"]), float(S["sft_min_lr"]),
                  int(S["sft_warmup"]), ga, int(S["sft_eval_every"]))
        self.stage = "sft"
        if input("Run the instruction-following evaluation now? (Y/n) ").lower() != "n":
            self.evaluate_instructions()
        if input(f"Save to '{self.save_path}' now? (Y/n) ").lower() != "n":
            self.save()

    # ---- generation ---------------------------------------------------------
    def _stream(self, prompt_ids, stop_ids, ban_ids=()):
        S = self.S
        out_ids, printed, text = [], "", ""
        gen = self.model.generate(prompt_ids, int(S["max_new_tokens"]), float(S["temperature"]),
                                  int(S["top_k"]), float(S["top_p"]), float(S["rep_penalty"]),
                                  stop_ids=stop_ids, ban_ids=ban_ids)
        for t in gen:
            if t in stop_ids:
                break
            out_ids.append(t)
            text = self.tok.decode(out_ids)
            safe = text[:-1] if text.endswith("\ufffd") else text
            if len(safe) > len(printed):
                print(safe[len(printed):], end="", flush=True)
                printed = safe
        if len(text) > len(printed):
            print(text[len(printed):], end="", flush=True)
        print()
        return text.strip()

    def _generate_text(self, ids, max_new, temperature):
        """Non-streaming generation for evaluation. Returns (text, ended_with_stop_token)."""
        S, tok = self.S, self.tok
        out, finished = [], False
        for t in self.model.generate(ids, max_new, temperature, int(S["top_k"]), float(S["top_p"]),
                                     float(S["rep_penalty"]), stop_ids=(tok.end_id, tok.eos_id),
                                     ban_ids=tuple(tok.role_ids.values())):
            if t in (tok.end_id, tok.eos_id):
                finished = True
                break
            out.append(t)
        return tok.decode(out).strip(), finished

    def _ready(self):
        if self.model is None or self.tok is None:
            print("Train or load a model first."); return False
        return True

    def complete(self):
        """Raw text continuation (no chat template)."""
        if not self._ready():
            return
        prompt = input("Prompt: ")
        ids = self.tok.encode(prompt)[-self.model.c["block_size"]:] or [self.tok.eos_id]
        print(prompt, end="")
        try:
            self._stream(ids, stop_ids=(self.tok.eos_id, self.tok.end_id))
        except KeyboardInterrupt:
            print("\n[stopped]")

    def chat(self):
        if not self._ready():
            return
        S, tok = self.S, self.tok
        system = S["system_prompt"]
        hist = []
        print("Chat started. Commands: /reset  /system <text>  /quit   (Ctrl+C stops a reply)")
        if self.stage != "sft":
            print("Note: this model has not been instruction-tuned yet (stage 2), so replies may be poor.")
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
                hist = []; print("[history cleared]"); continue
            if u.startswith("/system"):
                system = u[7:].strip(); hist = []; print("[system prompt set, history cleared]"); continue
            hist.append({"role": "user", "content": u})
            budget = self.model.c["block_size"] - min(int(S["max_new_tokens"]), self.model.c["block_size"] // 2)
            ids = chat_prompt_ids(tok, with_system(hist, system), budget)
            print("Bot:", end=" ", flush=True)
            try:
                reply = self._stream(ids, (tok.end_id, tok.eos_id), tuple(tok.role_ids.values()))
            except KeyboardInterrupt:
                print("\n[stopped]"); hist.pop(); continue
            hist.append({"role": "assistant", "content": reply or "..."})

    # ---- evaluation ---------------------------------------------------------
    @torch.no_grad()
    def heldout_response_loss(self):
        """Loss over assistant-response tokens of the held-out instruction examples."""
        if not self.sft_files:
            return None, 0
        _, val_ex, _ = self._sft_examples(quiet=True)
        if not val_ex:
            return None, 0
        amp, _ = self._amp()
        self.model.eval()
        tot, cnt = 0.0, 0
        cap = min(len(val_ex), 512)
        for i in range(0, cap, 16):
            x, y = collate(val_ex[i:min(i + 16, cap)], self.tok.eos_id, self.device)
            n = int((y != IGNORE).sum())
            if n:
                with amp():
                    tot += self.model(x, y)[1].item() * n
                cnt += n
        return (tot / cnt if cnt else None), cnt

    def evaluate_instructions(self):
        if not self._ready():
            return
        S, tok = self.S, self.tok
        path = S["eval_file"].strip()
        if path and os.path.isfile(path):
            items, name = load_eval_file(path), path
        else:
            if path:
                print(f"eval_file '{path}' not found - using the built-in suite.")
            items, name = builtin_suite(), "built-in suite"
        if not items:
            print("No evaluation prompts."); return
        print(f"Evaluating on {len(items)} prompts ({name}), temperature {S['eval_temperature']}...")
        budget = self.model.c["block_size"] - min(int(S["eval_max_new_tokens"]), self.model.c["block_size"] // 2)
        responses, finished = [], []
        try:
            for i, it in enumerate(items):
                msgs = with_system([{"role": "user", "content": it["prompt"]}], S["system_prompt"])
                text, fin = self._generate_text(chat_prompt_ids(tok, msgs, budget),
                                                int(S["eval_max_new_tokens"]), float(S["eval_temperature"]))
                responses.append(text); finished.append(fin)
                print(f"\r  {i + 1}/{len(items)}", end="", flush=True)
        except KeyboardInterrupt:
            print("\n[cancelled]"); return
        print()
        res = score_instruction_following(items, responses, finished)
        res["heldout_loss"], res["heldout_tokens"] = self.heldout_response_loss()
        print_eval_report(res, int(S["eval_show"]))
        summary = {k: res[k] for k in ("n_prompts", "prompt_strict", "prompt_loose", "inst_strict", "inst_loose",
                                       "stop_rate", "ref_f1", "heldout_loss")}
        summary.update({"set": name, "pretrain_steps": self.steps["pretrain"], "sft_steps": self.steps["sft"],
                        "time": time.strftime("%Y-%m-%d %H:%M:%S")})
        self.eval_log.append(summary)
        if S["eval_report"]:
            try:
                with open(S["eval_report"], "w", encoding="utf-8") as f:
                    json.dump({**summary, "by_type": res["by_type"], "details": res["details"]},
                              f, indent=2, ensure_ascii=False)
                print(f"Full report written to {S['eval_report']}")
            except OSError as e:
                print(f"Could not write report: {e}")

    # ---- save / load --------------------------------------------------------
    def save(self, path=None):
        if self.model is None or self.tok is None:
            print("Nothing to save (need tokenizer and model)."); return
        path = path or input(f"Save path [{self.save_path}]: ").strip() or self.save_path
        self.save_path = path
        tmp = path + ".tmp"
        torch.save({"version": 2, "config": self.model.c, "model": self.model.state_dict(),
                    "merges": [list(m) for m in self.tok.merges], "specials": SPECIALS, "settings": self.S,
                    "stage": self.stage, "steps": self.steps, "history": self.hist, "eval_log": self.eval_log,
                    "pt_files": self.pt_files, "sft_files": self.sft_files}, tmp)
        os.replace(tmp, path)
        print(f"Saved to {path} ({os.path.getsize(path) / 1e6:.1f} MB)")

    def load(self, path=None):
        path = path or input(f"Load path [{self.save_path}]: ").strip() or self.save_path
        if not os.path.isfile(path):
            print("File not found."); return
        try:
            ck = torch.load(path, map_location=self.device, weights_only=False)
            tok = ByteBPE(ck["merges"])
            cfg, sd = dict(ck["config"]), ck["model"]
            legacy = ck.get("version", 1) < 2
            if cfg["vocab_size"] != tok.vocab_size:
                if legacy and cfg["vocab_size"] == len(ck["merges"]) + 257:
                    cfg["vocab_size"] = tok.vocab_size
                    model = GPT(cfg)
                    grown = model.tok.weight.data.clone()
                    grown[:sd["tok.weight"].shape[0]] = sd["tok.weight"].to(grown.device)
                    sd = dict(sd); sd["tok.weight"] = grown; sd["head.weight"] = grown
                    print(f"Migrated legacy checkpoint: added {len(SPECIALS) - 1} special tokens "
                          f"(new embeddings start untrained; run stage 2 to teach them).")
                else:
                    raise ValueError("checkpoint vocabulary does not match its tokenizer")
            else:
                model = GPT(cfg)
            model = model.to(self.device)
            model.load_state_dict(sd)
            model.eval()
            self.tok, self.model = tok, model
            self.S.update({k: v for k, v in ck.get("settings", {}).items() if k in self.S})
            for k in ARCH_KEYS:
                self.S[k] = cfg[k]
            if legacy:
                n = ck.get("total_steps", 0)
                self.steps = {"pretrain": n, "sft": 0}
                self.hist = {"pretrain": ck.get("history", []), "sft": []}
                self.stage = "pretrained" if n else "new"
                self.pt_files, self.sft_files = list(ck.get("files", [])), []
                self.eval_log = []
            else:
                self.steps = ck.get("steps", {"pretrain": 0, "sft": 0})
                self.hist = ck.get("history", {"pretrain": [], "sft": []})
                self.stage = ck.get("stage", "new")
                self.pt_files, self.sft_files = list(ck.get("pt_files", [])), list(ck.get("sft_files", []))
                self.eval_log = ck.get("eval_log", [])
            self.save_path = path
            print(f"Loaded {path}: {model.n_params() / 1e6:.2f}M params, vocab {tok.vocab_size}, stage '{self.stage}' "
                  f"(pretrain {self.steps['pretrain']} steps, SFT {self.steps['sft']} steps).")
        except Exception as e:
            print(f"Failed to load: {e}")

    # ---- info / settings ----------------------------------------------------
    def info(self):
        print("\n=== AI INFO ===")
        print(f"Device          : {self.device}" + (f" ({torch.cuda.get_device_name(0)})" if self.device == "cuda" else ""))
        print(f"PyTorch         : {torch.__version__}")
        print("Tokenizer       : " + (f"byte-level BPE, vocab {self.tok.vocab_size} "
                                      f"({len(self.tok.merges)} merges + {len(SPECIALS)} special)" if self.tok else "not trained"))
        if self.model:
            c = self.model.c
            print(f"Model           : GPT, {self.model.n_params() / 1e6:.2f}M params")
            print(f"  layers/heads  : {c['n_layer']} / {c['n_head']}   embd {c['n_embd']}   "
                  f"context {c['block_size']}   dropout {c['dropout']}")
            print(f"Stage           : {self.stage}")
            for st, label in (("pretrain", "Pretraining"), ("sft", "Instruction SFT")):
                line = f"{label:16s}: {self.steps[st]} steps"
                if self.hist[st]:
                    s, tl, vl = self.hist[st][-1]
                    line += f"   last eval step {s}: train {tl:.4f}  val {vl:.4f}  (ppl {math.exp(min(vl, 20)):.1f})"
                print(line)
            if self.eval_log:
                e = self.eval_log[-1]
                print(f"Last IF eval    : prompt-strict {_pc(e.get('prompt_strict'))}  inst-strict "
                      f"{_pc(e.get('inst_strict'))}  ({e.get('set')}, {e.get('time')})")
        else:
            print("Model           : none")
        print(f"Pretrain files  : {len(self.pt_files)}    Instruction files: {len(self.sft_files)}")
        print(f"Save path       : {self.save_path}")

    def settings_menu(self):
        groups = [("Model architecture (new models only)", ("n_layer", "n_head", "n_embd", "block_size", "dropout")),
                  ("Tokenizer", ("vocab_size", "tok_train_chars")),
                  ("Stage 1 pretraining (weight_decay, grad_clip, eval_iters, mixed_precision are shared)",
                   ("steps", "batch_size", "grad_accum", "lr", "min_lr", "warmup", "weight_decay",
                    "grad_clip", "eval_every", "eval_iters", "val_split", "mixed_precision")),
                  ("Stage 2 instruction fine-tuning",
                   ("sft_steps", "sft_batch_size", "sft_grad_accum", "sft_lr", "sft_min_lr", "sft_warmup",
                    "sft_eval_every", "sft_val_split", "sft_response_only")),
                  ("Data parsing (comma-separate several; CSV accepts names or 0-based numbers)",
                   ("csv_prompt_col", "csv_response_col", "csv_text_col", "json_prompt_key", "json_response_key", "json_key")),
                  ("Generation / chat", ("temperature", "top_k", "top_p", "rep_penalty", "max_new_tokens", "system_prompt")),
                  ("Evaluation", ("eval_file", "eval_max_new_tokens", "eval_temperature", "eval_show", "eval_report"))]
        while True:
            print("\n=== SETTINGS ===")
            keys = []
            for title, ks in groups:
                print(f"-- {title}")
                for k in ks:
                    keys.append(k)
                    print(f"  {len(keys):2d}. {k:20s} = {self.S[k]!r}")
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
        ("Add PRETRAINING data (.txt / .csv / .jsonl, folders, globs)", app.add_pretrain_files),
        ("Add INSTRUCTION data (.csv / .jsonl prompt-response or chat)", app.add_sft_files),
        ("List data files", app.list_files),
        ("Clear data files", app.clear_files),
        ("Set up CSV / JSONL prompt & response columns", app.column_setup),
        ("Preview data (shows which tokens are trained on)", app.preview),
        ("Train / retrain tokenizer (byte-level BPE + special tokens)", app.train_tokenizer),
        ("STAGE 1: Pretrain (raw text, loss on all tokens)", app.pretrain),
        ("STAGE 2: Instruction fine-tune (loss on responses only)", app.finetune),
        ("Evaluate instruction following", app.evaluate_instructions),
        ("Chat", app.chat),
        ("Text completion (raw prompt)", app.complete),
        ("Settings", app.settings_menu),
        ("Save model", app.save),
        ("Load model", app.load),
        ("AI info", app.info),
        ("Discard model (keep tokenizer)", app.reset_model),
    ]
    while True:
        print("\n==================== Sylvie LLM ====================")
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
