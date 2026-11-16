"""Shared helpers: device constants, exact-size pinned host memory, a synchronizing timer, sampling."""

import os
import time
from collections import defaultdict

import torch

os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DEFAULT_PROMPT = os.path.join(REPO_ROOT, "data", "prompt.txt")
GiB = 2**30
DEV = "cuda"


def mem_available_bytes():
    """MemAvailable from /proc/meminfo (Linux)."""
    with open("/proc/meminfo") as f:
        for line in f:
            if line.startswith("MemAvailable"):
                return int(line.split()[1]) * 1024
    return 0


_PINNED = {}  # data_ptr -> mmap, keeps the mapping alive


def pinned_empty(shape, dtype):
    """Allocate page-locked host memory of exactly the requested size.

    torch.empty(..., pin_memory=True) rounds each allocation up to a power of two
    (0.625 GiB becomes 1 GiB), which costs ~12 GiB extra at 32 beams. Instead we
    mmap anonymous memory and register it with cudaHostRegister.
    """
    import mmap

    n = 1
    for d in shape:
        n *= d
    nbytes = n * torch.empty(0, dtype=dtype).element_size()
    mm = mmap.mmap(-1, nbytes)
    t = torch.frombuffer(mm, dtype=dtype).view(*shape)
    err = torch.cuda.cudart().cudaHostRegister(t.data_ptr(), nbytes, 0)
    if int(err) != 0:
        raise MemoryError(f"cudaHostRegister failed: {err}")
    _PINNED[t.data_ptr()] = mm
    return t


def pinned_free(t):
    ptr = t.data_ptr()
    if ptr in _PINNED:
        torch.cuda.cudart().cudaHostUnregister(ptr)
        del _PINNED[ptr]


class Timer:
    """switch(name): wait for the GPU, charge the elapsed time to the previous bucket, start `name`."""

    def __init__(self):
        self.t = defaultdict(float)
        self.cur = None
        self.last = None

    def switch(self, name):
        torch.cuda.synchronize()
        now = time.perf_counter()
        if self.cur is not None:
            self.t[self.cur] += now - self.last
        self.cur, self.last = name, now


def sample(logits, gumbel, forced=None):
    """Gumbel-max sampling: token = argmax(log-prob + pre-generated Gumbel noise), i.e. a draw from softmax(logits).

    If `forced` is given, that token is used instead (for comparing two runs on the same tree).
    Returns (token the sampler would pick, token actually used, log-prob of the used token).
    """
    lp = torch.log_softmax(logits.float(), dim=-1)
    own = (lp + gumbel).argmax(-1)
    used = own if forced is None else forced
    return own, used, lp.gather(1, used[:, None]).squeeze(1).double()


def read_prompt_ids(tokenizer, n_tokens, path=None):
    """Tokenize the prompt text and keep the first n_tokens ids."""
    with open(path or DEFAULT_PROMPT) as f:
        ids = tokenizer(f.read())["input_ids"]
    assert len(ids) >= n_tokens, f"the prompt has only {len(ids)} tokens"
    return ids[:n_tokens]
