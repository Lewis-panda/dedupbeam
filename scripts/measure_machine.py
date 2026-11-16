"""Measure the machine: PCIe link, pinned host<->GPU bandwidth, how much host memory can be pinned, free GPU memory
after loading OPT-2.7B, one large FP16 matmul. Writes results/machine.json. Usage: python scripts/measure_machine.py"""

import datetime, gc, json, os, statistics, subprocess, threading, time

import _common  # noqa: F401  (sets up paths)
import torch

from dedupbeam.common import GiB, mem_available_bytes, pinned_empty, pinned_free
from dedupbeam.opt import OPT

MiB = 2**20
out = {}


def smi(*a):
    return subprocess.run(
        ["nvidia-smi", *a], capture_output=True, text=True, timeout=20
    ).stdout.strip()


def pci_dir():
    bus = (
        smi("--query-gpu=pci.bus_id", "--format=csv,noheader")
        .splitlines()[0]
        .strip()
        .lower()
    )  # 00000000:01:00.0
    return "/sys/bus/pci/devices/" + bus[-12:]


PCI = pci_dir()


def sysfs(name):
    try:
        return open(f"{PCI}/{name}").read().strip()
    except Exception as e:
        return f"not readable ({e.__class__.__name__})"


def smi_link():
    try:
        lines = [l.strip() for l in smi("-q").splitlines()]
        i = lines.index("GPU Link Info")
        return " / ".join(l.replace(" ", "") for l in lines[i + 1 : i + 10] if ":" in l)
    except Exception as e:
        return f"not readable ({e.__class__.__name__})"


def med(fn, n, warm=2):
    for _ in range(warm):
        fn()
    ts = []
    for _ in range(n):
        torch.cuda.synchronize()
        t = time.perf_counter()
        fn()
        torch.cuda.synchronize()
        ts.append(time.perf_counter() - t)
    return statistics.median(ts), min(ts), max(ts)


# ---------- 1. PCIe link (idle) ----------
out["pcie_idle"] = dict(
    current_link_speed=sysfs("current_link_speed"),
    current_link_width=sysfs("current_link_width"),
    max_link_speed=sysfs("max_link_speed"),
    max_link_width=sysfs("max_link_width"),
    nvidia_smi=smi_link(),
)
try:
    lspci = subprocess.run(
        ["lspci", "-vv", "-s", PCI[-7:]], capture_output=True, text=True, timeout=20
    ).stdout
    lnk = [l.strip() for l in lspci.splitlines() if "LnkSta:" in l or "LnkCap:" in l]
    out["lspci"] = (
        lnk if lnk else "not readable (lspci shows LnkCap/LnkSta only with root)"
    )
except Exception as e:
    out["lspci"] = f"not readable ({e.__class__.__name__})"

# ---------- 2. how much host memory can be pinned (0.625 GiB blocks = one OPT-2.7B beam at 2048 tokens) ----------
torch.cuda.init()
avail0 = mem_available_bytes()
blocks, t0, stopped = [], time.perf_counter(), None
block_elems = int(0.625 * GiB) // 2
for i in range(32):
    if mem_available_bytes() < 4 * GiB + block_elems * 2:
        stopped = f"stopped at block {i}: MemAvailable would drop below 4 GiB"
        break
    try:
        blocks.append(pinned_empty((block_elems,), torch.half))
    except Exception as e:
        stopped = f"failed at block {i}: {e.__class__.__name__}"
        break
out["pin_test"] = dict(
    mem_total_gib=os.sysconf("SC_PAGE_SIZE") * os.sysconf("SC_PHYS_PAGES") / GiB,
    mem_available_before_gib=avail0 / GiB,
    block_gib=0.625,
    blocks_pinned=len(blocks),
    pinned_gib=len(blocks) * 0.625,
    seconds=time.perf_counter() - t0,
    mem_available_while_pinned_gib=mem_available_bytes() / GiB,
    stopped=stopped,
)
for b in blocks:
    pinned_free(b)
del blocks
gc.collect()
a = mem_available_bytes()
tmp = torch.empty(
    block_elems, dtype=torch.half, pin_memory=True
)  # PyTorch's allocator rounds up to a power of two
out["pin_test"]["torch_pin_memory_one_block_costs_gib"] = (
    a - mem_available_bytes()
) / GiB
del tmp
gc.collect()

# ---------- 3. host <-> GPU bandwidth ----------
link_seen, stop = set(), False


def watch():
    while not stop:
        link_seen.add((sysfs("current_link_speed"), sysfs("current_link_width")))
        time.sleep(0.05)


th = threading.Thread(target=watch)
th.start()
xfer = {}
for label, nbytes, reps in [
    ("1 GiB", GiB, 30),
    ("320 MiB", 320 * MiB, 30),
    ("84 MiB", 84 * MiB, 50),
    ("10 MiB", 10 * MiB, 100),
]:
    src = torch.empty(nbytes // 2, dtype=torch.half, pin_memory=True).fill_(1)
    dst = torch.empty(nbytes // 2, dtype=torch.half, device="cuda")
    m_, lo, hi = med(lambda: dst.copy_(src), reps)
    xfer[f"h2d_pinned_{label}"] = dict(
        bytes=nbytes,
        median_s=m_,
        gib_per_s=nbytes / GiB / m_,
        gb_per_s=nbytes / 1e9 / m_,
        best_gib_per_s=nbytes / GiB / lo,
        worst_gib_per_s=nbytes / GiB / hi,
        reps=reps,
    )
    if label == "1 GiB":
        smi_during = smi_link()
        m_, lo, hi = med(lambda: src.copy_(dst), reps)
        xfer["d2h_pinned_1 GiB"] = dict(
            bytes=nbytes,
            median_s=m_,
            gib_per_s=nbytes / GiB / m_,
            gb_per_s=nbytes / 1e9 / m_,
            reps=reps,
        )
        pg = torch.ones(nbytes // 2, dtype=torch.half)
        m_, lo, hi = med(lambda: dst.copy_(pg), 10)
        xfer["h2d_pageable_1 GiB"] = dict(
            bytes=nbytes,
            median_s=m_,
            gib_per_s=nbytes / GiB / m_,
            gb_per_s=nbytes / 1e9 / m_,
            reps=10,
        )
        del pg
    del src, dst
stop = True
th.join()
out["transfer"] = xfer
out["pcie_during_transfer"] = dict(sysfs_seen=sorted(link_seen), nvidia_smi=smi_during)
torch.cuda.empty_cache()

# ---------- 4. free GPU memory after loading FP16 OPT-2.7B ----------
free0, total = torch.cuda.mem_get_info()
m = OPT(torch.float16)
torch.cuda.synchronize()
gc.collect()
torch.cuda.empty_cache()
free1, _ = torch.cuda.mem_get_info()
out["gpu_memory"] = dict(
    total_mib=total / MiB,
    free_before_model_mib=free0 / MiB,
    free_after_model_mib=free1 / MiB,
    model_allocated_mib=torch.cuda.memory_allocated() / MiB,
    n_params=sum(p.numel() for p in m.hf.parameters()),
)

# ---------- 5. FP16 matmul (131072 x 2560) x (2560 x 2560), and one decode layer at 32 beams ----------
with torch.inference_mode():
    a = torch.randn(131072, 2560, dtype=torch.half, device="cuda")
    b = torch.randn(2560, 2560, dtype=torch.half, device="cuda")
    m_, lo, hi = med(lambda: a @ b, 20, warm=3)
    flops = 2 * 131072 * 2560 * 2560
    out["matmul"] = dict(
        median_s=m_, best_s=lo, worst_s=hi, flops=flops, tflops=flops / m_ / 1e12
    )
    N, s = 32, 2047
    K = torch.randn(s + 1, N, m.E, dtype=torch.half, device="cuda") * 0.1
    V = torch.randn(s + 1, N, m.E, dtype=torch.half, device="cuda") * 0.1
    h = torch.randn(N, m.E, dtype=torch.half, device="cuda")
    m_, lo, hi = med(lambda: m.layer(0, h, K, V, s), 30, warm=3)
    out["one_layer_32beams_s2047"] = dict(
        median_s=m_,
        times_32_layers_s=m_ * 32,
        kv_gib_read=2 * (s + 1) * N * m.E * 2 / GiB,
    )

out["versions"] = dict(
    torch=torch.__version__,
    cuda=torch.version.cuda,
    gpu=torch.cuda.get_device_name(0),
    driver=smi("--query-gpu=driver_version", "--format=csv,noheader"),
    date=datetime.datetime.now().isoformat(timespec="seconds"),
)
json.dump(out, open("results/machine.json", "w"), indent=1)
print(json.dumps(out, indent=1))
