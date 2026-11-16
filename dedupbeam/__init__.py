"""DedupBeam: shared-prefix KV deduplication for step-wise beam search on a consumer GPU.

Modules:
  common    device constants, exact-size pinned host memory, timer, Gumbel-max sampling
  opt       OPT-2.7B decoder with an explicit per-layer KV interface
  gqa       Qwen2.5 (GQA) decoder with the same interface
  perbeam   PerBeamEngine: one KV copy per beam; modes ingpu / baseline (layer-wise offloading) / lbs
  segbook   bookkeeping of shared KV segments, plus GPU-free replay of a beam tree
  dedup     DedupEngine: one copy per shared segment, pool + mask attention; modes baseline / lbs
  predict   the LBS paper's transfer-volume model (Eq. 1-5)
"""

__version__ = "0.1.0"
