# Phase 8 — Query conditioning for DSNet

Date: 2026-08-12.

## What changed

`src/query_head.py` adds `QueryConditioner`, fused into the frame features at the top of
DSNet's `forward`. The persona query is encoded once by CLIP's **text** encoder; CLIP's
shared image/text space is what makes a query comparable to a frame feature at all
(the reason Phase 4 exists).

Total change to DSNet for this phase: **18 insertions, 6 deletions** across two files.
Cumulative DSNet diff including the Phase 2 compatibility renames: 32 insertions,
20 deletions across 5 files. Patch: `results/phase8/dsnet_query_head.patch`
(base commit `1804176`).

The architectural delta is kept minimal on purpose. The contribution of this project is
the **supervision**; a larger architectural change would make it impossible to attribute
a Phase 9 result to persona labels rather than to a better network.

### Insertion

```python
def forward(self, x, query_emb=None):
    qcond = getattr(self, 'qcond', None)
    if qcond is not None and query_emb is not None:
        x = qcond(x, query_emb)
    ...                       # everything below is untouched DSNet
```

`__init__` is not modified; the conditioner is attached externally
(`model.qcond = QueryConditioner(...)`). So an unmodified DSNet, or one called without a
query, is byte-for-byte the original code path — which is what keeps the Phase 2/3/4
baselines valid.

## Identity initialisation — a correction to the runbook

The runbook's checkpoint says to verify the original model is recovered "when
`query_emb=0` and `mode="concat"` with zero-init proj bias". **That cannot hold.** For
concat, `proj([f ; 0]) = W_f f + b`; zeroing only `b` leaves `W_f f`, and `W_f` is
randomly initialised — a random linear map of the features, not the features.

Both modes are therefore initialised to the exact identity:

| mode | init | at init |
|---|---|---|
| concat | `W = [I \| 0]`, `b = 0` | `proj([f ; q]) = f` for any `q` |
| film | `gamma` weights 0 / bias 1, `beta` weights 0 / bias 0 | `1*f + 0 = f` |

Why this is better than what was asked for:

* the checkpoint becomes genuinely verifiable — **bit-identical**, not approximately equal;
* training starts from exactly the Phase 2 baseline rather than a randomly perturbed
  version of it, so the optimiser never has to first undo random damage;
* if persona conditioning helps nothing, the model can remain at identity — a clean null
  result instead of a confounded one.

## Checkpoint results

```
=== anchor-based ===
  concat identity-init: max|diff| = 0.000e+00   bit-identical: True
  film   identity-init: max|diff| = 0.000e+00   bit-identical: True
  gradient reaches the QUERY half of proj: 2.11e+04  (yes)
=== anchor-free ===
  concat identity-init: max|diff| = 0.000e+00   bit-identical: True
  film   identity-init: max|diff| = 0.000e+00   bit-identical: True
  gradient reaches the QUERY half of proj: 1.47e+03  (yes)
```

The gradient check matters as much as the identity check: an identity init could have
been **dead** (zero gradient on the query path, query permanently ignored). It is not,
because the query input is non-zero, so `dL/dW_q = delta (x) q != 0`.

End-to-end on real persona data (two `tvsum:VT` personas, same video):

```
at identity init, two personas give identical output: True   (query ignored until trained)
after perturbing ONLY the query weights: max|diff| = 5.99e-03
  -> the architecture CAN express persona-dependent predictions
```

## Query embeddings

```
data/query_emb.npz   (128, 512) float32, unit-normalised, ViT-B-32/laion2b_s34b_b79k
                     pairwise cosine mean 0.352, max 0.849
```

Unit-normalised to match the scale of CLIP image features. The similarity statistics match
the Phase 6 audit exactly, as they should — same encoder, same strings.

Keyed by `persona_id`, which is also stored as an attr on every persona h5 group, so
Phase 9's loader can join them without any filename parsing.

## Reproduce

```bash
source env/activate-train.sh
python src/query_head.py --precompute --out data/query_emb.npz
# verification
python - <<'EOF'
import sys, torch; sys.path[:0]=["third_party/DSNet/src","src"]
from modules.model_zoo import get_model
from query_head import QueryConditioner
m = get_model("anchor-based", base_model="attention", num_feature=512,
              num_hidden=128, num_head=8, anchor_scales=[4,8,16,32]).eval()
x, q = torch.randn(1,40,512), torch.randn(512)
base = m(x); m.qcond = QueryConditioner(512,512,"concat").eval()
assert all(torch.equal(a,b) for a,b in zip(base, m(x,q)))
print("identity check OK")
EOF
```

## For Phase 9

- `--num-feature 512` on every run (CLIP features).
- The DSNet data loader must be extended to return the query embedding per sample:
  read `persona_id` from the h5 group attrs, look it up in `data/query_emb.npz`, and pass
  it to `model(x, query_emb)` / `model.predict(seq, query_emb)`.
- `mode` is the Axis-1 ablation: `concat` (default) vs `film` vs `none`.
  `none` with persona labels isolates "supervision" from "query".
- Planned replacement for the dropped CLIP-It control (Phase 3): the same
  query-conditioned model trained on the **stock generic** `gtscore`. Same architecture,
  query head on, generic supervision — isolates "reads a query" from "persona
  supervision" within one model rather than across two codebases.
