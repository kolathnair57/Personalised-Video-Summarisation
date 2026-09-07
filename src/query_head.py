
import argparse
import json
from pathlib import Path

import numpy as np
import torch
from torch import nn

ROOT = Path(__file__).resolve().parents[1]


class QueryConditioner(nn.Module):
    """Fuse a query embedding into per-frame features. Identity at initialisation."""

    def __init__(self, feat_dim: int = 512, query_dim: int = 512, mode: str = "concat"):
        super().__init__()
        self.mode = mode
        self.feat_dim = feat_dim
        self.query_dim = query_dim

        if mode == "concat":
            self.proj = nn.Linear(feat_dim + query_dim, feat_dim)
            with torch.no_grad():
                self.proj.weight.zero_()
                self.proj.weight[:, :feat_dim].copy_(torch.eye(feat_dim))
                self.proj.bias.zero_()
        elif mode == "film":
            self.gamma = nn.Linear(query_dim, feat_dim)
            self.beta = nn.Linear(query_dim, feat_dim)
            with torch.no_grad():
                self.gamma.weight.zero_(); self.gamma.bias.fill_(1.0)
                self.beta.weight.zero_(); self.beta.bias.zero_()
        elif mode == "none":
            pass                                   # explicit no-op, for ablations
        else:
            raise ValueError(f"unknown mode {mode!r}")

    def forward(self, frame_feats: torch.Tensor, query_emb: torch.Tensor) -> torch.Tensor:
        """frame_feats: (B, T, D) or (T, D);  query_emb: (D,), (1, D) or (B, D)."""
        if self.mode == "none" or query_emb is None:
            return frame_feats
        squeeze = frame_feats.dim() == 2
        if squeeze:
            frame_feats = frame_feats.unsqueeze(0)
        B, T, D = frame_feats.shape

        q = query_emb
        if q.dim() == 1:
            q = q.unsqueeze(0)
        q = q.to(frame_feats.dtype).to(frame_feats.device)
        if q.shape[0] == 1 and B > 1:
            q = q.expand(B, -1)
        q = q.unsqueeze(1).expand(B, T, q.shape[-1])       # (B, T, Dq)

        if self.mode == "concat":
            out = self.proj(torch.cat([frame_feats, q], dim=-1))
        else:                                               # film
            out = self.gamma(q) * frame_feats + self.beta(q)
        return out.squeeze(0) if squeeze else out


# --------------------------------------------------------------------------------------
# Precompute CLIP text embeddings for every persona query.
# --------------------------------------------------------------------------------------
def precompute(out_path, model_name="ViT-B-32", pretrained="laion2b_s34b_b79k",
               pool_path=None):
    import open_clip
    pool = json.loads(Path(pool_path or ROOT / "personas/pool.json").read_text())
    people = [p for v in pool.values() for p in v]
    ids = [p["persona_id"] for p in people]
    queries = [p["preference_query"] for p in people]

    tok = open_clip.get_tokenizer(model_name)
    model, _, _ = open_clip.create_model_and_transforms(model_name, pretrained=pretrained)
    dev = "cuda" if torch.cuda.is_available() else "cpu"
    model = model.eval().to(dev)
    with torch.no_grad():
        emb = model.encode_text(tok(queries).to(dev)).float()
        emb = emb / emb.norm(dim=-1, keepdim=True)          # unit norm, like image feats
    E = emb.cpu().numpy().astype(np.float32)

    Path(out_path).parent.mkdir(parents=True, exist_ok=True)
    np.savez(out_path, persona_ids=np.array(ids), embeddings=E,
             model=f"{model_name}/{pretrained}")
    print(f"wrote {out_path}: {E.shape} embeddings for {len(ids)} personas "
          f"({model_name}/{pretrained})")
    sim = E @ E.T
    iu = np.triu_indices(len(E), 1)
    print(f"  pairwise cosine: mean {sim[iu].mean():.3f}  max {sim[iu].max():.3f}")
    return E


def load_query_embeddings(path=None):
    """-> dict persona_id -> np.ndarray(D,)"""
    d = np.load(path or ROOT / "data/query_emb.npz", allow_pickle=True)
    return {str(k): v for k, v in zip(d["persona_ids"], d["embeddings"])}


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--precompute", action="store_true")
    ap.add_argument("--out", default=str(ROOT / "data/query_emb.npz"))
    ap.add_argument("--model", default="ViT-B-32")
    ap.add_argument("--pretrained", default="laion2b_s34b_b79k")
    a = ap.parse_args()
    if a.precompute:
        precompute(a.out, a.model, a.pretrained)
    else:
        ap.print_help()
