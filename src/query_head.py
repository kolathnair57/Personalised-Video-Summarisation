"""Phase 8: query-conditioning modification to the DSNet scoring head.

Encodes the persona query with CLIP's text encoder and fuses it into the frame
features before DSNet's existing scoring layers (concat by default, FiLM and
cross-attention as Axis-1 alternatives). Nothing else in DSNet changes.

Not yet implemented.
"""
