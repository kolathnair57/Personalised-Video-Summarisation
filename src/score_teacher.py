"""Phase 7.1: score each shot for each persona with Qwen3-VL (cached, resumable).

Reads the persona pool and the h5 change_points, sends one representative frame
per shot plus the persona query to the local vLLM server, and writes the
per-shot importances to labels_cache/ keyed by (video, persona, model, prompt).

Not yet implemented.
"""
