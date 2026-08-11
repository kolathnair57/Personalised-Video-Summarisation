"""Phase 7.2: broadcast teacher shot scores to gtscore and write persona h5 files.

Copies a base h5, replaces gtscore with the persona-conditioned scores
(broadcast from shots to the picks inside each change_points span), and attaches
the query / persona_id attributes. Everything else is left untouched.

Not yet implemented.
"""
