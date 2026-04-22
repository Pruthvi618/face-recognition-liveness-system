"""Shared math helpers for face embedding comparison.

Why this file exists:
- Keep small utility logic in one place so matching behavior stays consistent.
"""

import numpy as np

def cosine_similarity(a, b):
    """Return cosine similarity between two embedding vectors.

    Why we need this:
    - Face models output high-dimensional vectors.
    - Cosine similarity is a standard way to measure how close two embeddings are.
    """
    a = np.array(a)
    b = np.array(b)
    return np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b))
