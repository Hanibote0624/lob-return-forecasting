"""Small artifact fingerprints shared by packing, training, and inference."""

import hashlib


def sha256_file(path: str) -> str:
    digest = hashlib.sha256()
    with open(path, "rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def require_fingerprint(path: str, expected: str) -> str:
    actual = sha256_file(path)
    if not expected or actual != expected:
        raise ValueError(f"artifact fingerprint mismatch: {path}")
    return actual
