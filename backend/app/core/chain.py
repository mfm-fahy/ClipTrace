"""
Capture-Time Proof Module
Generates a tamper-evident hash chain for a video at registration time.
Each segment's proof links to the previous one, making any modification detectable.
"""
import hashlib
import hmac
import json
import time
from app.core.config import SECRET_KEY


def _sha256(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def generate_segment_proof(
    video_id: str,
    segment_index: int,
    fingerprint_hex: str,
    prev_hash: str,
) -> dict:
    """Create a single link in the hash chain for one segment."""
    payload = json.dumps({
        "video_id": video_id,
        "seg": segment_index,
        "fp": fingerprint_hex,
        "prev": prev_hash,
        "ts": int(time.time()),
    }, sort_keys=True).encode()

    chain_hash = _sha256(payload)
    signature = hmac.new(SECRET_KEY.encode(), chain_hash.encode(), hashlib.sha256).hexdigest()

    return {
        "chain_hash": chain_hash,
        "signature": signature,
        "payload": payload.decode(),
    }


def build_chain(video_id: str, fingerprints: list[str]) -> list[dict]:
    """Build a full hash chain for all segments of a video."""
    chain = []
    prev_hash = "0" * 64  # genesis block
    for idx, fp in enumerate(fingerprints):
        proof = generate_segment_proof(video_id, idx, fp, prev_hash)
        chain.append(proof)
        prev_hash = proof["chain_hash"]
    return chain


def verify_chain(chain: list[dict]) -> tuple[bool, str]:
    """Verify integrity of a stored hash chain. Returns (valid, reason)."""
    prev_hash = "0" * 64
    for idx, link in enumerate(chain):
        payload = json.loads(link["payload"])
        if payload["prev"] != prev_hash:
            return False, f"Chain broken at segment {idx}: prev_hash mismatch"

        expected_hash = _sha256(link["payload"].encode())
        if expected_hash != link["chain_hash"]:
            return False, f"Hash tampered at segment {idx}"

        expected_sig = hmac.new(
            SECRET_KEY.encode(), link["chain_hash"].encode(), hashlib.sha256
        ).hexdigest()
        if expected_sig != link["signature"]:
            return False, f"Signature invalid at segment {idx}"

        prev_hash = link["chain_hash"]

    return True, "Chain intact"
