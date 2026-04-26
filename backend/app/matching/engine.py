"""
Matching Engine
Compares query segment embeddings against the FAISS index,
aggregates results by video, applies temporal continuity bonuses,
and returns ranked source matches with confidence scores.
"""
import numpy as np
import json
from collections import defaultdict

from app.matching.vector_index import get_index
from app.core.config import SIMILARITY_THRESHOLD, MIN_MATCHING_SEGMENTS, CONFIDENCE_CONTINUITY_BONUS


async def match_segments(
    query_embeddings: list[np.ndarray],
    db,
    top_k: int = 10,
) -> list[dict]:
    """
    Match a list of query embeddings (one per query segment) against the index.
    Returns a ranked list of source matches.
    """
    index = get_index()
    if index.size == 0:
        return []

    # video_id → list of (query_seg_idx, db_seg_id, score)
    video_hits: dict[str, list[tuple[int, str, float]]] = defaultdict(list)

    for q_idx, emb in enumerate(query_embeddings):
        results = index.search(emb, top_k=top_k)
        for seg_id, score in results:
            if score >= SIMILARITY_THRESHOLD:
                video_hits[_video_id_for_segment(seg_id)].append((q_idx, seg_id, score))

    if not video_hits:
        return []

    # Fetch segment metadata for all matched segment IDs
    all_seg_ids = [sid for hits in video_hits.values() for _, sid, _ in hits]
    seg_rows = await _fetch_segments(all_seg_ids, db)
    seg_map = {row["id"]: row for row in seg_rows}

    matches = []
    for video_id, hits in video_hits.items():
        if len(hits) < MIN_MATCHING_SEGMENTS:
            continue

        video_row = await db["videos"].find_one({"id": video_id})
        if not video_row:
            continue

        confidence, matched_timestamps = _compute_confidence(hits, seg_map, len(query_embeddings))

        matches.append({
            "video_id": video_id,
            "title": video_row["title"],
            "owner": video_row["owner"],
            "confidence": round(confidence, 4),
            "matched_segments": len(hits),
            "matched_timestamps": matched_timestamps,
            "is_mixed": False,  # updated below
        })

    matches.sort(key=lambda x: x["confidence"], reverse=True)

    # Detect mixed-content: multiple sources each covering distinct query segments
    matches = _flag_mixed_content(matches, video_hits, len(query_embeddings))

    return matches[:5]  # return top 5 sources


def _video_id_for_segment(seg_id: str) -> str:
    """Lookup video_id from the in-memory segment→video map (populated at index build time)."""
    return _seg_to_video.get(seg_id, "unknown")


# Populated when index is rebuilt
_seg_to_video: dict[str, str] = {}


def register_segment_video(seg_id: str, video_id: str):
    _seg_to_video[seg_id] = video_id


def clear_segment_video_map():
    _seg_to_video.clear()


async def _fetch_segments(seg_ids: list[str], db) -> list[dict]:
    if not seg_ids:
        return []
    cursor = db["segments"].find({"id": {"$in": seg_ids}})
    return await cursor.to_list(length=None)


def _compute_confidence(
    hits: list[tuple[int, str, float]],
    seg_map: dict[str, dict],
    total_query_segs: int,
) -> tuple[float, list[dict]]:
    """
    Confidence = base_score × coverage_ratio × continuity_multiplier
    """
    base_score = np.mean([s for _, _, s in hits])
    coverage = len(hits) / max(total_query_segs, 1)

    # Temporal continuity: reward consecutive matching segments
    q_indices = sorted(set(q for q, _, _ in hits))
    continuity_bonus = 0.0
    for i in range(1, len(q_indices)):
        if q_indices[i] - q_indices[i - 1] == 1:
            continuity_bonus += CONFIDENCE_CONTINUITY_BONUS

    confidence = min(1.0, float(base_score) * coverage + continuity_bonus)

    matched_timestamps = []
    for q_idx, seg_id, score in sorted(hits, key=lambda x: x[0]):
        seg = seg_map.get(seg_id)
        if seg:
            matched_timestamps.append({
                "query_time": q_idx,
                "source_start": seg["timestamp_start"],
                "source_end": seg["timestamp_end"],
                "score": round(float(score), 4),
            })

    return confidence, matched_timestamps


def _flag_mixed_content(
    matches: list[dict],
    video_hits: dict[str, list[tuple[int, str, float]]],
    total_query_segs: int,
) -> list[dict]:
    """Mark results as mixed-content when multiple sources cover distinct query segments."""
    if len(matches) < 2:
        return matches

    coverage_sets = {}
    for m in matches:
        vid = m["video_id"]
        coverage_sets[vid] = set(q for q, _, _ in video_hits[vid])

    # Check pairwise overlap
    vids = list(coverage_sets.keys())
    for i in range(len(vids)):
        for j in range(i + 1, len(vids)):
            overlap = coverage_sets[vids[i]] & coverage_sets[vids[j]]
            union = coverage_sets[vids[i]] | coverage_sets[vids[j]]
            if len(overlap) / max(len(union), 1) < 0.3:
                # Low overlap → genuinely different segments → mixed content
                for m in matches:
                    if m["video_id"] in (vids[i], vids[j]):
                        m["is_mixed"] = True

    return matches
