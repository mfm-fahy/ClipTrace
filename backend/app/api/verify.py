import json
import aiofiles
import uuid
from fastapi import APIRouter, UploadFile, File, Depends, HTTPException
from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy import select

from app.db.database import get_db, Segment, Video
from app.processing.segmenter import cleanup_audio_files
from app.matching.engine import match_segments
from app.core.config import UPLOAD_DIR
from app.api.processing_helper import process_video

router = APIRouter()


@router.post("/clip")
async def verify_clip(
    file: UploadFile = File(...),
    db: AsyncSession = Depends(get_db),
):
    tmp_path = UPLOAD_DIR / f"verify_{uuid.uuid4()}_{file.filename}"
    async with aiofiles.open(tmp_path, "wb") as f:
        await f.write(await file.read())

    try:
        segments, embeddings, _ = await process_video(str(tmp_path))
    except Exception as e:
        tmp_path.unlink(missing_ok=True)
        raise HTTPException(status_code=400, detail=str(e))

    matches = await match_segments(embeddings, db)
    cleanup_audio_files(segments)
    tmp_path.unlink(missing_ok=True)

    if not matches:
        return {
            "is_original": True,
            "status": "no_match",
            "message": "No matching source found. Content appears original or unregistered.",
            "sources": [],
        }

    top = matches[0]
    is_edited = top["confidence"] < 0.95
    return {
        "is_original": not is_edited,
        "status": "edited" if is_edited else "exact_match",
        "message": (
            f"Content matched to '{top['title']}' with {top['confidence']*100:.1f}% confidence."
            + (" Content appears to have been edited." if is_edited else "")
        ),
        "sources": matches,
        "is_mixed_content": any(m["is_mixed"] for m in matches),
    }


@router.get("/chain/{video_id}")
async def verify_video_chain(video_id: str, db: AsyncSession = Depends(get_db)):
    video = await db.get(Video, video_id)
    if not video:
        raise HTTPException(status_code=404, detail="Video not found")

    result = await db.execute(
        select(Segment).where(Segment.video_id == video_id).order_by(Segment.segment_index)
    )
    segs = result.scalars().all()
    if not segs:
        raise HTTPException(status_code=404, detail="No segments found for this video")

    hashes = [s.chain_hash for s in segs if s.chain_hash]
    is_continuous = len(hashes) == len(segs) and len(set(hashes)) == len(hashes)

    return {
        "video_id": video_id,
        "title": video.title,
        "chain_root_hash": video.chain_root_hash,
        "total_segments": len(segs),
        "chain_intact": is_continuous,
        "status": "verified" if is_continuous else "tampered",
    }
