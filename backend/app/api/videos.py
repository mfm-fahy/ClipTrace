import uuid
import time
import json
import aiofiles
import numpy as np
from fastapi import APIRouter, UploadFile, File, Form, Depends, HTTPException
from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy import select

from app.db.database import get_db, Video, Segment, MonetizationRule
from app.processing.segmenter import cleanup_audio_files
from app.core.chain import build_chain
from app.matching.vector_index import get_index
from app.matching.engine import register_segment_video
from app.core.config import UPLOAD_DIR
from app.api.processing_helper import process_video

router = APIRouter()

MAX_UPLOAD_MB = 500


@router.post("/register")
async def register_video(
    file: UploadFile = File(...),
    title: str = Form(...),
    owner: str = Form(...),
    db: AsyncSession = Depends(get_db),
):
    video_id = str(uuid.uuid4())
    save_path = UPLOAD_DIR / f"{video_id}_{file.filename}"

    # Stream file to disk
    async with aiofiles.open(save_path, "wb") as f:
        await f.write(await file.read())

    try:
        # All CPU work runs in thread pool — never blocks event loop
        segments, embeddings, fingerprints_hex = await process_video(str(save_path))
    except Exception as e:
        save_path.unlink(missing_ok=True)
        raise HTTPException(status_code=400, detail=str(e))

    if not segments:
        save_path.unlink(missing_ok=True)
        raise HTTPException(status_code=400, detail="No segments could be extracted from this video.")

    chain = build_chain(video_id, fingerprints_hex)
    chain_root = chain[-1]["chain_hash"] if chain else ""

    video_row = Video(
        id=video_id,
        title=title,
        owner=owner,
        duration=segments[-1]["timestamp_end"],
        registered_at=time.time(),
        chain_root_hash=chain_root,
    )
    db.add(video_row)
    await db.flush()

    index = get_index()
    for i, (seg, emb, fp_hex) in enumerate(zip(segments, embeddings, fingerprints_hex)):
        seg_row = Segment(
            video_id=video_id,
            segment_index=seg["index"],
            timestamp_start=seg["timestamp_start"],
            timestamp_end=seg["timestamp_end"],
            fingerprint_hex=fp_hex,
            embedding_blob=json.dumps(emb.tolist()),
            chain_hash=chain[i]["chain_hash"] if i < len(chain) else None,
        )
        db.add(seg_row)
        await db.flush()
        index.add(seg_row.id, emb)
        register_segment_video(seg_row.id, video_id)

    db.add(MonetizationRule(video_id=video_id, owner=owner, revenue_share=1.0, action="monetize"))
    await db.commit()
    cleanup_audio_files(segments)

    return {
        "video_id": video_id,
        "title": title,
        "owner": owner,
        "segments_registered": len(segments),
        "chain_root_hash": chain_root,
        "duration": video_row.duration,
    }


@router.get("/")
async def list_videos(db: AsyncSession = Depends(get_db)):
    result = await db.execute(select(Video))
    videos = result.scalars().all()
    return [
        {
            "video_id": v.id,
            "title": v.title,
            "owner": v.owner,
            "duration": v.duration,
            "registered_at": v.registered_at,
            "chain_root_hash": v.chain_root_hash,
        }
        for v in videos
    ]


@router.delete("/{video_id}")
async def delete_video(video_id: str, db: AsyncSession = Depends(get_db)):
    video = await db.get(Video, video_id)
    if not video:
        raise HTTPException(status_code=404, detail="Video not found")
    await db.delete(video)
    await db.commit()
    return {"deleted": video_id}
