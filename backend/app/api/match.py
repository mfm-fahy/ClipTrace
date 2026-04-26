import aiofiles
import uuid
from fastapi import APIRouter, UploadFile, File, Depends, HTTPException
from sqlalchemy.ext.asyncio import AsyncSession

from app.db.database import get_db
from app.processing.segmenter import cleanup_audio_files
from app.matching.engine import match_segments
from app.core.config import UPLOAD_DIR
from app.api.processing_helper import process_video

router = APIRouter()


@router.post("/")
async def match_clip(
    file: UploadFile = File(...),
    db: AsyncSession = Depends(get_db),
):
    tmp_path = UPLOAD_DIR / f"query_{uuid.uuid4()}_{file.filename}"
    async with aiofiles.open(tmp_path, "wb") as f:
        await f.write(await file.read())

    try:
        segments, embeddings, _ = await process_video(str(tmp_path))
    except Exception as e:
        tmp_path.unlink(missing_ok=True)
        raise HTTPException(status_code=400, detail=str(e))

    if not segments:
        tmp_path.unlink(missing_ok=True)
        raise HTTPException(status_code=400, detail="No segments extracted from clip.")

    matches = await match_segments(embeddings, db)
    cleanup_audio_files(segments)
    tmp_path.unlink(missing_ok=True)

    return {
        "query_segments": len(segments),
        "matches_found": len(matches),
        "sources": matches,
    }
