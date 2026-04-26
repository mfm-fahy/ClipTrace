import aiofiles
import uuid
from fastapi import APIRouter, UploadFile, File, Depends, HTTPException, Body

from app.db.database import get_db
from app.processing.segmenter import cleanup_audio_files
from app.matching.engine import match_segments
from app.core.config import UPLOAD_DIR
from app.api.processing_helper import process_video

router = APIRouter()


@router.post("/route")
async def route_revenue(
    file: UploadFile = File(...),
    total_revenue: float = 0.0,
    db = Depends(get_db),
):
    tmp_path = UPLOAD_DIR / f"mono_{uuid.uuid4()}_{file.filename}"
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
        return {"status": "no_match", "allocations": [], "total_revenue": total_revenue}

    video_ids = [m["video_id"] for m in matches]
    cursor = db["monetization_rules"].find({"video_id": {"$in": video_ids}})
    rules_docs = await cursor.to_list(length=None)
    rules = {r["video_id"]: r for r in rules_docs}

    weights = {m["video_id"]: m["confidence"] * m["matched_segments"] for m in matches}
    total_weight = sum(weights.values()) or 1.0

    allocations = []
    for m in matches:
        rule = rules.get(m["video_id"])
        action = rule["action"] if rule else "monetize"
        share = (weights[m["video_id"]] / total_weight) * (rule["revenue_share"] if rule else 1.0)
        allocations.append({
            "video_id": m["video_id"],
            "title": m["title"],
            "owner": m["owner"],
            "action": action,
            "confidence": m["confidence"],
            "revenue_share_pct": round(share * 100, 2),
            "allocated_revenue": round(total_revenue * share, 4),
            "is_mixed_source": m["is_mixed"],
        })

    return {
        "status": "routed",
        "total_revenue": total_revenue,
        "allocations": allocations,
        "mixed_content_detected": any(m["is_mixed"] for m in matches),
    }


@router.put("/rules/{video_id}")
async def update_monetization_rule(
    video_id: str,
    action: str = Body(..., embed=True),
    revenue_share: float = Body(1.0, embed=True),
    db = Depends(get_db),
):
    if action not in ("monetize", "block", "allow"):
        raise HTTPException(status_code=400, detail="action must be monetize | block | allow")
        
    rule = await db["monetization_rules"].find_one({"video_id": video_id})
    if not rule:
        raise HTTPException(status_code=404, detail="No rule found for this video")
        
    await db["monetization_rules"].update_one(
        {"video_id": video_id},
        {"$set": {"action": action, "revenue_share": revenue_share}}
    )
    return {"video_id": video_id, "action": action, "revenue_share": revenue_share}


@router.get("/rules")
async def list_rules(db = Depends(get_db)):
    cursor = db["monetization_rules"].find()
    rules = await cursor.to_list(length=None)
    return [
        {
            "video_id": r["video_id"], 
            "owner": r["owner"], 
            "action": r["action"], 
            "revenue_share": r["revenue_share"]
        }
        for r in rules
    ]
