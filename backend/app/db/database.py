from sqlalchemy.ext.asyncio import create_async_engine, AsyncSession, async_sessionmaker
from sqlalchemy.orm import DeclarativeBase, Mapped, mapped_column, relationship
from sqlalchemy import String, Float, Integer, Text, ForeignKey, JSON
from app.core.config import DATABASE_URL


engine = create_async_engine(DATABASE_URL, echo=False)
SessionLocal = async_sessionmaker(engine, expire_on_commit=False, class_=AsyncSession)


class Base(DeclarativeBase):
    pass


class Video(Base):
    __tablename__ = "videos"

    id: Mapped[str] = mapped_column(String, primary_key=True)
    title: Mapped[str] = mapped_column(String)
    owner: Mapped[str] = mapped_column(String)
    duration: Mapped[float] = mapped_column(Float)
    registered_at: Mapped[float] = mapped_column(Float)
    chain_root_hash: Mapped[str] = mapped_column(String, nullable=True)
    segments: Mapped[list["Segment"]] = relationship(back_populates="video", cascade="all, delete")


class Segment(Base):
    __tablename__ = "segments"

    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    video_id: Mapped[str] = mapped_column(ForeignKey("videos.id"))
    segment_index: Mapped[int] = mapped_column(Integer)
    timestamp_start: Mapped[float] = mapped_column(Float)
    timestamp_end: Mapped[float] = mapped_column(Float)
    fingerprint_hex: Mapped[str] = mapped_column(Text)   # perceptual hash string
    embedding_blob: Mapped[str] = mapped_column(Text, nullable=True)  # JSON float list
    chain_hash: Mapped[str] = mapped_column(String, nullable=True)
    video: Mapped["Video"] = relationship(back_populates="segments")


class MonetizationRule(Base):
    __tablename__ = "monetization_rules"

    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    video_id: Mapped[str] = mapped_column(ForeignKey("videos.id"))
    owner: Mapped[str] = mapped_column(String)
    revenue_share: Mapped[float] = mapped_column(Float, default=1.0)
    action: Mapped[str] = mapped_column(String, default="monetize")  # monetize | block | allow
    metadata_json: Mapped[dict] = mapped_column(JSON, nullable=True)


async def init_db():
    async with engine.begin() as conn:
        await conn.run_sync(Base.metadata.create_all)


async def get_db():
    async with SessionLocal() as session:
        yield session
