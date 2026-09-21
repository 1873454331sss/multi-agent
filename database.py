import os
from datetime import datetime
from typing import List, Dict, Any
from sqlalchemy.ext.asyncio import create_async_engine, AsyncSession, async_sessionmaker
from sqlalchemy.orm import declarative_base, Mapped, mapped_column
from sqlalchemy import Text, DateTime, Integer, select
from dotenv import load_dotenv

load_dotenv()

# Render 给的是 postgresql://，SQLAlchemy 异步版需要 postgresql+asyncpg://
DATABASE_URL = os.getenv("DATABASE_URL", "").replace("postgresql://", "postgresql+asyncpg://")

engine = create_async_engine(DATABASE_URL, echo=False) if DATABASE_URL else None
AsyncSessionLocal = async_sessionmaker(engine, class_=AsyncSession, expire_on_commit=False) if engine else None
Base = declarative_base()


class AnalysisHistory(Base):
    __tablename__ = "analysis_history"

    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    query: Mapped[str] = mapped_column(Text, nullable=False)
    report: Mapped[str] = mapped_column(Text)
    sentiment_label: Mapped[str] = mapped_column(Text)
    risk_level: Mapped[str] = mapped_column(Text)
    created_at: Mapped[datetime] = mapped_column(DateTime, default=datetime.now)


async def init_db():
    """启动时建表"""
    if not engine:
        return
    async with engine.begin() as conn:
        await conn.run_sync(Base.metadata.create_all)


async def save_analysis(query: str, result: Dict[str, Any]):
    """保存一次分析记录"""
    if not AsyncSessionLocal:
        return
    async with AsyncSessionLocal() as session:
        record = AnalysisHistory(
            query=query,
            report=result.get("report", ""),
            sentiment_label=result.get("data", {}).get("sentiment", {}).get("label", "neutral"),
            risk_level=result.get("data", {}).get("risk_level", "low")
        )
        session.add(record)
        await session.commit()


async def get_history(limit: int = 10) -> List[Dict[str, Any]]:
    """查询最近的分析记录"""
    if not AsyncSessionLocal:
        return []
    async with AsyncSessionLocal() as session:
        result = await session.execute(
            select(AnalysisHistory).order_by(AnalysisHistory.created_at.desc()).limit(limit)
        )
        rows = result.scalars().all()
        return [
            {
                "id": r.id,
                "query": r.query,
                "report": r.report,
                "sentiment_label": r.sentiment_label,
                "risk_level": r.risk_level,
                "created_at": r.created_at.isoformat() if r.created_at else None,
            }
            for r in rows
        ]