from fastapi import FastAPI, HTTPException, Query
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from analyzer import run_full_analysis
from database import get_history, init_db

app = FastAPI(
    title="多 Agent 舆情分析 API",
    description="输入关键词，自动输出带情感分析和风险预警的舆情报告",
    version="1.0.0"
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=False,
    allow_methods=["*"],
    allow_headers=["*"],
)


class AnalyzeRequest(BaseModel):
    query: str


@app.on_event("startup")
async def startup():
    await init_db()


@app.get("/")
async def root():
    return {"message": "舆情分析 API 已启动，请访问 /docs 查看文档"}


@app.post("/analyze")
async def analyze(req: AnalyzeRequest):
    if not req.query or len(req.query.strip()) < 1:
        raise HTTPException(status_code=400, detail="查询关键词不能为空")
    result = await run_full_analysis(req.query)
    return result


@app.get("/history")
async def history(limit: int = Query(10, ge=1, le=50)):
    records = await get_history(limit)
    return {"count": len(records), "records": records}


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)