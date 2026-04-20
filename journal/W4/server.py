from contextlib import asynccontextmanager
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field
from transformers import pipeline
import torch

# 1) 全局容器：用 dict 装模型，避免用 global 关键字
state = {}

# 2) lifespan：服务启动时加载模型（只加载一次！），关闭时清理
@asynccontextmanager
async def lifespan(app: FastAPI):
    device = 0 if torch.cuda.is_available() else -1   # 0=cuda:0, -1=cpu
    state["pipe"] = pipeline(
        "text-generation",
        model="gpt2",
        device=device,
        torch_dtype=torch.float16 if device == 0 else torch.float32,
    )
    yield
    state.clear()

app = FastAPI(lifespan=lifespan)

# 3) 请求体（Pydantic 自动校验类型 + 生成 OpenAPI 文档）
class GenerateRequest(BaseModel):
    prompt: str = Field(..., min_length=1)
    max_new_tokens: int = Field(50, ge=1, le=512)
    temperature: float = Field(1.0, gt=0.0, le=2.0)
    do_sample: bool = True

# 4) 响应体
class GenerateResponse(BaseModel):
    prompt: str
    generated_text: str
    latency_ms: float

# 5) 接口
@app.post("/generate", response_model=GenerateResponse)
def generate(req: GenerateRequest):
    import time
    pipe = state["pipe"]
    try:
        t0 = time.perf_counter()
        outputs = pipe(
            req.prompt,
            max_new_tokens=req.max_new_tokens,
            temperature=req.temperature,
            do_sample=req.do_sample,
            return_full_text=False,   # 只返回新生成的部分，不重复 prompt
        )
        latency = (time.perf_counter() - t0) * 1000
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

    return GenerateResponse(
        prompt=req.prompt,
        generated_text=outputs[0]["generated_text"],
        latency_ms=round(latency, 2),
    )

@app.get("/health")
def health():
    return {"status": "ok", "model": "gpt2"}