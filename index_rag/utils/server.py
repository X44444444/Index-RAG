import time
from typing import Any, Optional

import uvicorn
from fastapi import FastAPI
from pydantic import BaseModel

from index_rag.utils.llm_utils import rag_chain

app = FastAPI(title="OpenAI-Compatible API", version="1.0")


class ChatCompletionRequest(BaseModel):
    model: str
    messages: list[dict[str, str]]
    max_tokens: Optional[int] = 512
    temperature: Optional[float] = 0.7


class PredictRequest(BaseModel):
    # model is optional; defaults to server MODEL_ID
    model: Optional[str] = None
    action: dict[str, Any]


model_name = "RAG-app"


@app.get("/v1/models")
def list_models():
    return {"data": [{"id": model_name, "object": "model"}]}


@app.post("/v1/chat/completions")
async def create_chat_completion(request: ChatCompletionRequest):
    # Generate response
    response = rag_chain.invoke(request.messages[-1]["content"])

    return {
        "id": f"chatcmpl-{int(time.time())}",
        "object": "chat.completion",
        "created": int(time.time()),
        "model": model_name,
        "choices": [
            {
                "index": 0,
                "message": {"role": "assistant", "content": response},
                "finish_reason": "stop",
            }
        ],
    }


def start_server(host="0.0.0.0", port=8001):
    uvicorn.run(app, host=host, port=port)


if __name__ == "__main__":
    start_server()
