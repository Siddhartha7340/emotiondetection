# from typing import Union

from fastapi import FastAPI
from service.api.api import main_router
import onnxruntime as rt
app = FastAPI(project_name="Emotions Detecttion")
app.include_router(main_router)
providers=['CPUExecutionProvider']
m_q=rt.InferenceSession(
        "service\core\logic\eff_quantized.onnx",providers=providers
    )


@app.get("/")
async def read_root():
    return {"Hello": "World"}


# @app.get("/items/{item_id}")
# def read_item(item_id: int, q: Union[str, None] = None):
#     return {"item_id": item_id, "q": q}