from fastapi import FastAPI
from model import MyModel
from data_types import MinerInput, MinerOutput
from utils import find_matching_question

app = FastAPI()
llm = MyModel()

@app.post("/solve")
async def solve(data: MinerInput):
    original_prompt = find_matching_question(data.modified_prompt)
    response = llm.generate(message=original_prompt)
    return MinerOutput(
        response=response
    )

@app.get("/health")
def health():
    return {"status": "ok"}
