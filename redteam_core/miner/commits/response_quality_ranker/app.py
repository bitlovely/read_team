from fastapi import FastAPI
from model import ResponseQualityHandler, MyModel
from data_types import MinerInput, MinerOutput
import json

app = FastAPI()
model = MyModel()

@app.post("/solve")
async def solve(data: MinerInput):
    challenge = {
        'prompt': data.prompt,
        'responses': data.responses
    }
    challenge_message = json.dumps(challenge)
    response_quality = model.generate(challenge_message)
    return MinerOutput(
        response_quality=response_quality
    )

@app.get("/health")
def health():
    return {"status": "ok"}
