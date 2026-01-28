from pydantic import BaseModel

class GenerateRequest(BaseModel):
    seed: int
    truncation:float =0.7
