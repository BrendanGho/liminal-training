from pydantic import BaseModel
from typing import Optional


class DatasetRow(BaseModel):
    prompt: str
    completion: str
    reference: Optional[str] = None
