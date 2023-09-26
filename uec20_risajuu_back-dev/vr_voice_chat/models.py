from typing import List, Optional
from pydantic import BaseModel
from enum import Enum

class Query(BaseModel):
  text: str
  emotion: bool