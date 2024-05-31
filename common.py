from pydantic import BaseModel


class BinModel(BaseModel):
    class Config:
        arbitrary_types_allowed = True

