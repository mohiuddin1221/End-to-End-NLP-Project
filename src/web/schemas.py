from pydantic import BaseModel

class UserInputRequest(BaseModel):
    title: str
    text: str
