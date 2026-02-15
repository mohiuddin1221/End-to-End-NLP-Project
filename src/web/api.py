import joblib
import pandas as pd
from fastapi import FastAPI
from .schemas import UserInputRequest


app = FastAPI()
model = joblib.load("./models/fake_news_full_pipeline.pkl")


@app.get("/health")
async def health_check():
    return {"status": "healthy"}


@app.post("/predict")
async def predict_score(user_input: UserInputRequest):
    title = user_input.title
    text = user_input.text

    title_len = len(title)
    text_len = len(text)

    title_word_count = len(title.split())
    text_word_count = len(text.split())

    inputs = {
        "title": title,
        "text": text,
        "title_len": title_len,
        "text_len": text_len,
        "title_word_count": title_word_count,
        "text_word_count": text_word_count,
    }
    input_data = pd.DataFrame([inputs])
    prediction = model.predict(input_data)[0]

   
    return {
        "prediction": prediction,
        "success": True
    }
    



# uvicorn src.web.api:app --reload