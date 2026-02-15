FROM python:3-slim

WORKDIR /app

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# NLTK library download
RUN python -m nltk.downloader punkt punkt_tab stopwords wordnet omw-1.4

COPY . /app

CMD ["uvicorn", "src.web.api:app", "--host", "0.0.0.0", "--port", "7860"]