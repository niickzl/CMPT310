"""FastAPI requests for YouTube comment analysis."""

import os

from dotenv import load_dotenv
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel

from preprocess import clean_batch
from youtube import (
    CommentsDisabledError,
    QuotaExceededError,
    VideoNotFoundError,
    extract_video_id,
    fetch_comments,
)

load_dotenv()

YOUTUBE_API_KEY = os.getenv("YOUTUBE_API_KEY")

app = FastAPI(title="YT Comment Analyzer")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)


class AnalyzeRequest(BaseModel):
    url: str

"""Check if api runs"""
@app.get("/health")
def health():
    return {"status": "ok"}

"""Fetch comments and use preprocess fn to get cleaned text"""
@app.post("/analyze")
def analyze(request: AnalyzeRequest):
    if not YOUTUBE_API_KEY:
        raise HTTPException(status_code=500, detail="YOUTUBE_API_KEY is not configured")

    try:
        video_id = extract_video_id(request.url)
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))

    try:
        raw_comments = fetch_comments(video_id, YOUTUBE_API_KEY)
    except VideoNotFoundError as e:
        raise HTTPException(status_code=404, detail=str(e))
    except CommentsDisabledError as e:
        raise HTTPException(status_code=403, detail=str(e))
    except QuotaExceededError as e:
        raise HTTPException(status_code=429, detail=str(e))

    cleaned_texts = clean_batch([c["text"] for c in raw_comments])

    comments = [
        {
            "author": raw["author"],
            "text": raw["text"],
            "cleaned_text": cleaned,
            "likes": raw["likes"],
            "published_at": raw["published_at"],
        }
        for raw, cleaned in zip(raw_comments, cleaned_texts)
    ]

    return {
        "video_id": video_id,
        "comment_count": len(comments),
        "comments": comments,
    }
