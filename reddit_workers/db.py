import os
from pymongo import MongoClient
from dotenv import load_dotenv

load_dotenv()

mongo = MongoClient(os.getenv("MONGO_URI"))
db = mongo["reddit_sentiment"]

queries_collection = db["queries"]
posts_collection = db["posts"]