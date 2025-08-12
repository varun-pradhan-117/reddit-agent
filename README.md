# Reddit Sentiment Bot

## Overview
Reddit Sentiment Bot is a Python-based tool for collecting Reddit posts and comments on a given topic and analyzing the sentiment of each post using a local LLM (DeepSeek via Ollama).  
It saves all query data, per-post comment data, sentiment scores, and summaries into **MongoDB** for later retrieval and analysis.

---

## Features
- **Topic-based search**: Collect posts from specific subreddits or across Reddit (`r/all`) for a given topic.
- **Customizable search scope**:
  - Time filter: `day`, `week`, `month`, `year`, `all`
  - Limit number of posts and comments
- **Dual comment sampling**:
  - Top comments
  - Controversial comments
- **Duplicate removal** for cleaner datasets.
- **Per-post sentiment analysis**:
  - Uses Deepseek-r1 for targeted sentiment classification.
  - Outputs score (1–5) and explanation for each post’s sentiment
- **Data persistence**:
  - Stores all raw and processed data in MongoDB
  - Tracks searches with a `query_id` for easy filtering
- **Summarization**:
  - Generates a short summary of each post and its discussion.

---

## Installation
1. **Clone the repository**  
```bash
git clone https://github.com/yourusername/reddit-sentibot.git
cd reddit-sentibot
