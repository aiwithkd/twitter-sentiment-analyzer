"""
Generates a simulated Twitter dataset for sentiment analysis.
Produces data/raw/tweets.csv — raw input for the ML pipeline.

Each tweet is assigned a ground-truth sentiment label (positive/negative/neutral)
and constructed from topic-specific vocabulary to simulate realistic patterns.
"""

import pandas as pd
import numpy as np
import random
from datetime import datetime, timedelta

random.seed(42)
np.random.seed(42)

TOPICS = ["tech", "sports", "politics", "entertainment", "finance"]

VOCAB = {
    "positive": {
        "tech":          ["love the new update", "amazing features", "so fast now", "best app ever", "game changer", "finally works perfectly", "huge improvement", "impressive release", "totally worth it", "solid performance"],
        "sports":        ["incredible match", "hat trick legend", "unstoppable season", "proud of the team", "what a comeback", "deserved win", "outstanding performance", "world class", "history made today", "brilliant goal"],
        "politics":      ["great policy", "strong leadership", "finally progress", "right decision", "inspiring speech", "much needed reform", "well handled", "positive change", "good governance", "finally action"],
        "entertainment": ["must watch", "brilliant acting", "loved every scene", "oscar worthy", "totally hooked", "best series ever", "amazing soundtrack", "cinematic masterpiece", "emotional rollercoaster", "highly recommend"],
        "finance":       ["great returns", "market is booming", "smart investment", "profits up", "bullish trend", "strong earnings", "portfolio growing", "outperformed expectations", "solid quarter", "beat estimates"],
    },
    "negative": {
        "tech":          ["keeps crashing", "worst update ever", "so many bugs", "total disaster", "data loss nightmare", "terrible UI", "broken again", "stopped working", "waste of money", "unusable now"],
        "sports":        ["disgraceful performance", "should be fired", "embarrassing loss", "worst season ever", "complete failure", "no tactics whatsoever", "gave up entirely", "shocking result", "unacceptable", "totally outplayed"],
        "politics":      ["corrupt as always", "broken promises again", "complete failure", "nothing changes", "total incompetence", "wasted opportunity", "ignored the public", "embarrassing policy", "no accountability", "disgraceful decision"],
        "entertainment": ["waste of time", "terrible plot", "worst movie ever", "boring throughout", "bad acting", "disappointing sequel", "walked out halfway", "overrated garbage", "zero character development", "complete letdown"],
        "finance":       ["lost everything", "market crash incoming", "terrible investment", "avoid this stock", "huge losses", "company bleeding money", "downgraded again", "missed badly", "poor guidance", "sell everything"],
    },
    "neutral": {
        "tech":          ["just updated the app", "trying the new feature", "reading about the release", "watching the keynote", "installed the patch", "testing on my device", "noticed a change", "checking the changelog", "downloaded the beta", "using it daily"],
        "sports":        ["match starts at 8", "watching the game tonight", "following the scores", "checked the standings", "team announced lineup", "saw the highlights", "reading the match report", "following the transfer news", "stadium looks full", "commentary ongoing"],
        "politics":      ["watching the debate", "read the policy document", "attended the town hall", "following the election", "heard the speech", "checked the poll numbers", "reading the report", "noticed the announcement", "following the vote", "watching the press conference"],
        "entertainment": ["watching the trailer", "reading reviews", "heard about the show", "checked the ratings", "saw it trending", "on my watchlist", "started episode one", "halfway through", "heard mixed reviews", "checked the cast"],
        "finance":       ["monitoring the market", "reading the report", "checked the ticker", "watching the index", "reviewed the portfolio", "read the earnings call", "following the news", "noted the price", "tracking the fund", "reading analyst notes"],
    }
}

HANDLES = [f"@user_{i:04d}" for i in range(1, 3001)]
HASHTAGS = {
    "tech":          ["#tech","#AI","#coding","#software","#startup"],
    "sports":        ["#sports","#football","#cricket","#NBA","#FIFA"],
    "politics":      ["#politics","#election","#policy","#government","#democracy"],
    "entertainment": ["#movies","#Netflix","#Hollywood","#Bollywood","#series"],
    "finance":       ["#stocks","#investing","#market","#crypto","#finance"],
}

NOISE = [
    "just saying", "tbh", "idk", "lol", "omg", "wtf", "smh",
    "ngl", "fwiw", "imo", "tbf", "literally", "actually"
]

AMBIGUOUS = [
    "not sure what to think", "could go either way", "interesting take",
    "we'll see how this plays out", "hard to say", "time will tell",
    "mixed feelings about this", "depends on your perspective"
]

def make_tweet(sentiment, topic):
    # 12% chance of cross-topic vocabulary (adds noise)
    actual_topic = topic if random.random() > 0.12 else random.choice(TOPICS)
    base = random.choice(VOCAB[sentiment][actual_topic])

    # 8% chance of injecting ambiguous phrase (confuses the model realistically)
    if random.random() < 0.08:
        base = random.choice(AMBIGUOUS)

    handle = random.choice(HANDLES)
    tag    = random.choice(HASHTAGS[topic])
    noise  = random.choice(NOISE) if random.random() < 0.15 else ""
    extras = ["", f" {tag}", f" cc {handle}", f" {tag} {handle}", f" via {handle}"]
    tweet  = (noise + " " if noise else "") + base + random.choice(extras)
    return tweet.strip()

def generate_tweets(n=3000):
    sentiments = random.choices(["positive","negative","neutral"], weights=[0.45,0.30,0.25], k=n)
    topics     = random.choices(TOPICS, k=n)
    base_date  = datetime(2024, 1, 1)

    records = []
    for i, (sent, topic) in enumerate(zip(sentiments, topics)):
        tweet = make_tweet(sent, topic)
        ts    = base_date + timedelta(days=random.randint(0,364), hours=random.randint(0,23), minutes=random.randint(0,59))
        records.append({
            "tweet_id":  f"tw_{i+1:05d}",
            "text":      tweet,
            "sentiment": sent,
            "topic":     topic,
            "timestamp": ts.strftime("%Y-%m-%d %H:%M:%S"),
            "likes":     int(np.random.exponential(scale=30)),
            "retweets":  int(np.random.exponential(scale=10)),
        })

    df = pd.DataFrame(records)
    df.to_csv("data/raw/tweets.csv", index=False)
    print(f"Generated {len(df)} tweets → data/raw/tweets.csv")
    print(df["sentiment"].value_counts())
    print(df["topic"].value_counts())
    return df

if __name__ == "__main__":
    generate_tweets(3000)
