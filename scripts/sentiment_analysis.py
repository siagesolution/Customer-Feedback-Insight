import json
import matplotlib.pyplot as plt
import pandas as pd

from bs4 import BeautifulSoup
from sklearn.feature_extraction.text import TfidfVectorizer
from transformers import pipeline


df = pd.read_excel("data/customer_feedback.xlsx", engine="openpyxl")
df = df[['OutletID', 'TemplateID', 'RegionDesc', 'FeedbackCol']].dropna()

def clean_html(text):
    return BeautifulSoup(text, "html.parser").get_text(separator=" ").strip()

df['CleanFeedback'] = df['FeedbackCol'].apply(clean_html)

classifier = pipeline(
    "sentiment-analysis",
    model="distilbert-base-uncased-finetuned-sst-2-english",
    tokenizer="distilbert-base-uncased-finetuned-sst-2-english",
    truncation=True,
    max_length=512
)


def batch_sentiment(texts, batch_size=32):
    results = []
    for i in range(0, len(texts), batch_size):
        batch = texts[i:i+batch_size]
        results.extend(classifier(batch))
    return results

sentiments = batch_sentiment(df['CleanFeedback'].tolist())
df['SentimentLabel'] = [s['label'] for s in sentiments]
df['SentimentScore'] = [s['score'] for s in sentiments]

outlet_sentiment = df.groupby('OutletID')['SentimentLabel'].value_counts(normalize=True).unstack().fillna(0)
region_sentiment = df.groupby('RegionDesc')['SentimentLabel'].value_counts(normalize=True).unstack().fillna(0)
template_sentiment = df.groupby('TemplateID')['SentimentLabel'].value_counts(normalize=True).unstack().fillna(0)

with pd.ExcelWriter("output/feedback_insights_summary.xlsx") as writer:
    df.to_excel(writer, sheet_name="Detailed Feedback", index=False)
    outlet_sentiment.to_excel(writer, sheet_name="By Outlet")
    region_sentiment.to_excel(writer, sheet_name="By Region")
    template_sentiment.to_excel(writer, sheet_name="By Template")


def extract_keywords(df, group_col, text_col, top_k=10):
    keywords = {}
    for group, group_df in df.groupby(group_col):
        texts = group_df[text_col].tolist()
        joined_text = " ".join(texts)
        vectorizer = TfidfVectorizer(stop_words="english", max_features=1000)
        tfidf_matrix = vectorizer.fit_transform([joined_text])
        feature_array = vectorizer.get_feature_names_out()
        tfidf_scores = tfidf_matrix.toarray().flatten()
        top_indices = tfidf_scores.argsort()[::-1][:top_k]
        top_keywords = [feature_array[i] for i in top_indices]
        keywords[group] = top_keywords
    return keywords

outlet_keywords = extract_keywords(df, group_col="OutletID", text_col="CleanFeedback")

with open("output/outlet_keywords.json", "w") as f:
    json.dump(outlet_keywords, f, indent=2)


def plot_outlet_sentiment_comparison(outlet_df, output_file="output/outlet_sentiment_comparison.png"):
    top_outlets = outlet_df.sum(axis=1).sort_values(ascending=False).head(10).index
    plot_df = outlet_df.loc[top_outlets]
    plot_df.plot(kind="bar", stacked=True, figsize=(12,6), colormap="coolwarm")
    plt.title("Top 10 Outlets by Sentiment Distribution")
    plt.ylabel("Proportion of Feedback")
    plt.xlabel("Outlet ID")
    plt.tight_layout()
    plt.savefig(output_file)

plot_outlet_sentiment_comparison(outlet_sentiment)


# Heatmap for all outlet sentiment comparison
def plot_sentiment_heatmap(outlet_df, output_file="output/outlet_sentiment_heatmap.png"):
    top_outlets = outlet_df.sum(axis=1).sort_values(ascending=False).head(20).index
    plot_df = outlet_df.loc[top_outlets]
    plt.figure(figsize=(12, 8))
    sns.heatmap(plot_df, annot=True, cmap="YlGnBu", fmt=".2f")
    plt.title("Sentiment Heatmap - Top 20 Outlets")
    plt.ylabel("Outlet ID")
    plt.xlabel("Sentiment")
    plt.tight_layout()
    plt.savefig(output_file)


plot_sentiment_heatmap(outlet_sentiment)

