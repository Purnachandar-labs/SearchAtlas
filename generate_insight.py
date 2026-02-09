import os
import pandas as pd
import logging
from groq import Groq

MODEL = "meta-llama/llama-4-scout-17b-16e-instruct"
DATA_DIR = "data"
OUTPUT_DIR = "outputs"

logging.basicConfig(level=logging.INFO)
client = Groq(api_key=os.getenv("GROQ_API_KEY"))

def load_csv(filename):
    path = os.path.join(DATA_DIR, filename)
    if not os.path.exists(path):
        logging.warning(f"Missing file: {filename}")
        return None
    return pd.read_csv(path)

def summarize_keywords(df):
    return {
        "total_keywords": len(df),
        "avg_position": df["Position"].mean(),
        "top_keywords": df.sort_values("Traffic", ascending=False)
                          .head(10)[["Keyword","Position","Traffic"]].to_dict("records")
    }

def summarize_competitors(df):
    return {
        "total_competitors": len(df),
        "competitors": df.head(10).to_dict("records")
    }

def summarize_pages(df):
    return {
        "total_pages": len(df),
        "top_pages": df.head(10).to_dict("records")
    }

def call_groq(prompt, data):
    try:
        response = client.chat.completions.create(
            model=MODEL,
            messages=[
                {"role": "system", "content": prompt},
                {"role": "user", "content": str(data)}
            ],
            temperature=0.2
        )
        return response.choices[0].message.content
    except Exception as e:
        logging.error(f"Groq API error: {e}")
        return "Error generating insights."

def save_output(filename, content):
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    with open(os.path.join(OUTPUT_DIR, filename), "w") as f:
        f.write(content)

def main():
    keywords = load_csv("KEYWORDS.csv")
    competitors = load_csv("COMPETITORS.csv")
    pages = load_csv("TOP_PAGES.csv")

    summaries = {}

    if keywords is not None:
        summaries["keywords"] = summarize_keywords(keywords)
    if competitors is not None:
        summaries["competitors"] = summarize_competitors(competitors)
    if pages is not None:
        summaries["pages"] = summarize_pages(pages)

    with open("Overall Analysis Prompt (All CSVs).md.md") as f:
        overall_prompt = f.read()
    output = call_groq(overall_prompt, summaries)
    save_output("overall_summary.md", output)

if __name__ == "__main__":
    main()
