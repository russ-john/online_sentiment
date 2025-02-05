import requests
import random
from bs4 import BeautifulSoup
import pandas as pd
import re
from dateutil import parser
from newspaper import Article
from newspaper import Config
import datetime as dt
import nltk
from googlenewsdecoder import new_decoderv1
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer
import os
import chardet

# Set dates for today and yesterday
now = dt.date.today()
yesterday = now - dt.timedelta(days=1)

# Setup requests configurations
nltk.download('punkt')

# Create a list of random user agents
user_agent_list = [
    'Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/605.1.15 (KHTML, like Gecko) Version/17.3.1 Safari/605.1.15',
    'Mozilla/5.0 (Windows NT 10.0; Win64; x64; rv:77.0) Gecko/20100101 Firefox/77.0',
    'Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_5) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/83.0.4103.97 Safari/537.36',
    'Mozilla/5.0 (Macintosh; Intel Mac OS X 10.15; rv:77.0) Gecko/20100101 Firefox/77.0',
    'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/83.0.4103.97 Safari/537.36'
]

config = Config()
config.browser_user_agent = random.choice(user_agent_list)
config.request_timeout = 20
header = {'User-Agent': config.browser_user_agent}

# load existing dataset to avoid re-fetching articles
script_dir = os.path.dirname(os.path.abspath(__file__))
output_dir = os.path.join(script_dir, 'online_sentiment/output')
os.makedirs(output_dir, exist_ok=True)
main_csv_path = os.path.join(output_dir, 'enterprise_risks_online_sentiment.csv')
if os.path.exists(main_csv_path):
    existing_df = pd.read_csv(main_csv_path, usecols=["LINK"], encoding="utf-8")
    existing_links = set(existing_df["LINK"].dropna().str.lower().str.strip())  # Normalize existing links
else:
    existing_links = set()

# encode-decode search terms
read_file = pd.read_csv('EnterpriseRisksListEncoded.csv', encoding='utf-8')
read_file['ENTERPRISE_RISK_ID'] = pd.to_numeric(read_file['ENTERPRISE_RISK_ID'], downcast='integer', errors='coerce')
def process_encoded_search_terms(term):
    try:
        encoded_number = int(term)
        byte_length = (encoded_number.bit_length() + 7) // 8
        byte_rep = encoded_number.to_bytes(byte_length, byteorder='little')
        decoded_text = byte_rep.decode('utf-8')
        return decoded_text
    except (ValueError, UnicodeDecodeError):
        return None

read_file['SEARCH_TERMS'] = read_file['ENCODED_TERMS'].apply(process_encoded_search_terms)

# prep lists for new entries
search_terms, title, published, link, domain, source, summary, keywords, sentiment, polarity = ([] for _ in range(10))

print('Fetching Google News articles...')

url_start = 'https://news.google.com/rss/search?q={'
url_end = '}%20when%3A1d'  # fetch only recent articles

# Grab Google News links
for term in read_file.SEARCH_TERMS.dropna():
    req = requests.get(url=url_start + term + url_end, headers=header)
    soup = BeautifulSoup(req.text, 'xml')

    for item in soup.find_all("item"):
        title_text = item.title.text.strip()
        encoded_url = item.link.text.strip()
        source_text = item.source.text.strip()
        decoded_url = new_decoderv1(encoded_url, interval=5)
        if decoded_url.get("status"):
            decoded_url = decoded_url['decoded_url'].strip().lower()  # normalize link
            # **SKIP IF LINK ALREADY EXISTS**
            if decoded_url in existing_links:
                continue  # Skip articles that have already been collected
            search_terms.append(term)
            title.append(title_text)
            source.append(source_text)
            link.append(decoded_url)
            published.append(parser.parse(item.pubDate.text).date())
            regex_pattern = re.compile('(https?):((|(\\\\))+[\w\d:#@%;$()~_?\+-=\\\.&]*)')
            domain_search = regex_pattern.search(str(item.source))
            domain.append(domain_search.group(0) if domain_search else None)

for article_link in link:
    article = Article(article_link, config=config)
    try:
        article.download()
        article.parse()
        article.nlp()
        summary.append(article.summary)
        keywords.append(article.keywords)
        analyzer = SentimentIntensityAnalyzer().polarity_scores(article.summary)
        neg, pos, neu = analyzer['neg'], analyzer['pos'], analyzer['neu']
        if neg > pos or neg == -1:
            sentiment.append('negative')
            polarity.append(f'-{neg}')
        elif pos > neg:
            sentiment.append('positive')
            polarity.append(f'+{pos}')
        else:
            sentiment.append('neutral')
            polarity.append(str(neu))
    except:
        summary.append(None)
        keywords.append(None)
        sentiment.append(None)
        polarity.append(None)

# Create DataFrame
alerts = pd.DataFrame({
    'SEARCH_TERMS': search_terms,
    'TITLE': title,
    'SUMMARY': summary,
    'KEYWORDS': keywords,
    'PUBLISHED_DATE': published,
    'LINK': link,
    'SOURCE': source,
    'SOURCE_URL': domain,
    'SENTIMENT': sentiment,
    'POLARITY': polarity
})

# merge new alerts with search terms data
joined_df = pd.merge(alerts, read_file, on='SEARCH_TERMS', how='left')
final_df = joined_df[['ENTERPRISE_RISK_ID', 'TITLE', 'SUMMARY', 'KEYWORDS', 'PUBLISHED_DATE', 'LINK', 'SOURCE', 'SOURCE_URL', 'SENTIMENT', 'POLARITY']]
final_df['LAST_RUN_TIMESTAMP'] = dt.datetime.now().strftime('%Y-%m-%d %H:%M:%S')

# load existing data and combine with new entries
if os.path.exists(main_csv_path):
    existing_main_df = pd.read_csv(main_csv_path, parse_dates=['PUBLISHED_DATE'], encoding='utf-8')
else:
    existing_main_df = pd.DataFrame()

combined_df = pd.concat([existing_main_df, final_df], ignore_index=True).drop_duplicates(subset=['TITLE', 'LINK'])

# rolling 6-month filter
six_months_ago = dt.datetime.now() - dt.timedelta(days=6 * 30)
combined_df['PUBLISHED_DATE'] = pd.to_datetime(combined_df['PUBLISHED_DATE'], errors='coerce')

# split recent and old data
recent_df = combined_df[combined_df['PUBLISHED_DATE'] >= six_months_ago].copy()
old_df = combined_df[combined_df['PUBLISHED_DATE'] < six_months_ago].copy()

# save recent data back to the main CSV
recent_df.sort_values(by='PUBLISHED_DATE', ascending=False).to_csv(main_csv_path, index=False, encoding='utf-8')

print(f"Updated main CSV with {len(recent_df)} records.")

# archive old data
archive_csv_path = os.path.join(output_dir, 'enterprise_risks_sentiment_archive.csv')
archive_data = old_df[['ENTERPRISE_RISK_ID', 'TITLE', 'PUBLISHED_DATE', 'LINK', 'SENTIMENT', 'POLARITY', 'LAST_RUN_TIMESTAMP']]
archive_data.to_csv(archive_csv_path, mode='a', index=False, header=not os.path.exists(archive_csv_path), encoding='utf-8')

print(f"Archived {len(old_df)} records.")
