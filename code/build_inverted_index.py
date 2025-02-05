import pickle
import pandas as pd
import requests
from collections import defaultdict
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from nltk.stem import WordNetLemmatizer
import re

# Initialize stopwords and lemmatizer
stop_words = set()
lemmatizer = WordNetLemmatizer()

# Load additional stopwords from the provided GitHub URL
stopwords_url = "https://gist.githubusercontent.com/sebleier/554280/raw/25c14de4563b0a464fa27a5d998fc1da7d8b19ec/stopwords.txt"
additional_stopwords_list = requests.get(stopwords_url).content.decode().splitlines()
stop_words.update(additional_stopwords_list)

# Load the original review DataFrame from reviews_segment.pkl
print("Loading reviews from reviews_segment.pkl...")
with open('reviews_segment.pkl', 'rb') as f:
    reviews_df = pickle.load(f)
print("Reviews loaded successfully.")

# Check if the loaded object is a DataFrame
if not isinstance(reviews_df, pd.DataFrame):
    raise TypeError("Expected reviews_segment.pkl to be a DataFrame")

# Ensure the dataset contains required columns
required_columns = ['review_id', 'review_text', 'customer_review_rating']
for column in required_columns:
    if column not in reviews_df.columns:
        raise KeyError(f"Missing required column '{column}' in the dataset.")

# Initialize the posting list (inverted index)
posting_list = defaultdict(list)

# Preprocessing function to clean, tokenize, remove stopwords, and lemmatize words
def preprocess(text):
    # Step 1: Replace smileys with sentiment placeholders
    text = re.sub(r'(:-\)|:\))', 'positive_smiley', text)  # Positive smileys
    text = re.sub(r'(:-\(|:\()', 'negative_smiley', text)  # Negative smileys

    # Step 2: Remove punctuation and normalize to lowercase
    text = re.sub(r'[^\w\s]', '', text.lower())

    # Step 3: Tokenize the text
    tokens = word_tokenize(text)

    # Step 4: Filter tokens to remove stopwords and apply lemmatization
    filtered_tokens = [lemmatizer.lemmatize(word) for word in tokens if word.isalnum() and word not in stop_words]
    
    return filtered_tokens

# Build the posting list by iterating through each review
if 'review_id' in reviews_df.columns and 'review_text' in reviews_df.columns:
    for idx, row in reviews_df.iterrows():
        review_id = row['review_id']
        review_text = row['review_text']
        review_rating = row['customer_review_rating']  # Extract the rating
        
        # Preprocess the review text
        tokens = preprocess(review_text)
        
        # Add each token to the posting list with the review ID, rating, and cleaned text
        for token in tokens:
            posting_list[token].append((review_id, review_rating, review_text))
else:
    raise KeyError("Expected 'review_id' and 'review_text' columns in the DataFrame")

# Save the posting list as posting_list.pkl
output_path = 'output/posting_list.pkl'
with open(output_path, 'wb') as f:
    pickle.dump(posting_list, f)
print(f"\nInverted index (posting list) saved to {output_path}")

