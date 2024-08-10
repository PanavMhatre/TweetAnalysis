##KMeans and TextBlob

import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.cluster import KMeans
import numpy as np

# Load the CSV file
data = pd.read_csv('tweet_10000_combined.csv', names=['tweet'])

# Extract the tweets from the CSV
tweets = data['tweet']

# Create a TF-IDF vectorizer to convert tweets into numerical features
vectorizer = TfidfVectorizer()
X = vectorizer.fit_transform(tweets)

# Apply KMeans clustering
num_clusters = 2  # Number of clusters (positive, negative, neutral)
kmeans = KMeans(n_clusters=num_clusters, random_state=42)
kmeans.fit(X)

# Get cluster predictions for the tweets
predictions = kmeans.predict(X)

# Store the predictions in an array
predictions_array = np.array(predictions)
print(predictions_array)

import sys
import numpy
numpy.set_printoptions(threshold=sys.maxsize)

silhouette = silhouette_score(X, predictions)
print(f"Silhouette Score: {silhouette}")

"""##BERT##

"""

from sklearn.metrics import silhouette_score, calinski_harabasz_score
!pip install transformers
!pip install sentencepiece

import pandas as pd
from transformers import AutoTokenizer, AutoModelForSequenceClassification

# Load the CSV file
data = pd.read_csv('tweet_10000_combined.csv',names=['tweet'])

# Extract the tweets from the CSV
tweets = data['tweet'].tolist()

# Load the pre-trained DistilBERT model and tokenizer
model_name = 'distilbert-base-uncased'
tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModelForSequenceClassification.from_pretrained(model_name)

# Set batch size for processing
batch_size = 16

# Tokenize and perform sentiment analysis in batches
for i in range(num_batches):
    start_idx = i * batch_size
    end_idx = (i + 1) * batch_size
    batch_tweets = tweets[start_idx:end_idx]

    # Tokenize the batch of tweets
    tokenized_inputs = tokenizer.batch_encode_plus(batch_tweets, padding=True, truncation=True, return_tensors='pt')

    # Perform sentiment analysis on the batch
    outputs = model(**tokenized_inputs)
    predictions = outputs.logits.argmax(dim=1)

    # Collect the predicted sentiment for each tweet in the batch
    sentiments.append(predictions)

sentiment_counts = pd.Series(sentiments).value_counts()
total_tweets = len(sentiments)
print(sentiments)
percentage = sentiment_counts / total_tweets * 100
num_tweets = len(tweets)
num_batches = (num_tweets - 1) // batch_size + 1

sentiments = []
