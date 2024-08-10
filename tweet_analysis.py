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
