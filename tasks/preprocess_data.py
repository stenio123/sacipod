import pandas as pd
import nltk
import json
import nltk
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from nltk.stem import PorterStemmer
from nltk.stem import WordNetLemmatizer
from nltk.sentiment import SentimentIntensityAnalyzer

# You might need to download the stopwords, punkt, and wordnet datasets by uncommenting the lines below
nltk.download('stopwords')
nltk.download('punkt')
nltk.download('wordnet')
nltk.download('vader_lexicon')

# Global Variable to toggle sentiment analysis stage - test to check best accuracy
SENTIMENT_BEFORE = True


def get_sentiment(sentence):
    sia = SentimentIntensityAnalyzer()
    return sia.polarity_scores(sentence)['compound']


def preprocess_sentence(sentence):
    # Tokenize the sentence
    words = word_tokenize(sentence)

    # Remove stopwords
    stop_words = set(stopwords.words('english'))
    words = [word for word in words if word.lower() not in stop_words]

    # Apply Stemming and Lemmatization
    stemmer = PorterStemmer()
    lemmatizer = WordNetLemmatizer()
    words = [stemmer.stem(lemmatizer.lemmatize(word)) for word in words]

    # Join words back together
    sentence = ' '.join(words)
    
    return sentence

def preprocess_data(json_data):
    # Load JSON
    podcasts = json.loads(json_data)
    
    # Initialize DataFrame
    df = pd.DataFrame(columns=['podcast', 'time_start', 'time_end', 'speaker', 'speaker_name', 'sentence'])

    for podcast in podcasts:
        podcast_name = podcast['podcast']
        for transcription in podcast['transcription']:
            time_start = transcription['time_start']
            time_end = transcription['time_end']
            speaker = transcription['speaker']
            sentence = transcription['sentence']
            
            if SENTIMENT_BEFORE:
                sentiment = get_sentiment(sentence)

            # Preprocess the sentence
            sentence = preprocess_sentence(sentence)
        
            if not SENTIMENT_BEFORE:
                sentiment = get_sentiment(sentence)
            
            # Add data to DataFrame
            new_data = {
                'podcast': podcast_name,
                'time_start': time_start,
                'time_end': time_end,
                'speaker': speaker,
                'speaker_name': f"Speaker_{speaker}",
                'sentence': sentence,
                'sentiment': sentiment
            }
            new_df = pd.DataFrame([new_data])
            df = pd.concat([df, new_df], ignore_index=True)
    
    return df
