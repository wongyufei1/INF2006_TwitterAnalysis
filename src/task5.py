# Import Relevant Libraries
import os
import sys

from pyspark.sql import SparkSession
from utils import load_tweets, convert_data_type

os.environ['PYSPARK_PYTHON'] = sys.executable
os.environ['PYSPARK_DRIVER_PYTHON'] = sys.executable


"""Convert PennTreebank tags to WordNet tags.

    Parameters:
        tag (str): A PennTreebank tag (e.g., 'JJ' for adjective, 'NN' for noun, 'RB' for adverb, 'VB' for verb).

    Returns:
        str or None: Corresponding WordNet tag if tag is recognized, otherwise None.
"""


def penn_to_wn(tag):
    # Import Relevant Libraries
    from nltk.corpus import wordnet as wn
    # Conditions for each tag
    if tag.startswith('J'):
        return wn.ADJ
    elif tag.startswith('N'):
        return wn.NOUN
    elif tag.startswith('R'):
        return wn.ADV
    elif tag.startswith('V'):
        return wn.VERB
    return None


"""
    Retrieve sentiment score of a word.

    Parameters:
        word (str): A single word for which sentiment score is to be determined.

    Returns:
        float or None: Sentiment score of the word if found, otherwise None.
"""


def get_word_sentiment_score(word):
    # Import Relevant Libraries

    import nltk
    from nltk.corpus import sentiwordnet as swn
    from nltk.corpus import wordnet as wn

    nltk.download('sentiwordnet')
    nltk.download('wordnet')
    nltk.download('omw-1.4')
    nltk.download('averaged_perceptron_tagger')
    from nltk.stem import WordNetLemmatizer

    # Initialize lemmatizer
    lemmatizer = WordNetLemmatizer()

    # Define sentiment score
    sentiment = 0.0

    # Tag the word with a POS tag
    tagged_word = nltk.tag.pos_tag([word])  # Pass word as a list

    # For each word in tags
    for word, tag in tagged_word:
        # Tag conversion
        wn_tag = penn_to_wn(tag)
        if wn_tag not in (wn.NOUN, wn.ADJ, wn.ADV):
            continue

        lemma = lemmatizer.lemmatize(word, pos=wn_tag)
        if not lemma:
            continue

        synsets = wn.synsets(lemma, pos=wn_tag)
        if not synsets:
            continue

        # Take the first sense, the most common
        synset = synsets[0]

        swn_synset = swn.senti_synset(synset.name())

        # Calculate score by taking pos - neg
        sentiment += swn_synset.pos_score() - swn_synset.neg_score()

        return sentiment

    # Return default scores if sentiment scores are not found or if POS tag is not mapped to WordNet tag
    return None


"""
    Retrieve sentiment scores of a word.

    Parameters:
        word (str): A single word for which sentiment score is to be retrieved.

    Returns:
        tuple: Tuple containing the word and its sentiment score if found, otherwise None.
"""


def retrieve_sentiment_scores(word):
    scores = get_word_sentiment_score(word)

    if scores:

        return word, scores
    else:

        return word, None  # Return default scores if sentiment scores are not found


"""
    Map sentiment score to sentiment label.

    Parameters:
        score (float): A sentiment score (positive, negative, or neutral).

    Returns:
        str: Corresponding sentiment label ('positive', 'negative', or 'neutral').
"""


def score_to_label(score):
    if score > 0:
        return "positive"

    elif score < 0:
        return "negative"

    else:
        return "neutral"


def run():
    # Initialize SparkSession
    spark = SparkSession.builder.appName("task5").getOrCreate()

    # Read the entire file
    tweets = spark.read.text("../data/Airline-Full-Non-Ag-DFE-Sentiment.csv")

    """Preprocessing of text"""
    # Split lines into parts
    parts = tweets.rdd.map(lambda line: line.value.split(","))

    # Filter for valid parts
    validParts = parts.filter(lambda line: len(line) >= 22 and line[14] is not None and line[21] is not None)

    # Filter out the header row
    header = validParts.first()

    # Take out header which is the first row
    validParts = validParts.filter(lambda line: line != header)

    """Obtaining sentiment scores for each tweet in tweet column"""

    # Split the tweet texts into individual words
    tweets_with_words = validParts.map(lambda tweet: (tweet[3], tweet[21].split(" ")))

    # Apply the function for each word
    word_scores = tweets_with_words.flatMapValues(
        lambda words: [(word, retrieve_sentiment_scores(word)) for word in words])

    # Filter out for any None values
    filter_scores = filtered_word_scores = word_scores.filter(lambda x: x[1][1][1] is not None)

    # Calculate the sentiment score for each word and aggregate them based on id
    tweet_scores = filter_scores.map(lambda x: (x[0], x[1][1][1])) \
        .reduceByKey(lambda score1, score2: score1 + score2)

    # Map sentiment scores to sentiment labels
    tweet_sentiment_labels = tweet_scores.mapValues(score_to_label)

    """Preprocessing of sentiment_gold"""

    # Filter out tweets where tweet[17] is empty or null (*Remove after preprocessing implemented)
    filtered_tweets = validParts.filter(lambda tweet: tweet[17] is not None and tweet[17] != '')

    # Split the sentiment_gold texts into individual words
    sentiment_gold = filtered_tweets.map(lambda tweet: (tweet[3], tweet[17].split(" ")))

    print(sentiment_gold.take(10))
    # Join tweet_sentiment_labels with sentiment_gold_filtered
    joined_data = tweet_sentiment_labels.join(sentiment_gold)

    # Calculate accuracy
    # Get the total count of tweets
    total_tweets = joined_data.count()
    print("Total tweets = ", total_tweets)

    # Get the count of tweets that match based on sentiment label
    correct_predictions = joined_data.filter(lambda x: x[1][0] == x[1][1][0]).count()

    print("Correctly predicted tweet sentiment = ", correct_predictions)
    accuracy = correct_predictions / total_tweets

    with open("../results/task5/accuracy.csv", "w") as file:
        file.write("accuracy\n")
        file.write(str(accuracy))


if __name__ == "__main__":
    run()
