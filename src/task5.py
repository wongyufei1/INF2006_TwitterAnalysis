# Contributor: Tan Yi Wei Isaac
# Import Relevant Libraries
import os
import sys

from pyspark.sql import SparkSession
from utils import load_tweets, convert_data_type
from pyspark.sql import functions as F

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
    # Uncomment to download relevant files first
    # nltk.download('sentiwordnet')
    # nltk.download('wordnet')
    # nltk.download('omw-1.4')
    # nltk.download('averaged_perceptron_tagger')
    from nltk.corpus import sentiwordnet as swn
    from nltk.corpus import wordnet as wn
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


def run():
    # Initialize SparkSession
    spark = SparkSession.builder.appName("task5").getOrCreate()

    # Load dataset
    tweets = load_tweets(spark, "../results/task1/part-00000-23db5c69-b7e0-4e42-b029-7d86ce677e0a-c000.csv")
    tweets.printSchema()
    """Obtaining sentiment scores for each tweet in tweet column"""

    # Split the tweet texts into individual words according to _unit_id
    tweets_with_words = tweets.select("_unit_id", F.split("text", " ").alias("words"))

    # Explode the array of words into separate rows
    tweets_with_words = tweets_with_words.select("_unit_id", F.explode("words").alias("word"))

    # Apply the function for each word
    word_scores = tweets_with_words.withColumn("sentiment", F.udf(retrieve_sentiment_scores)("word"))

    # Filter out for any None values
    filtered_word_scores = word_scores.filter(word_scores["sentiment"].isNotNull())

    # Calculate the sentiment score for each word and aggregate them based on id
    tweet_scores = filtered_word_scores.groupBy("_unit_id").agg(F.sum("sentiment").alias("total_sentiment"))

    # Map sentiment scores to sentiment labels based on certain conditions
    tweet_sentiment_labels = tweet_scores.select("_unit_id", F.when(F.col("total_sentiment") > 0, "positive")
                                                 .when(F.col("total_sentiment") < 0, "negative")
                                                 .otherwise("neutral").alias("sentiment_label"))

    """Preprocessing of sentiment_gold"""
    # Filter out tweets where tweet[17] is empty or null (*Remove after preprocessing implemented)
    filtered_tweets = tweets.filter(tweets["airline_sentiment_gold"].isNotNull() | (tweets["airline_sentiment_gold"] != ''))

    # Join tweet_sentiment_labels with sentiment_gold_filtered
    joined_data = tweet_sentiment_labels.join(filtered_tweets)

    # Calculate accuracy
    # Get the total count of tweets
    total_tweets = joined_data.count()
    print("Total tweets = ", total_tweets)

    # Get the count of tweets that match based on sentiment label
    correct_predictions = joined_data.filter(
        joined_data["sentiment_label"] == joined_data["airline_sentiment_gold"]).count()

    print("Correctly predicted tweet sentiment = ", correct_predictions)

    accuracy = correct_predictions / total_tweets
    print("Accuracy of prediction = ", accuracy)

    # with open("../results/task5/accuracy.csv", "w") as file:
    #     file.write("accuracy\n")
    #     file.write(str(accuracy))

    print("Analysis Completed")


if __name__ == "__main__":
    run()
