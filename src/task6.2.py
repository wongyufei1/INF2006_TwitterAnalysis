import os
import sys

from pyspark.sql import SparkSession
from pyspark.sql.functions import to_timestamp, year, month, count, dayofmonth, when

from INF2006_TwitterAnalysis.src.utils import load_tweets

spark = SparkSession.builder.appName("Temporal Analysis").config("spark.sql.legacy.timeParserPolicy",
                                                                 "LEGACY").getOrCreate()

from pyspark.sql.functions import sum


def time_analysis(tweets):
    # Convert the tweet_created column to a timestamp type
    tweets = tweets.withColumn("tweet_created", to_timestamp(tweets["tweet_created"], "dd/M/yyyy HH:mm"))

    # Extract year, month, and day from the timestamp and count the number of tweets for each sentiment label
    # Aggregate based on _unit_id
    tweet_counts = tweets.groupBy(year("tweet_created").alias("Year"),
                                  month("tweet_created").alias("Month"),
                                  dayofmonth("tweet_created").alias("Day"),
                                  "airline", "airline_sentiment_gold") \
        .agg(count("_unit_id").alias("Sentiment Count"))

    # Add a column to represent sentiment label as a string (positive, negative, neutral)
    # Remove any column that is not positive, negative or neutral
    tweet_counts = tweet_counts.withColumn("airline_sentiment_gold",
                                           when(tweet_counts["airline_sentiment_gold"] == "positive", "positive")
                                           .when(tweet_counts["airline_sentiment_gold"] == "negative", "negative")
                                           .otherwise("neutral"))

    # Aggregate the counts for each sentiment label for the same day
    tweet_counts = tweet_counts.groupBy("Year", "Month", "Day", "airline", "airline_sentiment_gold") \
        .agg(sum("Sentiment Count").alias("Total Sentiment Count"))

    # Order by the date
    tweet_counts = tweet_counts.orderBy("Year", "Month", "Day", "airline", "airline_sentiment_gold")

    return tweet_counts


def run():
    os.environ['PYSPARK_PYTHON'] = sys.executable
    os.environ['PYSPARK_DRIVER_PYTHON'] = sys.executable

    spark = SparkSession.builder.appName("task6.2").getOrCreate()
    tweets = load_tweets(spark, "../results/task1/part-00000-a158c61b-4a8e-4266-b079-18db75fc8da0-c000.csv")

    tweet_counts = time_analysis(tweets)

    tweet_counts.show()
    # save results
    tweet_counts.coalesce(1).write.csv(path="../results/task6.2", mode="overwrite", header=True)


if __name__ == "__main__":
    run()
