# Contributor: Cheryl Toh 

import os
import sys

from pyspark.sql import SparkSession
from pyspark.sql.functions import regexp_replace, udf

from utils import load_tweets, convert_data_type


def select_columns(dataframe):
    selected_columns = ["_unit_id", "_channel", "_trust", "_country", "airline", "airline_sentiment", "text",
                        "airline_sentiment_gold", "negativereason", "tweet_coord", "tweet_location", "tweet_created"]

    return dataframe.select(selected_columns)


def remove_duplicates(dataframe):
    # Remove duplicate rows
    deduplicated_tweets = dataframe.dropDuplicates()

    return deduplicated_tweets


def ascii_ignore(df_text):
    return df_text.encode('ascii', 'ignore').decode('ascii')


ascii_udf = udf(ascii_ignore)


def clean_text(dataframe, column_name):
    # Remove punctuation
    dataframe = dataframe.withColumn(column_name, regexp_replace(column_name, '[^\w\s]', ''))

    # Remove emojis (assuming emojis are represented as unicode characters)
    dataframe = dataframe.withColumn(column_name, regexp_replace(column_name, '[\U0001F600-\U0001F6FF]', ''))

    # Remove any urls found in the text
    dataframe = dataframe.withColumn(column_name, regexp_replace(column_name, r'\b\w*http\w*\b', ''))

    # Remove any UDF
    dataframe = dataframe.withColumn(column_name, ascii_udf(column_name))

    # Remove emoticons like :D or :p
    dataframe = dataframe.withColumn(column_name, regexp_replace(column_name, r":[\w]+\b", ''))

    # Remove any non-alphanumeric numbers
    dataframe = dataframe.withColumn(column_name, regexp_replace(column_name, '[^a-zA-Z0-9\s]', ''))

    return dataframe


def run():
    os.environ['PYSPARK_PYTHON'] = sys.executable
    os.environ['PYSPARK_DRIVER_PYTHON'] = sys.executable

    spark = SparkSession.builder.appName("task1").getOrCreate()
    tweets = load_tweets(spark, "../data/Twitter_Airline Dataset")

    # remove duplicates
    cleaned_tweets = remove_duplicates(tweets)

    # select columns needed for the tasks
    cleaned_tweets = select_columns(cleaned_tweets)

    # # clean the text column
    cleaned_tweets = clean_text(cleaned_tweets, "text")

    # convert to correct data type
    cleaned_tweets = convert_data_type(cleaned_tweets, "_trust", "float")

    # print schema to check if data type is correct
    cleaned_tweets.printSchema()

    print(cleaned_tweets.show(5))

    cleaned_tweets.coalesce(1).write.csv(path="../results/task1", mode="overwrite", header=True)


if __name__ == "__main__":
    run()
