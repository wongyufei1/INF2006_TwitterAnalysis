# Contributor: Derrick Lim 

import os
import sys

from pyspark.sql import SparkSession
from pyspark.sql.functions import count,col
from src.utils import load_tweets, load_country_codes


def compute_most_complaints_by_country(tweets, country_codes):

    # Filter only negative tweets
    negative_tweets = tweets.filter(col("airline_sentiment") == "negative")
    negative_tweets.show()

    # Extract country code from Twttier Dataset from the column '_country'
    twitter_country_count= negative_tweets.groupby(["_country"]).agg(count("_country").alias("count"))

    # Join the Twitter Dataset with Country Code Dataset to retrieve country name, country code and count in descending order
    complaints_by_country = twitter_country_count.join(country_codes, twitter_country_count["_country"] == country_codes["code"]) \
    .select(country_codes["name"].alias("country"),
            twitter_country_count["_country"].alias("code"),
            twitter_country_count["count"]) \
            .orderBy(col("count").desc())
    
    return complaints_by_country


def run():
    os.environ['PYSPARK_PYTHON'] = sys.executable
    os.environ['PYSPARK_DRIVER_PYTHON'] = sys.executable

    spark = SparkSession.builder.appName("task3").getOrCreate()

    tweets = load_tweets(spark, "../results/task1/*.csv")
    country_codes = load_country_codes(spark, "../data/ISO-3166-alpha3.tsv")

    tweets.printSchema()
    country_codes.printSchema()

    complaints_by_country = compute_most_complaints_by_country(tweets, country_codes)

    # Display the result
    complaints_by_country.show()
    complaints_by_country.coalesce(1).write.csv(path="../results/task3", mode="overwrite", header=True)

if __name__ == "__main__":
    run()
