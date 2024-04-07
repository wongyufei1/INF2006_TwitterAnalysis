# Contributor: Wong Yu Fei
import os
import sys

from pyspark.sql import SparkSession, Window
from pyspark.sql.functions import count, rank, col

from utils import load_tweets, load_country_codes


def compute_popular_airline_by_country(tweets, country_codes):
    # count number of tweets for each airline and country pair
    tweets_count = tweets.groupby(["_country", "airline"]).agg(count("_country").alias("count"))

    # get country names using country codes
    tweets_count = tweets_count.join(country_codes, tweets_count["_country"] == country_codes["code"])\
        .select(country_codes["name"].alias("country"), tweets_count["airline"], tweets_count["count"])

    # get the top ranked airline within each country with the most tweets in descending order
    window = Window.partitionBy(tweets_count["country"]).orderBy(tweets_count["count"].desc())
    popular_airlines = tweets_count.select("*", rank().over(window).alias("rank")).filter(col("rank") <= 1).drop("rank")

    return popular_airlines


def run():
    os.environ['PYSPARK_PYTHON'] = sys.executable
    os.environ['PYSPARK_DRIVER_PYTHON'] = sys.executable

    # create new spark session and load data
    spark = SparkSession.builder.appName("task6").getOrCreate()
    tweets = load_tweets(spark, "../results/task1/*.csv")
    country_codes = load_country_codes(spark, "../data/ISO-3166-alpha3.tsv")

    print(f"Total number of tweets: {tweets.count()}")

    # compute most tweeted about airline in each country
    popular_airlines = compute_popular_airline_by_country(tweets, country_codes)

    # display results
    popular_airlines.show(popular_airlines.count())

    # save results
    popular_airlines.coalesce(1).write.csv(path="../results/task6.1", mode="overwrite", header=True)


if __name__ == "__main__":
    run()
