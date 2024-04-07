# Contributor: Wong Yu Fei 

import os
import sys

from pyspark.sql import SparkSession
from pyspark.sql.functions import median, mean, round

from utils import load_tweets, convert_data_type


def compute_channel_trustpoints_stats(tweets):
    # convert dtype of column for processing
    tweets = convert_data_type(tweets, "_trust", "float")

    # get trust points mean and median of every channel
    channel_trustpoints = tweets.groupby("_channel")\
        .agg(mean("_trust").alias("mean_trust"), median("_trust").alias("median_trust"))

    # round to 2 d.p.
    channel_trustpoints = channel_trustpoints.withColumn("mean_trust", round(channel_trustpoints["mean_trust"], 2))
    channel_trustpoints = channel_trustpoints.withColumn("median_trust", round(channel_trustpoints["median_trust"], 2))

    return channel_trustpoints


def run():
    os.environ['PYSPARK_PYTHON'] = sys.executable
    os.environ['PYSPARK_DRIVER_PYTHON'] = sys.executable

    # create new spark session and load data
    spark = SparkSession.builder.appName("task4").getOrCreate()
    tweets = load_tweets(spark, "../results/task1/*.csv")

    print(f"Total number of tweets: {tweets.count()}")

    # compute mean and median
    channel_trustpoints = compute_channel_trustpoints_stats(tweets)

    # display results
    channel_trustpoints.show(channel_trustpoints.count())

    # save results
    channel_trustpoints.coalesce(1).write.csv(path="../results/task4", mode="overwrite", header=True)


if __name__ == "__main__":
    run()
