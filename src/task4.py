import os
import sys

from pyspark.sql import SparkSession
from pyspark.sql.functions import median, mean, round

from src.utils import load_tweets


def compute_channel_trustpoints_stats(tweets):
    # get trust points mean and median of every channel
    channel_trustpoints = tweets.groupby("_channel")\
        .agg(mean("_trust").alias("mean_trust"), median("_trust").alias("median_trust"))

    # clean up NULLs
    channel_trustpoints = channel_trustpoints.na.drop(subset=["_channel"]).fillna(value=0.0)

    # round to 2 d.p.
    channel_trustpoints = channel_trustpoints.withColumn("mean_trust", round(channel_trustpoints["mean_trust"], 2))
    channel_trustpoints = channel_trustpoints.withColumn("median_trust", round(channel_trustpoints["median_trust"], 2))

    return channel_trustpoints


def run():
    os.environ['PYSPARK_PYTHON'] = sys.executable
    os.environ['PYSPARK_DRIVER_PYTHON'] = sys.executable

    spark = SparkSession.builder.appName("task4").getOrCreate()
    tweets = load_tweets(spark, "../data/Twitter_Airline Dataset")

    tweets.printSchema()
    tweets = tweets.withColumn("_trust", tweets["_trust"].cast("float"))

    channel_trustpoints = compute_channel_trustpoints_stats(tweets)

    channel_trustpoints.show(100)
    channel_trustpoints.coalesce(1).write.csv(path="../results/task4", mode="overwrite")


if __name__ == "__main__":
    run()
