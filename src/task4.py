import os
import sys

from pyspark.sql import SparkSession
from pyspark.sql.functions import median, mean


def run():
    os.environ['PYSPARK_PYTHON'] = sys.executable
    os.environ['PYSPARK_DRIVER_PYTHON'] = sys.executable

    spark = SparkSession.builder.appName("task4").getOrCreate()

    data = spark.read.csv("../data/Twitter_Airline Dataset",
                          header=True, inferSchema=True)
    data.printSchema()

    data = data.withColumn("_trust", data["_trust"].cast("float"))

    channel_tp_mean_median = data.groupby("_channel").agg(mean("_trust").alias("mean_trust"),
                                                          median("_trust").alias("median_trust"))
    channel_tp_mean_median = channel_tp_mean_median.na.drop(subset=["_channel"]).fillna(value=0.0)

    channel_tp_mean_median.show(100)
    channel_tp_mean_median.coalesce(1).write.csv(path="../results/task4")


if __name__ == "__main__":
    run()
