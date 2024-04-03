import os
import sys

from pyspark.sql import SparkSession, Window
from pyspark.sql.functions import count, rank, col


def run():
    os.environ['PYSPARK_PYTHON'] = sys.executable
    os.environ['PYSPARK_DRIVER_PYTHON'] = sys.executable

    spark = SparkSession.builder.appName("task4").getOrCreate()

    tweets = spark.read.csv("../data/Twitter_Airline Dataset",
                          header=True, inferSchema=True)
    country_names = spark.read.csv("../data/ISO-3166-alpha3.tsv",
                          sep="\t", header=False, inferSchema=True).toDF("_code", "_name")
    tweets.printSchema()
    country_names.printSchema()

    country_count = tweets.groupby(["_country", "airline"]).agg(count("_country").alias("count"))
    country_count = country_count.join(country_names, country_count["_country"] == country_names["_code"])\
        .select(country_names["_name"].alias("country"), country_count["airline"], country_count["count"])

    window = Window.partitionBy(country_count["country"]).orderBy(country_count["count"].desc())
    country_count = country_count.select("*", rank().over(window).alias("rank")).filter(col("rank") <= 1).drop("rank")
    country_count.show()
    country_count.coalesce(1).write.csv(path="../results/task5", mode="overwrite", header=True)


if __name__ == "__main__":
    run()
