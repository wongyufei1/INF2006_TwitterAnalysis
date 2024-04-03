def load_tweets(spark, path):
    return spark.read.csv(path, header=True, inferSchema=True)


def load_country_codes(spark, path):
    return spark.read.csv(path, sep="\t", header=False, inferSchema=True)\
        .toDF("code", "name")
