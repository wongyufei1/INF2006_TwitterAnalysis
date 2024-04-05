def load_tweets(spark, path):
    return spark.read.format("csv").option("header", "true").option("delimiter", ",").option("quote", "\"") \
        .option("escape", "\"").option("multiline", True).load(path)


def load_country_codes(spark, path):
    return spark.read.csv(path, sep="\t", header=False, inferSchema=True) \
        .toDF("code", "name")


def convert_data_type(dataframe, column_name, data_type):
    dataframe = dataframe.withColumn(column_name, dataframe[column_name].cast(data_type))
    return dataframe
