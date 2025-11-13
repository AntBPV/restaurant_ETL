from pyspark.sql import SparkSession
from pyspark.sql import functions as F
from pyspark.sql.types import *

def create_spark_session():
    return SparkSession.builder.appName("Restaurant_ETL").getOrCreate()

def cleaner(df):
    for col in df.columns:
        df = df.withColumnRenamed(col, col.strip().replace(" ", "_"))

    df = df.dropDuplicates()

    categorical_cols = [
        "Gender", "VisitFrequency", "PreferredCuisine",
        "TimeOfVisit", "DiningOccasion", "MealType"
    ]

    categorical_cols = [c for c in categorical_cols if c in df.columns]

    for c in categorical_cols:
        df = df.withColumn(c, F.trim(F.initcap(F.col(c))))

    numeric_cols = [
        "Age", "Income", "AverageSpend", "GroupSize", "WaitTime",
        "ServiceRating", "FoodRating", "AmbianceRating", "HighSatisfaction"
    ]

    for c in numeric_cols:
        if c in df.columns:
            df = df.withColumn(c, F.col(c).try_cast(DoubleType()))

    df = df.filter((F.col("Age") > 0) & (F.col("Age") < 100))
    df = df.filter(F.col("Income") > 0)
    df = df.filter(F.col("AverageSpend") > 0)
    df = df.filter(F.col("WaitTime") >= 0)

    if "Gender" in df.columns:
        df = df.withColumn(
            "Gender",
            F.when(F.lower(F.col("Gender")).isin("male", "m"), "Male")
            .when(F.lower(F.col("Gender")).isin("female", "f"), "Female")
            .otherwise("Other")
        )

    df = df.withColumn("Income", F.round(F.col("Income"), 2))
    df = df.withColumn("AverageSpend", F.round(F.col("AverageSpend"), 2))

    return df

def main():
    spark = create_spark_session()
    df = spark.read.csv("restaurant_customer_satisfaction.csv", header = True, inferSchema = True)

    df_clean = cleaner(df)

    # You'll get a folder with the name written below
    df_clean.write.mode("overwrite").option("header", "true").csv("restaurant_customer_satisfaction_clean")

if __name__ == "__main__":
    main()