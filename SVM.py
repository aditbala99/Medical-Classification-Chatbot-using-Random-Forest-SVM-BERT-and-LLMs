from pyspark.sql import SparkSession
from pyspark.ml.feature import Tokenizer, HashingTF, IDF
from pyspark.ml.classification import LinearSVC, OneVsRest
from pyspark.ml import Pipeline
from pyspark.ml.evaluation import MulticlassClassificationEvaluator

# Create a Spark session
spark = SparkSession.builder.appName("SVMModel").getOrCreate()

# Load data from GCS
train_df = spark.read.csv("gs://sabigdataclass/train2.csv", header=True, inferSchema=True)
validation_df = spark.read.csv("gs://sabigdataclass/validation.csv", header=True, inferSchema=True)

# Drop rows with missing values in the 'question' column
train_df = train_df.na.drop(subset=["question"])
validation_df = validation_df.na.drop(subset=["question"])

# Select relevant columns
train_df = train_df.select("question", "cop", "exp")
validation_df = validation_df.select("question", "cop", "exp")

# Use TF-IDF for text vectorization
tokenizer = Tokenizer(inputCol="question", outputCol="words")
hashing_tf = HashingTF(inputCol="words", outputCol="raw_features", numFeatures=5000)
idf = IDF(inputCol="raw_features", outputCol="features")

# Build a pipeline for text vectorization
pipeline = Pipeline(stages=[tokenizer, hashing_tf, idf])
pipeline_model = pipeline.fit(train_df)

# Transform the training and validation data
train_df = pipeline_model.transform(train_df)
validation_df = pipeline_model.transform(validation_df)

# Build a LinearSVC model for 'cop'
lsvc = LinearSVC(labelCol="cop", featuresCol="features", maxIter=100, regParam=0.01)

# Wrap LinearSVC in a OneVsRest wrapper for multi-class classification
ovr = OneVsRest(classifier=lsvc, labelCol="cop", featuresCol="features")

# Fit the OneVsRest model
ovr_model = ovr.fit(train_df)

# Predict 'cop' values on the validation set
predictions = ovr_model.transform(validation_df)

# Evaluate the model for 'cop' on the validation set
evaluator = MulticlassClassificationEvaluator(labelCol="cop", predictionCol="prediction", metricName="accuracy")
cop_val_accuracy = evaluator.evaluate(predictions)
print(f'Accuracy for cop prediction on validation set: {cop_val_accuracy}')
