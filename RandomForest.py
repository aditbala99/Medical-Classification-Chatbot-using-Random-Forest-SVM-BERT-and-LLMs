from pyspark.sql import SparkSession
from pyspark.ml.feature import Tokenizer, HashingTF, IDF
from pyspark.ml.classification import RandomForestClassifier
from pyspark.ml import Pipeline
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from pyspark.ml.evaluation import MulticlassClassificationEvaluator
from pyspark.sql.functions import col

# Create a Spark session
spark = SparkSession.builder.appName("RandomForest").getOrCreate()


# Load data from GCS
train_df = spark.read.csv("train.csv", header=True, inferSchema=True)
validation_df = spark.read.csv("validation.csv", header=True, inferSchema=True)

train_df = train_df.na.drop(subset=["question"])
train_df = validation_df.na.drop(subset=["question"])

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

# Build an SVM model for 'cop'
rf = RandomForestClassifier(labelCol="cop", featuresCol="features", numTrees=100, seed=42)
rf = rf.fit(train_df)

# Predict 'cop' values on the validation set
predictions = rf.transform(validation_df)

# Evaluate the model for 'cop' on the validation set
evaluator = MulticlassClassificationEvaluator(labelCol="cop", predictionCol="prediction", metricName="accuracy")
cop_val_accuracy = evaluator.evaluate(predictions)
print(f'Accuracy for cop prediction on validation set: {cop_val_accuracy}')
#print(f"Accuracy for cop prediction on validation set: {accuracy}")