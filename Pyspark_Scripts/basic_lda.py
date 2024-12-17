from pyspark.sql import functions as F
from pyspark.sql.window import Window
import sparknlp
from sparknlp.base import DocumentAssembler, Finisher
from sparknlp.annotator import Tokenizer, Normalizer, StopWordsCleaner, Stemmer
from pyspark.ml import Pipeline
import os

# Start Spark NLP
spark = sparknlp.start()

# Load XML with nested structure
data_source = '/proj/cse349-449/djd225/final/exports/*/*.xml'
df = spark.read.format("xml") \
    .option("rowTag", "page") \
    .load(data_source)

# Select relevant columns and handle nested structure
df = df.select(
    F.col("_title").alias("title"),  # Page title
    F.col("revisions.rev._timestamp").alias("timestamp"),
    F.col("revisions.rev._VALUE").alias("content")
)

# Convert timestamp to timestamp type
df = df.withColumn("timestamp", F.to_timestamp("timestamp"))

# Extract year from the timestamp
df = df.withColumn("year", F.year("timestamp"))

# Define time period ranges and assign a period label
df = df.withColumn(
    "period",
    F.when((F.col("year") >= 2001) & (F.col("year") <= 2005), "2001-2005")
     .when((F.col("year") >= 2006) & (F.col("year") <= 2011), "2006-2011")
     .when((F.col("year") >= 2012) & (F.col("year") <= 2018), "2012-2018")
     .when((F.col("year") >= 2019) & (F.col("year") <= 2024), "2019-2024")
)

# Group by title and period, and concatenate content within each group
grouped_df = df.groupBy("title", "period") \
    .agg(F.concat_ws(" ", F.collect_list("content")).alias("content"))

# NLP processing for topic modeling
document_assembler = DocumentAssembler() \
    .setInputCol("content") \
    .setOutputCol("document") \
    .setCleanupMode("shrink")

tokenizer = Tokenizer() \
    .setInputCols(["document"]) \
    .setOutputCol("token")

normalizer = Normalizer() \
    .setInputCols(["token"]) \
    .setOutputCol("normalized") \
    .setLowercase(True)

stopwords_cleaner = StopWordsCleaner()\
    .setInputCols("normalized")\
    .setOutputCol("cleanTokens")\
    .setCaseSensitive(False)

stemmer = Stemmer() \
    .setInputCols(["cleanTokens"]) \
    .setOutputCol("stem")

finisher = Finisher() \
    .setInputCols(["stem"]) \
    .setOutputCols(["tokens"]) \
    .setOutputAsArray(True) \
    .setCleanAnnotations(False)

# Build and apply the pipeline
nlp_pipeline = Pipeline(
    stages=[document_assembler, tokenizer, normalizer, stopwords_cleaner, stemmer, finisher]
)
nlp_model = nlp_pipeline.fit(grouped_df)
processed_df = nlp_model.transform(grouped_df)

# Prepare data for LDA topic modeling
tokens_df = processed_df.select("title", "period", "tokens")

# Vectorize tokens
cv = CountVectorizer(inputCol="tokens", outputCol="raw_features")
cv_model = cv.fit(tokens_df)
vectorized_tokens = cv_model.transform(tokens_df)

idf = IDF(inputCol="raw_features", outputCol="features")
idf_model = idf.fit(vectorized_tokens)
vectorized_df = idf_model.transform(vectorized_tokens).drop("raw_features")

# Directory to save results
output_dir = "/proj/cse349-449/djd225/final/lda_results"
os.makedirs(output_dir, exist_ok=True)

# Apply LDA to each period separately
lda = LDA(k=10, maxIter=50)
periods = ["2001-2005", "2006-2011", "2012-2018", "2019-2024"]

with open('output.txt', 'w') as output_file:
    for period in periods:
        # Writing the period header to the file
        output_file.write(f"\nTopics for period {period}:\n")

        # Filtering and fitting the LDA model for the current period
        period_df = vectorized_df.filter(F.col("period") == period)
        lda_model = lda.fit(period_df)

        # Getting vocabulary and topic indices
        vocab = cv_model.vocabulary
        raw_topics = lda_model.describeTopics().collect()
        topic_inds = [ind.termIndices for ind in raw_topics]

        # Extracting topics
        topics = []
        for topic in topic_inds:
            _topic = [vocab[ind] for ind in topic]
            topics.append(_topic)

        # Writing each topic to the file
        for i, topic in enumerate(topics, start=1):
            output_file.write(f"Topic {i}: {topic}\n")

        # Transforming period_df with LDA model to get topic distribution
        lda_df = lda_model.transform(period_df)

        # Redirecting show output to the file
        show_output = lda_df.select(F.col("title"), F.col("topicDistribution")).toPandas().to_string(index=False)
        output_file.write(show_output + "\n")
