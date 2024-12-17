from pyspark.sql import functions as F
import sparknlp
from sparknlp.base import DocumentAssembler, Finisher
from sparknlp.annotator import Tokenizer, Normalizer, StopWordsCleaner, Stemmer
from pyspark.ml import Pipeline
from pyspark.ml.feature import CountVectorizer, IDF
from pyspark.ml.clustering import LDA
from gensim.models import CoherenceModel
from gensim.corpora import Dictionary
import os
import re
import html
from pyspark.sql.functions import udf
from pyspark.sql.types import StringType
import matplotlib.pyplot as plt

# Start Spark NLP
spark = sparknlp.start()

# Load XML with nested structure
data_source = '/proj/cse349-449/djd225/final/exports/*/*.xml'
df = spark.read.format("xml") \
    .option("rowTag", "page") \
    .load(data_source)

# Select relevant columns and handle nested structure
df = df.select(
    F.col("_title").alias("title"),
    F.col("revisions.rev._timestamp").alias("timestamp"),
    F.col("revisions.rev._VALUE").alias("content")
)

# Remove null or empty pages
df = df.filter(F.col("content").isNotNull() & (F.length(F.col("content")) > 0))

# Convert timestamp to timestamp type and extract year
df = df.withColumn("timestamp", F.to_timestamp("timestamp")) \
       .withColumn("year", F.year("timestamp"))

def preprocess_wikipedia(text):
    text = html.unescape(text)
    text = re.sub(r"<!--.*?-->", "", text)
    text = re.sub(r"\{\{.*?\}\}", "", text)
    text = re.sub(r"<ref.*?>.*?</ref>", "", text)
    text = re.sub(r"\[\[File:.*?\]\]", "", text)
    text = re.sub(r"http\S+|www\.\S+", "<URL>", text)
    text = re.sub(r"(?i)\bISBN\b", "", text)
    text = re.sub(r"(?i)\bREDIRECT\b", "", text)
    text = re.sub(r"<[^>]+>", "", text)
    return text

preprocess_udf = udf(preprocess_wikipedia, StringType())
df = df.withColumn("content", preprocess_udf(F.col("content")))

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
    .setCaseSensitive(False)\
    .setStopWords(["url", "ref", "date"])

stemmer = Stemmer() \
    .setInputCols(["cleanTokens"]) \
    .setOutputCol("stem")

finisher = Finisher() \
    .setInputCols(["stem"]) \
    .setOutputCols(["tokens"]) \
    .setOutputAsArray(True) \
    .setCleanAnnotations(False)

nlp_pipeline = Pipeline(
    stages=[document_assembler, tokenizer, normalizer, stopwords_cleaner, stemmer, finisher]
)

# Build and apply pipeline
nlp_model = nlp_pipeline.fit(df)
processed_df = nlp_model.transform(df)

# Prepare data for vectorization
tokens_df = processed_df.select("title", "year", "tokens")

# Vectorize tokens
cv = CountVectorizer(inputCol="tokens", outputCol="raw_features", minDF = 5)
cv_model = cv.fit(tokens_df)
vectorized_tokens = cv_model.transform(tokens_df)

# genism stuff
vocab = cv_model.vocabulary
tokens = tokens_df.select("tokens").rdd.flatMap(lambda x: x).collect()
dictionary = Dictionary(tokens)

idf = IDF(inputCol="raw_features", outputCol="features")
idf_model = idf.fit(vectorized_tokens)
vectorized_df = idf_model.transform(vectorized_tokens).select("title", "year", "features")

# Broadcast variables for parallel access
tokens_broadcast = spark.sparkContext.broadcast(tokens)
dictionary_broadcast = spark.sparkContext.broadcast(dictionary)

# Initialize lists
coherence_scores = []
perplexity_scores = []

# Number of topics to try
num_topics_list = [5, 7, 10, 13, 16, 20, 30, 50]

# Loop through each number of topics
for k in num_topics_list:
    print("Starting " + str(k))
    # Train LDA model
    lda = LDA(k=k, maxIter=50)
    lda_model = lda.fit(vectorized_df)

    # Extract topics
    topics = [[vocab[idx] for idx in topic.termIndices] for topic in lda_model.describeTopics(10).collect()]

    # Coherence calculation
    coherence_model = CoherenceModel(
        topics=topics,
        texts=tokens_broadcast.value,
        dictionary=dictionary_broadcast.value,
        coherence="c_v"
    )
    coherence_score = coherence_model.get_coherence()
    coherence_scores.append((k, coherence_score))

    perplexity = lda_model.logPerplexity(vectorized_df)
    perplexity_scores.append((k, perplexity))

    print(f"Done {k}: Coherence={coherence_score:.4f}, Perplexity={perplexity:.4f}")


# Extract values for plotting
x, coherence_y = zip(*coherence_scores)
_, perplexity_y = zip(*perplexity_scores)

# save results
import csv
with open('/proj/cse349-449/djd225/final/lda_scores.csv', 'w', newline='') as csvfile:
    writer = csv.writer(csvfile)
    writer.writerow(["Topics", "Coherence", "Perplexity"])
    writer.writerows(zip(x, coherence_y, perplexity_y))

# Plot coherence and perplexity on the same graph
plt.figure(figsize=(10, 6))
fig, ax1 = plt.subplots()

# Coherence score
color = "tab:blue"
ax1.set_xlabel("Number of Topics (k)")
ax1.set_ylabel("Coherence Score", color=color)
ax1.plot(x, coherence_y, marker="o", label="Coherence Score", color=color)
ax1.tick_params(axis="y", labelcolor=color)

# Perplexity score (secondary y-axis)
ax2 = ax1.twinx()
color = "tab:orange"
ax2.set_ylabel("Perplexity", color=color)
ax2.plot(x, perplexity_y, marker="s", linestyle="--", label="Perplexity", color=color)
ax2.tick_params(axis="y", labelcolor=color)

# Title and layout
plt.title("Coherence Score and Perplexity vs. Number of Topics")
fig.tight_layout()

# Save and show the plot
output_plot_path = "/proj/cse349-449/djd225/final/Coherence_and_Perplexity_vs_Number_of_Topics.png"
plt.savefig(output_plot_path)
print(f"Coherence and Perplexity plot saved to {output_plot_path}")

tokens_broadcast.unpersist()
dictionary_broadcast.unpersist()
