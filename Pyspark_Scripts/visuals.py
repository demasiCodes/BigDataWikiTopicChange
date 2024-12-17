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
from pyspark.sql.types import StringType, IntegerType
from pyspark.ml.linalg import DenseVector
from pyspark.sql.functions import col
import matplotlib.pyplot as plt
import pandas as pd
from pyspark.sql import Row
import seaborn as sns
from pyspark.sql.window import Window

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
    .setStopWords(["url", "ref"])

stemmer = Stemmer() \
    .setInputCols(["cleanTokens"]) \
    .setOutputCol("stem")

finisher = Finisher() \
    .setInputCols(["cleanTokens"]) \
    .setOutputCols(["tokens"]) \
    .setOutputAsArray(True) \
    .setCleanAnnotations(False)

nlp_pipeline = Pipeline(
    stages=[document_assembler, tokenizer, normalizer, stopwords_cleaner, finisher]
)

# Directory to save results
output_dir = "/proj/cse349-449/djd225/final/visual_results"
os.makedirs(output_dir, exist_ok=True)

num_topics = [16]

with open(f"{output_dir}/output.txt", 'w', encoding='utf-8') as output_file:   
    for num_topic in num_topics:
        output_file.write(f"\nFor number of topics, k={num_topic}\n")

        # Build and apply pipeline
        nlp_model = nlp_pipeline.fit(df)
        processed_df = nlp_model.transform(df)
        
        # Prepare data for vectorization
        tokens_df = processed_df.select("title", "year", "tokens")
        
        # Vectorize tokens
        cv = CountVectorizer(inputCol="tokens", outputCol="raw_features")
        cv_model = cv.fit(tokens_df)
        vectorized_tokens = cv_model.transform(tokens_df)
        
        idf = IDF(inputCol="raw_features", outputCol="features")
        idf_model = idf.fit(vectorized_tokens)
        vectorized_df = idf_model.transform(vectorized_tokens).select("title", "year", "features")
        
        # Apply LDA
        lda = LDA(k=num_topic, maxIter=100)
        lda_model = lda.fit(vectorized_df)
        
        # Extract topics
        vocab = cv_model.vocabulary
        raw_topics = lda_model.describeTopics(10).collect()
        topics = [[vocab[idx] for idx in topic.termIndices] for topic in raw_topics]
        
        # Write topics to file
        for j, topic in enumerate(topics, start=1):
            output_file.write(f"Topic {j}: {', '.join(topic)}\n")
        
        # Calculate Perplexity
        perplexity = lda_model.logPerplexity(vectorized_df)
        output_file.write(f"Perplexity: {perplexity}\n")
        
        # Prepare data for coherence calculation
        tokens = tokens_df.select("tokens").rdd.flatMap(lambda x: x).collect()
        dictionary = Dictionary(tokens)
        corpus = [dictionary.doc2bow(text) for text in tokens]
        
        coherence_model = CoherenceModel(
            topics=topics,
            texts=tokens,
            dictionary=dictionary,
            coherence='c_v'
        )
        coherence_score = coherence_model.get_coherence()
        
        # Write coherence score to file
        output_file.write(f"Coherence Score: {coherence_score}\n")

        lda_df = lda_model.transform(vectorized_df)

        # Create Stacked Bar Charts
        specific_titles = ["Intersectionality", "Feminism", "Racism in the United States", "Gender equality"]
        lda_results = lda_df.filter(col('title').isin(specific_titles)) \
                    .select('title', 'year', 'topicDistribution') \
                    .collect()

        # Prepare topic data for all titles
        all_topic_data = []

        for row in lda_results:
            title = row['title']
            year = row['year']
            topic_distribution = row['topicDistribution'].toArray()
            all_topic_data.append([title, year] + list(topic_distribution))

        # Convert to a pandas DataFrame
        columns = ["Title", "Year"] + [f"Topic_{i}" for i in range(len(all_topic_data[0]) - 2)]
        df = pd.DataFrame(all_topic_data, columns=columns)

        # Sort by Title and Year for correct plotting
        df = df.sort_values(by=["Title", "Year"])

        # Create a stacked bar chart for each title
        for title in specific_titles:
            title_df = df[df["Title"] == title]
            years = title_df["Year"]
            topic_distributions = title_df.drop(columns=["Title", "Year"])

            # Create the stacked bar chart
            plt.figure(figsize=(12, 8))
            topic_distributions.plot(kind="bar", stacked=True, figsize=(12, 8), colormap="viridis")

            # Chart formatting
            plt.title(f"Topic Distributions Over Years for '{title}'")
            plt.xlabel("Year")
            plt.ylabel("Topic Distribution")
            plt.xticks(range(len(years)), years, rotation=45)
            plt.legend(title="Topics", bbox_to_anchor=(1.05, 1), loc="upper left")
            plt.tight_layout()

            # Save the chart for each title
            output_plot_path = f"/proj/cse349-449/djd225/final/visual_results/Stacked_{title.replace(' ', '_')}_k{num_topic}.png"
            plt.savefig(output_plot_path)
            plt.close()

        print(f"\nStacked bar charts saved for all specified titles. k = {num_topic}\n")

        # Create Heatmap
        # Collect LDA results from the DataFrame
        lda_results = lda_df.select('year', 'title', 'topicDistribution').collect()

        # Process the collected data
        processed_data = []
        for row in lda_results:
            year = row['year']
            title = row['title']
            topic_distribution = row['topicDistribution']
            for topic_index, topic_value in enumerate(topic_distribution):
                # Ensure topic_value is converted to native float
                processed_data.append(Row(year=year, topicIndex=topic_index, topicValue=float(topic_value)))

        # Create a new DataFrame from the processed data
        exploded_df = spark.createDataFrame(processed_data)

        # Aggregate topic values by year and topic
        agg_df = exploded_df.groupBy("year", "topicIndex").agg(F.sum("topicValue").alias("totalDistribution"))

        # Normalize topic distribution by year
        window_spec = Window.partitionBy("year")
        agg_df = agg_df.withColumn(
            "normalizedDistribution",
            F.col("totalDistribution") / F.sum("totalDistribution").over(window_spec)
        )

        # Convert to Pandas DataFrame for visualization
        heatmap_data = agg_df.groupBy("year").pivot("topicIndex").agg(F.first("normalizedDistribution")).toPandas()
        heatmap_data.set_index("year", inplace=True)

        # Rename columns for visualization
        num_heat_topics = len(heatmap_data.columns)
        heatmap_data.columns = [f"Topic {i}" for i in range(num_heat_topics)]

        # Plot the heatmap
        plt.figure(figsize=(10, 8))
        sns.heatmap(heatmap_data, cmap="coolwarm", annot=False, fmt=".2f", cbar_kws={'label': 'Normalized Prevalence'})
        plt.title("Normalized Topic Prevalence by Year")
        plt.ylabel("Year")
        plt.xlabel("Topics")
        plt.tight_layout()

        # Save the heatmap to a file
        output_heat_path = f"/proj/cse349-449/djd225/final/visual_results/heatmap_k{num_topic}.png"
        plt.savefig(output_heat_path, dpi=300)
        plt.close()

        print(f"Heatmap saved to {output_heat_path}. k = {num_topic}\n")
