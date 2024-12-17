from pyspark.sql import functions as F
import sparknlp
from sparknlp.base import DocumentAssembler, Finisher
from sparknlp.annotator import Tokenizer, Normalizer, StopWordsCleaner, LemmatizerModel, Stemmer, NGramGenerator
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

# Define different NLP pipelines
def build_pipeline(use_lemmatizer=False, use_stemmer=False, lowercase=True, remove_stopwords=True, use_ngrams=None, cleanup_mode="shrink"):
    document_assembler = DocumentAssembler() \
        .setInputCol("content") \
        .setOutputCol("document") \
        .setCleanupMode(cleanup_mode)
    
    tokenizer = Tokenizer() \
        .setInputCols(["document"]) \
        .setOutputCol("token")
    
    normalizer = Normalizer() \
        .setInputCols(["token"]) \
        .setOutputCol("normalized") \
        .setLowercase(lowercase)
    
    stages = [document_assembler, tokenizer, normalizer]
    
    if remove_stopwords:
        stopwords_cleaner = StopWordsCleaner() \
            .setInputCols("normalized") \
            .setOutputCol("cleanTokens") \
            .setCaseSensitive(False) \
            .setStopWords(["url", "ref", "date", "accessed"])
        stages.append(stopwords_cleaner)
        token_input_col = "cleanTokens"
    else:
        token_input_col = "normalized"

    if use_lemmatizer:
        lemmatizer = LemmatizerModel.pretrained() \
            .setInputCols([token_input_col]) \
            .setOutputCol("lemma")
        stages.append(lemmatizer)
        final_output = "lemma"
    else:
        final_output = token_input_col
    
    if use_stemmer:
        stemmer = Stemmer() \
            .setInputCols([final_output]) \
            .setOutputCol("stem")
        stages.append(stemmer)
        final_output = "stem"
    
    if use_ngrams:
        ngram_generator = NGramGenerator() \
            .setInputCols([final_output]) \
            .setOutputCol("ngrams") \
            .setN(use_ngrams)
        stages.append(ngram_generator)
        final_output = "ngrams"
    
    finisher = Finisher() \
        .setInputCols([final_output]) \
        .setOutputCols(["tokens"]) \
        .setOutputAsArray(True) \
        .setCleanAnnotations(False)
    stages.append(finisher)
    
    return Pipeline(stages=stages)

# Define pipeline variants
pipeline_variants = [
    {"use_lemmatizer": False, "use_stemmer": False, "lowercase": True, "remove_stopwords": True, "use_ngrams": None, "cleanup_mode": "shrink"},
    {"use_lemmatizer": False, "use_stemmer": True, "lowercase": True, "remove_stopwords": True, "use_ngrams": None, "cleanup_mode": "shrink"},
    {"use_lemmatizer": False, "use_stemmer": True, "lowercase": True, "remove_stopwords": True, "use_ngrams": 2, "cleanup_mode": "shrink"},
]

# Directory to save results
output_dir = "/proj/cse349-449/djd225/final/lda_results"
os.makedirs(output_dir, exist_ok=True)

with open(f"{output_dir}/output.txt", 'w', encoding='utf-8') as output_file:
    for i, params in enumerate(pipeline_variants, start=1):
        print(f"\nPipeline Variant {i}\n")
        output_file.write(f"\nPipeline Variant {i}: {params}\n")
        
        # Build and apply pipeline
        pipeline = build_pipeline(**params)
        nlp_model = pipeline.fit(df)
        processed_df = nlp_model.transform(df)
        
        # Prepare data for vectorization
        tokens_df = processed_df.select("title", "year", "tokens")
        
        # Vectorize tokens
        cv = CountVectorizer(inputCol="tokens", outputCol="raw_features", minDF=10)
        cv_model = cv.fit(tokens_df)
        vectorized_tokens = cv_model.transform(tokens_df)
        
        idf = IDF(inputCol="raw_features", outputCol="features")
        idf_model = idf.fit(vectorized_tokens)
        vectorized_df = idf_model.transform(vectorized_tokens).select("title", "year", "features")
        
        # Apply LDA
        lda = LDA(k=5, maxIter=30)
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
