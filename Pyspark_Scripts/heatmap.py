from pyspark.sql import Row
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

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
from pyspark.sql.window import Window
window_spec = Window.partitionBy("year")
agg_df = agg_df.withColumn(
    "normalizedDistribution",
    F.col("totalDistribution") / F.sum("totalDistribution").over(window_spec)
)

# Convert to Pandas DataFrame for visualization
heatmap_data = agg_df.groupBy("year").pivot("topicIndex").agg(F.first("normalizedDistribution")).toPandas()
heatmap_data.set_index("year", inplace=True)

# Rename columns for visualization
num_topics = len(heatmap_data.columns)  # Dynamically detect the number of topics
heatmap_data.columns = [f"Topic {i}" for i in range(num_topics)]

# Plot the heatmap
plt.figure(figsize=(10, 8))
sns.heatmap(heatmap_data, cmap="coolwarm", annot=False, fmt=".2f", cbar_kws={'label': 'Normalized Prevalence'})
plt.title("Normalized Topic Prevalence by Year")
plt.ylabel("Year")
plt.xlabel("Topics")
plt.tight_layout()

# Save the heatmap to a file
output_file = "/proj/cse349-449/djd225/final/good_heatmap.png"
plt.savefig(output_file, dpi=300)
plt.close()

print(f"Heatmap saved to {output_file}")

