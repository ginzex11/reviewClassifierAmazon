Amazon Review Sentiment Analysis and Pattern Mining
Overview
This project performs sentiment analysis and pattern mining on Amazon review data (Reviews.csv) using PySpark for big data processing. It preprocesses reviews, classifies sentiments (positive, neutral, negative), identifies subjective summaries, and mines frequent patterns and association rules. The analysis is conducted in batches over quarterly time windows, with visualizations and statistical summaries saved for insights into product sentiments and trends.
Objectives

Preprocess Amazon review data to ensure quality and relevance (2000–2022).
Classify review sentiments and subjectivity using a two-phase machine learning pipeline.
Mine frequent patterns and association rules using Apriori and FP-Growth algorithms.
Generate visualizations and statistical summaries to analyze sentiment distributions, product trends, and pattern frequencies.


Outputs:

Processed Data: Preprocessed DataFrame with sentiment scores and features.
Visualizations:
Data analysis plots (data_analysis_plots.png): Review counts, score distribution, helpfulness trends, correlations.
Sentiment distributions, top products by sentiment, frequent itemsets, association rules, and patterns over time.



Methodology

Preprocessing (ReviewAnalyzer):

Load Reviews.csv (columns: Id, ProductId, UserId, Score, Time, Summary, Text, etc.).
Filter valid data (2000–2022, non-null key columns), compute HelpfulnessScore, and convert timestamps.
Output: Preprocessed DataFrame with Year, Month, and cleaned text.


Sentiment Analysis (SentimentAnalyzer):

Phase 1 (Subjectivity Detection):
Use Logistic Regression to classify summaries as subjective (TextBlob subjectivity > 0.5) or objective.
Features: Tokenized summaries, stop words removed, TF-IDF (1000 features).


Phase 2 (Sentiment Classification):
Classify subjective reviews as Positive, Neutral, or Negative using Logistic Regression.
Features: Tokenized review text, TF-IDF (500 features), class weights for balancing.
Balancing: Undersample majority (Positive) and oversample minority classes (Negative, Neutral).


Process data in quarterly batches, evaluate metrics (accuracy, precision, recall, F1).


Pattern Mining:

Apriori: Custom implementation with Trie to find frequent itemsets (min support: 0.05, max iterations: 3).
FP-Growth: PySpark ML implementation for frequent itemsets and association rules (min support: 0.05, min confidence: 0.3).
Extract unigrams and bigrams with CountVectorizer, paired with sentiments for pattern analysis.
Analyze patterns by year to identify temporal trends.


Visualizations (DataAnalyzer, BatchProcessor):

Review counts by year, score distribution, helpfulness trends, correlation heatmap.
Sentiment distributions, top products by positive/negative ratios, frequent itemsets, association rules.
Temporal analysis: Sentiment and pattern trends over years.
Algorithm comparison: Apriori vs. FP-Growth (itemset counts, runtime).



Key Results

Preprocessing: Filtered ~563,000 reviews (2000–2022), removed invalid entries.
Sentiment Analysis: Achieved balanced classification with ~73% accuracy for "Positive" sentiments in Phase 2.
Pattern Mining: Identified frequent sentiment patterns (e.g., co-occurring positive/negative terms) and rules (e.g., "great:Positive → quality:Positive").
Visualizations: Highlighted dominant positive sentiments, top products, and temporal shifts in patterns.
Performance: FP-Growth outperformed Apriori in runtime for large datasets.

Limitations

Limited to reviews from 2000–2022; newer data may alter patterns.
TextBlob sentiment scores missed nuanced sentiments.
Class imbalance (Positive-heavy) required aggressive balancing, potentially affecting generalization.
High memory usage for large datasets; requires optimized Spark configuration.

Future Work

Incorporate advanced NLP models (e.g., BERT) for sentiment analysis.
Expand pattern mining to include higher-order n-grams and cross-product patterns.
Implement real-time processing for streaming review data.
Enhance visualizations with interactive dashboards (e.g., Plotly).
