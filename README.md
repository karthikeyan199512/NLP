# Natural Language Processing

Repository containing portfolio of NLP projects completed by me for academic, self learning, and hobby purposes. Presented in the form of iPython Notebooks.

## 2-way polarity (positive, negative) classification system for reviews, using NLTK's sentiment analysis engine.
Tools : NLTK, scikit-learn

## Overview
We'll use a logistic regression, naive bayes and SVM classifier, bag-of-words features, and polarity lexicons (both in-built and external). We'll also create our own UDF to clean raw text present in the form of reviews.

## Brief description of the dataset
#### Dataset is scrapped from amazon for lenovo K8 mobile phones
- Review in the form of free text was scrapped and the user rating
- A user rating of 1,2,3 -> sentiment 0 -> negative sentiment
- A user rating of 4 and 5 -> sentiment 1 -> positive sentiment

### Topics:
1. Getting some visuals on text data
    - Wordcloud
    - bargraph
    - Frequency graph
2. Text cleaning tasks
3. Extarct features from text and convert text to numbers
4. n-gram analysis -> bigram, trigrams, obtain visuals on the n-grams
5. Sentiment analysis using AFFIN and VADER
6. Document classification
7. Document clustering
8. Document and word similarlity
