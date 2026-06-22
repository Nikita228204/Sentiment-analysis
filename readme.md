# Sentiment Analysis Web App 🎯

A Machine Learning web application that analyzes sentiment of text as Positive or Negative.

## Built With
- Python
- Flask
- Scikit-learn (Naive Bayes + TF-IDF)
- Pandas
- Matplotlib
- TextBlob

## Features
- Single sentence sentiment analysis
- Full paragraph analysis (sentence by sentence)
- Confidence score for every prediction
- 85.1% model accuracy trained on 50,000 IMDB reviews

## How to Run

### 1. Install libraries
pip install pandas textblob scikit-learn matplotlib flask

### 2. Download dataset
Download IMDB Dataset.csv from Kaggle and place in project folder

### 3. Train the model
python sentiment.py

### 4. Run the web app
python app.py

### 5. Open browser
Go to http://127.0.0.1:5000

## Results
| Model | Accuracy |
|-------|----------|
| TextBlob (baseline) | 47.7% |
| Naive Bayes ML Model | 85.1% |

## Project By
**Nikita Panda**
Pinnacle Labs Data Science Internship 2026
