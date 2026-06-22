import pandas as pd
import pickle
import matplotlib.pyplot as plt
from textblob import TextBlob
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.model_selection import train_test_split
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix

df = pd.read_csv('IMBD Dataset.csv')
print("Data Loaded!", df.shape)

def get_sentiment(text):
    score = TextBlob(str(text)).sentiment.polarity
    if score> 0.1: return 'positive'
    elif score < -0.1: return 'negative'
    else: return 'neutral'

df['predicted'] = df['review'].apply(get_sentiment)

print(f"TextBlob Accuracy: {accuracy_score(df['sentiment'], df['predicted']) *100:.1f}%")


X_train, X_test, y_train, y_test = train_test_split(
    df['review'], df['sentiment'], test_size=0.2, random_state=42)

vectorizer = TfidfVectorizer(max_features=5000, stop_words='english')
X_train_vec = vectorizer.fit_transform(X_train)
X_test_vec = vectorizer.transform(X_test)

model = MultinomialNB()
model.fit(X_train_vec, y_train)
preds = model.predict(X_test_vec)
print(f"ML Model Accuracy: {accuracy_score(y_test, preds) * 100:.1f}%")

pickle.dump(model,      open('model.pkl', 'wb'))
pickle.dump(vectorizer, open('vectorizer.pkl', 'wb'))



fig, axes = plt.subplots(1, 2, figsize=(10, 4))
counts = df['sentiment'].value_counts()
axes[0].bar(counts.index, counts.values, color=['#1D9E75', '#E24B4A'])
axes[0].set_title('Sentiment Distribution')
cm = confusion_matrix(y_test, preds, labels=['positive', 'negative'])
axes[1].imshow(cm, cmap='Greens')
axes[1].set_xticks([0,1]); axes[1].set_xticklabels(['positive','negative'])
axes[1].set_yticks([0,1]); axes[1].set_yticklabels(['positive','negative'])
axes[1].set_title('Confusion Matrix')
for i in range(2):
    for j in range(2):
        axes[1].text(j, i, cm[i,j], ha='center', va='center', fontsize=14)
plt.tight_layout()
plt.savefig('results.png')
plt.show()

fig, axes = plt.subplots(1, 2, figsize=(10, 4))
counts = df['sentiment'].value_counts()
axes[0].bar(counts.index, counts.values, color=['#1D9E75', '#E24B4A'])
axes[0].set_title('Sentiment Distribution')
cm = confusion_matrix(y_test, preds, labels=['positive', 'negative'])
axes[1].imshow(cm, cmap='Greens')
axes[1].set_xticks([0,1]); axes[1].set_xticklabels(['positive','negative'])
axes[1].set_yticks([0,1]); axes[1].set_yticklabels(['positive','negative'])
axes[1].set_title('Confusion Matrix')
for i in range(2):
    for j in range(2):
        axes[1].text(j, i, cm[i,j], ha='center', va='center', fontsize=14)
plt.tight_layout()
plt.savefig('results.png')
plt.show()