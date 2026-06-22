from flask import Flask, render_template, request
import pickle
import re

app = Flask(__name__)

model      = pickle.load(open('model.pkl', 'rb'))
vectorizer = pickle.load(open('vectorizer.pkl', 'rb'))

@app.route('/', methods=['GET', 'POST'])
def home():
    result = None
    confidence = None
    user_text = None
    if request.method == 'POST':
        user_text = request.form['text']
        vec  = vectorizer.transform([user_text])
        pred = model.predict(vec)[0]
        conf = max(model.predict_proba(vec)[0]) * 100
        result = pred.upper()
        confidence = round(conf, 1)
    return render_template('index.html',
                           result=result,
                           confidence=confidence,
                           user_text=user_text,
                           sentences=None)

@app.route('/multi', methods=['POST'])
def multi():
    paragraph = request.form['paragraph']
    raw = re.split(r'(?<=[.!?])\s+', paragraph.strip())
    sentences = []
    for s in raw:
        s = s.strip()
        if not s:
            continue
        vec  = vectorizer.transform([s])
        pred = model.predict(vec)[0].upper()
        conf = round(max(model.predict_proba(vec)[0]) * 100, 1)
        sentences.append({'text': s, 'sentiment': pred, 'confidence': conf})
    return render_template('index.html',
                           result=None,
                           confidence=None,
                           user_text=None,
                           sentences=sentences)

if __name__ == '__main__':
    app.run(debug=True)