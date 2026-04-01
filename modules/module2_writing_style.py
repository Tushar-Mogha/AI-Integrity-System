# Module 2 - Writing Style Analysis
# We are trying to detect if an essay is written by AI or a human
# by looking at HOW the essay is written, not WHAT it is about
# Dataset - DAIGT V2 from Kaggle (44868 essays)
# Team - Abhinandan, Stuti, Tushar

import os
import string
import pickle
import numpy as np
import pandas as pd
from scipy.sparse import hstack
import scipy.sparse as sp
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix

os.system('cls')

# loading our dataset
print("Loading dataset...")
data = pd.read_csv('data/train_v2_drcat_02(Model_1).csv')
print("Total essays:", len(data))
print("Human written:", len(data[data['label']==0]))
print("AI written:", len(data[data['label']==1]))


# we decided to extract 7 features from each essay
# these features tell us about the writing style of the author
# for example - how long are the words, how complex are the sentences etc.

def get_writing_features(essay):

    # if essay is empty just return zeros
    if not isinstance(essay, str) or len(essay.strip()) == 0:
        return [0, 0, 0, 0, 0, 0, 0]

    cleaned = essay.strip()

    # get individual words after removing punctuation
    raw = cleaned.split()
    words = [w.strip(string.punctuation).lower() for w in raw]
    words = [w for w in words if len(w) > 0]

    # split essay into sentences
    # we treat ! ? ; : same as a full stop
    for ch in ['!', '?', ';', ':']:
        cleaned = cleaned.replace(ch, '.')
    sentences = [s.strip() for s in cleaned.split('.') if len(s.strip()) > 5]

    # split into paragraphs
    paragraphs = [p.strip() for p in essay.split('\n') if len(p.strip()) > 10]

    # linking words are used more in AI writing
    linking = [
        'however', 'therefore', 'moreover', 'furthermore',
        'although', 'nevertheless', 'consequently', 'additionally',
        'meanwhile', 'otherwise', 'similarly', 'thus'
    ]
    all_words = essay.lower().split()
    linking_count = sum(1 for w in linking if w in all_words)

    # count only meaningful punctuation marks
    meaningful = set('.,!?;:')
    punct_count = sum(1 for ch in essay if ch in meaningful)

    # feature 1 - average word length
    avg_word_len = np.mean([len(w) for w in words]) if words else 0

    # feature 2 - average sentence length
    avg_sent_len = np.mean([len(s.split()) for s in sentences]) if sentences else 0

    # feature 3 - how rich is the vocabulary
    # more unique words means richer vocabulary
    vocab_richness = len(set(words)) / len(words) if words else 0

    # feature 4 - punctuation count (already calculated)

    # feature 5 - number of paragraphs
    para_count = len(paragraphs)

    # feature 6 - linking word count (already calculated)

    # feature 7 - how often capital letters appear
    capital_ratio = sum(1 for c in essay if c.isupper()) / len(essay) if essay else 0

    return [
        avg_word_len,
        avg_sent_len,
        vocab_richness,
        punct_count,
        para_count,
        linking_count,
        capital_ratio
    ]


# applying the feature extraction on all essays
print("\nExtracting features from essays...")
print("Please wait, this takes about a minute...")
extracted = data['text'].apply(get_writing_features)

# converting to dataframe
style_df = pd.DataFrame(extracted.tolist(), columns=[
    'avg_word_len',
    'avg_sent_len',
    'vocab_richness',
    'punct_count',
    'para_count',
    'linking_count',
    'capital_ratio'
])
style_df['label'] = data['label']

print("Done! Sample output:")
print(style_df.head(3))


# removing essays that have unrealistic values
# for example avg word length of 100 is clearly wrong data
print("\nRemoving bad data...")
style_df = style_df[style_df['avg_word_len'] < 20]
style_df = style_df[style_df['avg_sent_len'] < 200]
clean_data = data.loc[style_df.index]
print("Essays remaining:", len(style_df))


# TF-IDF converts essay text into numbers
# it finds which words are most important in each essay
# AI tends to use certain words more frequently than humans
print("\nApplying TF-IDF...")
tfidf = TfidfVectorizer(max_features=500, stop_words='english', ngram_range=(1, 2))
tfidf_matrix = tfidf.fit_transform(clean_data['text'])
print("TF-IDF shape:", tfidf_matrix.shape)


# scaling our 7 features and combining with tfidf
scaler = StandardScaler()
style_scaled = scaler.fit_transform(style_df.drop('label', axis=1))

# final feature matrix = tfidf features + our 7 style features
X = hstack([tfidf_matrix, sp.csr_matrix(style_scaled)])
y = style_df['label'].values
print("Final matrix shape:", X.shape)


# splitting data - 80% for training, 20% for testing
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)
print("\nTraining samples:", X_train.shape[0])
print("Testing samples:", X_test.shape[0])


# training Random Forest
# we tried Logistic Regression first but Random Forest gave better results
print("\nTraining model...")
model = RandomForestClassifier(n_estimators=100, random_state=42, n_jobs=-1)
model.fit(X_train, y_train)
print("Training done!")


# checking how well our model performs
y_pred = model.predict(X_test)

print("\n── Results ────────────────────────────────────────")
print("Accuracy:", round(accuracy_score(y_test, y_pred) * 100, 2), "%")
print("\nDetailed Report:")
print(classification_report(y_test, y_pred, target_names=['Human', 'AI']))
print("Confusion Matrix:")
print(confusion_matrix(y_test, y_pred))


# saving model files so we don't have to retrain every time
print("\nSaving model...")
pickle.dump(model,  open('models/module2_model.pkl', 'wb'))
pickle.dump(scaler, open('models/module2_scaler.pkl', 'wb'))
pickle.dump(tfidf,  open('models/module2_tfidf.pkl', 'wb'))
print("Saved!")
print("\nModule 2 done!")


# testing with sample essays to verify predictions
model_loaded  = pickle.load(open('models/module2_model.pkl', 'rb'))
scaler_loaded = pickle.load(open('models/module2_scaler.pkl', 'rb'))
tfidf_loaded  = pickle.load(open('models/module2_tfidf.pkl', 'rb'))

human_essay = "I think phones are really dangerous when people use them while driving. My uncle got into an accident because he was texting. I never use my phone when driving and everyone should follow this rule."
ai_essay    = "The proliferation of mobile telecommunication devices has engendered substantial deliberation regarding their utilization in contemporary academic environments. Research demonstrates that smartphone dependency significantly impairs cognitive performance among undergraduate students."

def predict(essay):
    features = get_writing_features(essay)
    style_input = pd.DataFrame([features], columns=[
        'avg_word_len', 'avg_sent_len', 'vocab_richness',
        'punct_count', 'para_count', 'linking_count', 'capital_ratio'
    ])
    style_scaled_input = scaler_loaded.transform(style_input)
    tfidf_input        = tfidf_loaded.transform([essay])
    X_input            = hstack([tfidf_input, sp.csr_matrix(style_scaled_input)])
    pred               = model_loaded.predict(X_input)[0]
    prob               = model_loaded.predict_proba(X_input)[0]
    label              = "AI" if pred == 1 else "Human"
    return {
        'prediction'       : label,
        'ai_probability'   : round(prob[1] * 100, 2),
        'human_probability': round(prob[0] * 100, 2)
    }

print("\n── Sample Predictions ──────────────────────────────")
print("\nHuman Essay Test:")
print(predict(human_essay))
print("\nAI Essay Test:")
print(predict(ai_essay))
