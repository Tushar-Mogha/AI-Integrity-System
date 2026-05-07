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
    linking = [ 'however', 'therefore', 'moreover', 'furthermore', 'although', 'nevertheless', 'consequently', 'additionally', 'meanwhile', 'otherwise', 
    'similarly', 'thus', 'hence', 'accordingly', 'in addition', 'as a result', 'on the other hand', 'for instance', 'for example', 'in conclusion', 
    'to conclude', 'overall', 'in summary']

    all_words = essay.lower().split()
    linking_count = sum(essay.lower().count(w) for w in linking)

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

casual_human_1 = "i think this topic is kinda confusing like i tried to understand it but honestly it still doesnt make full sense to me and maybe in missing something but yeah its not very clear"
casual_human_2 = "today in class we learned about this thing but i didnt really get it properly because teacher was going fast and i was also tired so yeah not sure what exactly happened"
structured_ai_style_1 = "Artificial intelligence has significantly transformed modern computational systems by enabling data-driven decision-making processes across various domains, including healthcare, education, and finance."
structured_ai_style_2 = "The rapid advancement of machine learning algorithms has facilitated improved predictive modeling capabilities, thereby enhancing efficiency and accuracy in numerous real-world applications."
mixed_style_1 = "I think AI is useful, but sometimes it feels like people depend on it too much, which can be risky in situations where human judgment is actually important."
mixed_style_2 = "In my opinion, technology has both advantages and disadvantages; however, its overuse can negatively impact critical thinking skills."
structured_human_1 = "The education system plays a crucial role in shaping an individual's future by providing knowledge, skills, and values necessary for personal and professional growth."
structured_human_2 = "Environmental conservation is essential for maintaining ecological balance and ensuring sustainable development for future generations."
very_short_text_1 = "yes i agree about this"
very_short_text_2 = "Artificial intelligence is evolving rapidly."
Gibberish_1 = "asdjkhaskjdh kajshdkajshd kjashdkajshd kjashdka"
Gibberish_2 = "i was today last night qweqweqwe weqweqwe weqwe"
repetitive_ai_style ="Technology is important. Technology is useful. Technology is growing rapidly. Technology is used in many fields. Technology is essential."
paragraph_level_1 ="I believe that mobile phones are useful, but they can also be distracting. Many students use their phones during class, which affects their concentration. Teachers should set clear rules so that students can focus better on their studies."
paragraph_level_2 = "The integration of digital communication technologies into educational environments has sparked significant debate regarding their effectiveness. Studies suggest that excessive reliance on smartphones may hinder student engagement and academic performance."
informal_slang ="bro honestly i dont think this topic is that deep like people are overthinking it too much and its kinda unnecessary"
borderline_formal_student_answer = "In my opinion, the use of mobile phones in schools should be limited because they can distract students from their studies and reduce their focus during lectures."
random ="My name is Tushar , how are you."


def is_short_text(text):
    return len(text.split()) < 8

def is_repetitive(text):
    words = text.lower().split()
    if len(words) == 0:
        return False
    return len(set(words)) / len(words) < 0.6

def is_gibberish(text):
    words = text.split()
    if len(words) == 0:
        return True
    
    # too many random / non-alphabetic tokens
    non_alpha = sum(1 for w in words if not w.isalpha())
    return (non_alpha / len(words)) > 0.4


def has_human_tone(text):
    patterns = [ "i think", "i feel", "i believe", "in my opinion", "i guess", "maybe", "honestly", "personally", "i mean", "i dont think", "i don't think", 
    "i am not sure", "im not sure", "kind of", "kinda", "sort of", "i guess so", "to be honest", "as far as i know", "from what i know"]
    text = text.lower()
    return any(p in text for p in patterns)


def predict(essay):
    features = get_writing_features(essay)
    style_input = pd.DataFrame([features], columns=[
        'avg_word_len', 'avg_sent_len', 'vocab_richness',
        'punct_count', 'para_count', 'linking_count', 'capital_ratio'
    ])
    style_scaled_input = scaler_loaded.transform(style_input)
    tfidf_input        = tfidf_loaded.transform([essay])
    X_input            = hstack([tfidf_input, sp.csr_matrix(style_scaled_input)])
    prob = model_loaded.predict_proba(X_input)[0]
    ai_prob = prob[1]
    human_prob = prob[0]

    word_count = len(essay.split())

    # ---------- DECISION LOGIC ----------

    # short text → unreliable → human
    if is_short_text(essay):
        label = "Human"

    # gibberish → human
    elif is_gibberish(essay):
        label = "Human"

    elif is_repetitive(essay) and ai_prob > 0.70:
        label = "AI"

    elif word_count > 15 and ai_prob < 0.97:
        label = "Human"

    # strong AI
    elif ai_prob > 0.91:
        label = "AI"

    # strong human
    elif ai_prob < 0.35:
        label = "Human"

    elif 0.75 < ai_prob < 0.92:
        label = "Uncertain"

    # structured human correction
    elif has_human_tone(essay) and ai_prob < 0.85:
        label = "Human"

    # long essay correction
    elif word_count > 40 and ai_prob < 0.80:
        label = "Human"

    # fallback → uncertain
    else:
        label = "Uncertain"

    return {
        'prediction'       : label,
        'ai_probability'   : round(prob[1] * 100, 2),
        'human_probability': round(prob[0] * 100, 2)
    }

print("\n── Sample Predictions ──────────────────────────────")

print("\ncasual_human_1 Test:")
print(predict(casual_human_1))

print("\ncasual_human_2 Test:")
print(predict(casual_human_2))

print("\nstructured_ai_style_1 Test:")
print(predict(structured_ai_style_1))

print("\nstructured_ai_style_2 Test:")
print(predict(structured_ai_style_2))

print("\nmixed_style_1 Test:")
print(predict(mixed_style_1))

print("\nmixed_style_2 Test:")
print(predict(mixed_style_2))

print("\nstructured_human_1 Test:")
print(predict(structured_human_1))

print("\nstructured_human_2 Test:")
print(predict(structured_human_2))

print("\nvery_short_text_1 Test:")
print(predict(very_short_text_1))

print("\nvery_short_text_2 Test:")
print(predict(very_short_text_2))

print("\nGibberish_1:")
print(predict(Gibberish_1))

print("\nGibberish_2:")
print(predict(Gibberish_2))

print("\nrepetitive_ai_style:")
print(predict(repetitive_ai_style))

print("\nparagraph_level_1:")
print(predict(paragraph_level_1))

print("\nparagraph_level_2:")
print(predict(paragraph_level_2))

print("\ninformal_slang:")
print(predict(informal_slang))

print("\nborderline_formal_student_answer:")
print(predict(borderline_formal_student_answer))

print("\nRandom:")
print(predict(random))