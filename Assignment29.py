# ============================================================
# SESSION 29 (AIML) - NLP VECTORIZATION ASSIGNMENT
# Q1 TO Q10
# ============================================================

import pandas as pd
import matplotlib.pyplot as plt
import csv
import joblib

from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import accuracy_score


print("=" * 60)
print("SESSION 29 - NLP VECTORIZATION")
print("=" * 60)


# ============================================================
# Q1. DATA PREPARATION
# ============================================================

print("\n========== Q1. DATA PREPARATION ==========")

# ------------------------------------------------------------
# Try to read Emotions.csv
# ------------------------------------------------------------

rows = []

try:
    with open("Emotions.csv", "r", encoding="utf-8-sig") as file:
        reader = csv.reader(file)

        for row in reader:

            if len(row) >= 2:

                text = ",".join(row[:-1]).strip()
                emotion = row[-1].strip()

                # Skip header
                if text.lower() == "text" and emotion.lower() == "emotion":
                    continue

                if text != "" and emotion != "":
                    rows.append([text, emotion])

except FileNotFoundError:
    print("\nEmotions.csv not found.")
    print("Dataset will be created automatically.")


# ------------------------------------------------------------
# Create DataFrame
# ------------------------------------------------------------

df = pd.DataFrame(
    rows,
    columns=["text", "emotion"]
)


# ------------------------------------------------------------
# If dataset is empty, create dataset automatically
# ------------------------------------------------------------

if len(df) == 0:

    print("\nCSV was empty.")
    print("Creating Emotions dataset automatically...")

    data = [

        # JOY
        ("I am feeling very happy today", "joy"),
        ("This is the best day of my life", "joy"),
        ("I am so excited about the trip", "joy"),
        ("I feel wonderful and cheerful", "joy"),
        ("What a beautiful and joyful moment", "joy"),
        ("I am smiling all day", "joy"),
        ("I am delighted with the result", "joy"),
        ("This news made me extremely happy", "joy"),
        ("I feel great about my success", "joy"),
        ("I am enjoying this wonderful evening", "joy"),

        # SADNESS
        ("I am sad and want to cry", "sadness"),
        ("Today I feel very lonely", "sadness"),
        ("I miss my old friends", "sadness"),
        ("This situation makes me unhappy", "sadness"),
        ("I feel disappointed about the result", "sadness"),
        ("I am crying because I lost my toy", "sadness"),
        ("My heart feels heavy today", "sadness"),
        ("I feel unhappy after the bad news", "sadness"),
        ("I am unhappy with what happened", "sadness"),
        ("I feel lonely at home", "sadness"),

        # ANGER
        ("I am very angry right now", "anger"),
        ("This behavior makes me furious", "anger"),
        ("I hate being treated unfairly", "anger"),
        ("I am frustrated with this problem", "anger"),
        ("That rude comment made me angry", "anger"),
        ("I cannot control my anger", "anger"),
        ("I am annoyed by the constant noise", "anger"),
        ("This unfair decision makes me furious", "anger"),
        ("I feel irritated with my brother", "anger"),
        ("I am angry about the mistake", "anger"),

        # FEAR
        ("I am scared of the dark", "fear"),
        ("I feel afraid of losing my job", "fear"),
        ("This strange sound makes me nervous", "fear"),
        ("I am worried about the exam", "fear"),
        ("I fear that something bad will happen", "fear"),
        ("I feel frightened when I am alone", "fear"),
        ("The storm made everyone afraid", "fear"),
        ("I am nervous about tomorrow", "fear"),
        ("I am terrified by the loud noise", "fear"),
        ("I feel anxious before the interview", "fear"),

        # SURPRISE
        ("I am surprised by this gift", "surprise"),
        ("The result was completely unexpected", "surprise"),
        ("I cannot believe what I just saw", "surprise"),
        ("What a surprising announcement", "surprise"),
        ("I was shocked by the news", "surprise"),
        ("This unexpected event amazed me", "surprise"),
        ("I am astonished by your talent", "surprise"),
        ("The sudden visit surprised me", "surprise"),
        ("I never expected this wonderful gift", "surprise"),
        ("The final score surprised everyone", "surprise"),

        # LOVE
        ("I love spending time with my family", "love"),
        ("You are very important to me", "love"),
        ("I care deeply about my parents", "love"),
        ("I love my best friend", "love"),
        ("My family gives me so much love", "love"),
        ("I feel affection for my little sister", "love"),
        ("I am grateful for your kindness", "love"),
        ("I really care about these people", "love"),
        ("Love makes life beautiful", "love"),
        ("I enjoy helping the people I love", "love"),

        # NEUTRAL
        ("I feel calm and peaceful today", "neutral"),
        ("The weather is normal today", "neutral"),
        ("I am going to college in the morning", "neutral"),
        ("I finished my homework on time", "neutral"),
        ("The book is on the table", "neutral"),
        ("I attended my class today", "neutral"),
        ("The train arrived at the station", "neutral"),
        ("I had lunch with my friend", "neutral"),
        ("The computer is working normally", "neutral"),
        ("I will study for two hours tonight", "neutral")
    ]

    df = pd.DataFrame(
        data,
        columns=["text", "emotion"]
    )

    # Repeat data to make 210 records
    df = pd.concat(
        [df, df, df],
        ignore_index=True
    )

    # Save correct dataset
    df.to_csv(
        "Emotions.csv",
        index=False
    )

    print("New Emotions.csv created successfully!")


# ------------------------------------------------------------
# Clean dataset
# ------------------------------------------------------------

df = df.dropna(
    subset=["text", "emotion"]
)

df["text"] = df["text"].astype(str)
df["emotion"] = df["emotion"].astype(str)


print("\nDataset loaded successfully!")
print("Dataset shape:", df.shape)
print("Columns:", list(df.columns))


# ------------------------------------------------------------
# Separate X and y
# ------------------------------------------------------------

X = df["text"]
y = df["emotion"]


# ------------------------------------------------------------
# Train-Test Split
# ------------------------------------------------------------

X_train, X_test, y_train, y_test = train_test_split(

    X,
    y,

    test_size=0.20,

    random_state=42,

    stratify=y
)


print("\nX_train shape:", X_train.shape)
print("X_test shape:", X_test.shape)
print("y_train shape:", y_train.shape)
print("y_test shape:", y_test.shape)


# ============================================================
# Q2. BAG OF WORDS
# ============================================================

print("\n========== Q2. BAG OF WORDS ==========")

bow_vectorizer = CountVectorizer()

X_train_bow = bow_vectorizer.fit_transform(
    X_train
)

X_test_bow = bow_vectorizer.transform(
    X_test
)

print("\nX_train_BOW shape:", X_train_bow.shape)
print("X_test_BOW shape:", X_test_bow.shape)

print("\nFirst 20 vocabulary words:")

print(
    bow_vectorizer.get_feature_names_out()[:20]
)


# ============================================================
# Q3. BAG OF WORDS + MULTINOMIAL NB
# ============================================================

print("\n========== Q3. BOW + MULTINOMIAL NB ==========")

bow_model = MultinomialNB()

bow_model.fit(
    X_train_bow,
    y_train
)

y_pred_bow = bow_model.predict(
    X_test_bow
)

bow_accuracy = accuracy_score(
    y_test,
    y_pred_bow
)

print(
    "\nBag of Words Accuracy:",
    round(bow_accuracy * 100, 2),
    "%"
)


# ============================================================
# Q4. UNDERSTANDING VOCABULARY
# ============================================================

print("\n========== Q4. UNDERSTANDING VOCABULARY ==========")

vocabulary = bow_vectorizer.get_feature_names_out()

print(
    "\nTotal vocabulary size:",
    len(vocabulary)
)

print("\nFirst 15 vocabulary words:")

print(
    vocabulary[:15]
)


sample_document = X_train.iloc[0]

print("\nSample training document:")
print(sample_document)

sample_vector = bow_vectorizer.transform(
    [sample_document]
)

print("\nSample document BOW vector:")

print(
    sample_vector.toarray()
)


# ============================================================
# Q5. N-GRAMS WITH BAG OF WORDS
# ============================================================

print("\n========== Q5. BIGRAM BAG OF WORDS ==========")

bigram_vectorizer = CountVectorizer(
    ngram_range=(1, 2)
)

X_train_bigram = bigram_vectorizer.fit_transform(
    X_train
)

X_test_bigram = bigram_vectorizer.transform(
    X_test
)

print(
    "\nBigram training matrix shape:",
    X_train_bigram.shape
)

print(
    "Bigram test matrix shape:",
    X_test_bigram.shape
)

print("\nFirst 20 unigram + bigram features:")

print(
    bigram_vectorizer.get_feature_names_out()[:20]
)


# ============================================================
# Q6. N-GRAMS + MODEL TRAINING
# ============================================================

print("\n========== Q6. BIGRAM + MULTINOMIAL NB ==========")

bigram_model = MultinomialNB()

bigram_model.fit(
    X_train_bigram,
    y_train
)

y_pred_bigram = bigram_model.predict(
    X_test_bigram
)

bigram_accuracy = accuracy_score(
    y_test,
    y_pred_bigram
)

print(
    "\nBigram Accuracy:",
    round(bigram_accuracy * 100, 2),
    "%"
)

print("\nComparison with Basic BOW:")

print(
    "Basic BOW Accuracy:",
    round(bow_accuracy * 100, 2),
    "%"
)

print(
    "Bigram Accuracy:",
    round(bigram_accuracy * 100, 2),
    "%"
)


# ============================================================
# Q7. TF-IDF VECTORIZATION
# ============================================================

print("\n========== Q7. TF-IDF VECTORIZATION ==========")

tfidf_vectorizer = TfidfVectorizer()

X_train_tfidf = tfidf_vectorizer.fit_transform(
    X_train
)

X_test_tfidf = tfidf_vectorizer.transform(
    X_test
)

print(
    "\nTF-IDF training matrix shape:",
    X_train_tfidf.shape
)

print(
    "TF-IDF test matrix shape:",
    X_test_tfidf.shape
)

print("\nFirst 15 TF-IDF feature names:")

print(
    tfidf_vectorizer.get_feature_names_out()[:15]
)


# ============================================================
# Q8. TF-IDF + MULTINOMIAL NB
# ============================================================

print("\n========== Q8. TF-IDF + MULTINOMIAL NB ==========")

tfidf_model = MultinomialNB()

tfidf_model.fit(
    X_train_tfidf,
    y_train
)

y_pred_tfidf = tfidf_model.predict(
    X_test_tfidf
)

tfidf_accuracy = accuracy_score(
    y_test,
    y_pred_tfidf
)

print(
    "\nTF-IDF Accuracy:",
    round(tfidf_accuracy * 100, 2),
    "%"
)


# ============================================================
# Q9. COMPARISON OF VECTORIZERS
# ============================================================

print("\n========== Q9. COMPARISON OF VECTORIZERS ==========")

comparison = pd.DataFrame({

    "Method": [

        "Bag of Words (Unigrams)",

        "Bag of Words (Unigrams + Bigrams)",

        "TF-IDF"
    ],

    "Accuracy (%)": [

        round(bow_accuracy * 100, 2),

        round(bigram_accuracy * 100, 2),

        round(tfidf_accuracy * 100, 2)
    ]
})


print("\nComparison Table:")

print(
    comparison.to_string(index=False)
)


# Find best method

accuracy_values = {

    "Bag of Words (Unigrams)": bow_accuracy,

    "Bag of Words (Unigrams + Bigrams)": bigram_accuracy,

    "TF-IDF": tfidf_accuracy
}


best_method = max(
    accuracy_values,
    key=accuracy_values.get
)

best_accuracy = accuracy_values[
    best_method
]


print(
    "\nBest Method:",
    best_method
)

print(
    "Best Accuracy:",
    round(best_accuracy * 100, 2),
    "%"
)


print("\nObservation:")

print(
    "The method with the highest accuracy "
    "performed best on the test data."
)


# ------------------------------------------------------------
# Q9 GRAPH
# ------------------------------------------------------------

methods = [

    "Unigram BOW",

    "Unigram + Bigram",

    "TF-IDF"
]


accuracies = [

    bow_accuracy * 100,

    bigram_accuracy * 100,

    tfidf_accuracy * 100
]


plt.figure(
    figsize=(8, 5)
)

plt.bar(
    methods,
    accuracies
)

plt.title(
    "Comparison of Text Vectorization Methods"
)

plt.xlabel(
    "Vectorization Method"
)

plt.ylabel(
    "Accuracy (%)"
)

plt.ylim(
    0,
    100
)

plt.xticks(
    rotation=15
)

plt.tight_layout()

plt.show()


# ============================================================
# Q10. COMPLETE VECTORIZATION PIPELINE
# ============================================================

print("\n========== Q10. COMPLETE VECTORIZATION PIPELINE ==========")


# ------------------------------------------------------------
# FINAL BAG OF WORDS
# ------------------------------------------------------------

final_bow_vectorizer = CountVectorizer()

final_X_train_bow = final_bow_vectorizer.fit_transform(
    X_train
)

final_X_test_bow = final_bow_vectorizer.transform(
    X_test
)

final_bow_model = MultinomialNB()

final_bow_model.fit(
    final_X_train_bow,
    y_train
)

final_bow_pred = final_bow_model.predict(
    final_X_test_bow
)

final_bow_accuracy = accuracy_score(
    y_test,
    final_bow_pred
)


# ------------------------------------------------------------
# FINAL TF-IDF
# ------------------------------------------------------------

final_tfidf_vectorizer = TfidfVectorizer()

final_X_train_tfidf = final_tfidf_vectorizer.fit_transform(
    X_train
)

final_X_test_tfidf = final_tfidf_vectorizer.transform(
    X_test
)

final_tfidf_model = MultinomialNB()

final_tfidf_model.fit(
    final_X_train_tfidf,
    y_train
)

final_tfidf_pred = final_tfidf_model.predict(
    final_X_test_tfidf
)

final_tfidf_accuracy = accuracy_score(
    y_test,
    final_tfidf_pred
)


# ------------------------------------------------------------
# Q10 RESULTS
# ------------------------------------------------------------

print(
    "\nFinal Bag of Words Accuracy:",
    round(final_bow_accuracy * 100, 2),
    "%"
)

print(
    "Final TF-IDF Accuracy:",
    round(final_tfidf_accuracy * 100, 2),
    "%"
)


# ------------------------------------------------------------
# Select Best Model
# ------------------------------------------------------------

if final_bow_accuracy >= final_tfidf_accuracy:

    best_vectorizer = final_bow_vectorizer

    best_model = final_bow_model

    best_name = "Bag of Words"

else:

    best_vectorizer = final_tfidf_vectorizer

    best_model = final_tfidf_model

    best_name = "TF-IDF"


# ------------------------------------------------------------
# Save Best Vectorizer and Model
# ------------------------------------------------------------

joblib.dump(
    best_vectorizer,
    "best_vectorizer.pkl"
)

joblib.dump(
    best_model,
    "best_model.pkl"
)


print(
    "\nBest Vectorizer:",
    best_name
)

print("\nSaved files:")

print("best_vectorizer.pkl")

print("best_model.pkl")


# ============================================================
# FINAL SUMMARY
# ============================================================

print("\n============================================================")

print("FINAL SUMMARY")

print("============================================================")

print(
    "BOW Accuracy:",
    round(bow_accuracy * 100, 2),
    "%"
)

print(
    "Bigram Accuracy:",
    round(bigram_accuracy * 100, 2),
    "%"
)

print(
    "TF-IDF Accuracy:",
    round(tfidf_accuracy * 100, 2),
    "%"
)

print(
    "Best Method:",
    best_method
)

print(
    "Best Accuracy:",
    round(best_accuracy * 100, 2),
    "%"
)

print(
    "\nAssignment completed successfully!"
)

print("============================================================")