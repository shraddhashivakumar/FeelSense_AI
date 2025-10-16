# model_utils.py
import os
import pandas as pd
from sklearn.pipeline import Pipeline
from sklearn.feature_extraction.text import TfidfVectorizer
# NEW: Import the SGDClassifier for better performance in text classification
from sklearn.linear_model import LogisticRegression, SGDClassifier 
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report
from joblib import dump, load
from numpy import unique # Added to help get all unique classes

DEFAULT_CSV_PATH = "data/emotion.csv"

def detect_columns(df):
    text_names = ['text','message','sentence','utterance','input','review','content','msg']
    label_names = ['mood','label','sentiment','emotion','target','class']
    text_col = None
    label_col = None

    cols_lower = {c.lower(): c for c in df.columns}
    for name in text_names:
        if name in cols_lower:
            text_col = cols_lower[name]
            break
    for name in label_names:
        if name in cols_lower:
            label_col = cols_lower[name]
            break

    if text_col is None:
        # pick the first object dtype column as text
        obj_cols = [c for c in df.columns if df[c].dtype == object]
        if len(obj_cols) >= 1:
            text_col = obj_cols[0]
    if label_col is None:
        # pick second object dtype column as label if exists
        obj_cols = [c for c in df.columns if df[c].dtype == object]
        if len(obj_cols) >= 2:
            label_col = obj_cols[1]

    return text_col, label_col

def load_dataset(path=DEFAULT_CSV_PATH):
    if not os.path.exists(path):
        print(f"[model_utils] CSV not found at {path}")
        return None, None
    df = pd.read_csv(path)
    text_col, label_col = detect_columns(df)
    if text_col is None or label_col is None:
        print("[model_utils] Could not auto-detect columns. Please ensure CSV has 'text' and 'mood' (or similar).")
        return None, None
    df = df[[text_col, label_col]].dropna()
    df.columns = ['text', 'label']
    # drop empty texts
    df = df[df['text'].str.strip().astype(bool)]
    if df.shape[0] < 10:
        print("[model_utils] Dataset too small after cleaning:", df.shape[0])
        return None, None
    return df['text'].tolist(), df['label'].tolist()

def build_and_train(texts, labels, save_to="models/mood_model.joblib", le_save_to="models/label_encoder.joblib"):
    os.makedirs(os.path.dirname(save_to), exist_ok=True)
    le = LabelEncoder()
    y = le.fit_transform(labels)

    # UPDATED PIPELINE: Added class_weight='balanced' to compensate for the joy bias
    pipeline = Pipeline([
        # sublinear_tf=True helps boost the signal of less frequent sentiment words
        ("tfidf", TfidfVectorizer(ngram_range=(1,2), max_features=15000, sublinear_tf=True)), 
        # FIX: class_weight='balanced' automatically weights less frequent classes (like fear/sadness) 
        # higher, penalizing the model for incorrectly predicting the majority class (joy).
        ("clf", SGDClassifier(loss='log_loss', max_iter=1200, tol=1e-3, class_weight='balanced'))
    ])

    # Kept the non-stratified split to avoid crashing on single-sample moods
    X_train, X_test, y_train, y_test = train_test_split(texts, y, test_size=0.15, random_state=42)
    pipeline.fit(X_train, y_train)

    # Evaluation
    preds = pipeline.predict(X_test)
    print("[model_utils] Classification report on held-out set:")
    
    # FIX: Explicitly passing the unique labels (0 to 6) to avoid crash 
    # when one class is missing from the test set.
    # We use le.transform(le.classes_) to get the numerical labels corresponding to the names.
    all_numeric_labels = le.transform(le.classes_)
    print(classification_report(y_test, preds, 
                                target_names=le.classes_, 
                                labels=all_numeric_labels, 
                                zero_division=0)) # zero_division=0 ensures we don't crash if a mood has zero samples

    dump(pipeline, save_to)
    dump(le, le_save_to)
    print(f"[model_utils] Saved model to {save_to} and label encoder to {le_save_to}")
    return pipeline, le

def load_model(model_path="models/mood_model.joblib", le_path="models/label_encoder.joblib"):
    if not os.path.exists(model_path) or not os.path.exists(le_path):
        return None, None
    model = load(model_path)
    le = load(le_path)
    return model, le

# small fallback dataset to allow app to run if CSV missing
def fallback_sample():
    texts = [
        "I am so happy and excited today!",
        "I feel very sad and down.",
        "I'm angry about what happened",
        "I'm feeling okay, just normal",
        "This is amazing, I'm thrilled",
        "I'm disappointed and upset",
        "I'm scared and nervous about it",
        "What a surprise, I didn't expect that"
    ]
    labels = ["happy","sad","angry","neutral","happy","sad","fear","surprise"]
    return texts, labels
