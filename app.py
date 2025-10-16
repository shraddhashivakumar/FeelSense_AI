# app.py
import os
import json
import random
import traceback
from textblob import TextBlob
import csv
from flask import Flask, render_template, request, jsonify
from model_utils import load_dataset, build_and_train, load_model, fallback_sample, DEFAULT_CSV_PATH

# Initialize Flask app
app = Flask(__name__, static_folder='static', template_folder='templates')

# Paths for model and label encoder
MODEL_PATH = "models/mood_model.joblib"
LE_PATH = "models/label_encoder.joblib"

FEEDBACK_LOG = "data/feedback_log.csv"

def analyze_sentiment(feedback_text):
    """Analyze sentiment polarity of feedback."""
    blob = TextBlob(feedback_text)
    polarity = blob.sentiment.polarity
    if polarity > 0.1:
        return "positive"
    elif polarity < -0.1:
        return "negative"
    else:
        return "neutral"

# --- Helper: map detected label to a broad mood category ---
BROAD_MAPPINGS = {
    'happy': {'syn': {'happy','joy','joyful','glad','content','pleased','delighted','cheerful','excited','positive'}},
    'sad': {'syn': {'sad','sadness','down','unhappy','depressed','miserable','gloomy'}},
    'angry': {'syn': {'angry','anger','mad','furious','irritated','annoyed'}},
    'neutral': {'syn': {'neutral','ok','fine','meh','indifferent'}},
    'fear': {'syn': {'fear','scared','terrified','anxious','nervous','afraid'}},
    'surprise': {'syn': {'surprise','surprised','astonished','shocked'}},
    'disgust': {'syn': {'disgust','disgusted','repulsed'}},
}

def to_broad(mood_label):
    """Map a specific mood label to a broad category."""
    if mood_label is None:
        return 'neutral'
    ml = str(mood_label).lower()
    for broad, info in BROAD_MAPPINGS.items():
        for s in info['syn']:
            if s in ml:
                return broad
    return 'neutral'

def generate_reply(broad_mood, user_text):
    """Generate a reply based on broad mood."""
    templates = {
        'happy': [
            "That's wonderful to hear! 😊 Tell me more!",
            "Love that energy — what's making you smile today?",
            "Great! Keep it up — anything fun going on?"
        ],
        'sad': [
            "I'm sorry you're feeling down. Do you want to talk about it?",
            "That sounds tough. I'm here for you — what's on your mind?",
            "I hear you. Small steps can help — would you like breathing tips?"
        ],
        'angry': [
            "I can tell you're upset. Want to vent or find a solution together?",
            "That's frustrating — tell me what happened and we'll work it out.",
            "Anger is valid. Do you want some ways to calm down right now?"
        ],
        'neutral': [
            "Thanks for sharing. Anything else you'd like to add?",
            "Got it. Want to dive deeper or change the topic?",
            "Okay — how can I help further?"
        ],
        'fear': [
            "That sounds scary. Do you want to describe what's worrying you?",
            "I'm here with you — would you like some coping suggestions?",
            "It's okay to be nervous. Want grounding or breathing exercise ideas?"
        ],
        'surprise': [
            "Wow — that is surprising! Tell me more!",
            "That's unexpected — how do you feel about it?",
            "Interesting! What happened next?"
        ],
        'disgust': [
            "That sounds unpleasant. Want to talk about it?",
            "I get why you'd feel that way. Do you want a change of topic?",
            "Ugh — I hear you. Anything I can do to help?"
        ],
    }
    return random.choice(templates.get(broad_mood, templates['neutral']))

# --- Load or train model at startup ---
print("[app] Starting up, loading or training model...")
model, le = load_model(MODEL_PATH, LE_PATH)

if model is None or le is None:
    texts, labels = load_dataset(DEFAULT_CSV_PATH)
    if texts is None or labels is None:
        print("[app] Using fallback sample dataset.")
        texts, labels = fallback_sample()
    model, le = build_and_train(texts, labels, save_to=MODEL_PATH, le_save_to=LE_PATH)
else:
    print("[app] Model and label encoder loaded from disk.")

print("[app] Ready.")

# --- Routes ---
@app.route("/")
def index():
    return render_template("index.html")

@app.route("/chat", methods=["POST"])
def chat():
    data = request.get_json(force=True)
    user_msg = data.get("message", "").strip()
    if not user_msg:
        return jsonify({"error": "Empty message"}), 400
    try:
        # Predict mood
        if hasattr(model, "predict_proba"):
            probs = model.predict_proba([user_msg])[0]
            pred_idx = probs.argmax()
            confidence = float(probs[pred_idx])
        else:
            pred_idx = int(model.predict([user_msg])[0])
            confidence = None

        mood_label = le.inverse_transform([pred_idx])[0]
        broad = to_broad(mood_label)
        reply = generate_reply(broad, user_msg)

        response = {
            "mood": str(mood_label),
            "broad_mood": broad,
            "reply": reply,
            "confidence": confidence
        }
        return jsonify(response)
    except Exception as e:
        print("[Error] /chat exception:", e)
        print(traceback.format_exc())
        return jsonify({"error": "internal server error"}), 500
    
@app.route("/feedback", methods=["POST"])
def feedback():
    """
    Receive feedback from the user and analyze its sentiment.
    Expected JSON format:
    {
        "text": "...",         # feedback message from user
        "predicted": "...",    # predicted emotion (optional)
        "actual": "..."        # actual emotion user felt (optional)
    }
    """
    try:
        data = request.get_json(force=True)
        user_feedback = data.get("text", "").strip()
        predicted = data.get("predicted", "")
        actual = data.get("actual", "")

        if not user_feedback:
            return jsonify({"error": "Empty feedback"}), 400

        # Perform sentiment analysis
        sentiment = analyze_sentiment(user_feedback)

        # Ensure feedback_log.csv exists and append entry
        os.makedirs(os.path.dirname(FEEDBACK_LOG), exist_ok=True)
        with open(FEEDBACK_LOG, "a", newline="", encoding="utf-8") as f:
            writer = csv.writer(f)
            writer.writerow([user_feedback, predicted, actual, sentiment])

        return jsonify({
            "message": "Feedback received successfully",
            "sentiment": sentiment
        })

    except Exception as e:
        print("[Error] /feedback exception:", e)
        print(traceback.format_exc())
        return jsonify({"error": "internal server error"}), 500


# --- Run app ---
if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000, debug=True)
