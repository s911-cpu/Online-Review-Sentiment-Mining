import streamlit as st
import joblib
import re
import nltk
from nltk.corpus import stopwords

# Download stopwords if needed
nltk.download("stopwords", quiet=True)

# Load saved model and TF-IDF vectorizer
lr_final = joblib.load("lr_final.pkl")
tfidf_final = joblib.load("tfidf_final.pkl")


# -----------------------------
# Text preprocessing
# -----------------------------

def clean_text(text):
    text = text.lower()
    text = re.sub(r"http\S+|www\S+|https\S+", "", text)
    text = re.sub(r"<.*?>", "", text)
    text = re.sub(r"[^a-zA-Z\s]", "", text)
    text = re.sub(r"\s+", " ", text).strip()
    return text


def remove_stopwords(text):
    stop_words = set(stopwords.words("english"))
    words = text.split()
    return " ".join(word for word in words if word not in stop_words)


# -----------------------------
# Final hybrid prediction
# -----------------------------

def predict_sentiment_final(review):
    text = review.lower().strip()

    negative_phrases = [
        "very bad",
        "really bad",
        "so bad",
        "extremely bad",
        "terrible",
        "terrible product",
        "awful",
        "horrible",
        "worst",
        "very disappointing",
        "completely disappointing",
        "hate it",
        "waste of money"
    ]

    positive_phrases = [
        "very good",
        "really good",
        "so good",
        "excellent",
        "excellent product",
        "amazing",
        "awesome",
        "perfect",
        "love it",
        "really love",
        "highly recommend"
    ]

    for phrase in negative_phrases:
        if phrase in text:
            return "Negative"

    for phrase in positive_phrases:
        if phrase in text:
            return "Positive"

    cleaned = clean_text(review)
    processed = remove_stopwords(cleaned)
    review_tfidf = tfidf_final.transform([processed])

    return lr_final.predict(review_tfidf)[0]


# -----------------------------
# Rating → sentiment
# -----------------------------

def rating_to_sentiment(rating):
    if rating <= 2:
        return "Negative"
    elif rating == 3:
        return "Neutral"
    else:
        return "Positive"


# -----------------------------
# Page configuration
# -----------------------------

st.set_page_config(
    page_title="Online Review & Sentiment Mining",
    page_icon="💬",
    layout="centered"
)


# -----------------------------
# Minimal UI styling
# -----------------------------

st.markdown("""
<style>

.block-container {
    max-width: 700px;
    padding-top: 3rem;
    padding-bottom: 3rem;
}

h1 {
    text-align: center;
    font-size: 2.1rem;
    margin-bottom: 2.5rem;
}

.rating-label {
    font-size: 1rem;
    font-weight: 600;
    margin-top: 1.2rem;
    margin-bottom: 0.5rem;
}

.result {
    margin-top: 2rem;
    padding-top: 1rem;
    border-top: 1px solid #dddddd;
}

.result-title {
    font-size: 1.25rem;
    font-weight: 600;
    margin-bottom: 1rem;
}

</style>
""", unsafe_allow_html=True)


# -----------------------------
# App title
# -----------------------------

st.title("Online Review & Sentiment Mining")


# -----------------------------
# Review input
# -----------------------------

st.markdown("### How do you feel about the product/service?")

review = st.text_area(
    "Enter your review here",
    placeholder="Enter your review here...",
    height=120
)


# -----------------------------
# Rating
# -----------------------------

st.markdown(
    '<div class="rating-label">Rating</div>',
    unsafe_allow_html=True
)

rating = st.feedback("stars")


# -----------------------------
# Submit
# -----------------------------

if st.button("Submit", use_container_width=True):

    if not review.strip():
        st.warning("Please enter your review.")

    elif rating is None:
        st.warning("Please select a rating.")

    else:
        review_sentiment = predict_sentiment_final(review)

        # st.feedback stars are indexed from 0
        actual_rating = rating + 1

        rating_sentiment = rating_to_sentiment(actual_rating)

        st.markdown(
            '<div class="result"><div class="result-title">'
            'Sentiment Analysis'
            '</div></div>',
            unsafe_allow_html=True
        )

        st.write(
            f"**Review sentiment:** {review_sentiment}"
        )

        st.write(
            f"**Rating sentiment:** {rating_sentiment}"
        )