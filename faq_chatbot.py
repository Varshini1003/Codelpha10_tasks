import nltk
import string
from nltk.corpus import stopwords

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

# Download required datasets (first time only)
nltk.download('punkt')
nltk.download('stopwords')
# FAQ Dataset
faq_data = {
    "What is your return policy?":
        "You can return products within 30 days of purchase.",

    "How can I track my order?":
        "You can track your order using the tracking link sent to your email.",

    "Do you offer international shipping?":
        "Yes, we offer international shipping to many countries.",

    "How do I contact customer support?":
        "You can contact customer support at support@example.com.",

    "What payment methods do you accept?":
        "We accept credit cards, debit cards, UPI, and net banking."
}

# Text Preprocessing Function
def preprocess(text):

    text = text.lower()

    text = text.translate(
        str.maketrans('', '', string.punctuation)
    )

    words = text.split()

    stop_words = set(stopwords.words('english'))

    words = [
        word for word in words
        if word not in stop_words
    ]

    return " ".join(words)

# Prepare FAQ Questions
questions = list(faq_data.keys())

processed_questions = [
    preprocess(question)
    for question in questions
]

# TF-IDF Vectorization
vectorizer = TfidfVectorizer()

faq_vectors = vectorizer.fit_transform(
    processed_questions
)

# Chatbot Function
def chatbot(user_question):

    processed_input = preprocess(user_question)

    input_vector = vectorizer.transform(
        [processed_input]
    )

    similarity_scores = cosine_similarity(
        input_vector,
        faq_vectors
    )

    best_match_index = similarity_scores.argmax()

    best_score = similarity_scores[0][best_match_index]

    if best_score > 0.2:
        return faq_data[
            questions[best_match_index]
        ]
    else:
        return "Sorry, I couldn't understand your question."

# Chat Loop
print("=" * 50)
print("FAQ CHATBOT")
print("Type 'exit' to quit")
print("=" * 50)

while True:

    user_input = input("\nYou: ")

    if user_input.lower() == "exit":
        print("Chatbot: Goodbye!")
        break

    response = chatbot(user_input)

    print("Chatbot:", response)