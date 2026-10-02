from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression

texts = [
    "I love this movie", "I love this phone", "This is amazing",
    "Very happy with the service", "Great product",
    "I hate this", "This is terrible", "Very bad experience", "Worst app ever",
]

labels = [
    "positive", "positive", "positive", "positive", "positive",
    "negative", "negative", "negative", "negative",
]

vectorizer = TfidfVectorizer()
x = vectorizer.fit_transform(texts)

model = LogisticRegression()
model.fit(x, labels)

while True:
    user_text = input("Enter your review: ")

    if user_text.strip().lower() == "quit":
        print("Bye Tata")
        break

    if user_text.strip() == "":
        print("Bhai kuch to likho malik!")
        continue

    prediction = model.predict(vectorizer.transform([user_text]))
    print("Sentiment:", prediction[0])