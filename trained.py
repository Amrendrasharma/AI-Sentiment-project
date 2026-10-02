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


def train_model():
    """Texts aur labels se naya model banata hai."""
    vectorizer = TfidfVectorizer()
    x = vectorizer.fit_transform(texts)
    model = LogisticRegression()
    model.fit(x, labels)
    return vectorizer, model


vectorizer, model = train_model()

print("Sentiment checker shuru! Band karne ke liye 'quit' likho.")

while True:
    user_text = input("\nEnter your review: ")

    if user_text.strip().lower() == "quit":
        print("Bye Tata")
        break

    if user_text.strip() == "":
        print("Bhai kuch to likho malik!")
        continue

    prediction = model.predict(vectorizer.transform([user_text]))[0]
    print("Sentiment:", prediction)

    # Feedback lena
    feedback = input("Kya ye sahi tha? (y/n): ").strip().lower()

    if feedback == "n":
        # Sirf 2 labels hain, to galat hone par sahi label ulta hoga
        correct_label = "negative" if prediction == "positive" else "positive"

        # 3 baar add kiya taaki model par asar zyada pade
        for _ in range(3):
            texts.append(user_text)
            labels.append(correct_label)

        vectorizer, model = train_model()   # dobara train
        print(f"Thanks! Maine seekh liya: '{user_text}' = {correct_label}")

    elif feedback == "y":
        print("Badhiya!")
    else:
        print("Sirf y ya n likho.")