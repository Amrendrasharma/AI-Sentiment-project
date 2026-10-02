"""Interactive sentiment checker that learns from repeatedly confirmed reviews."""

import csv
import json
import re
from collections import Counter
from pathlib import Path


BASE_DIR = Path(__file__).resolve().parent
DATA_FILE = BASE_DIR / "data.csv"
PENDING_FILE = BASE_DIR / "pending_votes.json"
LABELS = ("p", "n", "n")

# Used until enough verified examples have been collected in data.csv.
STARTER_EXAMPLES = [
	("I love this product, it works wonderfully", "positive"),
	("Excellent quality and fantastic service", "positive"),
	("Very happy with my purchase", "positive"),
	("This is great and easy to use", "positive"),
	("It is okay, nothing special", "neutral"),
	("The product is average and does the job", "neutral"),
	("I have no strong opinion about this", "neutral"),
	("It arrived as described", "neutral"),
	("I hate this product, it is terrible", "negative"),
	("Poor quality and very disappointing", "negative"),
	("The service was awful and slow", "negative"),
	("I am unhappy with this purchase", "negative"),
]

POSITIVE_WORDS = {"nice","good","awesome", "best", "brilliant", "enjoy", "excellent", "fantastic",
	"amazing", "awesome", "best", "brilliant", "enjoy", "excellent", "fantastic",
	"good", "great", "happy", "like", "love", "perfect", "recommend", "wonderful",
}
NEGATIVE_WORDS = {
	"awful", "bad", "broken", "disappointing", "hate", "horrible", "poor", "terrible",
	"unhappy", "waste", "worst",
}
NEGATIONS = {"no", "not", "never", "isn't", "wasn't", "don't", "didn't", "can't"}


def load_training_data():
	examples = list(STARTER_EXAMPLES)
	if DATA_FILE.exists():
		try:
			with DATA_FILE.open("r", newline="", encoding="utf-8") as file:
				for row in csv.DictReader(file):
					text = row.get("review", "").strip()
					sentiment = row.get("sentiment", "").strip().lower()
					try:
						verified = int(row.get("confirmations", "0")) >= 3
					except (TypeError, ValueError):
						verified = False
					if text and sentiment in LABELS and verified:
						examples.append((text, sentiment))
		except (OSError, csv.Error):
			print("Warning: could not read data.csv; using built-in examples.")
	return examples


def predict_sentiment(review):
	examples = load_training_data()
	try:
		from sklearn.feature_extraction.text import TfidfVectorizer
		from sklearn.naive_bayes import MultinomialNB
		from sklearn.pipeline import make_pipeline

		texts, labels = zip(*examples)
		model = make_pipeline(TfidfVectorizer(ngram_range=(1, 2)), MultinomialNB())
		model.fit(texts, labels)
		return str(model.predict([review])[0])
	except ImportError:
		return lexicon_prediction(review)
	except ValueError:
		return lexicon_prediction(review)


def lexicon_prediction(review):
	words = re.findall(r"[a-z']+", review.lower())
	score = 0
	for index, word in enumerate(words):
		value = 1 if word in POSITIVE_WORDS else -1 if word in NEGATIVE_WORDS else 0
		if value and index > 0 and words[index - 1] in NEGATIONS:
			value *= -1
		score += value
	return "positive" if score > 0 else "negative" if score < 0 else "neutral"


def load_pending():
	try:
		with PENDING_FILE.open("r", encoding="utf-8") as file:
			pending = json.load(file)
		return pending if isinstance(pending, dict) else {}
	except (OSError, json.JSONDecodeError):
		return {}


def save_verified_review(review, sentiment):
	rows = []
	if DATA_FILE.exists():
		try:
			with DATA_FILE.open("r", newline="", encoding="utf-8") as file:
				rows = list(csv.DictReader(file))
		except (OSError, csv.Error):
			rows = []
	# Keep the newest verified label for an identical review.
	rows = [row for row in rows if row.get("review", "").strip().casefold() != review.strip().casefold()]
	rows.append({"review": review, "sentiment": sentiment, "confirmations": "3"})
	with DATA_FILE.open("w", newline="", encoding="utf-8") as file:
		writer = csv.DictWriter(file, fieldnames=("review", "sentiment", "confirmations"))
		writer.writeheader()
		writer.writerows(rows)


def collect_feedback(review, prediction, pending):
	answer = input("Is this prediction correct? [y/n]: ").strip().lower()
	if answer in {"y", "yes"}:
		feedback = prediction
	elif answer in {"n", "no"}:
		feedback = input("Choose the correct sentiment (p/nut/n): ").strip().lower()
		if feedback not in LABELS:
			print("Invalid sentiment; feedback was not recorded.")
			return
	else:
		print("Please enter y or n; feedback was not recorded.")
		return

	key = review.strip().casefold()
	record = pending.setdefault(key, {"review": review.strip(), "votes": {}})
	votes = record.setdefault("votes", {})
	votes[feedback] = int(votes.get(feedback, 0)) + 1
	total = sum(votes.values())
	if total >= 3:
		# Resolve disagreement by majority; ties are resolved by the latest vote.
		highest = max(votes.values())
		winners = [label for label, count in votes.items() if count == highest]
		final_label = feedback if feedback in winners else winners[0]
		save_verified_review(record["review"], final_label)
		pending.pop(key, None)
		print(f"Review verified as {final_label}. Saved to data.csv for future predictions.")
	else:
		print(f"Feedback recorded ({total}/3 confirmations for this review).")
	with PENDING_FILE.open("w", encoding="utf-8") as file:
		json.dump(pending, file, ensure_ascii=False, indent=2)


def main():
	print("AI Sentiment Checker (type 'quit' to exit)")
	pending = load_pending()
	while True:
		review = input("\nEnter a review: ").strip()
		if review.lower() in {"quit", "exit"}:
			break
		if not review:
			print("Please enter a review.")
			continue
		sentiment = predict_sentiment(review)
		print(f"Predicted sentiment: {sentiment}")
		collect_feedback(review, sentiment, pending)


if __name__ == "__main__":
	main()
