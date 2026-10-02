# AI Sentiment Project

A small machine learning project that classifies a review as **positive** or **negative**.
It also learns from user feedback: if the prediction is wrong, the user can correct it and the model retrains itself.

## Features

- Converts text into numbers using TF-IDF
- Predicts sentiment using Logistic Regression
- Asks for feedback after every prediction (y/n)
- Saves corrections to `data.csv`, so the model remembers them next time

## Tech Stack

- Python 3
- scikit-learn
- Git and GitHub

## Project Structure

```
AI-Sentiment-project/
├── app.py             # Main program
├── data.csv           # Training data (grows with user feedback)
├── requirements.txt   # Python dependencies
└── README.md          # Project documentation
```

## Getting Started

### 1. Clone the repository

```bash
git clone https://github.com/your-username/AI-Sentiment-project.git
cd AI-Sentiment-project
```

### 2. Create and activate a virtual environment

```bash
python -m venv venv
source venv/Scripts/activate    # Git Bash on Windows
```

### 3. Install dependencies

```bash
pip install -r requirements.txt
```

### 4. Run the app

```bash
python app.py
```

## Example

```
Enter your review: I love this app
Sentiment: positive
Was this correct? (y/n): y
```

Type `quit` to exit.

## How the Feedback Loop Works

1. The model predicts the sentiment of your text.
2. You tell it whether the prediction was right.
3. If it was wrong, the corrected example is saved to `data.csv`.
4. The model retrains with the updated data.

## Future Improvements

- [ ] Add accuracy evaluation and unit tests
- [ ] Save and load the trained model
- [ ] Build a REST API with FastAPI
- [ ] Containerize with Docker
- [ ] Add CI with GitHub Actions

## Author

Amrendra