
# 📧 Spam Email Prediction Using Machine Learning

This project is a machine learning-based spam email detection system that classifies emails as **Spam** or **Not Spam**. It uses natural language processing (NLP) techniques and a **Multinomial Naive Bayes** classifier trained on real-world email data.

---

## 📂 Dataset

The dataset used in this project was sourced from **Kaggle**:
- **Name**: Email Spam Classification Dataset
- **Link**: [Kaggle Dataset](https://www.kaggle.com/datasets/balaka18/email-spam-classification-dataset-csv)

The dataset contains labeled email messages marked as `spam (1)` or `ham (0)` along with their text content.

---

## ⚙️ Project Features

- Data cleaning and preprocessing using **NLTK**
- Text vectorization using **TF-IDF Vectorizer**
- Classification using **Multinomial Naive Bayes**
- Model evaluation using **accuracy, precision, confusion matrix**
- Visual analysis of word frequency in spam and ham emails
- Simple CLI (Command Line Interface) for real-time spam prediction
- Model and vectorizer are saved using **pickle** for reuse

---

## 🛠️ Technologies Used

- **Python 3.10+**
- **Scikit-learn** – for machine learning models
- **NLTK** – for natural language processing
- **Pandas & NumPy** – for data handling
- **Matplotlib & Seaborn** – for data visualization
- **Pickle** – to save and load the model/vectorizer

---

## 🧠 How the Model Works

1. **Preprocessing**:
   - Lowercasing
   - Tokenization
   - Removing stopwords and punctuation
   - Stemming using PorterStemmer

2. **Vectorization**:
   - Emails are converted to numerical format using **TF-IDF**.

3. **Classification**:
   - A **Multinomial Naive Bayes** model is trained to classify emails as spam or ham.

4. **Prediction**:
   - The user enters an email via the terminal.
   - The system preprocesses and vectorizes it.
   - The trained model predicts if it's spam or not.

---

## 🚀 How to Run the Project

1. Clone the repository:
   ```bash
   git clone https://github.com/your-username/spam-email-prediction.git
   cd spam-email-prediction
````

2. Install dependencies:

   ```bash
   pip install -r requirements.txt
   ```

3. Run the program:

   ```bash
   python app.py
   ```

4. Follow the prompt to input an email for prediction.

---

## 📦 Files in the Repository

* `app.py` – Main Python script for predicting spam
* `vectorizer.pkl` – Saved TF-IDF vectorizer
* `model.pkl` – Trained Naive Bayes model
* `emails.csv` – The dataset (or link in `data/` directory)
* `README.md` – Project documentation

---

This project was developed as part of an academic semester project on **Artificial Intelligence**.

---

## 📜 License

This project is licensed for educational purposes. Dataset is publicly available on Kaggle and subject to their licensing terms.

```

```



