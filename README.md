# 🧠 Mental Health Text Classifier

This is a **Streamlit-based NLP machine learning project** that classifies mental health-related text into categories like *depression*, *positive*, *neutral*, etc. Built to detect the emotional or psychological state based on user-submitted Reddit-style posts.

---

## 🚀 Features

- 🔍 Clean, minimal **Streamlit interface**
- 🤖 Trained **Logistic Regression model** for classification
- 🧼 Robust **text preprocessing**
- 📊 EDA and model training done in Jupyter Notebook
- 💾 Reusable `preprocess.py` and `train_model.py` scripts
- 📦 Joblib saved models: `text_classifier.pkl`, `vectorizer.pkl`, and `label_encoder.pkl`

---

## 📁 Project Structure

mental-health-classifier/
│
├── app/
│ └── streamlit_app.py # Streamlit app
│
├── data/
│ └── raw/
│ └── data_to_be_cleansed.csv
│
├── models/
│ ├── text_classifier.pkl
│ ├── vectorizer.pkl
│ └── label_encoder.pkl
│
├── notebooks/
│ ├── 1_EDA.ipynb
│ └── 2_Model_Training.ipynb
│
├── src/
│ ├── preprocess.py
│ └── train_model.py
│
└── README.md (you are here)

yaml
Copy
Edit

---

## ⚙️ How to Run Locally

```bash
# Step 1: Go to project folder
cd mental-health-classifier

# Step 2: Run Streamlit app
streamlit run app/streamlit_app.py
Make sure you have Python + required libraries installed:
pip install streamlit pandas scikit-learn joblib matplotlib seaborn

🧠 Sample Inputs
You can try examples like:

"I feel empty and numb inside"

"I had an amazing day, feeling grateful"

"Why does everything feel pointless lately?"

The model will predict the emotional tone or mental state behind the post.

📌 Tech Stack
Python

Streamlit

Pandas, NumPy

scikit-learn

Joblib

Jupyter Notebook

🙌 Contribution
This project was built by Bharat Mandge
Feel free to fork, star ⭐, and contribute ideas or improvements!

“Mental health matters — let's use AI to support awareness & understanding.”
