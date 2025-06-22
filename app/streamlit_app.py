import streamlit as st
import joblib
import re
import string
import random

# ---------- Clean text ----------
def clean_text(text):
    text = str(text).lower()
    text = re.sub(r"http\S+|www\S+|https\S+", '', text)
    text = re.sub(r'\@w+|\#','', text)
    text = re.sub(r'[%s]' % re.escape(string.punctuation), '', text)
    text = re.sub(r'\w*\d\w*', '', text)
    return text

# ---------- Load Model Stuff ----------
model = joblib.load("models/text_classifier.pkl")
vectorizer = joblib.load("models/vectorizer.pkl")
label_encoder = joblib.load("models/label_encoder.pkl")

# ---------- Streamlit Config ----------
st.set_page_config(page_title="Mental Health Classifier", page_icon="🧠", layout="centered")

# ---------- Title Section ----------
st.markdown("<h1 style='text-align: center; color: #6C63FF;'>🧠 Mental Health Post Classifier</h1>", unsafe_allow_html=True)
st.markdown("<p style='text-align: center; font-size: 16px;'>Enter your thoughts — and let AI guide your mental clouds toward clarity.</p>", unsafe_allow_html=True)
st.markdown("---")

# ---------- Sample Examples ----------
examples = [
    "I can't sleep at night and I feel like I'm drowning in my own thoughts.",
    "Had a great day today! Felt really motivated after a long time.",
    "I just feel numb. Nothing excites me anymore.",
    "Life’s been really hard, but I’m trying to stay positive.",
    "No one understands how tired I am, even when I do nothing."
]

if st.button("✨ Try Random Thought"):
    st.session_state.input_text = random.choice(examples)

# ---------- Text Input ----------
text = st.text_area("📝 Write your thoughts below:", key="input_text")

# ---------- Predict ----------
if st.button("🔮 Analyze My Mental State"):
    if not text.strip():
        st.warning("Bhai... kuch likh toh! AI Baba ke paas blank thought nahi chalta 😤")
    else:
        cleaned = clean_text(text)
        vect = vectorizer.transform([cleaned])
        pred = model.predict(vect)[0]
        
        # ✅ Fix: Convert label to string
        label_str = str(label_encoder.inverse_transform([pred])[0])

        st.success(f"🧠 **Predicted Mental State:** `{label_str}`")

        if "depress" in label_str.lower():
            st.markdown("😔 **You're strong, even if it doesn't feel like it right now.**")
            st.info("🌱 *“Even the darkest night will end, and the sun will rise.”* — Victor Hugo")
        elif "anx" in label_str.lower() or "stress" in label_str.lower():
            st.markdown("💨 **Breathe. You’re doing better than you think.**")
            st.info("🧘 *“Do not believe everything you think. Thoughts are just thoughts.”*")
        elif "happy" in label_str.lower() or "positive" in label_str.lower():
            st.markdown("😄 **Yessir! Your vibe is immaculate!**")
            st.info("☀️ *“Happiness is contagious — spread it like stardust.”*")
        else:
            st.markdown("🧘‍♂️ **Keep checking in with yourself. Awareness is the first step.**")
            st.info("🌻 *“Sometimes not being okay is part of the journey toward healing.”*")
