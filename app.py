import streamlit as st
import json
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import numpy as np

st.set_page_config(page_title="CodeAlpha FAQ Bot", page_icon="🤖", layout="centered")

st.title("🤖 CodeAlpha FAQ Chatbot")
st.write("Ask me anything about CodeAlpha internship, tasks, or submission!")

# Load FAQs
@st.cache_data
def load_faqs():
    with open('faqs.json', 'r', encoding='utf-8') as f:
        return json.load(f)

faqs = load_faqs()
questions = [item['question'] for item in faqs]
answers = [item['answer'] for item in faqs]

# TF-IDF Vectorizer
vectorizer = TfidfVectorizer()
question_vectors = vectorizer.fit_transform(questions)

# User input
user_question = st.text_input("💬 Your Question:", placeholder="e.g. How many tasks to complete?")

col1, col2 = st.columns([1,4])
with col1:
    ask_button = st.button("Ask 🚀", use_container_width=True)

if ask_button and user_question:
    user_vector = vectorizer.transform([user_question])
    similarities = cosine_similarity(user_vector, question_vectors)
    best_match_idx = np.argmax(similarities)
    
    if similarities[0][best_match_idx] > 0.25:
        st.success("**Answer:**")
        st.info(answers[best_match_idx])
        with st.expander("Matched FAQ"):
            st.write(f"**Q:** {questions[best_match_idx]}")
    else:
        st.warning("Sorry, I couldn't find a good match. Try asking about: tasks, certificate, submission, or deadline.")

st.sidebar.markdown("### 📌 Sample Questions")
for q in questions:
    st.sidebar.write(f"• {q}")

st.sidebar.markdown("---")
st.sidebar.markdown("**Built by:** Kinza Saleem")
st.sidebar.markdown("**Task 2:** CodeAlpha AI Internship")
