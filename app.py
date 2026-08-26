"""
app.py

Simple Streamlit interface for the fine-tuned T5-based grammar corrector.

Run with:
    streamlit run app.py
"""

import streamlit as st
from grammar_corrector import correct_grammar

st.set_page_config(page_title="Grammar Corrector", page_icon="✍️")

st.title("✍️ Grammar Corrector")
st.write("Enter a sentence below and get a grammar-corrected version, powered by a fine-tuned T5 model.")

user_input = st.text_area("Your sentence", height=120, placeholder="e.g. She go to school every day.")

if st.button("Correct Grammar", type="primary"):
    if not user_input.strip():
        st.warning("Please enter a sentence first.")
    else:
        with st.spinner("Correcting..."):
            corrected = correct_grammar(user_input)
        st.subheader("Corrected Sentence")
        st.success(corrected)
