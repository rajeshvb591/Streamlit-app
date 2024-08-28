import streamlit as st
from transformers import pipeline

# Initialize the summarization pipeline
summarizer = pipeline("summarization", model="facebook/bart-large-cnn")

def summarize_text(text):
    """Generates a summary of the input text."""
    # The pipeline might require a minimum length for summarization, so we check if the input is sufficient.
    if len(text) < 50:
        return "Text is too short to summarize. Please provide more content."
    
    # Generate the summary
    summary = summarizer(text, max_length=150, min_length=50, do_sample=False)
    return summary[0]['summary_text']

def main():
    st.title("Text Summarization App")
    
    # User input
    user_input = st.text_area("Enter text to summarize:", height=300)
    
    if st.button("Summarize"):
        if user_input:
            summary = summarize_text(user_input)
            st.subheader("Summary")
            st.write(summary)
        else:
            st.warning("Please enter some text to summarize.")

if __name__ == "__main__":
    main()
