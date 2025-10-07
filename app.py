import streamlit as st
import torch
from transformers import T5Tokenizer, T5ForConditionalGeneration

# Load model and tokenizer
st.title("📝 Text Summarizer using T5")

@st.cache_resource
def load_model():
    tokenizer = T5Tokenizer.from_pretrained("t5-small")
    model = T5ForConditionalGeneration.from_pretrained("t5-small", low_cpu_mem_usage=True)
    return tokenizer, model

tokenizer, model = load_model()

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model.to(device)

# Input box
text_input = st.text_area("Enter your text here:", height=200)

if st.button("Summarize"):
    if len(text_input.strip()) == 0:
        st.warning("⚠️ Please enter some text to summarize.")
    else:
        st.write("⏳ Summarizing...")
        input_text = "summarize: " + text_input

        inputs = tokenizer.encode(input_text, return_tensors="pt", truncation=True, max_length=512)
        inputs = inputs.to(device)

        # Generate summary
        summary_ids = model.generate(
            inputs,
            max_length=150,
            min_length=40,
            length_penalty=2.0,
            num_beams=4,
            early_stopping=True
        )

        summary = tokenizer.decode(summary_ids[0], skip_special_tokens=True)
        st.subheader("Summary:")
        st.success(summary)
