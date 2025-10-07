import streamlit as st
from transformers import T5Tokenizer, T5ForConditionalGeneration
import torch


st.set_page_config(page_title="Text Summarizer", layout="wide")
st.title("📝 Text Summarization App")
st.markdown("### Powered by T5-small — Summarize long articles into concise summaries!")


@st.cache_resource
def load_model():
   
    model_path = "./t5_summarizer_model_manual"  
    try:
        tokenizer = T5Tokenizer.from_pretrained(model_path)
        model = T5ForConditionalGeneration.from_pretrained(model_path)
    except:
        st.info("Using pretrained T5-small model for summarization.")
        tokenizer = T5Tokenizer.from_pretrained("t5-small")
        model = T5ForConditionalGeneration.from_pretrained("t5-small")
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = model.to(device)
    return tokenizer, model, device

tokenizer, model, device = load_model()


st.subheader("Enter the text you want to summarize 👇")
input_text = st.text_area(
    "Paste your long article or paragraph here:",
    height=300,
    placeholder="Enter your ctext (long document)..."
)

if st.button("✨ Generate Summary"):
    if input_text.strip() == "":
        st.warning("Please enter some text before summarizing!")
    else:
        with st.spinner("Generating summary... please wait ⏳"):
            
            text = "summarize: " + input_text
            inputs = tokenizer.encode(text, return_tensors="pt", max_length=512, truncation=True).to(device)

            
            summary_ids = model.generate(
                inputs,
                max_length=150,
                min_length=40,
                length_penalty=2.0,
                num_beams=4,
                early_stopping=True
            )
            summary = tokenizer.decode(summary_ids[0], skip_special_tokens=True)

        
        st.success("✅ Summary Generated Successfully!")
        st.subheader("🧾 Summary:")
        st.write(summary)


st.markdown("---")
st.caption("Developed by Sharmi 🌸 | Model: T5-small | Runs on CPU")

