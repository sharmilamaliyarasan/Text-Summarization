# 🔍 Text Summarization Dashboard

## App Link:
https://text-summarization-4vnwr8stqlnjfaqr9ro5nv.streamlit.app/

## 📌 Project Overview

This project is an End-to-End NLP application that summarizes long documents or articles into concise, human-readable summaries using T5-small (Text-to-Text Transfer Transformer).

It provides an interactive Streamlit dashboard where users can paste long text and get instant summaries. The app works fully on CPU and can also use a fine-tuned T5 model for better results.

## 📂 Data Preprocessing

Clean dataset: Keep only required columns (ctext as input, text as output)

Drop missing values

Limit sample size for CPU-friendly training

Split dataset into training and validation sets

Tokenize input (ctext) and output (text) for T5 model

Pad and truncate sequences to maximum lengths

## ⚙️ Model (T5-small)

Pretrained T5-small model from Hugging Face

Optional fine-tuning on custom dataset

Encoder-decoder architecture for text summarization

CPU-friendly configuration (batch size = 1, limited epochs)

## 📊 Evaluation Metrics

Loss during training

Generated summaries compared with reference text

Optional: ROUGE/L metrics for performance evaluation

## 📈 Visualization

Display original text vs generated summary

Compare target vs predicted summaries during validation

## 🚀 Features

✅ Summarize long articles or documents into concise summaries

✅ Interactive Streamlit dashboard for real-time summarization

✅ Automatic fallback to pretrained T5-small if no fine-tuned model is found

✅ Optional fine-tuning support with custom datasets (ctext → text)

✅ CPU-friendly setup with small batch sizes for low-memory environments

✅ Save/load fine-tuned T5 model for repeated use

## 🛠️ Tech Stack

Python 🐍

PyTorch → Model training and inference

Transformers (Hugging Face) → T5 model and tokenizer

Pandas / NumPy → Data handling

Streamlit → Interactive dashboard frontend

## 📂 Project Structure

Text_Summarization_App/
├── app.py                    
├── t5_summarizer_model_manual/ 
├── news_summary.csv          
├── requirements.txt          
└── README.md                 


## Install dependencies:

pip install -r requirements.txt


## Run the Streamlit app:

streamlit run app.py

## 📊 Example Workflow

Preprocess dataset → Clean, tokenize, truncate/pad sequences

(Optional) Fine-tune T5-small → ctext → text

Train model → Track loss per epoch

Save fine-tuned model → t5_summarizer_model_manual

Open Streamlit dashboard → Paste text → Click Summarize → View summary

## 📊 Evaluation Metrics

Loss → Training / validation loss per epoch

Generated summaries → Compare with target text

## 🎯 Future Enhancements

Support batch summarization from CSV upload

Add ROUGE / BLEU visualization in dashboard

Deploy publicly on Streamlit Cloud / Hugging Face Spaces

Enable multi-language summarization

Add role-based access (Admin / User) for managing models
