# 📧 AI-Powered Phishing Email Detector

A modern, intelligent, and visually engaging phishing email detection system built using **RoBERTa**, **Streamlit**, and **LIME** — with real-time alerts via email.

---

## 🚀 Overview

With phishing attacks becoming increasingly sophisticated, this tool empowers individuals and organizations to detect malicious email content using a fine-tuned Transformer model. The system not only predicts whether an email is suspicious but also offers **visual explanations**, **confidence scores**, and **email alerting** to notify concerned parties immediately.

---

## 🧠 Key Features

- ✅ **Transformer-Powered**: Uses a fine-tuned [RoBERTa](https://huggingface.co/transformers/model_doc/roberta.html) model for binary classification.
- 🎛️ **Streamlit Interface**: Intuitive web UI for pasting email content and reviewing results.
- ⚠️ **Confidence Flagging**: Alerts if the model is uncertain (confidence < 0.5).
- 📩 **Email Notifications**: Automatically sends alerts when phishing content is detected.

---

## 🛠️ Tech Stack

| Tool/Library        | Purpose                                      |
|---------------------|----------------------------------------------|
| `Streamlit`         | Frontend web interface                       |
| `transformers`      | RoBERTa-based model for phishing detection   |
| `torch`             | Model serving using CUDA if available        |
| `smtplib` + `email` | Email alerts to predefined recipients        |

---
