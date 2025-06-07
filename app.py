import streamlit as st
import torch
import numpy as np
from lime.lime_text import LimeTextExplainer
from transformers import AutoTokenizer, AutoModelForSequenceClassification
import torch.nn.functional as F
import smtplib
from email.mime.text import MIMEText
from email.mime.multipart import MIMEMultipart

# Email Alert Configuration
SMTP_SERVER = "smtp.gmail.com"
SMTP_PORT = 587
USERNAME = "rugwedyawalkar01@gmail.com"
PASSWORD = "vbeisxcixlfwblex"
TO_EMAILS = [
    "rugwedyawalkar06@gmail.com",
    "likhartirthak@gmail.com",
    "himanshusayankar230@gmail.com"
]

# Send Email Alert Function
def send_email_alert(subject, body):
    msg = MIMEMultipart()
    msg['From'] = USERNAME
    msg['To'] = ", ".join(TO_EMAILS)
    msg['Subject'] = subject

    msg.attach(MIMEText(body, 'plain'))
    
    try:
        with smtplib.SMTP(SMTP_SERVER, SMTP_PORT) as server:
            server.starttls()
            server.login(USERNAME, PASSWORD)
            server.sendmail(USERNAME, TO_EMAILS, msg.as_string())
    except Exception as e:
        st.error(f"❌ Failed to send email alert: {str(e)}")

# Styling
st.markdown("""
    <style>
        .title {
            font-size: 45px;
            font-weight: bold;
            color: #4CAF50;
            text-align: center;
        }
        .prediction {
            font-size: 24px;
            font-weight: bold;
            text-align: center;
        }
        .highlight {
            font-size: 18px;
            color: #f44336;
            font-weight: bold;
        }
        .warning {
            color: #ff9800;
            font-size: 18px;
            font-weight: bold;
        }
    </style>
""", unsafe_allow_html=True)

# Load model and tokenizer
model_path = "C:/Users/rugwe/OneDrive/Desktop/AI_LAB/models/roberta_finetuned"
tokenizer_path = "C:/Users/rugwe/OneDrive/Desktop/AI_LAB/models/roberta_finetuned_tokenizer"

model = AutoModelForSequenceClassification.from_pretrained(model_path)
tokenizer = AutoTokenizer.from_pretrained(tokenizer_path)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model.to(device)
model.eval()

label_names = ["Not Phishing", "Phishing"]
explainer = LimeTextExplainer(class_names=label_names)

def predict_proba(texts):
    inputs = tokenizer(texts, padding=True, truncation=True, return_tensors="pt").to(device)
    with torch.no_grad():
        logits = model(**inputs).logits
        probs = F.softmax(logits, dim=1).detach().cpu().numpy()
    return probs

def get_prediction(text):
    inputs = tokenizer(text, return_tensors="pt", truncation=True, padding=True).to(device)
    with torch.no_grad():
        logits = model(**inputs).logits
        probs = F.softmax(logits, dim=1).squeeze()
        pred_class = torch.argmax(probs).item()
        confidence = probs[pred_class].item()
        needs_manual_review = confidence < 0.5
    return label_names[pred_class], confidence, needs_manual_review

# Streamlit UI
st.markdown("<h1 class='title'>📧 AI-Powered Phishing Email Detector</h1>", unsafe_allow_html=True)

st.sidebar.image("https://cdn-icons-png.flaticon.com/128/3518/3518775.png", width=150)
st.sidebar.header("About")
st.sidebar.write("This app detects phishing emails using an AI model trained on real-world email data. The model predicts if the email is phishing or not.")

text = st.text_area("Paste your email text here:", height=200)

if text:
    label, conf, flag = get_prediction(text)

    st.markdown(f"<h2 class='prediction'>{label} </h2>", unsafe_allow_html=True)

    if label == "Phishing":
        st.markdown("<p class='highlight'>🚨 This is likely a phishing email.</p>", unsafe_allow_html=True)

        # Send alert email
        subject = "⚠️ Phishing Email Detected!"
        body = f"""A phishing email was detected with the following content:\n\n{text}\n\n"""
        send_email_alert(subject, body)

    elif flag:
        st.markdown("<p class='warning'>⚠️ This prediction is uncertain and may need manual review.</p>", unsafe_allow_html=True)
else:
    st.info("📝 Paste an email above to get started!")

# Footer
st.markdown("""
    <footer>
        <p style="text-align: center;">Made with ❤️ by RTH</p>
    </footer>
""", unsafe_allow_html=True)