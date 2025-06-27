import streamlit as st
import json
from transformers import AutoTokenizer, AutoModelForCausalLM
from huggingface_hub import login
from dotenv import load_dotenv
import torch
import os

# Load environment and HF token
load_dotenv()
hf_token = os.getenv("HF_TOKEN")
login(hf_token)

# Load model and tokenizer
@st.cache_resource
def load_model():
    tokenizer = AutoTokenizer.from_pretrained("ibm-granite/granite-3.3-2b-instruct", use_auth_token=True)
    model = AutoModelForCausalLM.from_pretrained(
        "ibm-granite/granite-3.3-2b-instruct",
        torch_dtype=torch.float32,
        use_auth_token=True
    )
    return tokenizer, model

tokenizer, model = load_model()

# Streamlit Page Config
st.set_page_config(page_title="SDLC Admin Dashboard", page_icon="🛠️", layout="wide")
st.title("🛠️ SDLC-AI Admin Dashboard")

# Tabs for organization
tab1, tab2 = st.tabs(["Chat Assistant", "User Info"])

# Tab 1: Chat Assistant
with tab1:
    st.subheader("💬 Chat with Model")
    user_input = st.text_area("Prompt", placeholder="Type your SDLC-related prompt here...")
    if st.button("Generate Response"):
        with st.spinner("Generating response..."):
            inputs = tokenizer(user_input, return_tensors="pt")
            outputs = model.generate(
                **inputs,
                max_new_tokens=250,
                temperature=0.7,
                do_sample=True,
                top_p=0.95
            )
            response = tokenizer.decode(outputs[0], skip_special_tokens=True)
            response = response.replace(user_input, "").strip()
            st.success("Model Response")
            st.write(response)

# Tab 2: View Registered Users
with tab2:
    st.subheader("📋 Registered User Emails")
    try:
        with open("users.json", "r") as f:
            users = json.load(f)
            if users:
                for i, email in enumerate(users.keys(), start=1):
                    st.markdown(f"**{i}.** {email}")
            else:
                st.info("No users found.")
    except FileNotFoundError:
        st.warning("users.json file not found. Please ensure it exists.")
