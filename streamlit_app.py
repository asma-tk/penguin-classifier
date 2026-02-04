import streamlit as st
import requests
from PIL import Image
import os

# Configuration
st.set_page_config(page_title="Penguin ID", page_icon="🐧", layout="wide")

# Chargement du CSS externe
if os.path.exists("style.css"):
    with open("style.css") as f:
        st.markdown(f"<style>{f.read()}</style>", unsafe_allow_html=True)

st.title("🐧 Penguin Identifier")

# --- SIDEBAR MINIMALE ---
with st.sidebar:
    try:
        response = requests.get("http://localhost:8000/api/health", timeout=2)
        if response.status_code == 200:
            st.success("● API Online")
        else:
            st.error("● API Error")
    except:
        st.error("○ API Offline")

# --- ZONE PRINCIPALE ---
uploaded_file = st.file_uploader("Upload a photo", type=["jpg", "jpeg", "png", "webp"])

if uploaded_file:
    col1, col2 = st.columns([1, 1])
    
    with col1:
        image = Image.open(uploaded_file)
        st.image(image, use_container_width=True, caption="Uploaded Image")
    
    with col2:
        if st.button("🔍 Analyze Image", type="primary", use_container_width=True):
            try:
                files = {"file": (uploaded_file.name, uploaded_file.getvalue(), uploaded_file.type)}
                res = requests.post("http://localhost:8000/api/identify/image", files=files)
                
                if res.status_code == 200:
                    data = res.json()
                    if data.get("success"):
                        preds = data["predictions"]
                        top = preds[0]
                        
                        # Résultat principal
                        name = top["species"].replace("_", " ").upper()
                        st.markdown(f'<div class="big-result">{name}</div>', unsafe_allow_html=True)
                        
                        # Barre de confiance
                        prob = top["probability"]
                        st.markdown(f"""
                            <div class="confidence-bar-container">
                                <div class="confidence-bar-fill" style="width:{prob}%">{prob}% Confidence</div>
                            </div>
                        """, unsafe_allow_html=True)
                        
                        st.write(f"**Scientific Name:** *{top.get('scientific', 'N/A')}*")
                        st.write(f"**Classification:** {top.get('category', 'N/A')}")

                        # Autres suggestions
                        if len(preds) > 1:
                            with st.expander("Other possibilities"):
                                for p in preds[1:3]:
                                    st.markdown(f"""
                                        <div class="species-card">
                                            <strong>{p['species'].replace('_', ' ')}</strong><br>
                                            <small>{p['probability']}% match</small>
                                        </div>
                                    """, unsafe_allow_html=True)
                else:
                    st.error("API Error")
            except Exception as e:
                st.error(f"Connection failed: {e}")