import streamlit as st
from PIL import Image
import torch
import time
from streamlit_webrtc import webrtc_streamer
import cv2
import numpy as np
from pathlib import Path
import plotly.graph_objects as go
import plotly.express as px
import pandas as pd

# Import our classifier
from waste_classifier import DualWasteClassifier

class WasteVisionApp:
    def __init__(self):
        self.classifier = DualWasteClassifier()
        self.setup_page()
        self.main()
    
    def setup_page(self):
        st.set_page_config(page_title="Waste Vision", layout="wide")
        
        # Enhanced Custom CSS
        st.markdown("""
        <style>
        .main {
            background-color: #f8f9fa;
            padding: 2rem;
        }
        .stButton>button {
            background-color: #28a745;
            color: white;
            border-radius: 10px;
            padding: 0.5rem 2rem;
            border: none;
            transition: all 0.3s ease;
        }
        .stButton>button:hover {
            background-color: #218838;
            transform: translateY(-2px);
            box-shadow: 0 4px 6px rgba(0,0,0,0.1);
        }
        .success-box {
            padding: 1.5rem;
            border-radius: 10px;
            background-color: #e8f5e9;
            border: 2px solid #28a745;
            margin: 1rem 0;
            box-shadow: 0 4px 6px rgba(0,0,0,0.1);
        }
        .metric-card {
            background-color: white;
            padding: 1.5rem;
            border-radius: 10px;
            box-shadow: 0 4px 6px rgba(0,0,0,0.1);
            margin: 1rem 0;
        }
        h1, h2, h3 {
            color: #2c3e50;
        }
        .stPlotlyChart {
            background-color: white;
            padding: 1rem;
            border-radius: 10px;
            box-shadow: 0 4px 6px rgba(0,0,0,0.1);
        }
        .upload-section {
            background-color: white;
            padding: 2rem;
            border-radius: 15px;
            box-shadow: 0 4px 6px rgba(0,0,0,0.1);
        }
        .how-it-works {
            background-color: white;
            padding: 2rem;
            border-radius: 15px;
            box-shadow: 0 4px 6px rgba(0,0,0,0.1);
            margin-top: 1rem;
        }
        </style>
        """, unsafe_allow_html=True)
    
    def main(self):
        st.title("🌍 Waste Vision: Revolutionizing Waste Management")
        
        # Navigation with custom styling
        with st.sidebar:
            st.markdown("### 📌 Navigation")
            section = st.radio("", ["Live Demo", "Impact and Vision"])
        
        if section == "Live Demo":
            self.show_live_demo()
        else:
            self.show_future_vision()
    
    def show_live_demo(self):
        st.header("🎯 Live Waste Classification Demo")
        
        col1, col2 = st.columns(2, gap="large")
        
        with col1:
            st.markdown("""
            <div class='upload-section'>
                <h3>📤 Upload Image</h3>
            </div>
            """, unsafe_allow_html=True)
            uploaded_file = st.file_uploader("Choose an image...", type=["jpg", "jpeg", "png"])
            if uploaded_file is not None:
                image = Image.open(uploaded_file)
                st.image(image, caption="Uploaded Image", use_column_width=True)
        
        with col2:
            st.markdown("""
            <div class='how-it-works'>
                <h3>🔍 How it Works</h3>
            </div>
            """, unsafe_allow_html=True)
            st.write("Our AI-powered waste classifier uses computer vision to identify the type of waste in the uploaded image.")
            st.write("The model is trained on a large dataset of images and can classify waste into four categories: recyclable, organic, hazardous, and other.")
        
        if uploaded_file is not None:
            with st.spinner("Classifying..."):
                time.sleep(2)  # Simulate processing time
                classification_result = self.classifier.classify(image)
                st.write(f"**Classification Result:** {classification_result}")
    
# ... (previous imports and initial setup remain the same)

    def show_future_vision(self):
        st.header("🌟 Impact and Business Value")
        
        # Cost Breakdown Section
        st.subheader("💰 Implementation Cost Breakdown")
        
        cost_df = pd.DataFrame({
            'Component': [
                'Hardware Setup (Cameras & Computing Units)',
                'AI Model Implementation & Integration',
                'Staff Training',
                'Maintenance (Annual)',
                'Software Licenses & Updates (Annual)'
            ],
            'Estimated Cost (USD)': [
                '$15,000 - $25,000',
                '$30,000 - $50,000',
                '$5,000 - $8,000',
                '$5,000 - $10,000',
                '$2,000 - $5,000'
            ]
        })
        
        st.table(cost_df)
        
        # ROI and Savings Section
        st.subheader("📈 Return on Investment")
        
        col1, col2 = st.columns(2)
        
        with col1:
            st.markdown("""
            #### Annual Savings Potential:
            * Labor cost reduction: $40,000 - $60,000
            * Improved sorting accuracy: $25,000 - $35,000
            * Reduced contamination penalties: $15,000 - $25,000
            * Increased recycling revenue: $20,000 - $30,000
            
            **Total Annual Savings: $100,000 - $150,000**
            """)
        
        with col2:
            st.markdown("""
            #### Additional Benefits:
            * 🎯 95% sorting accuracy (vs. 70-80% manual)
            * ⚡ 3x faster processing speed
            * 🌱 Reduced environmental impact
            * 📊 Real-time waste analytics
            * 🔄 24/7 operation capability
            """)
        
        # Break-even Analysis
        st.subheader("📊 Break-even Analysis")
        st.write("""
        Based on the implementation costs and potential savings:
        * Initial Investment: $50,000 - $83,000
        * Annual Recurring Costs: $7,000 - $15,000
        * Annual Savings: $100,000 - $150,000
        
        **Expected Break-even Period: 6-10 months**
        """)
        
        # Contact Section
        st.header("📬 Get In Touch")
        st.markdown("""
        Interested in implementing this solution? Or just find other applications?
        
        **Contact:**
        * 📧 Email: Krish.ubc.j@gmail.com
        * 💼 Open to collaboration and consulting opportunities
        * 🤝 Available for proof-of-concept demonstrations
        
        Whether you're looking to:
        * Implement a complete waste sorting solution
        * Conduct a pilot project
        * Discuss customization options
        * Learn more about the technology
        * Other verticals
        """)
        
        # FAQ Section
        st.header("❓ Frequently Asked Questions")
        
        with st.expander("How long does implementation typically take?"):
            st.write("""
            The typical implementation timeline is 2-3 months, including:
            * Initial assessment: 1-2 weeks
            * Hardware setup: 2-3 weeks
            * Software integration: 2-3 weeks
            * Testing and optimization: 2-3 weeks
            * Staff training: 1 week
            """)
            
        with st.expander("What kind of support is provided after implementation?"):
            st.write("""
            * 24/7 technical support
            * Regular system maintenance
            * Quarterly performance reviews
            * Software updates and improvements
            * Ongoing staff training as needed
            """)
            
        with st.expander("Can this system be integrated with existing facility management software?"):
            st.write(""" Yes, our system can be integrated with most existing facility management software, allowing for seamless data exchange and optimization.""")
    
if __name__ == "__main__":
    app = WasteVisionApp()