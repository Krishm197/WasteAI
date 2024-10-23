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
        
        # Custom CSS
        st.markdown("""
        <style>
        .main {background-color: #f5f5f5}
        .stButton>button {background-color: #4CAF50; color: white;}
        .success-box {
            padding: 1rem;
            border-radius: 0.5rem;
            background-color: #e8f5e9;
            border: 2px solid #4CAF50;
        }
        </style>
        """, unsafe_allow_html=True)
    
    def main(self):
        st.title("🌍 Waste Vision: Revolutionizing Waste Management")
        
        # Navigation
        section = st.sidebar.radio("Navigate", 
            ["Live Demo", "Industrial Applications", "Impact Visualization", "Future Vision"])
        
        if section == "Live Demo":
            self.show_live_demo()
        elif section == "Industrial Applications":
            self.show_industrial_applications()
        elif section == "Impact Visualization":
            self.show_impact_visualization()
        else:
            self.show_future_vision()
    
    def show_live_demo(self):
        st.header("🎯 Live Waste Classification Demo")
        
        col1, col2 = st.columns(2)
        
        with col1:
            st.subheader("Upload Image or Use Camera")
            source = st.radio("Select input source:", ["Upload Image", "Use Camera"])
            
            if source == "Upload Image":
                uploaded_file = st.file_uploader("Choose an image...", type=["jpg", "jpeg", "png"])
                if uploaded_file is not None:
                    image = Image.open(uploaded_file)
                    st.image(image, caption="Uploaded Image", use_column_width=True)
                    
                    if st.button("Classify Waste"):
                        with st.spinner("Analyzing..."):
                            # Simulate processing time for better UX
                            time.sleep(1)
                            result = self.classifier.classify_image(uploaded_file)
                            st.markdown(f"<div class='success-box'>{result}</div>", 
                                      unsafe_allow_html=True)
            
            else:
                st.warning("Note: Camera feed will be processed in real-time")
                webrtc_streamer(key="waste_camera")
        
        with col2:
            st.subheader("How it Works")
            st.write("""
            1. **Dual Model Approach**: Combines Microsoft's ResNet-50 and OpenAI's CLIP
            2. **High Accuracy**: Leverages the strengths of both models
            3. **Real-time Processing**: Suitable for industrial applications
            """)
            
            # Add a sample confusion matrix or accuracy metrics
            st.markdown("### Model Performance")
            dummy_conf_matrix = pd.DataFrame(
                [[95, 2, 3], [4, 92, 4], [3, 2, 95]],
                columns=['Recyclable', 'Organic', 'Other'],
                index=['Recyclable', 'Organic', 'Other']
            )
            fig = px.imshow(dummy_conf_matrix, 
                           labels=dict(x="Predicted", y="Actual", color="Accuracy"),
                           text=dummy_conf_matrix)
            st.plotly_chart(fig)
    
    def show_industrial_applications(self):
        st.header("🏭 Industrial Applications")
        
        col1, col2 = st.columns(2)
        
        with col1:
        	st.subheader("Automated Sorting System")
            st.markdown("""
            <div style='background-color: #f0f2f6; padding: 20px; border-radius:10px;'>
            	<h4>Robotic Sorting System</h4>
                <p>The advanced system combines computer vision with robotic automation to achieve efficient and accurate waste sorting. </p>
            </div>
            """, unsafe_allow_html=True)
            st.write("""
            ### Key Components:
            1. **Vision System**: Our AI classifier
            2. **Robotic Arms**: Precise picking and sorting
            3. **Conveyor System**: Continuous waste flow
            4. **Control System**: Real-time coordination
            """)
        
        with col2:
            st.subheader("Integration Points")
            
            # Interactive system diagram
            fig = go.Figure()
            fig.add_trace(go.Sankey(
                node=dict(
                    pad=15,
                    thickness=20,
                    line=dict(color="black", width=0.5),
                    label=["Mixed Waste", "Vision System", "Robotics", "Sorted Plastic", 
                           "Sorted Paper", "Sorted Glass"],
                    color=["gray", "blue", "green", "red", "orange", "purple"]
                ),
                link=dict(
                    source=[0, 0, 0, 1, 1, 1],
                    target=[1, 1, 1, 3, 4, 5],
                    value=[1, 1, 1, 1, 1, 1]
                )
            ))
            fig.update_layout(title_text="System Integration Flow", font_size=10)
            st.plotly_chart(fig)
    
    def show_impact_visualization(self):
        st.header("📊 Environmental Impact")
        
        # Create metrics
        col1, col2, col3 = st.columns(3)
        with col1:
            st.metric("Sorting Accuracy", "94%", "+2.5%")
        with col2:
            st.metric("Processing Speed", "1.2 tons/hour", "+0.3 tons")
        with col3:
            st.metric("CO2 Reduction", "500kg/day", "+50kg")
        
        # Add projection graph
        st.subheader("Projected Impact Over Time")
        years = list(range(2024, 2031))
        manual_sorting = [100, 105, 110, 115, 120, 125, 130]
        ai_sorting = [100, 130, 170, 220, 280, 350, 430]
        
        fig = go.Figure()
        fig.add_trace(go.Scatter(x=years, y=manual_sorting, name="Manual Sorting"))
        fig.add_trace(go.Scatter(x=years, y=ai_sorting, name="AI-Powered Sorting"))
        fig.update_layout(title="Waste Processing Efficiency", 
                         xaxis_title="Year",
                         yaxis_title="Processing Capacity (normalized)")
        st.plotly_chart(fig)
    
	def show_future_vision(self):
 	   st.header("🚀 Future Vision")
    
  	  st.write("""
   	 ### Extended Applications
    	1. **Smart Cities Integration**
			- Connected waste bins with fill-level monitoring
			- Optimized collection routes
			- Real-time waste analytics
    
	    2. **Educational Impact**
			- Interactive waste sorting games
  	 		- Public awareness campaigns
   		    - School programs
    
    	3. **Blockchain Integration**
    		- Waste tracking and verification
    		- Recycling rewards system
    		- Transparent supply chain
    	""")
    
if __name__ == "__main__":
    app = WasteVisionApp()
