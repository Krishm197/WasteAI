import streamlit as st
from PIL import Image
import torch
import time
import tempfile
import cv2
import numpy as np
from pathlib import Path
import plotly.graph_objects as go
import pandas as pd
import traceback  # To capture detailed error info

# Import our classifier
from waste_classifier import DualWasteClassifier

# Static content moved to separate functions
def load_custom_css():
    """Load custom CSS for the app."""
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

class WasteVisionApp:
    def __init__(self):
        self.classifier = DualWasteClassifier()
        self.setup_page()
        self.main()
    
    def setup_page(self):
        """Initial page configuration."""
        st.set_page_config(page_title="Waste Vision", layout="wide")
        load_custom_css()
    
    def main(self):
        """Main function to handle app logic and navigation."""
        st.title("🌍 Waste Vision: Revolutionizing Waste Management")
        
        # Use tabs instead of sidebar for navigation
        tab1, tab2 = st.tabs(["Live Demo", "Impact and Vision"])
        
        with tab1:
            self.show_live_demo()
        with tab2:
            self.show_future_vision()

    def upload_and_display_image(self):
        """Function to handle image uploading and displaying."""
        uploaded_file = st.file_uploader("Choose an image...", type=["jpg", "jpeg", "png"])
        if uploaded_file:
            image = Image.open(uploaded_file)
            st.image(image, caption="Uploaded Image", use_column_width=True)
            return uploaded_file
        return None

    @st.cache_resource
    def classify_image(self, image_path):
        """Cache the classification process for efficiency."""
        return self.classifier.classify_image(image_path)

    def classify_image_safely(self, image_path):
        """Handle image classification with error catching and detailed logging."""
        try:
            # Log the image path to make sure it's being passed correctly
            st.write(f"Classifying image from path: {image_path}")
            
            # Perform classification with the image path
            result = self.classify_image(image_path)
            
            # Check if result is None (if classifier returns None on failure)
            if result is None:
                st.error("Classifier returned no result. Please check the input.")
                return None
            
            return result
        except Exception as e:
            # Log the full traceback to help with debugging
            st.error(f"Error during classification: {str(e)}")
            st.error(traceback.format_exc())  # Print full error details for debugging
            return None
    
    def show_live_demo(self):
        """Display the live demo section."""
        st.header("🎯 Live Waste Classification Demo")
        col1, col2 = st.columns(2)

        with col1:
            st.subheader("Upload Image")
            uploaded_file = self.upload_and_display_image()
            
            if uploaded_file and st.button("Classify Waste"):
                with st.spinner("Analyzing..."):
                    # Save the uploaded image to a temporary file
                    try:
                        with tempfile.NamedTemporaryFile(delete=False, suffix=".png") as temp_file:
                            # Ensure the image is saved in the correct format
                            image = Image.open(uploaded_file)
                            image.save(temp_file.name)
                            temp_file_path = temp_file.name
                        
                        # Pass the temporary file path to the classifier
                        result = self.classify_image_safely(temp_file_path)
                        
                        if result:
                            st.markdown(f"<div class='success-box'>{result}</div>", unsafe_allow_html=True)
                        else:
                            st.error("Classification failed. Please try again.")
                    
                    except Exception as e:
                        # Log any issues with saving or processing the image
                        st.error(f"Error processing the image: {str(e)}")
                        st.error(traceback.format_exc())

        with col2:
            st.subheader("How it Works")
            st.write("""
            1. **Multi-Model Approach**: Combines Microsoft's ResNet-50, OpenAI's CLIP and a fine-tuned Waste CNN model
            2. **High Accuracy**: Leverages the strengths of all models
            3. **Real-time Processing**: Suitable for industrial applications and creating an environmental impact
            """)

    def display_metrics(self, metrics_dict):
        """Display metrics in columns for a clean layout."""
        cols = st.columns(len(metrics_dict))
        for i, (label, value) in enumerate(metrics_dict.items()):
            cols[i].metric(label, value)

    def show_future_vision(self):
        """Display the future vision section."""
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

        st.header("📊 Environmental Impact")
        # Create metrics
        self.display_metrics({
            "Sorting Accuracy": "94%",
            "Processing Speed": "1.2 tons/hour",
            "CO2 Reduction": "500kg/day"
        })

        # Projected impact over time graph
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

        st.header("📮 Contact")
        st.write("Feel free to contact me for any suggestions or just to connect!")
        st.write("📧 krish.ubc.j@gmail.com")
        st.markdown("[LinkedIn](https://www.linkedin.com/in/krish-mehta-172559202/)")

if __name__ == "__main__":
    app = WasteVisionApp()
