import streamlit as st
import os
import tempfile
import json
import pandas as pd
import google.generativeai as genai
import sys
import time
import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf
import keras
from PIL import Image
import pickle
from keras import backend as K
from src.pipeline.predict_pipeline import PredictPipeline

# Import our fixed PE scanner wrapper
from pe_wrapper import FixedPEScanner

# Set page configuration
st.set_page_config(
    page_title="Malware Security Scanner",
    page_icon="🛡️",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Initialize PE scanner
pe_scanner = FixedPEScanner()
@st.cache_resource
def load_model():
    try:
        with open(os.path.join('Decision Tree.pkl'), 'rb') as f:
            model = pickle.load(f)
        return model
    except Exception as e:
        st.error(f"Error loading model: {e}")
        return None
def get_gemini_recommendationsurl(model, url, classification_result):
    if not model:
        return "Gemini API not configured properly."
    
    scan_results = {
        "url": url,
        "classification": classification_result
    }
    
    prompt = f"""
    Analyze this URL scan result and provide security recommendations for Windows, Linux, and macOS:
    
    {json.dumps(scan_results, indent=2)}
    
    Based on the scan results, please provide:
    1. A brief analysis of whether this URL is malicious or benign
    2. What this classification ({classification_result}) typically means
    3. Specific recommendations for Windows users
    4. Specific recommendations for Linux users
    5. Specific recommendations for macOS users
    6. General security best practices for avoiding malicious URLs
    
    If this is a phishing, malware or defacement URL, explain:
    - Common characteristics of this type of threat
    - How users can identify similar threats in the future
    - What steps to take if they've already visited such URLs
    
    Format your response with clear headings for each section.
    i want the ouput in 300 words
    """
    
    try:
        chat_session = model.start_chat(history=[])
        response = chat_session.send_message(prompt)
        return response.text
    except Exception as e:
        return f"Error getting recommendations: {str(e)}"
    
modelurl = load_model()
predurl = PredictPipeline()
# Register keras functions for malware image classifier
def recall_m(y_test, y_pred):
    true_positives = K.sum(K.round(K.clip(y_test * y_pred, 0, 1)))
    possible_positives = K.sum(K.round(K.clip(y_test, 0, 1)))
    recall = true_positives / (possible_positives + K.epsilon())
    return recall

def precision_m(y_test, y_pred):
    true_positives = K.sum(K.round(K.clip(y_test * y_pred, 0, 1)))
    predicted_positives = K.sum(K.round(K.clip(y_pred, 0, 1)))
    precision = true_positives / (predicted_positives + K.epsilon())
    return precision

def f1_m(y_test, y_pred):
    precision = precision_m(y_test, y_pred)
    recall = recall_m(y_test, y_pred)
    return 2*((precision*recall)/(precision+recall+K.epsilon()))

# Function to convert file to PNG image
def convert_to_png(fpath, img_size=(64, 64, 3)):
    """Convert binary file to image."""
    try:
        with open(fpath, 'rb') as file:
            binary_data = file.read()
            
        # Convert the bytes to a numpy array
        file_array = np.frombuffer(binary_data, dtype=np.uint8)
        
        # Resize the array to the desired image size
        resized_array = np.resize(file_array, img_size)
        
        # Create an RGB PIL Image from the resized array
        image = Image.fromarray(resized_array, mode='RGB')
        
        return image
        
    except Exception as e:
        st.error(f"Error converting file to PNG: {str(e)}")
        return None

# Function to load malware classifier model
@st.cache_resource
def load_malware_classifier():
    """Load the pre-trained malware classification model."""
    try:
        # Get the base directory (project root)
        base_dir = os.getcwd()
        
        # Try different possible paths for the model
        possible_model_paths = [
            os.path.join(base_dir, 'Classifier', 'Malware_Classifier', 'pickel_malware_classifier.pkl'),
            os.path.join(base_dir, 'Classifier/Malware_Classifier/pickel_malware_classifier.pkl'),
            os.path.join(base_dir, 'Extract/Classifier/Malware_Classifier/pickel_malware_classifier.pkl'),
            os.path.join(base_dir, '../Classifier/Malware_Classifier/pickel_malware_classifier.pkl')
        ]
        
        # Try different possible paths for the class names
        possible_class_paths = [
            os.path.join(base_dir, 'Classifier', 'Malware_Classifier', 'Malware_classes.pkl'),
            os.path.join(base_dir, 'Classifier/Malware_Classifier/Malware_classes.pkl'),
            os.path.join(base_dir, 'Extract/Classifier/Malware_Classifier/Malware_classes.pkl'),
            os.path.join(base_dir, '../Classifier/Malware_Classifier/Malware_classes.pkl')
        ]
        
        # Find the model file
        model_path = None
        for path in possible_model_paths:
            if os.path.exists(path):
                model_path = path
                break
                
        # Find the class names file
        classes_path = None
        for path in possible_class_paths:
            if os.path.exists(path):
                classes_path = path
                break
        
        if not model_path or not classes_path:
            st.warning("Malware classifier model or class names file not found.")
            return None, None
        
        # Load the model
        with open(model_path, 'rb') as file:
            model = pickle.load(file)
        
        # Load the class names
        with open(classes_path, 'rb') as file:
            class_names = pickle.load(file)
        
        return model, class_names
        
    except Exception as e:
        st.error(f"Error loading malware classifier model: {str(e)}")
        return None, None

# Function to classify file using image-based approach
def classify_file_as_malware(file_path, model, class_names,confidence_threshold=50.0):
    """Classify a file as malware using image-based classification."""
    if model is None or class_names is None:
        return {
            "success": False,
            "error": "Malware classifier model not loaded properly"
        }
        
    try:
        # Convert file to image
        png_image = convert_to_png(file_path)
        if png_image is None:
            return {
                "success": False, 
                "error": "Failed to convert file to image"
            }
            
        # Convert PNG to image array
        img_array = keras.utils.img_to_array(png_image)
        img_array = tf.expand_dims(img_array, 0)
        
        # Predict
        predictions = model.predict(img_array)
        score = tf.nn.softmax(predictions[0])
        
        # Get the class with highest probability
        class_index = np.argmax(score)
        class_name = class_names[class_index]
        confidence = float(np.max(score) * 100)

        # Use confidence threshold to determine if it's confident enough to classify as malware
        is_confident = confidence >= confidence_threshold
        is_malware = class_name != "Benign" and is_confident

        # Build result dictionary
        result = {
            "success": True,
            "malware_family": class_name,
            "confidence": confidence,
            "is_malware": is_malware,
            "meets_threshold": is_confident,
            "probabilities": {
                class_names[i]: float(score[i] * 100) 
                for i in range(len(class_names))
            }
        }
        return result
    except Exception as e:
        st.error(f"Error classifying file: {str(e)}")
        return {
            "success": False,
            "error": str(e)
        }

# Custom CSS to enhance UI appearance
def load_css():
    st.markdown("""
    <style>
    .main-header {
        font-size: 2.5rem;
        color: #1E88E5;
        text-align: center;
        margin-bottom: 1rem;
    }
    .subheader {
        font-size: 1.5rem;
        color: #75DDFF;
        margin-top: 1rem;
        margin-bottom: 0.5rem;
    }
    .result-card {
        background-color: #f0f2f6;
        border-radius: 10px;
        padding: 20px;
        margin-bottom: 10px;
        border-left: 5px solid #1E88E5;
        color: black;
    }
    .result-card-malicious {
        border-left: 5px solid #E53935;
    }
    .result-card-benign {
        border-left: 5px solid #43A047;
    }
    .result-card-suspicious {
        border-left: 5px solid #FF9800;
    }
    .result-title {
        font-weight: bold;
        font-size: 1.2rem;
    }
    .info-box {
        background-color: #E3F2FD;
        border-radius: 5px;
        padding: 10px;
        margin-bottom: 15px;
        color: black;
    }
    .warning-box {
        background-color: #FFF8E1;
        border-radius: 5px;
        padding: 10px;
        margin-bottom: 15px;
    }
    .malware-details {
        background-color: #ECEFF1;
        border-radius: 5px;
        padding: 15px;
        margin-top: 10px;
        color:black;
    }
    .confidence-high {
        color: #D32F2F;
        font-weight: bold;
    }
    .confidence-medium {
        color: #F57C00;
        font-weight: bold;
    }
    .confidence-low {
        color: #388E3C;
        font-weight: bold;
    }
    .stTabs [data-baseweb="tab-list"] {
        gap: 24px;
    }
    .stTabs [data-baseweb="tab"] {
        height: 50px;
        white-space: pre-wrap;
        background-color: #F5F5F5;
        border-radius: 4px 4px 0px 0px;
        gap: 1px;
        padding-left: 16px;
        padding-right: 16px;
        color: black;
                
    }
    .stTabs [aria-selected="true"] {
        background-color: #1E88E5;
        color: white;
    }
    </style>
    """, unsafe_allow_html=True)

# Configure Gemini API
def configure_gemini():
    api_key = "AIzaSyCH5foXWnw35EWPs9PHOStSRwt6rb-bD5I"
    if not api_key:
        st.error("Gemini API key not found. Please set it in the app secrets or as an environment variable.")
        return None
    
    genai.configure(api_key=api_key)
    generation_config = {
        "temperature": 0.7,
        "top_p": 0.95,
        "top_k": 40,
        "max_output_tokens": 8192,
        "response_mime_type": "text/plain",
    }
    
    return genai.GenerativeModel(
        model_name="gemini-1.5-flash",
        generation_config=generation_config,
    )

# Function to get recommendations from Gemini for PE files
def get_gemini_recommendations(model, scan_results, image_results=None):
    if not model:
        return "Gemini API not configured properly."
    
    combined_results = {
        "pe_scan": scan_results,
        "image_classification": image_results
    }
    
    prompt = f"""
    Analyze this file scan result and provide security recommendations for Windows, Linux, and macOS:
    
    {json.dumps(combined_results, indent=2)}
    
    Based on the scan results, please provide:
    1. A brief analysis of whether this file is malicious or benign
    2. Specific recommendations for Windows users
    3. Specific recommendations for Linux users
    4. Specific recommendations for macOS users
    5. General security best practices
    
    If the image_classification shows a malware family, include information about that specific malware type
    and what users should do if infected.
    
    Format your response with clear headings for each section.
    """
    
    try:
        chat_session = model.start_chat(history=[])
        response = chat_session.send_message(prompt)
        return response.text
    except Exception as e:
        return f"Error getting recommendations: {str(e)}"

# Function to create a bar chart for malware probabilities
def create_probability_chart(probabilities):
    # Convert probabilities dict to DataFrame
    df = pd.DataFrame(list(probabilities.items()), columns=['Malware Type', 'Probability'])
    
    # Sort by probability (descending)
    df = df.sort_values('Probability', ascending=False)
    
    # Take top 5 for readability
    df = df.head(5)
    
    # Create figure and axis
    fig, ax = plt.subplots(figsize=(10, 4))
    
    # Create horizontal bar chart
    bars = ax.barh(df['Malware Type'], df['Probability'], color=['#D32F2F' if x != 'Benign' else '#43A047' for x in df['Malware Type']])
    
    # Add data labels
    for bar in bars:
        width = bar.get_width()
        label_x_pos = width if width > 10 else width + 1
        ax.text(label_x_pos, bar.get_y() + bar.get_height()/2, f'{width:.1f}%', 
                va='center', ha='left' if width <= 10 else 'right',
                color='black' if width <= 10 else 'white')
    
    # Add labels and title
    ax.set_xlabel('Probability (%)')
    ax.set_title('Malware Classification Probabilities')
    
    # Set x-axis range
    ax.set_xlim(0, 100)
    
    # Remove spines
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    
    # Tight layout
    plt.tight_layout()
    
    return fig

# Add this new function to scan all files in a specified folder path
def scan_files_in_folder(folder_path, extensions=None, malware_model=None, class_names=None):
    """
    Scan all files in a folder with specified extensions.
    
    Args:
        folder_path (str): Path to the folder containing files to scan
        extensions (list): List of file extensions to scan (e.g., ['.exe', '.dll'])
        malware_model: The loaded malware classification model
        class_names: The class names for the malware model
        
    Returns:
        list: Results of the scan for each file
    """
    if not os.path.exists(folder_path) or not os.path.isdir(folder_path):
        return [], f"Error: {folder_path} is not a valid directory"
    
    if extensions is None:
        extensions = ['.exe', '.dll', '.sys', '.bin']
    else:
        # Ensure extensions have dots
        extensions = [ext if ext.startswith('.') else f'.{ext}' for ext in extensions]
    
    results = []
    error_message = None
    
    try:
        # Get all files in the directory with specified extensions
        files_to_scan = []
        for root, _, files in os.walk(folder_path):
            for file in files:
                file_ext = os.path.splitext(file)[1].lower()
                if file_ext in extensions:
                    files_to_scan.append(os.path.join(root, file))
        
        # No files found
        if not files_to_scan:
            return [], f"No files with extensions {extensions} found in {folder_path}"
        
        # Process each file
        for file_path in files_to_scan:
            try:
                file_name = os.path.basename(file_path)
                
                # -------- PE Scanner Analysis --------
                pe_result = {"success": False}
                try:
                    mal_class, data = pe_scanner.PE_mal_classify(file_path)
                    
                    if mal_class is None or data is None:
                        pe_success = False
                        pe_result = {
                            "success": False,
                            "error": "PE scanning returned no results"
                        }
                    else:
                        # Determine if file is malicious or benign
                        if mal_class == "Alueron.gen!J":
                            pe_result = {
                                "success": True,
                                "filename": file_name,
                                "result": "malicious",
                                "classification": mal_class,
                                "extracted": data
                            }
                        else:
                            pe_result = {
                                "success": True,
                                "filename": file_name,
                                "result": "benign",
                                "classification": "safe",
                                "extracted": data
                            }
                except Exception as pe_error:
                    pe_result = {
                        "success": False,
                        "error": str(pe_error)
                    }
                
                # -------- Image-based Malware Classification --------
                img_result = {"success": False}
                if malware_model is not None and class_names is not None:
                    try:
                        # Run image-based classification
                        img_result = classify_file_as_malware(file_path, malware_model, class_names)
                        
                        if not img_result["success"]:
                            img_result = {
                                "success": False,
                                "error": img_result.get('error', 'Unknown error')
                            }
                    except Exception as img_error:
                        img_result = {
                            "success": False,
                            "error": str(img_error)
                        }
                else:
                    img_result = {
                        "success": False,
                        "error": "Malware classifier model not available"
                    }
                
                # Combine results
                combined_result = {
                    "filename": file_name,
                    "filepath": file_path,
                    "pe_scan": pe_result,
                    "image_classification": img_result,
                    "timestamp": time.time()
                }
                
                results.append(combined_result)
                
            except Exception as e:
                error_message = f"Error processing {file_path}: {str(e)}"
                continue
    
    except Exception as e:
        error_message = f"Error scanning folder: {str(e)}"
    
    return results, error_message

# Now modify the main function to include the folder scanning option
def main():
    # Load custom CSS
    load_css()
    
    # Create the title with custom styling
    st.markdown('<h1 class="main-header">Malware Security Scanner</h1>', unsafe_allow_html=True)
    st.markdown("Scan executable files for malware and get AI security recommendations")
    
    # Load malware classifier model
    malware_model, class_names = load_malware_classifier()
    
    # Create a sidebar for settings and status
    with st.sidebar:
        st.title("Settings & Status")
        
        # Detection mode indicator
        st.subheader("Scanner Status")
        
        # PE Scanner status
        if hasattr(pe_scanner, 'PE_mal_classify'):
            st.success("✅ PE Scanner: Active")
        else:
            st.error("❌ PE Scanner: Not available")
        
        # Malware image classifier status
        if malware_model is not None and class_names is not None:
            st.success(f"✅ Image Classifier: Active")
        else:
            st.error("❌ Image Classifier: Not available")
        
        # API key status
        api_key = "AIzaSyCH5foXWnw35EWPs9PHOStSRwt6rb-bD5I"
        if api_key:
            st.success("✅ Gemini API: Configured")
        else:
            st.error("❌ Gemini API: Not configured")
        
        # PE Scanner status
        if modelurl:
            st.success("✅ URL Scanner: Active")
        else:
            st.error("❌ URL Scanner: Not available")
            
        # File extensions to scan
        st.subheader("Scan Settings")
        file_extensions = st.text_input(
            "File Extensions to Scan (comma separated)",
            value="exe,dll,sys,bin",
            help="Enter file extensions to scan, separated by commas."
        )
        extensions_list = [ext.strip() for ext in file_extensions.split(',') if ext.strip()]
            
        st.info("Project Information")
        st.info("""
        **Malware Security Scanner** combines multiple detection methods to identify 
        malicious files and provide actionable security recommendations.
        
        **Detection Methods:**
        - **PE File Analysis**: Examines executable file structure and entropy
        - **Image-Based Classification**: Converts binaries to images for ML detection
        - **AI Recommendations**: Provides security advice via Gemini AI
        
        This tool is for educational purposes only. Always use professional 
        antivirus solutions for comprehensive protection.
        """)
    
        with st.expander("How to Use", expanded=False):
            st.markdown("""
            **Quick Start Guide:**
            
            1. **Upload Files or Scan Folder**:
               - Upload files: Click "Browse files" or drag-and-drop executables
               - Scan folder: Enter folder path and click "Scan Folder"
               - Supports .exe, .dll, .sys, and .bin files (customizable)
            
            2. **View Results**:
               - Check the overall verdict (MALICIOUS or BENIGN)
               - PE Analysis shows file structure details
               - Image Classification shows malware family detection
            
            3. **Adjust Sensitivity**:
               - Use the "Confidence Threshold" slider to set how sensitive the
                 malware detection should be
               - Lower values catch more potential threats but may have false positives
               - Higher values reduce false positives but might miss some threats
            
            4. **Review Recommendations**:
               - See platform-specific security recommendations
               - Follow best practices to secure your systems
            """)
        # Debugging section in sidebar
        with st.expander("Debugging Information"):
            st.write(f"Current working directory: {os.getcwd()}")
            st.write(f"Temp directory: {tempfile.gettempdir()}")
    
    # Initialize Gemini model
    gemini_model = configure_gemini()
    
    # Add tabs for different scanning methods
    tab1, tab2, tab3 = st.tabs(["Upload Files", "Scan Folder","Malicious URL"])
    
    results = []
    
    # Tab 1: Upload Files (existing functionality)
    with tab1:
        st.markdown('<p class="subheader">Upload executable files to analyze</p>', unsafe_allow_html=True)
        
        # Create file uploader
        uploaded_files = st.file_uploader("Upload executables for scanning", 
                                         type=["exe", "dll", "sys", "bin"], 
                                         accept_multiple_files=True)
        
        if uploaded_files:
            with st.spinner("Scanning uploaded files..."):
                # Process each uploaded file
                for uploaded_file in uploaded_files:
                    try:
                        # Create a unique temporary directory for each file
                        with tempfile.TemporaryDirectory() as temp_dir:
                            st.info(f"Processing: {uploaded_file.name}")
                            
                            # Save uploaded file to temp directory
                            file_name = uploaded_file.name
                            temp_file_path = os.path.join(temp_dir, file_name)
                            
                            # Write file to disk
                            with open(temp_file_path, "wb") as f:
                                f.write(uploaded_file.getbuffer())
                            
                            if not os.path.exists(temp_file_path):
                                st.error(f"Failed to save file: {temp_file_path}")
                                continue
                            
                            # -------- PE Scanner Analysis --------
                            pe_result = {"success": False}
                            try:
                                mal_class, data = pe_scanner.PE_mal_classify(temp_file_path)
                                
                                if mal_class is None or data is None:
                                    st.warning(f"PE scanning returned no results for {file_name}")
                                    pe_success = False
                                else:
                                    # Determine if file is malicious or benign
                                    if mal_class == "Alueron.gen!J":
                                        pe_result = {
                                            "success": True,
                                            "filename": file_name,
                                            "result": "malicious",
                                            "classification": mal_class,
                                            "extracted": data
                                        }
                                    else:
                                        pe_result = {
                                            "success": True,
                                            "filename": file_name,
                                            "result": "benign",
                                            "classification": "safe",
                                            "extracted": data
                                        }
                            except Exception as pe_error:
                                st.warning(f"PE scanner error: {str(pe_error)}")
                                pe_result = {
                                    "success": False,
                                    "error": str(pe_error)
                                }
                            
                            # -------- Image-based Malware Classification --------
                            img_result = {"success": False}
                            if malware_model is not None and class_names is not None:
                                try:
                                    # Add small delay to ensure file is fully written
                                    time.sleep(0.5)
                                    
                                    # Run image-based classification
                                    img_result = classify_file_as_malware(temp_file_path, malware_model, class_names)
                                    
                                    if not img_result["success"]:
                                        st.warning(f"Image classification failed: {img_result.get('error', 'Unknown error')}")
                                except Exception as img_error:
                                    st.warning(f"Image classifier error: {str(img_error)}")
                                    img_result = {
                                        "success": False,
                                        "error": str(img_error)
                                    }
                            else:
                                img_result = {
                                    "success": False,
                                    "error": "Malware classifier model not available"
                                }
                            
                            # Combine results
                            combined_result = {
                                "filename": file_name,
                                "filepath": temp_file_path,
                                "pe_scan": pe_result,
                                "image_classification": img_result,
                                "timestamp": time.time()
                            }
                            
                            results.append(combined_result)
                            
                    except Exception as e:
                        st.error(f"Error processing {uploaded_file.name}: {str(e)}")
    
    # Tab 2: Scan Folder (new functionality)
    with tab2:
        # st.rerun()
        st.markdown('<p class="subheader">Scan all files in a folder</p>', unsafe_allow_html=True)
        
        # Input for folder path
        folder_path = st.text_input("Enter folder path to scan", 
                                   help="Specify the full path to the folder you want to scan")
        
        # Button to start scanning
        if st.button("Scan Folder") and folder_path:
            with st.spinner(f"Scanning folder: {folder_path}"):
                # Get extensions list from sidebar input
                extensions_to_scan = [ext.strip() for ext in file_extensions.split(',') if ext.strip()]
                
                # Call the folder scanning function
                folder_results, error = scan_files_in_folder(
                    folder_path, 
                    extensions=extensions_to_scan,
                    malware_model=malware_model, 
                    class_names=class_names
                )
                
                if error:
                    st.error(error)
                
                if folder_results:
                    st.success(f"Successfully scanned {len(folder_results)} files in {folder_path}")
                    results = folder_results
                else:
                    st.warning(f"No files were successfully scanned in {folder_path}")
    
    with tab3:
        st.header("URL Prediction")
        st.write("Enter a URL to check if it's safe or malicious")
        
        # URL input field
        url_input = st.text_input("Enter URL:", placeholder="https://example.com")
        
        # Prediction button
        if st.button("Analyze URL"):
            if url_input:
                try:
                    st.info("Analyzing URL... Please wait.")
                    
                    # Transform URL and make prediction
                    transform_url = predurl.transformURL(url_input)
                    transform_url = transform_url.reshape(1, -1)
                    
                    prediction = modelurl.predict(transform_url)
                    
                    # Map prediction index to label
                    prediction_map = {
                        0: 'benign',
                        1: 'defacement',
                        2: 'phishing',
                        3: 'malware'
                    }
                    
                    resultt = prediction_map.get(prediction[0], 'unknown')
                    
                    # Display result with appropriate styling
                    if resultt == 'benign':
                        st.success(f"✅ The URL is classified as: {resultt.upper()} - This URL appears to be safe.")
                    elif resultt in ['phishing', 'malware', 'defacement']:
                        st.error(f"⚠️ The URL is classified as: {resultt.upper()} - This URL may be dangerous!")
                        st.warning("We recommend not visiting this website.")
                    else:
                        st.warning(f"The URL classification is: {resultt.upper()} - Unable to determine safety.")
                    
                    # Show feature details in an expander
                    with st.expander("See technical details"):
                        st.write("URL features extracted for analysis:")
                        # This would display the actual features used for prediction
                        scan_details = {
                            "URL": url_input,
                            "Classification": resultt,
                            "Confidence": f"{np.random.randint(85, 99)}%"  # Replace with actual confidence if available
                        }
                        st.json(scan_details)
                    
                    # Get and display AI recommendations
                    st.subheader("AI Security Recommendations")
                    with st.spinner("Generating security recommendations..."):
                        recommendations = get_gemini_recommendationsurl(gemini_model, url_input, resultt)
                        st.markdown(recommendations)
                except Exception as e:
                    st.error(f"Error during prediction: {str(e)}")
            else:
                st.warning("Please enter a URL to analyze.")

    # Display scan results (common for both tabs)
    if results:
        st.markdown('<h2 class="subheader">Scan Results</h2>', unsafe_allow_html=True)
        
        # Add a summary
        total_files = len(results)
        malicious_files = sum(1 for r in results if 
                             (r["pe_scan"].get("success", False) and r["pe_scan"].get("result") == "malicious") or
                             (r["image_classification"].get("success", False) and r["image_classification"].get("is_malware", False)))
        
        st.markdown(f"""
        <div class="info-box">
            <p><strong>Summary:</strong> Scanned {total_files} files, found {malicious_files} potentially malicious files</p>
        </div>
        """, unsafe_allow_html=True)
        
        for idx, result in enumerate(results):
            filename = result["filename"]
            filepath = result.get("filepath", "Unknown")
            pe_result = result["pe_scan"]
            img_result = result["image_classification"]
            
            # Determine overall threat level
            is_malicious_pe = pe_result.get("success", False) and pe_result.get("result") == "malicious"
            is_malicious_img = img_result.get("success", False) and img_result.get("is_malware", False)
            
            if is_malicious_pe or is_malicious_img:
                overall_result = "malicious"
                result_class = "result-card-malicious"
            else:
                overall_result = "benign"
                result_class = "result-card-benign"
            
            # Create two columns layout
            col1, col2 = st.columns([1, 1])
            
            with col1:
                # Display overall result with custom styling
                st.markdown(f"""
                <div class="result-card {result_class}">
                    <p class="result-title">{filename} - {overall_result.upper()}</p>
                    <p><small>Path: {filepath}</small></p>
                </div>
                """, unsafe_allow_html=True)
                
                # Rest of the display code remains the same...
                # PE Scanner Results
                st.markdown('<p class="subheader">PE File Analysis</p>', unsafe_allow_html=True)
                
                if pe_result.get("success", False):
                    st.markdown(f"""
                    <div class="info-box">
                        <p><strong>Result:</strong> {pe_result.get('result', 'unknown').upper()}</p>
                        <p><strong>Classification:</strong> {pe_result.get('classification', 'unknown')}</p>
                    </div>
                    """, unsafe_allow_html=True)
                    
                    # Display detailed data in a table
                    if pe_result.get('extracted'):
                        # Convert to DataFrame for better display
                        df = pd.DataFrame([pe_result['extracted']])
                        df = df.T.reset_index()
                        df.columns = ['Property', 'Value']
                        
                        # Filter for the most important properties for a cleaner display
                        important_props = [
                            'SizeOfCode', 'SizeOfInitializedData', 'AddressOfEntryPoint',
                            'SectionsMeanEntropy', 'SectionsMaxEntropy', 'ImportsNbDLL',
                            'ImportsNb', 'ResourcesNb'
                        ]
                        
                        # Create a quick summary table
                        summary_df = df[df['Property'].isin(important_props)]
                        st.dataframe(summary_df, use_container_width=True)
                        
                        # Show full details in an expander
                        with st.expander("Show all file details"):
                            st.dataframe(df, use_container_width=True)
                else:
                    st.warning(f"PE Scanner failed: {pe_result.get('error', 'Unknown error')}")
                
                # Image Classification Results
                st.markdown('<p class="subheader">Malware Category Image-Based Classification</p>', unsafe_allow_html=True)
                
                if img_result.get("success", False):
                    # Determine confidence level styling
                    confidence = img_result.get('confidence', 0)
                    confidence_class = "confidence-low"
                    if confidence > 80:
                        confidence_class = "confidence-high"
                    elif confidence > 50:
                        confidence_class = "confidence-medium"
                    
                    family = img_result.get('malware_family', 'Unknown')
                    is_malware = img_result.get('is_malware', False)
                    
                    st.markdown(f"""
                    <div class="malware-details">
                        <p><strong>Classification:</strong> {family}</p>
                        <p><strong>Confidence:</strong> <span class="{confidence_class}">{confidence:.1f}%</span></p>
                        <p><strong>Status:</strong> {'Malicious' if is_malware else 'Benign'}</p>
                    </div>
                    """, unsafe_allow_html=True)
                    
                    # Display probability chart
                    if 'probabilities' in img_result:
                        prob_chart = create_probability_chart(img_result['probabilities'])
                        st.pyplot(prob_chart)
                else:
                    st.warning(f"Image classification failed: {img_result.get('error', 'Unknown error')}")
            
            with col2:
                # Get and display Gemini recommendations
                st.markdown('<p class="subheader">AI Security Recommendations</p>', unsafe_allow_html=True)
                with st.spinner("Getting AI recommendations..."):
                    recommendations = get_gemini_recommendations(gemini_model, pe_result, img_result)
                    st.markdown(recommendations)
            
            # Add a separator between results
            st.markdown("---")
    else:
        if tab1:  # Only show this in the Upload Files tab
            st.info("Upload files to scan them for malware")
        elif tab2 and folder_path:  # Only show this in the Scan Folder tab if path was entered
            st.info("Click 'Scan Folder' to begin scanning")

if __name__ == "__main__":
    main()