RAIDEN ⚡ – A Hybrid AI-Powered Framework for Ransomware Detection and Mitigation

Real-time threat defense powered by Machine Learning, PE analysis, and image-based classification.**

🔍 Overview

RAIDEN is a hybrid intelligent system built for detecting and mitigating ransomware threats. It combines static PE file analysis with CNN-based image classification to identify suspicious binaries before they can harm a system. Designed using Python, Streamlit, and multiple ML techniques, this framework is optimized for both research and practical deployment.


💡 Features

- 🧠 Static Analysis** of PE headers for malware signatures  
- 🖼️ CNN-based Image Classification** for behavioral detection  
- ⚡ Real-time detection** in under 0.4s  
- 📁 File Quarantine** & alert mechanism  
- 📊 Streamlit dashboard** for user-friendly interaction  
- 🔐 Designed for **ransomware-specific detection**


🛠️ Tech Stack

- `Python 3.9.13`  
- `Streamlit`  
- `Scikit-learn`, `Keras`, `TensorFlow`  
- `PEfile`, `PIL`, `NumPy`, `Matplotlib`  
- `.pkl` models and `.keras` CNN models

📁 Project Structure
RAIDEN/
├── app.py                      # Streamlit dashboard entry point
├── Extract/                   # PE Header extraction scripts
├── Classifier/                # Traditional ML models (.pkl)
├── ML_Model/                  # CNN Keras models
├── malware_wrapper.py         # Ransomware mitigation logic
├── pe_wrapper.py              # PE feature analyzer
├── predict_pipeline.py        # Inference engine
├── requirements.txt           # Python dependencies
└── README.md

---

⚙️ Setup Instructions

1. Clone the repository
```bash
git clone https://github.com/saiv4/RAIDEN-Ransomware-Detection.git
cd RAIDEN-Ransomware-Detection

--Create virtual environment and activate

python -m venv venv
source venv/bin/activate      # MacOS/Linux
venv\Scripts\activate         # Windows
pip install -r requirements.txt



run the application

streamlit run app.py
