 # Satellite Land Risk Detection using Deep Learning

This project uses **satellite images** to **identify land type** (forest, residential, river, etc.) and **estimate risk level** using a **Deep Learning (CNN) model** trained on the **EuroSAT dataset**.

The system also shows **confidence score** and adjusts **risk level dynamically** based on prediction confidence.

---

## Why this project?

Satellite images are widely available, but **manual land monitoring is slow and expensive**.

This project helps in:
- Environmental monitoring
- Urban planning
- Disaster & land risk assessment
- Biodiversity & sustainability analysis

---

## Novelty 

✔ Uses **Confidence Score** to adjust risk  
✔ Converts **land classification → risk level**  
✔ Works on **real satellite images**  
✔ Deployable as a **Streamlit web app**  
✔ Fully automated end-to-end pipeline  

Unlike normal classifiers, this project **does not blindly trust predictions** —  
Low confidence = higher caution (risk ↑).

---

## How does the project work? 

1. User uploads a satellite image
2. Image is resized & normalized
3. CNN model predicts land category
4. Confidence score is calculated
5. Base risk is mapped from land type
6. Risk is adjusted using confidence
7. Final land type + confidence + risk is shown

---

## Dataset Used

**EuroSAT Dataset (Sentinel-2 satellite images)**  
Contains 10 land classes:

- AnnualCrop
- Forest
- HerbaceousVegetation
- Highway
- Industrial
- Pasture
- PermanentCrop
- Residential
- River
- SeaLake

### Dataset download (Kaggle)
```python
import kagglehub

path = kagglehub.dataset_download("apollo2506/eurosat-dataset")
print("Path to dataset files:", path)
```
## Structure
satellite-land-risk-detection/

│

├── app.py                           # Streamlit web app

├── train_model.py                   # Model training script

├── test_image.py                    # Image testing script

├── biodiversity_model.h5            # Trained CNN model

├── requirements.txt                 # Required libraries

├── test_images/                     # Sample test images

├── README.md                        # Project documentation

├── archieve/                       # Database container(EuroSAT)

## Technologies Used
- Python
- TensorFlow/Keras
- Numpy
- Streamlit
- Matplotlib
- KaggleHub
- Git & GitHub

### Installation & Setup
1. Clone the repository
   ```
   git clone https://github.com/codebeast367/satellite-land-risk-detection.git
   cd satellite-land-risk-detection
   ```
2. Create virtual environment
   ```
   python -m venv venv
   venv\Scripts\activate
   ```
3. Install required packages
   ```
   pip install -r requirements.txt
   ```
### Model Training
Run this to train the CNN model on EuroSAT dataset:
```
python train_model.py
```
This will generate:
```
biodiversity_model.h5
```
### Image Testing
To test a single satellite image:
```
python test_image.py
```
### Confidence Adjustment Logic
```
def adjust_risk(base_risk, confidence):
    if confidence < 60:
        if base_risk == "LOW":
            return "MEDIUM"
        if base_risk == "MEDIUM":
            return "HIGH"
    return base_risk
```
### Streamlit Web App
Run the web application:
```
streamlit run app.py
```






