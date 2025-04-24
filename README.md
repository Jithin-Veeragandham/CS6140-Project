# CS6140 Machine Learning Project - Microsoft Malware Prediction

> Predicting malware occurrences on machines using real-world telemetry data provided by Microsoft.

##  Contributors:

- Jithin Veeragandham  
- Rishi Srikaanth  
- Andrei Biswas  

---

## 📌 Problem Statement

With over a billion enterprise and consumer customers, Microsoft takes cybersecurity very seriously. As part of their initiative to improve global protection standards, Microsoft released a large-scale telemetry dataset, challenging the data science community to predict whether a machine will be affected by malware in the near future.

The goal of this project is to develop machine learning models that can effectively identify potential malware infections based on rich, anonymized telemetry data. The dataset reflects real-world challenges such as missing values, mixed data types, and distribution shifts between training and test sets.

This project is based on the official Kaggle competition:

🔗 [Microsoft Malware Prediction – Kaggle Competition](https://www.kaggle.com/competitions/microsoft-malware-prediction/overview)

---

## 📁 Project Structure & Components

Due to the large size of the Microsoft Malware Prediction dataset, we did not upload any raw or processed data files to this repository. Instead, all data processing was performed locally. To streamline the training pipeline, we saved processed versions of the data and preprocessing artifacts such as encoders and scalers.

### 🧪 Data Encoding

We employed two types of encodings for training our models:

1. **Initial Encoding**
2. **Advanced Encoding**

After feature engineering and scaling, we saved the relevant encoders and scalers as `.pkl` files:

🔗 [Encoder Pickle Files (Initial & Advanced)](https://github.com/Jithin-Veeragandham/CS6140-Project/tree/main/Encoder%20pickle%20Files)

This directory contains:
- Count Encoders  
- Label Encoders  
- Standard Scalers  

for both the initial and advanced encodings. These were reused across all model training pipelines.

---

### 🧠 Model Training

The [`Training`](https://github.com/Jithin-Veeragandham/CS6140-Project/tree/main/Training) folder contains Jupyter notebooks used to train:
- Neural networks  
- Simple feedforward models  
- Tree-based models (LightGBM, XGBoost)

We trained each model on both encoding schemes. The trained weights for all models are available here:

🔗 [Saved Models](https://github.com/Jithin-Veeragandham/CS6140-Project/tree/main/Saved%20Models)

To evaluate any model, you can download the corresponding weights and run them on the pre-encoded test data using the matching encoders.

---

### ⚙️ Encoding Scripts

The [`Encode Datasets`](https://github.com/Jithin-Veeragandham/CS6140-Project/tree/main/Encode%20Datasets) folder contains the scripts used to generate the encoded datasets and save the encoders/scalers used across training.

---

### 🔍 Data Exploration & Initial Analysis

The [`Data Exploration and Initial Approaches`](https://github.com/Jithin-Veeragandham/CS6140-Project/tree/main/Data%20Exploration%20and%20Initial%20Approaches) folder contains:
- Notebooks identifying **distribution shifts** between train/test sets  
- Initial exploratory analyses and brainstorming (Exploration1, Exploration2)

These helped us understand the structure and challenges of the dataset before finalizing our modeling approach.

---

### 📤 Submissions

The [`Submissions`](https://github.com/Jithin-Veeragandham/CS6140-Project/tree/main/Submissions) folder includes all scripts used to generate `.csv` submission files for Kaggle.

These scripts read predictions from the trained models, decode any label encodings (if necessary), and format the outputs according to the Kaggle submission requirements.


