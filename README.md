# CS6140 Machine Learning Project - Microsoft Malware Prediction

> Predicting malware occurrences on machines using real-world telemetry data provided by Microsoft.

## 👥 Contributors

- Jithin Veeragandham  
- Rishi Srikaanth  
- Andrei Biswas  

---

## 📌 Problem Statement

With over a billion enterprise and consumer customers, Microsoft takes cybersecurity very seriously. As part of their initiative to improve global protection standards, Microsoft released a large-scale telemetry dataset, challenging the data science community to predict whether a machine will be affected by malware in the near future.

The goal of this project is to develop machine learning models that can effectively identify potential malware infections based on rich, anonymized telemetry data. The dataset reflects real-world challenges such as missing values, mixed data types, and distribution shifts between training and test sets.

🔗 [Microsoft Malware Prediction – Kaggle Competition](https://www.kaggle.com/competitions/microsoft-malware-prediction/overview)

---

## 🗂️ Project Structure & Components

Due to the massive size of the dataset, we couldn’t upload raw or processed data directly to GitHub. To make modeling feasible and reproducible, we preprocessed the datasets locally and reused the saved encoders and scalers for consistent encoding across training and testing.

---

### ⚙️ Data Encoding & Preprocessing

We used two encoding strategies across the project:

- **Initial Encoding**: Basic transformations to quickly get started.
- **Advanced Encoding**: Enhanced feature engineering and scaling.

Encoding and data transformation scripts can be found here:  
🔗 [`Encode Datasets`](https://github.com/Jithin-Veeragandham/CS6140-Project/tree/main/Encode%20Datasets)

As part of this, we generated and stored pickle files (`.pkl`) for:
- Count encoders  
- Label encoders  
- Standard scalers  

These were **not used directly in training** but are crucial for **re-encoding test data** consistently. You can find them here:  
🔗 [`Encoder Pickle Files`](https://github.com/Jithin-Veeragandham/CS6140-Project/tree/main/Encoder%20pickle%20Files)

---

### 🧠 Model Training

The actual training of machine learning models was performed on the locally preprocessed datasets. We trained:
- Neural Networks  
- Feedforward Deep Learning models  
- Tree-based models (LightGBM, XGBoost)

Training notebooks are available in this folder:  
🔗 [`Training`](https://github.com/Jithin-Veeragandham/CS6140-Project/tree/main/Training)

Corresponding model weights are saved under:  
🔗 [`Saved Models`](https://github.com/Jithin-Veeragandham/CS6140-Project/tree/main/Saved%20Models)

These can be loaded to reproduce or evaluate predictions using the matching encoded test data.

---

### 📊 Data Exploration & Feature Validation

Our early investigations into the dataset involved identifying key issues like **distribution shift** between training and test datasets. This also helped us drive our feature selection decisions.

All notebooks related to EDA and initial exploration can be found here:  
🔗 [`Data Exploration and Initial Approaches`](https://github.com/Jithin-Veeragandham/CS6140-Project/tree/main/Data%20Exploration%20and%20Initial%20Approaches)

---

### 📤 Kaggle Submissions

The final `.csv` files generated for submission to Kaggle were built using scripts that:
- Load trained models
- Apply consistent encodings
- Format predictions per competition guidelines

All such scripts are stored here:  
🔗 [`Submissions`](https://github.com/Jithin-Veeragandham/CS6140-Project/tree/main/Submissions)

---

## ✅ Summary

This repository encapsulates our end-to-end pipeline for tackling Microsoft’s malware prediction problem — from encoding strategies and data exploration to model training and final submissions. Due to storage constraints, raw data and processed datasets were handled locally, but the full codebase is provided to ensure full reproducibility and transparency.

---
