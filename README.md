# CS6140 Machine Learning Project - Microsoft Malware Prediction

> Leveraging machine learning to predict malware infections using Microsoft's extensive telemetry data. Our project encompasses rigorous data preprocessing, insightful feature engineering, and a comparative analysis of various classification models, including tree-based algorithms and neural networks.

## 👥 Contributors

- Jithin Veeragandham
- Rishi Srikaanth
- Andrei Biswas

---

## 📌 Problem Statement

The escalating sophistication and proliferation of malware pose a significant threat to global cybersecurity. Traditional signature-based detection methods struggle to keep pace with evolving threats. This project addresses this challenge by applying advanced machine learning techniques to predict malware infections using a large-scale, anonymized telemetry dataset provided by Microsoft through the Kaggle competition. The objective is to build robust models capable of identifying subtle patterns indicative of impending malware attacks.

🔗 [Microsoft Malware Prediction – Kaggle Competition](https://www.kaggle.com/competitions/microsoft-malware-prediction/overview)

---

## 🗂️ Project Structure & Components

Due to the substantial size of the Microsoft Malware Prediction dataset, the raw and processed data are managed locally. To ensure reproducibility and efficient experimentation, we implemented a strategy of saving preprocessing artifacts (encoders, scalers) which are then consistently applied across training and testing phases.

---

### ⚙️ Data Encoding & Preprocessing

We employed two distinct encoding strategies to prepare the data for our machine learning models:

- **Initial Encoding**: A foundational approach involving frequency/count encoding for high-cardinality categorical features (e.g., `AppVersion`, `Census_OSVersion`), label encoding for lower-cardinality string features after text normalization, and imputation of missing values with -1. This strategy aimed for a quick baseline for tree-based models.
- **Advanced Encoding**: An enhanced strategy incorporating domain-inspired feature engineering (e.g., screen area, aspect ratio, disk usage ratios, a binary "magic\_4" feature based on antivirus states) and count encoding for high-cardinality features. This was designed to potentially improve model performance by capturing more complex relationships in the data.

Encoding and data transformation scripts, along with the logic for saving preprocessing artifacts, can be found here:
🔗 [`Encode Datasets`](https://github.com/Jithin-Veeragandham/CS6140-Project/tree/main/Encode%20Datasets)

The generated and stored pickle files (`.pkl`) for count encoders, label encoders, and standard scalers (used for consistent test data encoding) are located here:
🔗 [`Encoder Pickle Files`](https://github.com/Jithin-Veeragandham/CS6140-Project/tree/main/Encode%20Datasets/Encoder%20pickle%20Files)

---

### 🧠 Model Training

The training of our machine learning models was conducted on the locally preprocessed datasets. We explored a range of architectures:

- **Neural Networks**: Feedforward networks with 2, 3, and 4 hidden layers, employing ReLU activation, Batch Normalization, and Dropout for regularization. Trained using `BCEWithLogitsLoss` and the Adam optimizer with learning rate scheduling based on validation loss.
- **Simple Linear Classifiers**: Logistic Regression, Support Vector Machine (Linear kernel), and Perceptron, used as baseline models to assess data linear separability.
- **Tree-based Models**: LightGBM and XGBoost, chosen for their ability to handle categorical features directly and their strong performance in similar tasks. XGBoost hyperparameters were tuned using Optuna.

Training notebooks, detailing the model architectures and training procedures, are available in this directory:
🔗 [`Training`](https://github.com/Jithin-Veeragandham/CS6140-Project/tree/main/Training)

The trained model weights and configurations are saved for reproducibility and evaluation:
🔗 [`Saved Models`](https://github.com/Jithin-Veeragandham/CS6140-Project/tree/main/Saved%20Models)

---

### 📊 Data Exploration & Feature Validation

Our initial exploration focused on understanding the dataset's characteristics, including data types, missing values, and the crucial issue of distribution shift between the training and test sets. We employed techniques like visualization and statistical tests (KS 2-sample and Chi-square) to quantify these differences, which informed our feature selection strategies.

Notebooks documenting our exploratory data analysis and feature validation processes can be found here:
🔗 [`Data Exploration and Initial Approaches`](https://github.com/Jithin-Veeragandham/CS6140-Project/tree/main/Data%20Exploration)

---

### 📤 Kaggle Submissions

The generation of `.csv` files for submission to the Kaggle competition involved loading our trained models, applying the corresponding saved encoders to the test data, and formatting the predictions according to the competition's requirements.

Scripts used for generating Kaggle submission files are located here:
🔗 [`Submissions`](https://github.com/Jithin-Veeragandham/CS6140-Project/tree/main/submission-noteboooks)

---


## 🛠️ Potential Improvements

Future work could explore:

- More sophisticated techniques for addressing the observed distribution shift
- Investigating alternative neural network architectures like Resnet to keep training on deeper architectures.
- Further experimentation with ensemble methods, combining the strengths of different models (e.g., stacking).
- If access were available, incorporating external datasets (as done by top competitors) to enrich the feature set with information like software release dates and update intervals.


---

