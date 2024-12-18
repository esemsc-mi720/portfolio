# 🌟 Data Science & Machine Learning Projects Portfolio

Welcome to my portfolio! This space showcases data science and machine learning projects completed during my academic journey and free time.

---

## 📂 **Academic Projects**

### 1. **Imputing Missing Portions of MRI Scans**
- **Description**:
  - Developed a neural network architecture to recover missing portions of MRI images of human heads.
  - This addressed a key imputation problem in magnetic resonance imaging (MRI) where incomplete scans prevent detailed patient analysis. 
- **Technologies**: Python, PyTorch, NumPy, Matplotlib
- **Data**:
  - Artificially generated realistic MRI images for model training (using a pre-trained generative model).
  - Corrupted MRI images stored in test_set.npy (100 images of size 64x64 pixels).
- **Approach**:
  - Designed a neural network to learn from artificially-generated datasets.
  - Implemented preprocessing pipelines to apply corruption mask to clean MRI image data for training. 
  - Trained and evaluated a U-net architecture.
- **Results**:
  - Successfully reconstructed missing portions of MRI images:
  
![MRI Imputation - Before and After](images/mri_imputation.png)

- **Repository**: [GitHub Link](https://github.com/ese-ada-lovelace-2024/dl-module-coursework-1-esemsc-mi720)

---

### 2. **Time Series Weather Data Imputation**
- **Description**:
  - Developed a neural network to recover missing daily weather measurements from sequential time-series data.
  - Addressed corrupted weather data over multiple decades, using both clean and corrupted datasets for training and testing.
- **Technologies**: Python, LSTM/Transformer models, Pandas, TensorFlow, Matplotlib, Seaborn.
- **Data**:
  - ***Training***:  
    - training_set_0.csv, training_set_1.csv, training_set_2.csv (corrupted).
    - training_set_0_nogaps.csv, training_set_1_nogaps.csv, training_set_2_nogaps.csv (clean versions).
  - ***Testing***:
    - test_set.csv (corrupted time-series data for one unseen decade).
  - ***Variables***:
    - cloud_cover, sunshine, global_radiation, max_temp, mean_temp, min_temp, precipitation, pressure.
- **Approach**:
  -Exploratory Data Analysis (EDA)to visualize time-series trends and compare clean vs. corrupted data.
    - Created line plots for each variable across 365 days to highlight gaps.
    - Generated histograms to show distribution shifts due to corruption.
  - Preprocessing pipeline to handle differently distributed data for model stability.
  - Investigate several models like RNNS, LSTMs and Transformer architectures for time-series data imputation. LSTM model was ultimately selected due to time constraints to tune the highly parameter-sensitive transformer. 
	- Implemented an LSTM-based neural network to capture temporal dependencies and impute missing measurements.
	- Trained the model using corrupted inputs and clean targets.
	- Validated on unseen test data to evaluate recovery performance.
- **Repository**: [GitHub Link](https://github.com/ese-ada-lovelace-2024/dl-module-coursework-2-esemsc-mi720)
- **Results**:
  - Successfully imputed missing values across all weather variables with high accuracy:

![Time Series Missing Values Imputation - Before and After](images/cloud_cover_time_series.png)

---

### 3. **Predicting Significant Wave Heights**
- **Description**: Build a regression model to predict significant wave heights based on environmental and oceanic conditions. Perform	data preprocessing and EDA, feature engineering, model selection and evaluation with performance metrics.
- **Technologies**: Python, Pandas, Scikit-learn, Matplotlib
- **Repository**: [GitHub Link](https://github.com/ese-ada-lovelace-2024/dsml-2024-esemsc-mi720)
- **Notebooks**: Q1.ipynb, Q1_answer.ipynb.

### 4. **Predicting Passenger Transportation**
- **Description**: Build a classification problrm to predict whether passengers were transported based on personal and travel-related data. Model selection and threshold tuning. Use AUC-ROC and Precision-Recall to evaluate models.
- **Technologies**: Python, Pandas, Scikit-learn, Matplotlib, 
- **Repository**: [GitHub Link](https://github.com/ese-ada-lovelace-2024/dsml-2024-esemsc-mi720)
- **Notebooks**: Q2.ipynb, Q2_answer.ipynb.

---

## 📂 **Free-Time Personal Projects**

### 1. **Streamlit Dashboard for Environmental Analysis**
- **Description**: Interactive dashboard for visualizing environmental parameters.
- **Technologies**: Streamlit, Python, Plotly.
- **Demo**: [Link to Streamlit app](https://link-to-demo)
  

---

## 🛠️ **Tools and Technologies**
- **Languages**: Python
- **Frameworks/Libraries**: Scikit-learn, TensorFlow, Pandas, Matplotlib, Plotly, Seaborn, Pytorch
- **Tools**: Streamlit, Jupyter Notebook, GitHub
- **Platforms**: GitHub, Streamlit Cloud

---

## 🔗 **Links**
- **GitHub Profile**: [github.com/your-username](https://github.com/your-username)
