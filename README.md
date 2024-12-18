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
  - Implemented preprocessing pipelines to load and clean MRI image data.
  - Trained and evaluated the a U-net architecture on 64x64 pixel MRI images.
- **Results**:
  - Successfully reconstructed missing portions of MRI images:
  
![MRI Imputation - Before and After](images/mri_imputation.png)

- **Repository**: [GitHub Link](https://github.com/ese-ada-lovelace-2024/dl-module-coursework-1-esemsc-mi720)

---

### 2. **Time Series Weather Data Imputation**
- **Description**: Recover missing weather measurements using a neural network. Data includes weather variables recorded daily. Models are trained on corrupted and clean datasets split by decade. Decades are not sequential or related to one another. 
- **Technologies**: Python, LSTM/Transformer, Pandas, TensorFlow.
- **Data**:
  - training_set/ (Corrupted + Clean data for three decades).
  - test_set.csv (Corrupted data for one final decade).
- **Repository**: [GitHub Link](https://github.com/ese-ada-lovelace-2024/dl-module-coursework-2-esemsc-mi720)
- **Results**: Modeled pathways for efficient fuel generation with optimized yield.
- **Visuals**: [Add dashboard link or images]

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
