# Credit Card Default Prediction

## Project Overview
This project builds a predictive machine learning model to identify credit card customers at risk of defaulting on their payments. It employs advanced techniques, including SMOTE (Synthetic Minority Over-sampling Technique) for handling class imbalance, and compares multiple algorithms to find the best performer.

## Business Objective
The model aims to:
- Predict default risk
- Enable proactive intervention
- Reduce financial losses
- Improve credit decisions
- Provide explainability of factors contributing to default risk

### Business Impact
- 79.4% reduction in default rates when approving the top 10% safest customers
- Default rate reduced from 21.88% (overall) to 4.50% (approved customers)
- Enables data-driven credit decisions, saving millions in bad debt

## Dataset Description
- **Source**: `default of credit card clients.xls`
- **Records**: 30,000 credit card customers from Taiwan (April - September 2005)
- **Target Variable**: Default payment next month (Binary: 0 = No Default, 1 = Default)
- **Class Distribution**: 
  - Non-defaulters (0): ~78% (23,364 customers)
  - Defaulters (1): ~22% (6,636 customers)

## Data Pipeline
1. **Load Data**: Read the dataset and preserve features.
2. **Exploratory Data Analysis (EDA)**: Analyze credit limits by demographics and default distribution.
3. **Feature Engineering**: Create new features like credit utilization and payment ratios.
4. **Data Cleaning**: Drop unnecessary columns and prepare the dataset for modeling.

## Model Development
- **Algorithms Used**: Logistic Regression, Random Forest, XGBoost
- **Evaluation Metrics**: AUC-ROC Score, Gini Coefficient, KS Statistic
- **Best Model**: Random Forest with an AUC of 0.7643

## Deployment
- The model can be saved and loaded for predictions.
- A Streamlit web app is available for interactive predictions.

## Installation
1. Install Python 3.8+
2. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```

## Running the Project
- To run the full pipeline:
   ```bash
   python fraud_detection_full.py
   ```
- To run the Streamlit app:
   ```bash
   streamlit run app.py
   ```

## Project Structure
```
Credit Card Default Prediction/
├── default of credit card clients.xls
├── credit_risk.ipynb
├── app.py
├── requirements.txt
├── best_credit_risk_model.pkl
└── README.md
```

## Key Takeaways
- The project effectively identifies credit card default risk.
- The best-performing model significantly reduces default rates.
- The deployment is ready for production use.

## Contact & Support
For questions or improvements, refer to the Jupyter notebook, check the requirements.txt for dependencies, or consult the scikit-learn documentation for algorithm details.