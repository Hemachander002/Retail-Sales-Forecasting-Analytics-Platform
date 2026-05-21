# Retail Sales Forecasting Analytics Platform
The Retail Sales Forecasting Analytics Platform is an end-to-end machine learning solution developed to forecast future retail sales using historical sales data and external influencing factors. 
The platform automates the complete ML workflow, starting from data ingestion to model evaluation and prediction generation, making it scalable and production-ready.

# Work Completed in the Project
### 1. Exploratory Data Analysis (EDA)

==> Performed detailed Exploratory Data Analysis on the retail datasets to understand:

==> Sales trends across different years
==> Seasonal patterns and fluctuations
==> Store-wise and product-wise sales behavior
==> Correlation between external factors and sales performance
==> Data quality issues such as missing values and outliers

==> The EDA phase helped identify important business insights and prepare the data effectively for model training.

### 2. End-to-End Machine Learning Pipeline

==> Developed a complete automated ML pipeline that includes:

==> Data ingestion
==> Data preprocessing
==> Feature engineering
==> Model training
==> Hyperparameter tuning
==> Model evaluation with dagshub and MLflow for Experiment Tracking
==> Model saving and deployment preparation

==> The pipeline architecture makes the system reusable, maintainable, and production-ready.

### 3. Model Selection and Forecasting

Different forecasting models such as Prophet, ARIMA, and XGBoost were evaluated. 
Among them, the XGBRegressor model delivered the best performance in terms of prediction accuracy and generalization capability.
Therefore, XGBRegressor was selected as the primary forecasting model for the platform.

### 4. Hyperparameter Tuning

Implemented automated hyperparameter tuning inside the ML pipeline to identify the best-performing model configuration dynamically.

This ensures:

==> Improved forecasting accuracy
==> Better model optimization
==> Flexibility for future retraining as data changes over time

The pipeline can continuously retrain and produce updated optimal models whenever new data becomes available.

### 5. Web-Based Forecasting Application

==> Built a lightweight and user-friendly web application using:

==> Streamlit for frontend and backend integration
==> Custom CSS for UI styling and better user experience

==> The application allows users to interact with the forecasting system easily without requiring technical knowledge.

### 6. Future Data Input System

==> The platform accepts future input datasets in CSV format, including:

==> Future sales-related information
==> External datasets such as oil prices and other influencing factors

==> This design enables the model to generate more realistic and context-aware sales forecasts.

### 7. Prediction and Forecast Generation

After uploading the required datasets, the application processes the data through the trained ML pipeline and generates future sales predictions automatically.
The platform provides:
==> Forecasted sales values
==> Prediction outputs for future dates
==> Complete prediction report in csv format


## Screenshots of the Application
<img width="1918" height="997" alt="forecasting_front_page" src="https://github.com/user-attachments/assets/4cd936d1-5a5d-43a9-bfa7-64787b8e4bf8" />
<img width="1918" height="1001" alt="forecasting_store" src="https://github.com/user-attachments/assets/1c8e3274-6955-4ea3-90de-546b564a1e1c" />
<img width="1910" height="999" alt="forecasting_product" src="https://github.com/user-attachments/assets/91aa641d-5f0a-4458-b83b-e9d1e1ddbad2" />
<img width="1919" height="1040" alt="forecasting_feature_imp" src="https://github.com/user-attachments/assets/30b72b80-d868-4e73-a2e4-1bfee5325690" />

  


