# Car Price Prediction using Machine Learning

A comprehensive machine learning project that predicts car prices using various regression algorithms. This project compares the performance of Random Forest and XGBoost regressors to determine the best model for car price prediction.

## 📊 Dataset Overview

The dataset contains **8,128 car records** with the following features:
- **name**: Car model name
- **year**: Manufacturing year
- **selling_price**: Target variable (price in INR)
- **km_driven**: Distance driven in kilometers
- **fuel**: Fuel type (Petrol, Diesel, CNG, LPG)
- **seller_type**: Type of seller (Individual, Dealer, Trustmark Dealer)
- **transmission**: Transmission type (Manual, Automatic)
- **owner**: Ownership history (First Owner, Second Owner, etc.)
- **mileage**: Fuel efficiency (kmpl or km/kg)
- **engine**: Engine capacity (CC)
- **max_power**: Maximum power (bhp)
- **torque**: Torque specifications
- **seats**: Number of seats

## 🛠️ Technologies Used

- **Python 3.x**
- **Pandas** - Data manipulation and analysis
- **NumPy** - Numerical computing
- **Matplotlib** - Data visualization
- **Seaborn** - Statistical data visualization
- **Scikit-learn** - Machine learning algorithms and metrics
- **XGBoost** - Gradient boosting framework

## 📋 Project Structure

```
car-price-prediction/
│
├── Car_Price_Prediction_Final_One.ipynb    # Main Jupyter notebook
└── README.md                               # Project documentation
```

## 🔧 Installation & Setup

1. **Clone the repository**
   ```bash
   git clone https://github.com/beater35/ml-regression-car-price.git
   cd car-price-prediction
   ```

2. **Run the Jupyter notebook**
   ```bash
   jupyter notebook Car_Price_Prediction_Final_One.ipynb
   ```

## 📈 Data Preprocessing

### Data Cleaning
- Removed rows with missing values (222 rows dropped)
- Final dataset: **7,906 records**

### Feature Engineering
- Extracted **brand** from car name
- Cleaned numerical features by removing units:
  - Mileage: Removed 'kmpl' and 'km/kg'
  - Engine: Removed 'CC'
  - Max Power: Removed 'bhp'
  - Torque: Extracted numerical values using regex

### Encoding
- Applied **Label Encoding** to categorical variables:
  - Brand, Fuel, Seller Type, Transmission, Owner

## 🤖 Machine Learning Models

### 1. Random Forest Regressor
- **Training R² Score**: 0.9960
- **Test R² Score**: 0.9846
- **Test RMSE**: 108,142.39
- **Test MAE**: 60,258.27

### 2. XGBoost Regressor
- **Training R² Score**: 0.9958
- **Test R² Score**: 0.9869
- **Test RMSE**: 99,760.99
- **Test MAE**: 58,762.22

## 📊 Model Performance Comparison

| Model | R² Score (Test) | RMSE (Test) | MAE (Test) |
|-------|----------------|-------------|------------|
| **XGBoost** | **0.9869** | **99,760.99** | **58,762.22** |
| Random Forest | 0.9846 | 108,142.39 | 60,258.27 |

**Winner**: XGBoost performs better with higher R² score and lower error metrics.

## 🎯 Key Features Importance

Based on the model analysis, the most important features for car price prediction are:
1. **Year** - Manufacturing year
2. **Brand** - Car manufacturer
3. **Engine** - Engine capacity
4. **Max Power** - Maximum power output
5. **Km Driven** - Distance traveled

## 📊 Visualizations

The project includes several visualizations:
- **Correlation Heatmap** - Shows relationships between features
- **Feature Importance Charts** - For both Random Forest and XGBoost
- **Actual vs Predicted Scatter Plots** - Model performance visualization

## 🚀 Usage

```python
# Load the trained model
import pickle

# For new car price prediction
new_car_features = [[2020, 50000, 1, 1, 1, 0, 18.5, 1500, 120.0, 200.0, 5.0, 15]]
predicted_price = xgb_model.predict(new_car_features)
print(f"Predicted Car Price: ₹{predicted_price[0]:,.2f}")
```

## 📝 Results & Insights

- **XGBoost** achieved the best performance with **98.69% accuracy** (R² score)
- The model can predict car prices with an average error of ₹58,762
- **Year** and **Brand** are the most influential factors in determining car prices
- Both models show excellent performance with minimal overfitting

## 📚 References

- Dataset: [Kaggle - Vehicle Dataset](https://www.kaggle.com/datasets/nehalbirla/vehicle-dataset-from-cardekho)
