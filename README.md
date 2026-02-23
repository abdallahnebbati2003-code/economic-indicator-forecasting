# Economic Indicator Forecasting with Machine Learning

## Overview
This project applies Machine Learning techniques to forecast economic indicators,
specifically predicting inflation rates based on macroeconomic variables.
It demonstrates the intersection of Artificial Intelligence and Economics.

## Features
- Data simulation of key macroeconomic indicators
- Exploratory Data Analysis (EDA) with visualizations
- Linear Regression model (baseline)
- Random Forest Regressor (advanced model)
- Model evaluation and comparison

## Economic Indicators Used
| Feature | Description |
|---|---|
| GDP Growth Rate (%) | Annual GDP growth |
| Unemployment Rate (%) | National unemployment |
| Money Supply Growth (%) | M2 money supply change |
| Interest Rate (%) | Central bank rate |
| **Inflation Rate (%)** | **Target variable to predict** |

## Technologies
- Python 3.x
- scikit-learn
- pandas
- matplotlib / seaborn

## How to Run
```bash
pip install -r requirements.txt
python forecasting.py
```

## Results
The Random Forest model outperforms the Linear Regression baseline,
capturing non-linear relationships between macroeconomic variables.

## Author
Master's in Artificial Intelligence and its Applications
```

---

**File 2: `requirements.txt`**
```
pandas
numpy
scikit-learn
matplotlib
seaborn
