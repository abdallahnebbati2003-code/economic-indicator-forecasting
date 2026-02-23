import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.preprocessing import StandardScaler

# ── 1. SIMULATE MACROECONOMIC DATA ───────────────────────────────────────────
np.random.seed(42)
n_samples = 500

gdp_growth       = np.random.normal(2.5, 1.5, n_samples)       # GDP growth rate (%)
unemployment     = np.random.normal(6.0, 2.0, n_samples)        # Unemployment rate (%)
money_supply     = np.random.normal(4.0, 1.2, n_samples)        # Money supply growth (%)
interest_rate    = np.random.normal(3.5, 1.0, n_samples)        # Central bank rate (%)

# Inflation follows an economic relationship + noise (based on simplified Phillips Curve)
inflation = (
    0.5  * gdp_growth
    - 0.3 * unemployment
    + 0.6 * money_supply
    - 0.4 * interest_rate
    + np.random.normal(0, 0.5, n_samples)
    + 2.0  # base inflation
)

# Build DataFrame
data = pd.DataFrame({
    'GDP_Growth_Rate':    gdp_growth,
    'Unemployment_Rate':  unemployment,
    'Money_Supply_Growth': money_supply,
    'Interest_Rate':      interest_rate,
    'Inflation_Rate':     inflation
})

print("=" * 60)
print("  ECONOMIC INDICATOR FORECASTING WITH MACHINE LEARNING")
print("=" * 60)
print("\n[1] Dataset Overview")
print(data.describe().round(2))

# ── 2. EXPLORATORY DATA ANALYSIS (EDA) ───────────────────────────────────────
print("\n[2] Generating EDA plots...")

fig, axes = plt.subplots(2, 2, figsize=(12, 8))
fig.suptitle("Economic Indicators — Distribution Analysis", fontsize=14, fontweight='bold')

features = ['GDP_Growth_Rate', 'Unemployment_Rate', 'Money_Supply_Growth', 'Interest_Rate']
colors   = ['steelblue', 'tomato', 'seagreen', 'darkorange']

for ax, feature, color in zip(axes.flatten(), features, colors):
    ax.hist(data[feature], bins=30, color=color, alpha=0.7, edgecolor='white')
    ax.set_title(feature.replace('_', ' '))
    ax.set_xlabel('Value (%)')
    ax.set_ylabel('Frequency')

plt.tight_layout()
plt.savefig('eda_distributions.png', dpi=150)
plt.show()
print("   → Saved: eda_distributions.png")

# Correlation heatmap
plt.figure(figsize=(8, 6))
sns.heatmap(data.corr().round(2), annot=True, cmap='coolwarm',
            linewidths=0.5, fmt='.2f')
plt.title("Correlation Matrix — Macroeconomic Indicators", fontweight='bold')
plt.tight_layout()
plt.savefig('correlation_matrix.png', dpi=150)
plt.show()
print("   → Saved: correlation_matrix.png")

# ── 3. PREPARE DATA ──────────────────────────────────────────────────────────
X = data.drop('Inflation_Rate', axis=1)
y = data['Inflation_Rate']

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

scaler  = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled  = scaler.transform(X_test)

# ── 4. MODEL 1 — LINEAR REGRESSION (Baseline) ────────────────────────────────
print("\n[3] Training Linear Regression (Baseline)...")
lr_model = LinearRegression()
lr_model.fit(X_train_scaled, y_train)
lr_preds = lr_model.predict(X_test_scaled)

lr_rmse = np.sqrt(mean_squared_error(y_test, lr_preds))
lr_r2   = r2_score(y_test, lr_preds)

print(f"   RMSE : {lr_rmse:.4f}")
print(f"   R²   : {lr_r2:.4f}")

# ── 5. MODEL 2 — RANDOM FOREST ───────────────────────────────────────────────
print("\n[4] Training Random Forest Regressor...")
rf_model = RandomForestRegressor(n_estimators=100, random_state=42)
rf_model.fit(X_train_scaled, y_train)
rf_preds = rf_model.predict(X_test_scaled)

rf_rmse = np.sqrt(mean_squared_error(y_test, rf_preds))
rf_r2   = r2_score(y_test, rf_preds)

print(f"   RMSE : {rf_rmse:.4f}")
print(f"   R²   : {rf_r2:.4f}")

# ── 6. MODEL COMPARISON ──────────────────────────────────────────────────────
print("\n[5] Model Comparison")
print("-" * 40)
print(f"{'Model':<25} {'RMSE':>8} {'R²':>8}")
print("-" * 40)
print(f"{'Linear Regression':<25} {lr_rmse:>8.4f} {lr_r2:>8.4f}")
print(f"{'Random Forest':<25} {rf_rmse:>8.4f} {rf_r2:>8.4f}")
print("-" * 40)

# ── 7. VISUALIZE PREDICTIONS ─────────────────────────────────────────────────
fig, axes = plt.subplots(1, 2, figsize=(14, 5))
fig.suptitle("Actual vs Predicted Inflation Rate", fontsize=14, fontweight='bold')

for ax, preds, title, color in zip(
    axes,
    [lr_preds, rf_preds],
    ['Linear Regression', 'Random Forest'],
    ['steelblue', 'seagreen']
):
    ax.scatter(y_test, preds, alpha=0.5, color=color, edgecolors='white', linewidth=0.5)
    ax.plot([y_test.min(), y_test.max()],
            [y_test.min(), y_test.max()], 'r--', linewidth=2, label='Perfect Fit')
    ax.set_xlabel("Actual Inflation Rate (%)")
    ax.set_ylabel("Predicted Inflation Rate (%)")
    ax.set_title(f"{title}\nR² = {r2_score(y_test, preds):.4f}")
    ax.legend()

plt.tight_layout()
plt.savefig('predictions_comparison.png', dpi=150)
plt.show()
print("\n   → Saved: predictions_comparison.png")

# ── 8. FEATURE IMPORTANCE (Random Forest) ────────────────────────────────────
importances = pd.Series(rf_model.feature_importances_, index=X.columns).sort_values()

plt.figure(figsize=(8, 5))
importances.plot(kind='barh', color='seagreen', edgecolor='white')
plt.title("Feature Importance — Random Forest", fontweight='bold')
plt.xlabel("Importance Score")
plt.tight_layout()
plt.savefig('feature_importance.png', dpi=150)
plt.show()
print("   → Saved: feature_importance.png")

print("\n✅ Project 1 Complete — Economic Indicator Forecasting with ML")
print("=" * 60)
