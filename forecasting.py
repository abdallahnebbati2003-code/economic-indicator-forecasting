import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error
from sklearn.preprocessing import StandardScaler
import time
import warnings
warnings.filterwarnings('ignore')

# ── 1. توليد بيانات أضخم وأكثر واقعية ───────────────────────────────────────
print("=" * 65)
print("   ECONOMIC INDICATOR FORECASTING — VERSION 2.0 ENHANCED")
print("=" * 65)

np.random.seed(42)
n_samples = 50000  # ⬆️ من 500 إلى 50,000 سجل

print(f"\n[1] Generating {n_samples:,} economic records...")

# مؤشرات اقتصادية أكثر واقعية
gdp_growth        = np.random.normal(2.5,  1.8,  n_samples)
unemployment      = np.random.normal(6.0,  2.5,  n_samples)
money_supply      = np.random.normal(4.0,  1.5,  n_samples)
interest_rate     = np.random.normal(3.5,  1.2,  n_samples)
trade_balance     = np.random.normal(-1.0, 3.0,  n_samples)   # ميزان تجاري
govt_spending     = np.random.normal(20.0, 5.0,  n_samples)   # إنفاق حكومي %GDP
oil_price_change  = np.random.normal(0.0,  15.0, n_samples)   # تغير سعر النفط
consumer_confidence = np.random.normal(100, 15.0, n_samples)  # ثقة المستهلك

# علاقة أكثر تعقيداً وواقعية (غير خطية)
inflation = (
    0.45  * gdp_growth
    - 0.28 * unemployment
    + 0.58 * money_supply
    - 0.38 * interest_rate
    + 0.05 * trade_balance
    + 0.03 * govt_spending
    + 0.04 * oil_price_change
    - 0.01 * consumer_confidence
    + 0.02 * gdp_growth ** 2          # علاقة غير خطية
    - 0.01 * unemployment * interest_rate  # تفاعل بين المتغيرات
    + np.random.normal(0, 0.8, n_samples)  # ضوضاء أكبر
    + 2.0
)

data = pd.DataFrame({
    'GDP_Growth_Rate':      gdp_growth,
    'Unemployment_Rate':    unemployment,
    'Money_Supply_Growth':  money_supply,
    'Interest_Rate':        interest_rate,
    'Trade_Balance':        trade_balance,
    'Govt_Spending_GDP':    govt_spending,
    'Oil_Price_Change':     oil_price_change,
    'Consumer_Confidence':  consumer_confidence,
    'Inflation_Rate':       inflation
})

print(f"   ✅ Dataset created: {data.shape[0]:,} rows × {data.shape[1]} columns")
print(f"   Features: {list(data.columns[:-1])}")
print(f"\n   Dataset Overview:")
print(data.describe().round(2).to_string())

# ── 2. EDA ────────────────────────────────────────────────────────────────────
print("\n[2] Generating EDA visualizations...")

fig, axes = plt.subplots(2, 4, figsize=(18, 8))
fig.suptitle("Economic Indicators — Distribution Analysis (50,000 Records)",
             fontsize=13, fontweight='bold')

features = data.columns[:-1]
colors   = ['steelblue','tomato','seagreen','darkorange',
            'purple','brown','teal','crimson']

for ax, feature, color in zip(axes.flatten(), features, colors):
    ax.hist(data[feature], bins=60, color=color, alpha=0.7, edgecolor='white')
    ax.set_title(feature.replace('_', ' '), fontsize=9)
    ax.set_xlabel('Value')
    ax.set_ylabel('Frequency')

plt.tight_layout()
plt.savefig('eda_distributions.png', dpi=150)
plt.show()
print("   → Saved: eda_distributions.png")

plt.figure(figsize=(10, 8))
sns.heatmap(data.corr().round(2), annot=True, cmap='coolwarm',
            linewidths=0.5, fmt='.2f', annot_kws={"size": 8})
plt.title("Correlation Matrix — All Economic Indicators", fontweight='bold')
plt.tight_layout()
plt.savefig('correlation_matrix.png', dpi=150)
plt.show()
print("   → Saved: correlation_matrix.png")

# ── 3. تحضير البيانات ────────────────────────────────────────────────────────
print("\n[3] Preparing data...")
X = data.drop('Inflation_Rate', axis=1)
y = data['Inflation_Rate']

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

scaler         = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled  = scaler.transform(X_test)

print(f"   Training set : {X_train.shape[0]:,} samples")
print(f"   Testing set  : {X_test.shape[0]:,} samples")
print(f"   Features     : {X_train.shape[1]}")

# ── 4. النموذج الأول — Linear Regression ──────────────────────────────────────
print("\n[4] Training Model 1 — Linear Regression...")
start = time.time()

lr_model = LinearRegression()
lr_model.fit(X_train_scaled, y_train)
lr_preds = lr_model.predict(X_test_scaled)

lr_time = time.time() - start
lr_rmse = np.sqrt(mean_squared_error(y_test, lr_preds))
lr_mae  = mean_absolute_error(y_test, lr_preds)
lr_r2   = r2_score(y_test, lr_preds)

print(f"   ⏱️  Training time : {lr_time:.2f}s")
print(f"   RMSE            : {lr_rmse:.4f}")
print(f"   MAE             : {lr_mae:.4f}")
print(f"   R²              : {lr_r2:.4f}")

# ── 5. النموذج الثاني — Random Forest (تدريب أعمق) ────────────────────────────
print("\n[5] Training Model 2 — Random Forest (Enhanced)...")
print("   ⚙️  n_estimators=300, max_depth=20 — this may take a moment...")
start = time.time()

rf_model = RandomForestRegressor(
    n_estimators=300,      # ⬆️ من 100 إلى 300 شجرة
    max_depth=20,          # ⬆️ عمق أكبر لكل شجرة
    min_samples_split=5,
    min_samples_leaf=2,
    n_jobs=-1,             # استخدام كل cores المعالج
    random_state=42
)
rf_model.fit(X_train_scaled, y_train)
rf_preds = rf_model.predict(X_test_scaled)

rf_time = time.time() - start
rf_rmse = np.sqrt(mean_squared_error(y_test, rf_preds))
rf_mae  = mean_absolute_error(y_test, rf_preds)
rf_r2   = r2_score(y_test, rf_preds)

print(f"   ⏱️  Training time : {rf_time:.2f}s")
print(f"   RMSE            : {rf_rmse:.4f}")
print(f"   MAE             : {rf_mae:.4f}")
print(f"   R²              : {rf_r2:.4f}")

# ── 6. النموذج الثالث — Gradient Boosting (أقوى نموذج) ────────────────────────
print("\n[6] Training Model 3 — Gradient Boosting (Most Powerful)...")
print("   ⚙️  n_estimators=500, learning_rate=0.05 — this will take longer...")
start = time.time()

gb_model = GradientBoostingRegressor(
    n_estimators=500,       # 500 مرحلة تدريب
    learning_rate=0.05,     # تعلم بطيء = أدق
    max_depth=6,
    min_samples_split=5,
    min_samples_leaf=3,
    subsample=0.8,          # يتدرب على 80% من البيانات كل مرة
    random_state=42,
    verbose=1               # يطبع التقدم أثناء التدريب
)
gb_model.fit(X_train_scaled, y_train)
gb_preds = gb_model.predict(X_test_scaled)

gb_time = time.time() - start
gb_rmse = np.sqrt(mean_squared_error(y_test, gb_preds))
gb_mae  = mean_absolute_error(y_test, gb_preds)
gb_r2   = r2_score(y_test, gb_preds)

print(f"\n   ⏱️  Training time : {gb_time:.2f}s")
print(f"   RMSE            : {gb_rmse:.4f}")
print(f"   MAE             : {gb_mae:.4f}")
print(f"   R²              : {gb_r2:.4f}")

# ── 7. مقارنة النماذج ─────────────────────────────────────────────────────────
print("\n[7] Final Model Comparison")
print("=" * 62)
print(f"{'Model':<28} {'RMSE':>8} {'MAE':>8} {'R²':>8} {'Time':>8}")
print("-" * 62)
print(f"{'Linear Regression':<28} {lr_rmse:>8.4f} {lr_mae:>8.4f} {lr_r2:>8.4f} {lr_time:>6.2f}s")
print(f"{'Random Forest (300 trees)':<28} {rf_rmse:>8.4f} {rf_mae:>8.4f} {rf_r2:>8.4f} {rf_time:>6.2f}s")
print(f"{'Gradient Boosting (500)':<28} {gb_rmse:>8.4f} {gb_mae:>8.4f} {gb_r2:>8.4f} {gb_time:>6.2f}s")
print("=" * 62)

# ── 8. رسم المقارنة ───────────────────────────────────────────────────────────
fig, axes = plt.subplots(1, 3, figsize=(18, 5))
fig.suptitle("Actual vs Predicted Inflation Rate — Model Comparison",
             fontsize=13, fontweight='bold')

sample_idx = np.random.choice(len(y_test), 500, replace=False)
y_sample   = np.array(y_test)[sample_idx]

for ax, preds, title, color in zip(
    axes,
    [lr_preds, rf_preds, gb_preds],
    ['Linear Regression', 'Random Forest\n(300 trees)', 'Gradient Boosting\n(500 stages)'],
    ['steelblue', 'seagreen', 'darkorange']
):
    p_sample = preds[sample_idx]
    ax.scatter(y_sample, p_sample, alpha=0.3, color=color, s=15)
    ax.plot([y_sample.min(), y_sample.max()],
            [y_sample.min(), y_sample.max()], 'r--', linewidth=2)
    ax.set_xlabel("Actual Inflation Rate (%)")
    ax.set_ylabel("Predicted Inflation Rate (%)")
    ax.set_title(f"{title}\nR² = {r2_score(y_sample, p_sample):.4f}")

plt.tight_layout()
plt.savefig('predictions_comparison.png', dpi=150)
plt.show()
print("\n   → Saved: predictions_comparison.png")

# ── 9. Feature Importance ─────────────────────────────────────────────────────
fig, axes = plt.subplots(1, 2, figsize=(14, 5))
fig.suptitle("Feature Importance Comparison", fontsize=13, fontweight='bold')

rf_imp = pd.Series(rf_model.feature_importances_, index=X.columns).sort_values()
gb_imp = pd.Series(gb_model.feature_importances_, index=X.columns).sort_values()

rf_imp.plot(kind='barh', ax=axes[0], color='seagreen', edgecolor='white')
axes[0].set_title("Random Forest — Feature Importance")
axes[0].set_xlabel("Importance Score")

gb_imp.plot(kind='barh', ax=axes[1], color='darkorange', edgecolor='white')
axes[1].set_title("Gradient Boosting — Feature Importance")
axes[1].set_xlabel("Importance Score")

plt.tight_layout()
plt.savefig('feature_importance.png', dpi=150)
plt.show()
print("   → Saved: feature_importance.png")

# ── 10. Cross Validation ──────────────────────────────────────────────────────
print("\n[8] Cross-Validation (5-Fold) on best model...")
cv_scores = cross_val_score(rf_model, X_train_scaled, y_train,
                             cv=5, scoring='r2', n_jobs=-1)
print(f"   CV R² Scores : {cv_scores.round(4)}")
print(f"   Mean R²      : {cv_scores.mean():.4f}")
print(f"   Std R²       : {cv_scores.std():.4f}")

print("\n✅ Project 1 — Version 2.0 Complete!")
print("=" * 65)
