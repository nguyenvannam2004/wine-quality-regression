import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
import joblib
import numpy as np

sns.set(style="whitegrid")

# Load dữ liệu
df = pd.read_csv("winequality-red.csv", sep=";")
X = df.drop("quality", axis=1)
y = df["quality"]

# Tách train/test
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Train mô hình
model = RandomForestRegressor(n_estimators=100, random_state=42)
model.fit(X_train, y_train)

# Dự đoán
y_pred = model.predict(X_test)

# Đánh giá
mse = mean_squared_error(y_test, y_pred)
mae = mean_absolute_error(y_test, y_pred)
r2 = r2_score(y_test, y_pred)

print(f"✅ MAE: {mae:.4f}")
print(f"✅ MSE: {mse:.4f}")
print(f"✅ R² Score: {r2:.4f}")

# Lưu model
joblib.dump(model, "wine_model.pkl")

# === 🔍 1. Biểu đồ dự đoán vs thực tế ===
# plt.figure(figsize=(8, 6))
# sns.scatterplot(x=y_test, y=y_pred, alpha=0.6)
# plt.plot([y.min(), y.max()], [y.min(), y.max()], '--r')
# plt.xlabel("Giá trị thực tế")
# plt.ylabel("Dự đoán")
# plt.title("So sánh giá trị thực tế và dự đoán")
# plt.savefig("plot_actual_vs_predicted.png")
# plt.close()

# === 🔍 2. Phân phối sai số ===
errors = y_test - y_pred
# plt.figure(figsize=(8, 6))
# sns.histplot(errors, bins=20, kde=True)
# plt.xlabel("Sai số (thực tế - dự đoán)")
# plt.title("Phân phối sai số")
# plt.savefig("plot_error_distribution.png")
# plt.close()

# === 🔍 3. Boxplot sai số theo giá trị thực tế ===
# df_result = pd.DataFrame({"actual": y_test, "error": errors})
# plt.figure(figsize=(10, 6))
# sns.boxplot(x="actual", y="error", data=df_result)
# plt.title("Phân phối sai số theo giá trị thực tế")
# plt.xlabel("Chất lượng rượu (thực tế)")
# plt.ylabel("Sai số")
# plt.savefig("plot_boxplot_error_by_quality.png")
# plt.close()

# === 🔍 4. Heatmap tương quan giữa các đặc trưng ===
plt.figure(figsize=(12, 10))
corr_matrix = df.corr()
sns.heatmap(corr_matrix, annot=True, cmap="coolwarm", fmt=".2f", square=True)
plt.title("Ma trận tương quan giữa các đặc trưng")
plt.savefig("plot_feature_correlation_heatmap.png")
plt.close()

# === 🔍 5. Feature importance ===
# importances = model.feature_importances_
# features = X.columns
# indices = np.argsort(importances)[::-1]

# plt.figure(figsize=(10, 6))
# sns.barplot(x=importances[indices], y=features[indices])
# plt.title("Mức độ quan trọng của các đặc trưng")
# plt.xlabel("Tỉ lệ ảnh hưởng")
# plt.ylabel("Đặc trưng")
# plt.savefig("plot_feature_importance.png")
# plt.close()
