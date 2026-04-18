import pandas as pd

from logistic_regression import gradient_descent

df = pd.read_csv("class_imbalance/dataset_865_analcatdata_neavote.csv")
X_df = pd.get_dummies(df[["Party", "Favorable"]], columns=["Party"])
X_df["Favorable"] = X_df["Favorable"] / 10.0

y_series = df["binaryClass"].map({"P": 1, "N": 0})

X = X_df.values.astype(float)
y = y_series.values.astype(float)

weight, bias = gradient_descent(X, y)
print(weight, bias)
