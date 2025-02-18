import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA

# 1. Read the data (named "output.py" as per your question)
df = pd.read_csv("output.csv")

# 2. Select numeric columns
numeric_cols = df.select_dtypes(include=['float64', 'int64']).columns
df_numeric = df[numeric_cols]

# 3. Standardize the numeric columns
scaler = StandardScaler()
scaled_data = scaler.fit_transform(df_numeric)

# 4. Apply PCA to retain >= 99% variance
pca = PCA(n_components=0.99)  # 0.99 => retain at least 99% variance
pca_data = pca.fit_transform(scaled_data)

# 5. Wrap the PCA output into a DataFrame
#    Name columns PC1, PC2, ...
df_pca = pd.DataFrame(
    pca_data,
    columns=[f"PC{i+1}" for i in range(pca_data.shape[1])]
)

# 6. Save the PCA-transformed data to "std_output.py"
df_pca.to_csv("stdPCA_output.csv", index=False)
print("done")
