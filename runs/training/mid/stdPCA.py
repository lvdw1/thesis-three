import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA

# 1. Read the data
df = pd.read_csv("session3/run1.csv")

# 2. Select numeric columns but exclude "steering", "throttle", and "brake"
exclude_cols = ["steering", "throttle", "brake"]
numeric_cols = [col for col in df.select_dtypes(include=['float64', 'int64']).columns 
                if col not in exclude_cols]
df_numeric = df[numeric_cols]

# 3. Standardize the selected numeric columns
scaler = StandardScaler()
scaled_data = scaler.fit_transform(df_numeric)

# 4. Apply PCA to retain >= 99% variance
pca = PCA(n_components=0.99)  # retain at least 99% variance
pca_data = pca.fit_transform(scaled_data)

# 5. Wrap the PCA output into a DataFrame
df_pca = pd.DataFrame(
    pca_data,
    columns=[f"PC{i+1}" for i in range(pca_data.shape[1])]
)

# Optionally, if you want to add the excluded columns back to the output:
df_final = pd.concat([df_pca, df[exclude_cols].reset_index(drop=True)], axis=1)

# 6. Save the final DataFrame
df_final.to_csv("../processed/session3/run1.csv", index=False)
print("done")
