import csv
import math
import numpy as np
import pandas as pd

from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error

import joblib

class FeatureTransformer:
    def __init__(self):
        self.scaler = None
        self.pca = None
        self.numeric_cols = []
        self.exclude_cols = []

    def fit_transform(self, df, exclude_cols=None, pca_variance=0.99):
        if exclude_cols is None:
            exclude_cols = ["steering", "throttle", "brake"]
        numeric_cols = [c for c in df.select_dtypes(include=[np.number]).columns if c not in exclude_cols]
        self.numeric_cols = numeric_cols
        self.exclude_cols = exclude_cols
        self.scaler = StandardScaler()
        scaled_data = self.scaler.fit_transform(df[numeric_cols])
        self.pca = PCA(n_components=pca_variance)
        pca_data = self.pca.fit_transform(scaled_data)
        pc_columns = [f"PC{i+1}" for i in range(pca_data.shape[1])]
        df_pca = pd.DataFrame(pca_data, columns=pc_columns)
        df_excl = df[exclude_cols].reset_index(drop=True)
        df_out = pd.concat([df_pca, df_excl], axis=1)
        return df_out

    def transform(self, df):
        if self.scaler is None or self.pca is None:
            raise RuntimeError("FeatureTransformer not fitted yet!")
        # If missing columns, fill with 0.0
        for c in self.numeric_cols:
            if c not in df.columns:
                df[c] = 0.0
        scaled_data = self.scaler.transform(df[self.numeric_cols])
        pca_data = self.pca.transform(scaled_data)
        pc_columns = [f"PC{i+1}" for i in range(pca_data.shape[1])]
        df_pca = pd.DataFrame(pca_data, columns=pc_columns)
        df_excl = pd.DataFrame()
        for c in self.exclude_cols:
            if c in df.columns:
                df_excl[c] = df[c].values
        df_excl = df_excl.reset_index(drop=True)
        df_out = pd.concat([df_pca, df_excl], axis=1)
        return df_out

    def save(self, path="transformer.joblib"):
        joblib.dump({
            "scaler": self.scaler,
            "pca": self.pca,
            "numeric_cols": self.numeric_cols,
            "exclude_cols": self.exclude_cols
        }, path)

    def load(self, path="transformer.joblib"):
        data = joblib.load(path)
        self.scaler = data["scaler"]
        self.pca = data["pca"]
        self.numeric_cols = data["numeric_cols"]
        self.exclude_cols = data["exclude_cols"]

