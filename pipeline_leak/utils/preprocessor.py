import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler, LabelEncoder


class PipelinePreprocessor:
    NUMERIC_FEATURES = [
        "pipeline_length_km", "pipeline_diameter_mm",
        "pressure_upstream_mpa", "pressure_downstream_mpa",
        "flow_rate_m3h", "temperature_c", "ambient_temp_c",
        "soil_moisture_pct", "pipe_wall_thickness_mm",
        "pressure_drop_mpa", "flow_anomaly_m3h",
        "acoustic_emission_db", "temperature_diff_c", "vibration_level_g",
    ]
    CATEGORICAL_FEATURES = ["pipeline_type"]

    def __init__(self):
        self.scaler = StandardScaler()
        self.label_encoders = {}
        self.fitted = False

    def fit(self, df):
        self.scaler.fit(df[self.NUMERIC_FEATURES])
        for col in self.CATEGORICAL_FEATURES:
            le = LabelEncoder()
            le.fit(df[col])
            self.label_encoders[col] = le
        self.fitted = True
        return self

    def transform(self, df):
        X_num = self.scaler.transform(df[self.NUMERIC_FEATURES])
        X_cat = np.column_stack([
            self.label_encoders[col].transform(df[col])
            for col in self.CATEGORICAL_FEATURES
        ])
        return np.hstack([X_num, X_cat])

    def fit_transform(self, df):
        self.fit(df)
        return self.transform(df)
