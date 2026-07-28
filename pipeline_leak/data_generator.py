import numpy as np
import pandas as pd


class PipelineDataGenerator:
    PIPELINE_TYPES = ["crude_oil", "natural_gas", "refined_product", "water_injection"]
    LEAK_SIZES = ["none", "small", "medium", "large"]

    def __init__(self, n_samples=5000, random_state=2024):
        self.n_samples = n_samples
        self.rng = np.random.RandomState(random_state)

    def generate(self):
        n = self.n_samples
        rng = self.rng

        pipeline_type = rng.choice(self.PIPELINE_TYPES, n)
        pipeline_length_km = rng.uniform(1, 100, n)
        pipeline_diameter_mm = rng.choice([100, 150, 200, 300, 400], n)
        pressure_upstream_mpa = rng.uniform(0.5, 10.0, n)
        pressure_downstream_mpa = rng.uniform(0.3, 8.0, n)
        flow_rate_m3h = rng.uniform(10, 500, n)
        temperature_c = rng.uniform(-10, 60, n)
        ambient_temp_c = rng.uniform(-20, 50, n)
        soil_moisture_pct = rng.uniform(5, 80, n)
        pipe_wall_thickness_mm = rng.uniform(5, 25, n)

        has_leak = rng.choice([0, 1], n, p=[0.7, 0.3])
        leak_size = np.where(
            has_leak,
            rng.choice([1, 2, 3], n, p=[0.5, 0.3, 0.2]),
            0,
        )
        leak_size_label = np.where(
            leak_size == 0, "none",
            np.where(leak_size == 1, "small",
            np.where(leak_size == 2, "medium", "large"))
        )

        pressure_drop = (
            (pressure_upstream_mpa - pressure_downstream_mpa)
            * (1 + 0.5 * has_leak * leak_size / 3)
            + rng.normal(0, 0.1, n)
        )

        flow_anomaly = np.where(
            has_leak,
            flow_rate_m3h * (0.02 + 0.08 * leak_size / 3) + rng.normal(0, 0.5, n),
            rng.normal(0, 0.2, n),
        )

        acoustic_emission = np.where(
            has_leak,
            20 + 30 * leak_size / 3 + rng.normal(0, 3, n),
            rng.normal(5, 2, n),
        )
        acoustic_emission = np.clip(acoustic_emission, 0, 100)

        temperature_diff = np.where(
            has_leak,
            (5 + 10 * leak_size / 3) * (1 + 0.1 * (temperature_c - ambient_temp_c) / 30),
            rng.normal(0, 0.5, n),
        )

        vibration_level = np.where(
            has_leak,
            0.5 + 2.0 * leak_size / 3 + rng.normal(0, 0.2, n),
            rng.normal(0.2, 0.1, n),
        )
        vibration_level = np.clip(vibration_level, 0, 5)

        df = pd.DataFrame({
            "pipeline_type": pipeline_type,
            "pipeline_length_km": np.round(pipeline_length_km, 2),
            "pipeline_diameter_mm": pipeline_diameter_mm,
            "pressure_upstream_mpa": np.round(pressure_upstream_mpa, 3),
            "pressure_downstream_mpa": np.round(pressure_downstream_mpa, 3),
            "flow_rate_m3h": np.round(flow_rate_m3h, 2),
            "temperature_c": np.round(temperature_c, 1),
            "ambient_temp_c": np.round(ambient_temp_c, 1),
            "soil_moisture_pct": np.round(soil_moisture_pct, 1),
            "pipe_wall_thickness_mm": np.round(pipe_wall_thickness_mm, 1),
            "pressure_drop_mpa": np.round(pressure_drop, 4),
            "flow_anomaly_m3h": np.round(flow_anomaly, 4),
            "acoustic_emission_db": np.round(acoustic_emission, 2),
            "temperature_diff_c": np.round(temperature_diff, 2),
            "vibration_level_g": np.round(vibration_level, 3),
            "has_leak": has_leak,
            "leak_size": leak_size_label,
        })
        return df

    def save(self, path="outputs/data/pipeline_data.csv"):
        import os
        os.makedirs(os.path.dirname(path), exist_ok=True)
        df = self.generate()
        df.to_csv(path, index=False)
        return df
