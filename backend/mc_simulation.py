import numpy as np
import pandas as pd

def run_mc_simulation_uniform(
        left_sev: float,
        right_sev: float,
        brain_sev: float,
        bounds: dict,
        samples: int = 10000,
        best_pipe=None
    ):

    mode_features = ['Mode_ACP','Mode_ACV','Mode_CMP','Mode_CMV']

    # Mode-specific feature sets
    pressure_feats = ['inspiratory_pressure', 'peep', 'fio2', 'respiration_rate']
    volume_feats   = ['flow', 'tidal_volume', 'peep', 'fio2', 'respiration_rate']

    # Core continuous features (from bounds)
    continuous_features = list(bounds.keys())
    all_core_feats = set(continuous_features)

    expected_features = [
        'fio2','peep','inspiratory_pressure','respiration_rate',
        'flow','tidal_volume',
        'left_severity','right_severity','brain_injury_severity',
        'Mode_ACP','Mode_ACV','Mode_CMP','Mode_CMV'
    ]

    N_per_mode = samples // 4
    mc_blocks = []

    for mode in mode_features:

        block = pd.DataFrame()

        # Select active features
        if mode in ['Mode_ACP', 'Mode_CMP']:
            active_feats = pressure_feats
        else:
            active_feats = volume_feats

        # Sample active features
        for feat in active_feats:
            low, high = bounds[feat]
            block[feat] = np.random.uniform(low, high, N_per_mode)

        # Inactive continuous features → ZERO-FILL
        inactive_feats = list(all_core_feats - set(active_feats))
        for feat in inactive_feats:
            block[feat] = np.zeros(N_per_mode)

        # Fixed severities
        block['left_severity'] = left_sev
        block['right_severity'] = right_sev
        block['brain_injury_severity'] = brain_sev

        # One-hot modes (1 for this mode, else 0)
        for m in mode_features:
            block[m] = 1 if m == mode else 0

        # --- Ensure all expected columns exist ---
        for col in expected_features:
            if col not in block.columns:
                block[col] = 0.0

        # Reorder columns correctly
        block = block[expected_features]

        mc_blocks.append(block)

    # --- Combine all blocks ---
    mc_samples = pd.concat(mc_blocks, ignore_index=True)

    # --- NO NaN allowed in StandardScaler ---
    mc_samples = mc_samples.fillna(0)

    # --- Predict ---
    vital_names = [
        "Respiratory Rate","Heart Rate","SpO2","PaO2","Systolic BP",
        "Diastolic BP","MAP","PIP","pH","PaCO2","Hematocrit","Hemoglobin"
    ]

    preds = best_pipe.predict(mc_samples)
    preds_df = pd.DataFrame(preds, columns=vital_names)

    return mc_samples, preds_df
