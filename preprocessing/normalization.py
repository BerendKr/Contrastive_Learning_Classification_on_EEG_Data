import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler


EEG_df = pd.read_pickle("data/EEG_Timeseries_data_100_4D_C34_F78_vlabel_84_patients.pkl")


# Check for and remove None values
old_length = EEG_df.shape[0]
print("rows to be removed: ", EEG_df[EEG_df.isna().any(axis=1)])
EEG_df = EEG_df[~EEG_df.isna().any(axis=1)].reset_index(drop=True)
new_length = EEG_df.shape[0]
dif_length = old_length - new_length
print("How many rows were removed because of NaN/None: ", dif_length)



# remove patients with too many artifacts
EEG_df = EEG_df[EEG_df['patient_number'] != '151'] # Remove patient 151, as it has too many artifacts (vlabel 0)
EEG_df = EEG_df[EEG_df['patient_number'] != '153'] # Remove patient 153, as it has too many artifacts (vlabel 0)

# remove every other epoch to make sure there is no more overlap
EEG_df1 = EEG_df.groupby('patient_number').apply(lambda x: x.iloc[::2]).reset_index(drop=True)

all_eeg = np.stack(EEG_df1['eeg_array'].values)  # shape: (N, T, 4)

# Compute mean and std over samples and time (axis 0 and 1)
print("mean: ", np.mean(all_eeg.mean(axis=(0, 1))))
print("std. dev.: ", np.mean(all_eeg.std(axis=(0, 1))))


scaler = StandardScaler()

all_values = np.concatenate([arr.reshape(-1, arr.shape[1]) for arr in EEG_df1['eeg_array'].values], axis=0)
scaler.fit(all_values) # or uncomment lines below if not enough ram is abailable. These are the precomputed parameters.
# means = np.array([float.fromhex('-0x1.5043b9fa0df37p-36'), float.fromhex('-0x1.153f9cf626ffdp-31'), float.fromhex('0x1.cef5e4d65364dp-33'), float.fromhex('0x1.06aac33176d7cp-31')])
# scales = np.array([float.fromhex('0x1.532e4fb52ea04p-16'), float.fromhex('0x1.56ee7152f7dc3p-16'), float.fromhex('0x1.752506649ae3cp-16'), float.fromhex('0x1.57c0653547313p-16')])
# varis = np.array([float.fromhex('0x1.c163af7a976a8p-32'), float.fromhex('0x1.cb61f4e096af8p-32'), float.fromhex('0x1.0ff274fe024cbp-31'), float.fromhex('0x1.cd951fccbcb3cp-32')])
# scaler.mean_ = means
# scaler.scale_ = scales
# scaler.var_ = varis
# scaler.n_samples_seen_ = 67880000


def normalize_array(arr):
    return scaler.transform(arr)


EEG_df1['eeg_array'] = EEG_df1['eeg_array'].apply(normalize_array)

print(EEG_df1)

all_eeg = np.stack(EEG_df1['eeg_array'].values)  # shape: (N, T, 4)

# Compute mean and std over samples and time (axis 0 and 1)
print("After normalization")
print("mean: ", np.mean(all_eeg.mean(axis=(0, 1))))
print("std. dev.: ", np.mean(all_eeg.std(axis=(0, 1))))

EEG_df1.to_pickle("data/EEG_Timeseries_data_100_4D_C34_F78_vlabel_normalized_84.pkl")