import os
import warnings
import mne
import pandas as pd
import numpy as np

# Path of folder with EEG .set files and extra feature excels and path to outcome excel
eeg_folder = "Z:/06_PostOHCA_Pediatric/3-Experiments/Post_OHCA_MNE_Epoched_20sec_10secOverlap_EEGonly/"
patient_features_excel_path = "Z:/06_PostOHCA_Pediatric/OLD_files/01_Clinical_Parameters/ML_OHCA_Patients_20230328_CoD_Class_PreArrestPCPCCompleteMH_SurvivalHospitalDischarge.xlsx"
vlabels_excel_path = "Z:/06_PostOHCA_Pediatric/4-Metadata/Combined_EEG_Epoch_Classifications_30min.xlsx"

# Output path for DataFrames
output_folder = "./data/"



# Loop over all files in the EEG data folder, only using the .set files and extracting patient number from the file name
filename_list = []
file_path_list = []
for filename in os.listdir( eeg_folder ):
    if filename.endswith(".set"):
        # Create file_path and filename_parts list
        file_path = os.path.join(eeg_folder, filename)
        file_path_list.append(file_path)
        filename_parts = filename.split('_')
        filename_parts[-1] = filename_parts[-1].replace('.set','')
        filename_list.append(filename_parts)


# In this DataFrame the rows are the different EEG epochs and the columns are: ['patient_number','epoch_number']
filename_array = np.delete(np.array(filename_list), [0,3,4], axis=1)
file_array = np.append(filename_array, np.array(file_path_list).reshape(-1,1), axis=1)
file_df = pd.DataFrame(file_array, columns = ['patient_number', 'recording_session_number', 'epoch_number', 'file_path'])

# Create a dictionary of unique patient numbers with the key being the PCPC12m score.
unique_patients = np.unique(file_array[:,0])
patient_dict = {patient: None for patient in unique_patients}
print('patient_dict with PCPC12m score: ', patient_dict)

# Load patient features into df and extract PCPC12m outcome (Later we can also extract features like gender and age here)
patient_features_df = pd.read_excel(patient_features_excel_path)

# Here we create a dictionary where the keys are the patient numbers and the items are the PCPC12m outcome
patients_with_no_pcpm = []
for _, row in patient_features_df.iterrows():
    res_key = row['Res_Key'][4:]
    pcpc12m_value = row['PCPC12m']

    if res_key in patient_dict and pd.notna(pcpc12m_value): # Only update if there is a PCPC outcome
        patient_dict[res_key] = int(pcpc12m_value)
    elif not pd.notna(pcpc12m_value):
        patients_with_no_pcpm.append(res_key)
print(f"Patients {patients_with_no_pcpm} have no PCPC12m value and will not be included")

# Map the PCPC12m outcome to the rows ot the file_df
file_df['PCPC12m'] = file_df['patient_number'].map(patient_dict)


# Here we create a dictionary where the keys are the patient numbers and the items are the visual label
patient_dict = {patient: None for patient in unique_patients}
vlabels_df = pd.read_excel(vlabels_excel_path)
vlabels_df['Consensus'] = vlabels_df['Consensus'].fillna(vlabels_df['Classification_RvdB'])
# print(vlabels_df)
patients_with_no_vlabel = []
for _, row in vlabels_df.iterrows():
    EEG_name = row['EEG_name'][4:7]
    vlabel = row['Consensus']

    if EEG_name in patient_dict and pd.notna(pcpc12m_value): # Only update if there is a PCPC outcome
        patient_dict[EEG_name] = int(vlabel)
    elif not pd.notna(vlabel):
        patients_with_no_vlabel.append(EEG_name)
print(f"Patients {patients_with_no_vlabel} have no vlabel")


# Map the PCPC12m outcome to the rows ot the file_df
file_df['vlabel'] = file_df['patient_number'].map(patient_dict)



# Create row for the eeg_array (fill with object type cells for np.array)
file_df['eeg_array'] = None
file_df_4 = file_df.astype({'eeg_array': 'object'})

patient_number_F7 = '0'
patient_number_F8 = '0'

# Load the EEG data into the file_DF
print('Every 500 rows, the current row number is printed.')
for i, row in file_df.iterrows():
    # Load EEG data from .set file
    eeg_file_path = row['file_path']
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", category=RuntimeWarning) # Surpress a runtime warning we get when loading the data (omitting 1 annotation)
        eeg_data_raw = mne.io.read_raw_eeglab(eeg_file_path, preload=True)

    # Resample to 100 Hz after filtering for 0.5-35 Hz
    eeg_data_raw.filter(l_freq=0.5, h_freq=35., verbose=False)
    eeg_data_raw.resample(100., npad="auto")
    
    # Convert the EEG data to a DF
    eeg_data_array = eeg_data_raw.get_data() # extract data to put into DF
    eeg_channel_names, eeg_timestamps = eeg_data_raw.ch_names, eeg_data_raw.times # Load channel names and timestamps
    eeg_df = pd.DataFrame(eeg_data_array.T, columns = eeg_channel_names, index=eeg_timestamps)
    
    # If the needed channels are not present, we skip the epoch
    if not set(['F7']).issubset(eeg_channel_names) and not set(['Fp1']).issubset(eeg_channel_names):
        print('Patient excluded for not having F7 or Fp1: ', row['patient_number'], 'Channels: ', eeg_channel_names)
        continue
    if not set(['F8']).issubset(eeg_channel_names) and not set(['Fp2']).issubset(eeg_channel_names):
        print('Patient excluded for not having F8 or Fp2: ', row['patient_number'], 'Channels: ', eeg_channel_names)
        continue
    


    # Get channels C3 and C4
    eeg_df_C3_only = eeg_df['C3']
    eeg_df_C4_only = eeg_df['C4']

    # Get channel F7 (or Fp1)
    if set(['F7']).issubset(eeg_channel_names):
        eeg_df_F7_only = eeg_df['F7']
    else:
        # if patient_number_F7 != row['patient_number']:
        print("Patient number ", row['patient_number'], ' does not have F7')
        patient_number_F7 = row['patient_number']
        eeg_df_F7_only = eeg_df['Fp1']

    # Get channel F8 (or Fp2)
    if set(['F8']).issubset(eeg_channel_names):
        eeg_df_F8_only = eeg_df['F8'] # for 7
    else:
        # if patient_number_F8 != row['patient_number']:
        print("Patient number ", row['patient_number'], ' does not have F8')
        patient_number_F8 = row['patient_number']
        eeg_df_F8_only = eeg_df['Fp2']
    
    # Load the needed channels and add the array to the dataframe
    eeg_data_array_C3_only = eeg_df_C3_only.to_numpy()
    eeg_data_array_C4_only = eeg_df_C4_only.to_numpy()
    eeg_data_array_F7_only = eeg_df_F7_only.to_numpy()
    eeg_data_array_F8_only = eeg_df_F8_only.to_numpy()
    file_df_4.at[i, 'eeg_array'] = np.column_stack([eeg_data_array_C3_only, eeg_data_array_C4_only, eeg_data_array_F7_only, eeg_data_array_F8_only])
    
    # Print progress every 500 epochs
    if i % 500 == 0:
        print(i, end=' ', flush=True) # (roughly 5 to 6 minutes per 1000 epochs)
print('Done with loading EEG data')

print('shape before filtering nan eeg values: ', file_df_4.shape)
file_df_4_filtered = file_df_4[file_df_4['PCPC12m'].notna()].reset_index(drop=True)
print('shape after filtering nan eeg values: ', file_df_4_filtered.shape)

print('Saving datasets to pickled files: ')
print(file_df_4_filtered)
file_df_4_filtered.to_pickle(output_folder + "EEG_Timeseries_data_100_4D_C34_F78_vlabel_84_patients.pkl")
print('Dataframe is saved succesfully.')