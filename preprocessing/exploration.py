import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
# from more_itertools import powerset
import seaborn as sns
from matplotlib import cm
import mne

###
# This file contains multiple (commented) blocks of code that can be used individually to explore the data and create visualizations.
# The code is not meant to be run as a whole, but rather to be executed in parts as needed.
###

# df 
path = "./data/EEG_Timeseries_data_100_4D_C34_F78_vlabel_normalized_84.pkl"
df = pd.read_pickle(path)

colors = cm.get_cmap('PuBu', 7)  # Blue-Purple colormap with x discrete colors
Color = colors(6) # set color for all plots






# # Plot example epoch, unnormalized, 4 channels
# eeg_data_raw = mne.io.read_raw_eeglab(r"Z:\06_PostOHCA_Pediatric\3-Experiments\Post_OHCA_MNE_Epoched_20sec_10secOverlap_EEGonly\EEG_086_01_A_eeg_02.set", preload=True)

# eeg_data_raw.resample(250., npad="auto")
# eeg_data_array = eeg_data_raw.get_data() # extract data to put into DF
# eeg_channel_names, eeg_timestamps = eeg_data_raw.ch_names, eeg_data_raw.times # Load channel names and timestamps
# print(eeg_channel_names)

# time_axis = np.arange(5000) / 250
# fig, axs = plt.subplots(4, 1, figsize=(14, 2.5*4), sharex=True)
# channels = {2:'C3', 3:'C4', 6:'F7', 7:'F8'}
# i=0
# for c in [2,3,6,7]:
#     signal = eeg_data_array[c,:]
#     axs[i].plot(time_axis, signal, label=f'Channel {channels[c]}', linewidth=0.6)
#     axs[i].set_ylabel('Amplitude')
#     axs[i].legend(loc='upper right')
#     i += 1
# axs[-1].set_xlabel('Time (s)')
# global_min = signal.min()
# global_max = signal.max()
# max_val = max(-1*global_min, global_max)
# margin = 1.7*(max_val)
# for ax in axs:
#     ax.set_ylim(-1*margin, margin)
# plt.tight_layout(rect=[0,0.03,1,0.95])
# plt.show()










# # df = df.groupby('patient_number').apply(lambda x: x.iloc[::2]).reset_index(drop=True)
# print(df['eeg_array'].iloc[0].shape)
# pd.set_option('display.max_rows', None)  # Show all rows in the DataFrame
# print(df.drop_duplicates('patient_number'))

# print(df.drop_duplicates('patient_number')['vlabel'].value_counts())



# # Check the distribution of the vlabels over the patient outcomes
# # Define outcome groups
# df['outcome'] = df['PCPC12m'].apply(lambda x: 'Survival' if x in [1,2,3,4,5] else 'Death')

# # Count how many patients in each outcome group for each vlabel
# counts = df.drop_duplicates('patient_number').groupby(['outcome', 'vlabel']).size().unstack(fill_value=0)

# # Reorder rows to ensure 'Survival' comes first
# counts = counts.reindex(['Survival', 'Death'])

# vlabel_names = {
#     1: '1: Continuous',
#     2: '2: Suppressed',
#     3: '3: No Activity',
#     4: '4: Burst Identical',
#     5: '5: Burst Non-Identical',
#     6: '6: GPDs Flat',
#     7: '7: GPDs Non-Flat'
# }

# # Plot
# colors = cm.get_cmap('PuBu', 7)  # Blue-Purple colormap with 6 discrete colors
# color_list = [colors(i) for i in range(7)]
# ax = counts.plot(kind='bar', stacked=True, color=color_list[::-1], edgecolor='black', figsize=(8,6))

# # Update legend with custom labels
# handles, labels = ax.get_legend_handles_labels()
# new_labels = [vlabel_names.get(int(label), label) for label in labels]
# ax.legend(handles, new_labels, title='EEG background pattern', bbox_to_anchor=(1.05, 1), loc='upper left')

# plt.xlabel('Outcome')
# plt.ylabel('Number of Patients')
# # plt.legend(title='EEG background pattern', bbox_to_anchor=(1.05, 1), loc='upper left')
# plt.tight_layout()
# plt.show()







# all_eeg = np.stack(df['eeg_array'].values)  # shape: (N, 5000, 4)

# # Compute mean and std over samples and time (axis 0 and 1), per channel
# mean_per_channel = all_eeg.mean(axis=(0, 1))  # shape: (4,)
# std_per_channel = all_eeg.std(axis=(0, 1))    # shape: (4,)

# # Print results
# for i, (mean, std) in enumerate(zip(mean_per_channel, std_per_channel)):
#     print(f"Channel {i+1}: mean = {mean:.4f}, std = {std:.4f}")


# # mean and std deviation of epochs per patient
# patient_counts = df['patient_number'].value_counts()
# print("mean epochs per patient: ", patient_counts.mean())
# print("std. deviation epochs per patient: ", patient_counts.std())

# how many patients have 1-4 / 5-6 as PCPC12m
# 1-12, 2-8, 3-11, 4-6, 5-0, 6-47
# how_many_patients_with_this_pcpc = df.drop_duplicates('patient_number')[df.drop_duplicates('patient_number')['PCPC12m'] == 6].shape[0]
# print(df.drop_duplicates('patient_number')[df.drop_duplicates('patient_number')['PCPC12m'] != 6])
# print("how many patients with PCPC12m == 6: ", how_many_patients_with_this_pcpc)


# # Plot
# plt.figure(figsize=(10, 5))
# plt.hist(patient_counts.values, bins=15, color='#4BA3C3', edgecolor='black', alpha=0.9)

# # Add labels and title
# plt.title('Histogram of Epoch Counts per Patient', fontsize=14, weight='bold')
# plt.xlabel('Number of Epochs', fontsize=12)
# plt.ylabel('Number of Patients', fontsize=12)





# # ##### create histogram of PCPC12m label distributions both normal and binary
# # pcpc_scores = [1,2,3,4,5,6]
# # patient_counts = [12,8,11,6,0,47]

# pcpc_scores = ['survival (class 0)', 'death (class 1)']
# patient_counts = [37, 47]

# # vlabels = ["1","2","3","4","5","6","7"]
# # patient_counts = [52, 2, 18, 4, 6, 1, 1]

# # Plot
# plt.figure(figsize=(10, 6))
# bars = plt.bar(pcpc_scores, patient_counts, color=Color, edgecolor='black', width=0.6)

# # Add count labels on top of bars
# for bar in bars:
#     height = bar.get_height()
#     plt.text(bar.get_x() + bar.get_width()/2, height + 0.8, str(height), ha='center', va='bottom', fontsize=10)

# # Labels and title
# # plt.title('Number of Patients per PCPC Score at 12 Months', fontsize=14)
# plt.xlabel('Background pattern at 24 Hours', fontsize=12)
# plt.ylabel('Number of Patients', fontsize=12)
# plt.xticks(pcpc_scores, fontsize=11)
# plt.grid(axis='y', linestyle='--', alpha=0.6)

# plt.tight_layout()
# plt.show()


# # histogram and boxplot of epochs per patient
# # Get counts
# patient_counts = df['patient_number'].value_counts()

# # Use seaborn styling for prettier visuals
# sns.set(style="whitegrid")

# # Create the figure and axis
# plt.figure(figsize=(10, 2.5))
# sns.boxplot(x=patient_counts.values, color=Color)

# # Add visual details
# # plt.title('Distribution of Epoch Counts per Patient', fontsize=14, weight='bold')
# plt.xlabel('Number of Epochs', fontsize=12)
# plt.yticks([])  # Remove y-axis ticks for a cleaner horizontal plot
# plt.tight_layout()
# plt.show()

# # Plot
# plt.figure(figsize=(10, 5))
# plt.hist(patient_counts.values, bins=15, color=Color, edgecolor='black', alpha=0.9)

# # Add labels and title
# # plt.title('Histogram of Epoch Counts per Patient', fontsize=14, weight='bold')
# plt.xlabel('Number of Epochs', fontsize=12)
# plt.ylabel('Number of Patients', fontsize=12)

# # Add grid and style tweaks
# plt.grid(axis='y', linestyle='--', alpha=0.7)
# plt.tight_layout()
# plt.show()






#eeg_data_raw = mne.io.read_raw_eeglab(r"Z:\06_PostOHCA_Pediatric\3-Experiments\Post_OHCA_MNE_Epoched_20sec_10secOverlap_EEGonly\EEG_145_01_A_eeg_10.set", preload=True)

#eeg_data_array = eeg_data_raw.get_data() # extract data to put into DF
#eeg_channel_names, eeg_timestamps = eeg_data_raw.ch_names, eeg_data_raw.times # Load channel names and timestamps
#print(eeg_channel_names)



# EEG_df_norm = pd.read_pickle("data/EEG_Timeseries_data_250_4D_C34_F78_vlabel_normalized.pkl")

# dfdf = EEG_df_norm[['patient_number', 'vlabel']].copy()
# print(dfdf)
# dfdf.to_pickle("data/EEG_Timeseries_data_250_4D_C34_F78_pnumber_vlabels_only.pkl")



# all_channels = ['A1', 'A2', 'C3', 'C4', 'Cz', 'F3', 'F4', 'F7', 'F8', 'Fp1', 'Fp2', 'O1', 'O2', 'P3', 'P4', 'P7', 'P8', 'Pz', 'T3', 'T4', 'T5', 'T6', 'T7', 'T8', 'T9', 'T10']
# all_channels_realistic = ['C3', 'C4', 'Cz', 'F3', 'F4', 'F7', 'F8', 'Fp1', 'Fp2', 'O1', 'O2', 'P3', 'P4', 'T3', 'T4', 'T5', 'T6']
# all_channels123 = ['A1', 'A2', 'C3', 'C4']

# EEG_df_norm['vlabel'] = EEG_df_vlab['vlabel']
# EEG_df_norm = EEG_df_norm[['patient_number', 'recording_session_number', 'epoch_number', 'file_path', 'PCPC12m', 'vlabel', 'eeg_array']]
# EEG_df_norm.to_pickle("data/EEG_Timeseries_data_250_4D_C34_F78_vlabel_normalized.pkl")


# # Check the distribution of the channels over the epochs
# dict_channels = {
#     'C3_C4_Cz_F3_F4_F7_F8_Fp1_Fp2_O1_O2_P3_P4_T3_T4_T5_T6': 5069,
#     'C3_C4_Cz_F3_F4_F7_F8_Fp1_Fp2_O1_O2_P3_P4_Pz_T10_T3_T4_T5_T6': 1240,
#     'C3_C4_Cz_F3_F4_F7_F8_Fp1_Fp2_Fz_O1_O2_P3_P4_Pz_T3_T4_T5_T6': 1447,
#     'C3_C4_Cz_F7_F8_Fp1_Fp2_O1_O2_T3_T4_T5_T6': 536,
#     'C3_C4_Cz_F3_F4_F7_F8_Fp1_Fp2_O1_O2_P3_P4_Pz_T10_T3_T4_T5_T6_T9': 145,
#     'C3_C4_Cz_F3_F4_F7_F8_Fp1_Fp2_O1_O2_P4_T3_T4_T5_T6': 175,




#eeg_data_raw = mne.io.read_raw_eeglab(r"Z:\06_PostOHCA_Pediatric\3-Experiments\Post_OHCA_MNE_Epoched_20sec_10secOverlap_EEGonly\EEG_145_01_A_eeg_10.set", preload=True)

#eeg_data_array = eeg_data_raw.get_data() # extract data to put into DF
#eeg_channel_names, eeg_timestamps = eeg_data_raw.ch_names, eeg_data_raw.times # Load channel names and timestamps
#print(eeg_channel_names)


# EEG_df_norm = pd.read_pickle("data/EEG_Timeseries_data_250_4D_C34_F78_vlabel_normalized.pkl")

# dfdf = EEG_df_norm[['patient_number', 'vlabel']].copy()
# print(dfdf)
# dfdf.to_pickle("data/EEG_Timeseries_data_250_4D_C34_F78_pnumber_vlabels_only.pkl")



# all_channels = ['A1', 'A2', 'C3', 'C4', 'Cz', 'F3', 'F4', 'F7', 'F8', 'Fp1', 'Fp2', 'O1', 'O2', 'P3', 'P4', 'P7', 'P8', 'Pz', 'T3', 'T4', 'T5', 'T6', 'T7', 'T8', 'T9', 'T10']
# all_channels_realistic = ['C3', 'C4', 'Cz', 'F3', 'F4', 'F7', 'F8', 'Fp1', 'Fp2', 'O1', 'O2', 'P3', 'P4', 'T3', 'T4', 'T5', 'T6']
# all_channels123 = ['A1', 'A2', 'C3', 'C4']

# EEG_df_norm['vlabel'] = EEG_df_vlab['vlabel']
# EEG_df_norm = EEG_df_norm[['patient_number', 'recording_session_number', 'epoch_number', 'file_path', 'PCPC12m', 'vlabel', 'eeg_array']]
# EEG_df_norm.to_pickle("data/EEG_Timeseries_data_250_4D_C34_F78_vlabel_normalized.pkl")


# # Check the distribution of the channels over the epochs
# dict_channels = {
#     'C3_C4_Cz_F3_F4_F7_F8_Fp1_Fp2_O1_O2_P3_P4_T3_T4_T5_T6': 5069,
#     'C3_C4_Cz_F3_F4_F7_F8_Fp1_Fp2_O1_O2_P3_P4_Pz_T10_T3_T4_T5_T6': 1240,
#     'C3_C4_Cz_F3_F4_F7_F8_Fp1_Fp2_Fz_O1_O2_P3_P4_Pz_T3_T4_T5_T6': 1447,
#     'C3_C4_Cz_F7_F8_Fp1_Fp2_O1_O2_T3_T4_T5_T6': 536,
#     'C3_C4_Cz_F3_F4_F7_F8_Fp1_Fp2_O1_O2_P3_P4_Pz_T10_T3_T4_T5_T6_T9': 145,
#     'C3_C4_Cz_F3_F4_F7_F8_Fp1_Fp2_O1_O2_P4_T3_T4_T5_T6': 175,
#     'C3_C4_F3_F4_F7_F8_Fp1_Fp2_O1_O2_P3_P4_T3_T4_T5_T6': 179,
#     'C3_C4_Cz_F7_F8_Fp1_Fp2_Fz_O1_O2_T3_T4_T5_T6': 179,
#     'C3_C4_Cz_F3_F4_F7_F8_Fp1_Fp2_O1_O2_P3_P4_T3_T4_T5': 172,
#     'C3_C4_Cz_F3_F4_F7_F8_Fp1_Fp2_O1_P4_T3_T4_T5_T6': 170,
#     'C3_C4_Cz_F3_F4_F7_F8_Fp1_Fp2_Fz_O1_O2_P3_P4_T3_T4_T5_T6': 485,
#     'C3_C4_F3_F4_F7_F8_Fp1_Fp2_O1_P3_P4_T3_T4_T5_T6': 117,
#     'C3_C4_Cz_F3_F4_F8_Fp1_Fp2_O1_O2_P3_P4_T3_T4_T5_T6': 267,
#     'C3_C4_Cz_F7_F8_Fp1_Fp2_O1_O2_Pz_T3_T4_T5_T6': 177,
#     'C3_C4_Cz_F3_F4_F7_F8_Fp1_Fp2_Fz_P3_P4_Pz_T3_T4_T5_T6': 174,
#     'C3_C4_F3_F4_F7_F8_Fp1_Fp2_Fz_O1_O2_P3_P4_Pz_T3_T4_T5_T6': 111,
#     'C3_C4_Cz_Fp1_Fp2_O1_O2_T3_T4': 266,
#     'A1_A2_C3_C4_Cz_F3_F4_F7_F8_Fp1_Fp2_O1_O2_P3_P4_Pz_T10_T4_T5_T6_T9': 58,
#     'A1_A2_C3_C4_Cz_F3_F4_F7_F8_Fp1_Fp2_O1_O2_P3_P4_Pz_T3_T4_T5_T6': 118,
#     'C3_C4_Cz_F3_F4_F7_F8_Fp2_O1_O2_P3_P4_Pz_T3_T4_T5_T6': 169,
#     'C3_C4_Cz_F7_F8_Fp1_Fp2_O1_O2_T4_T5_T6': 174,
#     'C3_C4_Cz_F3_F4_F7_F8_Fp1_Fp2_O1_O2_P3_Pz_T3_T4_T5_T6': 173,
#     'C3_C4_Cz_F7_Fp1_Fp2_O1_O2_T3_T4_T5_T6': 173,
#     'C3_C4_Cz_F3_F4_F7_F8_Fp1_Fp2_O1_O2_T3_T4_T5_T6': 159,
#     'C3_C4_F3_F4_F7_F8_Fp1_Fp2_Fz_O1_O2_P3_P4_T4_T5_T6': 153,
#     'C3_C4_F3_F4_F7_F8_Fp1_Fp2_O1_O2_P3_P4_Pz_T3_T4_T5_T6': 171,
#     'C3_C4_Cz_F3_F4_F7_F8_Fp1_Fp2_O1_O2_P4_Pz_T3_T4_T5_T6': 155,
#     'C3_C4_Cz_F3_F4_F7_F8_Fp1_Fp2_Fz_O1_O2_P3_P4_P7_P8_Pz_T7_T8': 1012,
#     'C3_C4_Cz_F3_F4_F8_Fp2_Fz_O1_O2_P3_P4_P7_P8_Pz_T7_T8': 168,
#     'C3_C4_Cz_F3_F7_F8_Fp1_Fp2_O1_O2_P4_P7_P8_T7_T8': 179,
#     'C3_C4_Cz_F3_F8_Fp1_Fp2_Fz_O1_O2_P3_P4_P7_P8': 116
# }

# all_channels = ['A1', 'A2', 'C3', 'C4', 'Cz', 'F3', 'F4', 'F7', 'F8', 'Fp1', 'Fp2', 'O1', 'O2', 'P3', 'P4', 'P7', 'P8', 'Pz', 'T3', 'T4', 'T5', 'T6', 'T7', 'T8', 'T9', 'T10']
# all_channels_realistic = ['C3', 'C4', 'Cz', 'F3', 'F4', 'F7', 'F8', 'Fp1', 'Fp2', 'O1', 'O2', 'P3', 'P4', 'T3', 'T4', 'T5', 'T6']
# all_channels123 = ['A1', 'A2', 'C3', 'C4']

# # Create powerset of all_channels calles all_channel_combinations to check which channels are used in the epochs
# all_channel_combinations = list(powerset(all_channels))
# channels_dict = dict.fromkeys(all_channel_combinations, 0)

# for channel_combination in channels_dict.keys():
#     for key, item in dict_channels.items():
#         if set(channel_combination).issubset(set(key.split('_'))):
#             channels_dict[channel_combination] += item


# total = 13887

# sorted_items = sorted(channels_dict.items(), key=lambda kv: (kv[1], kv[0]))

# for i in range(30):
#     for j in range(len(sorted_items)-1, 0, -1):
#         if len(sorted_items[j][0]) == i:
#             print(i, ' : ', sorted_items[j][1]/total, sorted_items[j])
#             break


# # for item in sorted_items:
# #     if len(item[0]) >= 8: 
# #         print(item, item[1]/total)
# # for key in sorted(channels_dict):
# #     print(channels_dict[key]/total, ' : ', channels_dict[key], ' : ', key)




# # df = pd.read_pickle("data/EEG_Timeseries_data_normalized.pkl")

# # # Take 10 patients, 10 different epochs from each patient

# # unique_patients_10 = df['patient_number'].unique()[1:10] # 005=6., 037=6., 038=1., 041=3., 047=2.
# # print(unique_patients_10)
# # filtered_df = df[df['patient_number'].isin(['007', '037', '038', '041', '047' ])]
# # df_10 = filtered_df.groupby('patient_number').head(10)

# # print(df_10)

# # # patient_outcomes = df.groupby('patient_number')['PCPC12m'].first()
# # # outcome_counts = patient_outcomes.value_counts().reindex([1.0, 2.0, 3.0, 4.0, 5.0, 6.0], fill_value=0)
# # # print(outcome_counts)
# # # # 1.0 - 12, 2.0 - 8, 3.0 - 10, 4.0 - 5, 5.0 - 0, 6.0 - 43 
# # # #
# # # #

# # # first_non_6_patient = df[df['PCPC12m'] != 6].sort_values(by='patient_number').iloc[600] 
# # # print(first_non_6_patient)
# # # # <38=6, 38=1, 41=3, 46=2, 47=2


# # for patient in ['007', '037', '038', '041', '047' ]:
# #     patient_data = df_10[df_10['patient_number'] == patient]
# #     print(patient)
# #     patient_outcome = patient_data['PCPC12m'].iloc[0]
# #     max_amplitude = patient_data['max_abs_value'].iloc[0]

# #     plt.figure(figsize=(12,8))
# #     plt.suptitle(f'Patient {patient} - PCPC12m {patient_outcome} - Max_Amplitude {max_amplitude}', fontsize=16)

# #     for i, (_, row) in enumerate(patient_data.iterrows(), start=1):
# #         plt.subplot(5, 2, i)
# #         plt.plot(row['eeg_array'], color='blue', linewidth=0.3)
# #         plt.title(f'Epoch {row["epoch_number"]}')
# #         # plt.ylim(-0.00002, 0.00002)
# #         plt.xlabel('Time')
# #         plt.ylabel('Amplitude')
    
# #     plt.tight_layout(rect=[0,0,1,0.96])
# #     plt.show()