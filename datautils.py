import numpy as np
import pandas as pd
from sklearn.model_selection import KFold



def load_EEG_per_patient(path, fold, random_seed=42, binary_labels=True, visual_labels=False, n_folds=5, non_neur_deaths_green=False, all_patients_visualized = False):
    ''' Here the train data is shuffled fully, also inter patient, but the test data is a dictionary with the patient number as key. '''

    all_data_df = pd.read_pickle(path)
    
    # Shuffle per patient to keep patient ordering
    unique_patients = all_data_df['patient_number'].unique()
    unique_patients = np.sort(unique_patients)
    kf = KFold(n_splits=n_folds, shuffle=True, random_state=random_seed)
    for i, (train_index, test_index) in enumerate(kf.split(unique_patients)):
        if i == fold:
            train_patients = unique_patients[train_index]
            test_patients = unique_patients[test_index]
            break

    print('patients in the test set: ', test_patients)

    train_data_df = all_data_df[all_data_df['patient_number'].isin(train_patients)].reset_index(drop=True)
    test_data_df = all_data_df[all_data_df['patient_number'].isin(test_patients)].reset_index(drop=True)
    train_data_df['patient_number1'] = pd.Categorical(train_data_df['patient_number']).codes
    test_data_df['patient_number1'] = pd.Categorical(test_data_df['patient_number']).codes


    ppppp = set(["028","033","084","095","097","102","104","123","142"]) # All non-neur deaths

    # Create train_data and train_labels as numpy arrays for training the encoder
    train_data = np.stack([arr[:, np.newaxis] if arr.ndim == 1 else arr for arr in train_data_df['eeg_array'].values])
    if visual_labels:
        train_labels = train_data_df['vlabel'].to_numpy()
        if binary_labels:
            train_labels = np.where(train_labels == 1, 0, 1) 
        if all_patients_visualized:
            train_labels = train_data_df['patient_number1'].to_numpy()
    else:
        train_labels = train_data_df['PCPC12m'].to_numpy()
        if binary_labels:
            train_labels = np.where(train_labels == 6, 1, 0)
            if non_neur_deaths_green:
                train_labels_mask = (train_data_df['patient_number'].isin(ppppp)).astype(int).to_numpy()
                train_labels = np.where(train_labels_mask == 1, 2, train_labels)

    # Create train_data_dict (used when picking p1 and p2 for the classifier)
    train_data_dict = {}
    train_labels_dict = {}
    for patient in train_patients:
        train_data_dict[patient] = train_data_df[train_data_df['patient_number'] == patient].reset_index(drop=True)
        if visual_labels:
            train_labels_dict[patient] = train_data_dict[patient]['vlabel'].to_numpy()
            if binary_labels:
                train_labels_dict[patient] = np.where(train_labels_dict[patient] == 1, 0, 1)
            if all_patients_visualized:
                train_labels_dict[patient] = train_data_dict[patient]['patient_number1'].to_numpy()
        else:
            train_labels_dict[patient] = train_data_dict[patient]['PCPC12m'].to_numpy()
            if binary_labels:
                train_labels_dict[patient] = np.where(train_labels_dict[patient] == 1, 1, 0)
                if non_neur_deaths_green:
                    if patient in ppppp:
                        train_labels_dict[patient][:] = 2
        train_data_dict[patient] = np.stack([arr[:, np.newaxis] if arr.ndim == 1 else arr for arr in train_data_dict[patient]['eeg_array'].values])


    # Create test_data_dict to be able to classify per patient
    test_data_dict = {}
    test_labels_dict = {}
    for patient in test_patients:
        test_data_dict[patient] = test_data_df[test_data_df['patient_number'] == patient].reset_index(drop=True)
        if visual_labels:
            test_labels_dict[patient] = test_data_dict[patient]['vlabel'].to_numpy()
            if binary_labels:
                test_labels_dict[patient] = np.where(test_labels_dict[patient] == 1, 0, 1)
            if all_patients_visualized:
                test_labels_dict[patient] = test_data_dict[patient]['patient_number1'].to_numpy()
        else:
            test_labels_dict[patient] = test_data_dict[patient]['PCPC12m'].to_numpy()
            if binary_labels:
                test_labels_dict[patient] = np.where(test_labels_dict[patient] == 6, 1, 0)
                if non_neur_deaths_green:
                    if patient in ppppp:
                        test_labels_dict[patient][:] = 2
        test_data_dict[patient] = np.stack([arr[:, np.newaxis] if arr.ndim == 1 else arr for arr in test_data_dict[patient]['eeg_array'].values])

    # shuffle the training data, also inter patient, so all epochs of all patients are shuffled
    fraction = 1
    train_length = train_labels.shape[0]
    train_permutation = np.random.permutation(train_length)
    train_data = train_data[train_permutation]
    train_labels = train_labels[train_permutation]
    train_data = train_data[:int(fraction * train_data.shape[0])]
    train_labels = train_labels[:int(fraction * train_labels.shape[0])]

    return train_data, train_labels, test_data_dict, test_labels_dict, train_data_dict, train_labels_dict