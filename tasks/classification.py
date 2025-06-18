import numpy as np
from . import _eval_protocols as eval_protocols
from sklearn.metrics import roc_curve, auc
from sklearn.metrics import confusion_matrix, classification_report
import seaborn as sns
import matplotlib
# matplotlib.use("Agg")  # uncomment to avoid some GUI crashes during parallel processing for classifier fitting
import matplotlib.pyplot as plt
from sklearn.model_selection import KFold
from sklearn.preprocessing import StandardScaler
from collections import Counter
from sklearn.neighbors import KNeighborsClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.pipeline import make_pipeline


def find_best_p1_p2_5fold(best_params, train_repr_dict, train_labels_dict, visual_labels, eval_protocol, random_seed):
    """
    Finds the best p1 and p2 using 5-fold cross-validation on training data.
    Splits the data by patient, trains KNN on 80% of patients, and validates on 20%.
    Returns the average best p1 and p2 values over the 5 folds.
    """

    patients = list(train_repr_dict.keys())
    kf = KFold(n_splits=5, shuffle=True, random_state=random_seed)

    p1_values = np.linspace(0, 1, num=101)  # Fixed number of p1 steps
    p2_values = np.linspace(0, 1, num=101)  # Fixed number of p2 steps

    best_p1_list = []
    best_p2_list = []

    for train_index, val_index in kf.split(patients):
        # Split patients into train and validation sets
        train_patients = [patients[i] for i in train_index]
        val_patients = [patients[i] for i in val_index]

        # Prepare training and validation data
        train_data = np.vstack([train_repr_dict[p] for p in train_patients])
        train_labels = np.hstack([train_labels_dict[p] for p in train_patients])

        val_data_dict = {p: train_repr_dict[p] for p in val_patients}
        val_labels_dict = {p: train_labels_dict[p] for p in val_patients}

        # Use classifier with best hyperparameters from training on the full training set
        if eval_protocol == 'knn':
            clf = make_pipeline(
                StandardScaler(),
                KNeighborsClassifier(**best_params)
            )
        elif eval_protocol == 'rf':
            clf = make_pipeline(
                StandardScaler(),
                RandomForestClassifier(**best_params)
            )
        elif eval_protocol == 'svm':
            clf = make_pipeline(
                StandardScaler(),
                SVC(**best_params)
            )
        clf.fit(train_data, train_labels)

        # Compute predictions for validation set
        val_results_dict = {}
        for patient, val_data in val_data_dict.items():
            y_score = clf.predict_proba(val_data)[:, 1]  # Get class 1 probabilities
            val_results_dict[patient] = {'y_score': y_score, 'true_label': val_labels_dict[patient][0]}

        # Find the best p1, p2 for this fold
        accuracy_matrix = np.zeros((len(p1_values), len(p2_values)))
        false_positive_matrix = np.zeros((len(p1_values), len(p2_values)))

        for i, p1 in enumerate(p1_values):
            for j, p2 in enumerate(p2_values):
                per_patient_prediction = []
                per_patient_labels = []
                false_positive_count = 0

                for patient, data in val_results_dict.items():
                    y_score = data['y_score']
                    true_label = data['true_label']

                    predictions = (y_score >= p1).astype(int)
                    death_percentage = np.mean(predictions)
                    patient_prediction = 1 if death_percentage >= p2 else 0

                    per_patient_prediction.append(patient_prediction)
                    per_patient_labels.append(true_label)

                    if patient_prediction == 1 and true_label == 0:
                        false_positive_count += 1

                accuracy = np.mean(np.array(per_patient_labels) == np.array(per_patient_prediction))
                accuracy_matrix[i, j] = accuracy
                false_positive_matrix[i, j] = false_positive_count

        # Select best p1 and p2 for this fold
        filtered_accuracy_matrix = np.where(false_positive_matrix == 0, accuracy_matrix, -1)

        
        # If we are using visual labels, we can use the full range of p1 and p2, else we have constraints
        if not visual_labels:
            filtered_accuracy_matrix[0:int(np.floor( (90/100) * filtered_accuracy_matrix.shape[0])), :] = -1 # Set range for p1 (0-100) that we want to exclude
            filtered_accuracy_matrix[:, 0:int(np.floor( (50/100) * filtered_accuracy_matrix.shape[1]))] = -1 # Set range for p2 (0-100) that we want to exclude

        # best_idx = np.unravel_index(np.argmax(filtered_accuracy_matrix), filtered_accuracy_matrix.shape)
        max_val = np.max(filtered_accuracy_matrix)
        all_max_indices = np.argwhere(filtered_accuracy_matrix == max_val)
        best_idx = all_max_indices[0] # Which one of the best idx do you want? Take first one for now
        best_p1_list.append(p1_values[best_idx[0]])
        best_p2_list.append(p2_values[best_idx[1]])


    # Compute final best p1 and p2 by removing lowest p1 and p2 and then taking the mean of the others
    best_p1_list = sorted(best_p1_list)
    best_p2_list = sorted(best_p2_list)
    best_p1_final = np.mean(best_p1_list)
    best_p2_final = np.mean(best_p2_list)
    print('in the CV, p1 and p2 list are:', best_p1_list, best_p2_list)

    return best_p1_final, best_p2_final





def eval_classification_per_patient(model, train_data, train_labels, test_data, test_labels, train_data_dict, train_labels_dict, figure_save_location, fold, visual_labels, eval_protocol='knn', nosave=False, random_seed=42):
    assert train_labels.ndim == 1 or train_labels.ndim == 2

    train_repr = model.encode(train_data, encoding_window='full_series' if train_labels.ndim == 1 else None)

    # compute the representations of the test and training data dicionaries (sorted per patient)
    # dictionary format for the test data as we want to classify per patient
    # dictionary format for the training data as we use 5 fold cross-validation to find the best hyperparameters
    test_repr = {}
    train_repr_dict = {}
    for patient, eeg in test_data.items():
        test_repr[patient] = model.encode(eeg, encoding_window='full_series' if train_labels.ndim == 1 else None)
    for patient, eeg in train_data_dict.items():
        train_repr_dict[patient] = model.encode(eeg, encoding_window='full_series' if train_labels.ndim == 1 else None)
    

    if eval_protocol == 'knn':
        fit_clf = eval_protocols.fit_knn
    elif eval_protocol == 'svm':
        fit_clf = eval_protocols.fit_svm
    elif eval_protocol == 'rf':
        fit_clf = eval_protocols.fit_random_forest
    else:
        assert False, 'unknown evaluation protocol'

    def merge_dim01(array):
        return array.reshape(array.shape[0]*array.shape[1], *array.shape[2:])

    if train_labels.ndim == 2:
        train_repr = merge_dim01(train_repr)
        train_labels = merge_dim01(train_labels)
        for patient, _ in test_repr.items():
            test_repr[patient] = merge_dim01(test_repr)
            test_labels[patient] = merge_dim01(test_labels)


    # Train the classifier on all the training data
    clf, best_params = fit_clf(train_repr, train_labels, random_seed=random_seed)
    
    best_p1, best_p2 = find_best_p1_p2_5fold(best_params, train_repr_dict, train_labels_dict, visual_labels, eval_protocol, random_seed)

    print('best_p1: ', best_p1, 'best_p2: ', best_p2)

    ### evaluate on the test data with the best p1 and p2 according to the training data
    # for the confusion matrix and ROC curve
    results_dict = {}
    for patient, test_repr_patient in test_repr.items():
        y_score = clf.predict_proba(test_repr_patient)
        results_dict[patient] = {'y_score': y_score[:, 1], 'true_label': test_labels[patient][0]} # class 1 probabilities

    per_patient_prediction = []
    per_patient_labels = []
    patients_wrongly_predicted = []
    for patient, data in results_dict.items():
        y_score = data['y_score']
        true_label = data['true_label']
        predictions = (y_score >= best_p1).astype(int) # per epoch thresholding with p1
        death_percentage = np.mean(predictions)
        patient_prediction = 1 if death_percentage >= best_p2 else 0 # per patient thresholding with p2
        per_patient_prediction.append(patient_prediction)
        per_patient_labels.append(true_label)
        if results_dict[patient]['true_label'] != patient_prediction:
            patients_wrongly_predicted.append(patient)
    # Accuracy for best p1 and p2 values
    acc = 1 - len(patients_wrongly_predicted) / len(results_dict)
    # Number of false positives for best p1 and p2 values
    number_of_fp = np.sum((np.array(per_patient_labels) != np.array(per_patient_prediction)) & (np.array(per_patient_prediction) == 1))
    print('accuracy: ', acc)
    print('number of false positives: ', number_of_fp)

    cm = confusion_matrix(per_patient_labels, per_patient_prediction)
    ppl = per_patient_labels
    ppp = per_patient_prediction

    sensitivity = cm[1, 1] / (cm[1, 1] + cm[1, 0]) if (cm[1, 1] + cm[1, 0]) > 0 else 0
    specificity = cm[0, 0] / (cm[0, 0] + cm[0, 1]) if (cm[0, 0] + cm[0, 1]) > 0 else 0
    precision = cm[1, 1] / (cm[1, 1] + cm[0, 1]) if (cm[1, 1] + cm[0, 1]) > 0 else 0
    F1_score = 2 * (precision * sensitivity) / (precision + sensitivity) if (precision + sensitivity) > 0 else 0

    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues')
    plt.xlabel('Predicted')
    plt.ylabel('True')
    if not nosave:
        plt.savefig(figure_save_location + f"ConfusionMatrix_Fold_{fold}.png", dpi=300, bbox_inches='tight')
        plt.close()
    else:
        plt.show()



    ### Completely recompute the test results for the ROC curve on the test data
    # (p1=best_p1, p2 is the curve variable)
    results_dict = {}
    for patient, test_repr_patient in test_repr.items():
        y_score = clf.predict_proba(test_repr_patient)[:, 1]  # Get class 1 probabilities
        results_dict[patient] = {'y_score': y_score, 'true_label': test_labels[patient][0]}

    p1_fixed = best_p1  # Fixing p1 at best_p1
    p2_values = np.linspace(0, 1, num=101)  # More granularity for p2

    per_patient_labels = []
    per_patient_prediction_percentages = []

    for patient, data in results_dict.items():
        y_score = data['y_score']
        true_label = data['true_label']

        predictions = (y_score >= best_p1).astype(int) # Classify epochs with p1
        death_percentage = np.mean(predictions) # Compute fraction of epochs classified as death

        per_patient_prediction_percentages.append(death_percentage)
        per_patient_labels.append(true_label)

    per_patient_labels = np.array(per_patient_labels)
    per_patient_prediction_percentages = np.array(per_patient_prediction_percentages)

    # Compute the ROC curve and AUC
    fpr, tpr, thresholds = roc_curve(per_patient_labels, per_patient_prediction_percentages)

    roc_auc = auc(fpr, tpr)

    # Save the ROC curve data
    roc_data = {
        'fpr': fpr,
        'tpr': tpr,
        'thresholds': thresholds,
        'auc': roc_auc
    }
    print("AUC: ", roc_auc)
    precision


    # Plot the new ROC curve
    plt.figure()
    plt.plot(fpr, tpr, color='blue', lw=2, label=f'ROC curve (AUC = {roc_auc:.2f})')
    plt.plot([0, 1], [0, 1], color='gray', linestyle='--')  # Diagonal reference line
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title(f'Receiver Operating Characteristic (ROC) Curve (p1={p1_fixed})')
    plt.legend(loc="lower right")

    if not nosave:
        plt.savefig(figure_save_location + f"ROC_Curve_Recomputed_fold_{fold}.png", dpi=300, bbox_inches='tight')
        plt.close()
    else:
        plt.show()
    
    print('Patients wrongly predicted ', patients_wrongly_predicted)
    eval_res = {'acc': acc, 'auc': roc_auc, 'sensitivity': sensitivity, 'specificity': specificity, 'precision': precision, 'F1_score': F1_score}
    
    return per_patient_prediction, eval_res, roc_data, ppl, ppp, best_p1, best_p2





def eval_classification_per_patient_multilabels(model, train_data, train_labels, test_data, test_labels, train_data_dict, train_labels_dict, figure_save_location, fold, eval_protocol='knn', nosave=False, vlabels=False, random_seed=42):

    # Compute represenation of all the shuffled training data (used to train the full classifier)
    train_repr = model.encode(train_data, encoding_window='full_series' if train_labels.ndim == 1 else None)
    
    # compute the representations of the test and training data dicionaries (sorted per patient)
    # dictionary format for the test data as we want to classify per patient
    # dictionary format for the training data as we use 5 fold cross-validation to find the best hyperparameters
    test_repr = {}
    train_repr_dict = {}
    for patient, eeg in test_data.items():
        test_repr[patient] = model.encode(eeg, encoding_window='full_series' if train_labels.ndim == 1 else None)
    for patient, eeg in train_data_dict.items():
        train_repr_dict[patient] = model.encode(eeg, encoding_window='full_series' if train_labels.ndim == 1 else None)

    # Select classifier
    if eval_protocol == 'knn':
        fit_clf = eval_protocols.fit_knn
    elif eval_protocol == 'svm':
        fit_clf = eval_protocols.fit_svm
    elif eval_protocol == 'rf':
        fit_clf = eval_protocols.fit_random_forest
    else:
        raise ValueError('Unknown evaluation protocol')

    # Merge dimensions if labels are multi-dimensional
    def merge_dim01(array):
        return array.reshape(array.shape[0] * array.shape[1], *array.shape[2:])

    if train_labels.ndim == 2:
        train_repr = merge_dim01(train_repr)
        train_labels = merge_dim01(train_labels)
        for patient in test_repr.keys():
            test_repr[patient] = merge_dim01(test_repr[patient])
            test_labels[patient] = merge_dim01(test_labels[patient])

    # Train classifier
    clf, _ = fit_clf(train_repr, train_labels, random_seed=random_seed)

    # Retrieve class labels from classifier
    class_labels = clf.classes_

    # Collect patient-level predictions
    per_patient_prediction = []
    per_patient_labels = []
    patients_wrongly_predicted = []

    for patient, test_repr_patient in test_repr.items():
        y_proba = clf.predict_proba(test_repr_patient)  # shape: (n_epochs, n_classes)
        epoch_indices = np.argmax(y_proba, axis=1)  # indices, not actual labels
        epoch_predictions = class_labels[epoch_indices]  # map back to actual labels

        # Majority vote across epochs
        majority_vote = Counter(epoch_predictions).most_common(1)[0][0]

        # For direct classification (y_true == y_pred)
        per_patient_prediction.append(majority_vote)
        per_patient_labels.append(test_labels[patient][0])  # true label of the patient (all epochs have same label)

        if majority_vote != test_labels[patient][0]:
            patients_wrongly_predicted.append(patient)
        

    per_patient_labels = np.array(per_patient_labels)
    per_patient_prediction = np.array(per_patient_prediction)
    ppl = per_patient_labels
    ppp = per_patient_prediction

    # Accuracy
    acc = np.mean(per_patient_labels == per_patient_prediction)
    print(f"Accuracy: {acc:.4f}")

    # Confusion matrix
    all_classes = [1,2,3,4,5,6,7] if vlabels else [1,2,3,4,5,6]
    cm = confusion_matrix(per_patient_labels, per_patient_prediction, labels=all_classes)

    plt.figure(figsize=(8, 6))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=all_classes, yticklabels=all_classes)
    plt.xlabel('Predicted')
    plt.ylabel('True')
    # plt.title('Confusion Matrix (Patient-Level)')
    if not nosave:
        plt.savefig(figure_save_location + f"ConfusionMatrix_Multiclass_fold_{fold}.png", dpi=300, bbox_inches='tight')
        plt.close()
    else:
        plt.show()

    # Classification report
    print("\nClassification Report:")
    print(classification_report(per_patient_labels, per_patient_prediction, labels=all_classes, zero_division=0))

    # Return results
    return per_patient_prediction, {
        'acc': acc,
        'patients_wrongly_predicted': patients_wrongly_predicted,
        'confusion_matrix': cm
    }, ppl, ppp