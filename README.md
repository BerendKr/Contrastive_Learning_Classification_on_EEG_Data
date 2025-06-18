# Contrastive_Learning_Classification_on_EEG_Data

This repository contains the code for the master's thesis of B.P.T.M. Krouwels, titled:

“Contrastive Learning for Predicting Neurological Outcome in Comatose Pediatric Patients After Cardiac Arrest”

The project leverages a TS2Vec encoder and a custom k-NN-based classification pipeline to perform patient-level outcome prediction from EEG data.

Read the full thesis here: [https://resolver.tudelft.nl/ce6d2e6f-6f28-4579-ba39-214da6dda375]


-----------------------------


To run open the repo, run the following command (replace --ARGS with arguments)

python train.py run_name --ARGS

followed by the arguments that you want (only run_name is required):


run_name (str) : name of the folder for saving model, outputs, and evaluation metrics (positional argument)

--gpu (int or str) : GPU device ID (e.g., 0); use 'cpu' to run on CPU (default: 'cpu')
--batch-size (int) : batch size used for training (default: 8)  
--lr (float) : learning rate (default: 0.001)  
--repr-dims (int) : dimensionality of the representation vector (default: 320)  
--max-train-length (int) : maximum sequence length; longer sequences are split (default: 3000)  
--iters (int or None) : number of training iterations (default: None)  
--epochs (int) : number of training epochs (default: 6)  
--save-every (int or None) : save a checkpoint every N iterations/epochs (default: None)  
--seed (int) : random seed for reproducibility (default: 42)  
--eval-protocol (str) : evaluation method: 'svm', 'knn', or 'linear' (default: 'knn')  
--max-threads (int or None) : maximum number of threads used by this process (default: None)
--scheduler (flag) : enable learning rate scheduler (default: off)  
--notrain (flag) : skip training; load existing model (default: off)  
--eval (flag) : run evaluation after training (default: off)  
--nosave (flag) : skip saving the model, output, evaluation metrics and figures (default: off)

More additional options/settings (paths, label selection and visualization) are in the train.py file on lines 57-83.

The models that were used for the thesis are provided in the folder /pretrained_models/ for folds 0 to 4 and random seed 42.
These can be loaded in the train.py file on line 146.

