import os
import matplotlib.pyplot as plt
import numpy as np
import argparse
import time
import datetime
from ts2vec import TS2Vec
import tasks
import datautils
from utils import init_dl_program, name_with_datetime
import matplotlib.pyplot as plt
import visualizations
import seaborn as sns
from sklearn.metrics import auc
from sklearn.metrics import confusion_matrix


def save_checkpoint_callback(
    save_every=1,
    unit='epoch'
):
    assert unit in ('epoch', 'iter')
    def callback(model, loss):
        n = model.n_epochs if unit == 'epoch' else model.n_iters
        if n % save_every == 0:
            model.save(f'{run_dir}/model_{n}.pkl')
    return callback

if __name__ == '__main__':

    def str_or_int(value):
        try:
            return int(value)
        except ValueError:
            return value
    
    parser = argparse.ArgumentParser()
    parser.add_argument('run_name', help='The folder name used to save model, output and evaluation metrics. This can be set to any word')
    parser.add_argument('--gpu', type=str_or_int, default='cpu', help='The gpu no. used for training and inference (defaults to 0) (for cpu do cpu, for gpu do 0')
    parser.add_argument('--batch-size', type=int, default=8, help='The batch size (defaults to 8)')
    parser.add_argument('--lr', type=float, default=0.001, help='The learning rate (defaults to 0.001)')
    parser.add_argument('--scheduler', action="store_true", help='Whether to use the learning rate scheduler')
    parser.add_argument('--repr-dims', type=int, default=320, help='The representation dimension (defaults to 320)')
    parser.add_argument('--max-train-length', type=int, default=3000, help='For sequence with a length greater than <max_train_length>, it would be cropped into some sequences, each of which has a length less than <max_train_length> (defaults to 3000)')
    parser.add_argument('--iters', type=int, default=None, help='The number of iterations')
    parser.add_argument('--epochs', type=int, default=6, help='The number of epochs')
    parser.add_argument('--save-every', type=int, default=None, help='Save the checkpoint every <save_every> iterations/epochs')
    parser.add_argument('--seed', type=int, default=42, help='The random seed')
    parser.add_argument('--eval-protocol', type=str, default='knn', help='The evaluation protocol used for evaluation. This can be set to, svm, knn or linear (not so good).')
    parser.add_argument('--max-threads', type=int, default=None, help='The maximum allowed number of threads used by this process')
    parser.add_argument('--notrain', action="store_true", help='Whether to skip training of the encoder, loading existing model')
    parser.add_argument('--eval', action="store_true", help='Whether to perform evaluation after training')
    parser.add_argument('--nosave', action="store_true", help='Whether to skip saving the model, output and evaluation metrics')
    args = parser.parse_args()
    

    visual_labels = False # True for visual labels, False for PCPC12m labels
    binary_labels = True # True for binary labels, False for multilabels
    n_folds = 5
    
    # Paths
    data_path = "./data/" + "EEG_Timeseries_data_100_4D_C34_F78_vlabel_normalized_84.pkl"
    save_location = "./output/" + f"/final_seed_{args.seed}/"

    # t-SNE map of the encoder (per epoch) in 2D or 3D
    tsne_ = False
    non_neur_deaths_green = False # requires binary PCPC labels to work
    all_patients_visualized = False # give each patient their own color
    dimension_tsne = 2 # 2 or 3 dimensions

    # Saliency maps
    saliency = False
    highlight_top_salient_regions = True  # Whether to have full salient graph or only the highlights
    per_channel = False
    top_salient_regions_range = 0.1 # Percentage of the signal to highlight
    num_samples = 50 # Nr samples for the smooth saliency
    noise_level = 0.05 # Noise level for the smooth saliency
    per_channel = False
    example_patient_number = 10
    example_epoch_number = 0

    # Feature heatmap (uses the same patient as saliency)
    feature_heatmap = False




    print("Arguments:", str(args))
    
    # Create the run directory to save the model and output
    if not args.nosave:
        run_dir = save_location + '/training/' + name_with_datetime(args.run_name)
        os.makedirs(save_location, exist_ok=True)
        os.makedirs(run_dir, exist_ok=True)


    # These lists will be used to store the results of the evaluation
    # for each fold, and then averaged over all folds
    roc_list = []
    ppl_list = np.array([])
    ppp_list = np.array([])
    p1_list = []
    p2_list = []
    eval_res_total = {'acc': [], 'auc': [], 'sensitivity': [], 'specificity': [], 'precision': [], 'F1_score': []}


    ### Start the 5 fold cross validation (train-test) ###
    for fold in range(n_folds):
        print("\n Starting fold: ", fold)
        ### First train encoder on all the data ###
        device = init_dl_program(args.gpu, seed=args.seed, max_threads=args.max_threads)

        # Load the data
        train_data, train_labels, test_data, test_labels, train_data_dict, train_labels_dict = datautils.load_EEG_per_patient(data_path, fold, args.seed, binary_labels, visual_labels, n_folds)
        # train data is an ndarray, test data is a dict with patient ids as keys and ndarrays as values
            

        config = dict(
            batch_size=args.batch_size,
            lr=args.lr,
            output_dims=args.repr_dims,
            hidden_dims=64,
            depth=10,
            max_train_length=args.max_train_length
        )
        
        if args.save_every is not None:
            unit = 'epoch' if args.epochs is not None else 'iter'
            config[f'after_{unit}_callback'] = save_checkpoint_callback(args.save_every, unit)
        
        # Keep time to report training time
        t = time.time()

        # Initialize the model
        model = TS2Vec(
            input_dims=train_data.shape[-1],
            device=device,
            **config
        )

        # Load a model if --notrain is set
        if args.notrain:
            # Specify model to load here for pretrained model (fold and seed sensitive)
            print('Skipping training of the encoder, loading existing model.')
            if args.seed == 42:
                model.load(f'./pretrained_models/model_seed_{args.seed}_fold_{fold}.pkl')
                print(f'Model loaded from pretrained_models/model_seed_{args.seed}_fold_{fold}.pkl')
            else:
                raise ValueError(f"No pretrained model for seed {args.seed}. Please set --notrain to False to train the model.")

        else:
            # Train your model here
            all_data = np.concatenate( ( train_data, np.concatenate(list(test_data.values()) , axis=0) ), axis=0)
            print('all_data.shape: ', all_data.shape)
            loss_log, optimizer = model.fit(
                all_data,
                use_scheduler=args.scheduler,
                n_epochs=args.epochs,
                n_iters=args.iters,
                verbose=True
            )
            if not args.nosave:
                model.save(f'{run_dir}/model_fold_{fold}.pkl', optimizer=optimizer)


        t1 = time.time() - t
        print(f"\nTraining time encoder: {datetime.timedelta(seconds=t1)}\n")



    
        # Evaluate / classification
        if args.eval:
            if binary_labels:
                out, eval_res, roc_data, ppl, ppp, best_p1, best_p2 = tasks.eval_classification_per_patient(model, train_data, train_labels, test_data, test_labels, train_data_dict, train_labels_dict, save_location, fold, visual_labels, args.eval_protocol, args.nosave, args.seed)
                roc_list.append(roc_data)
                ppl_list = np.concatenate([ppl_list, ppl])
                ppp_list = np.concatenate([ppp_list, ppp])
                p1_list.append(best_p1)
                p2_list.append(best_p2)
                for key, item in eval_res.items():
                    eval_res_total[key].append(item)
            else:
                out, eval_res, ppl, ppp = tasks.eval_classification_per_patient_multilabels(model, train_data, train_labels, test_data, test_labels, train_data_dict, train_labels_dict, save_location, fold, args.eval_protocol, args.nosave, visual_labels, args.seed)
                ppl_list = np.concatenate([ppl_list, ppl])
                ppp_list = np.concatenate([ppp_list, ppp])
            
            t2 = time.time() - t - t1
            print(f"\nEval time: {datetime.timedelta(seconds=t2)}\n")
            
    # Compute average results over all folds
    if args.eval:
        eval_res_mean = {k: np.mean(v) for k, v in eval_res_total.items()}
        eval_res_std_dev = {k: np.std(v) for k, v in eval_res_total.items()}
        print("Evaluation results: ")
        for key in eval_res_mean.keys():
            print(f" - {key}: {eval_res_mean[key]:.4f} ± {eval_res_std_dev[key]:.4f}")
        




    ##### Visualizations #####

    ### t-SNE ###
    if tsne_:
        # Load all data through the model
        # Last fold is used by default, for other folds, skip the folds that come after
        # (so for fold 2, skip fold 3 and 4 in the above loop.)
        train_data, train_labels, test_data, test_labels, train_data_dict, train_labels_dict = datautils.load_EEG_per_patient(data_path, fold, args.seed, binary_labels, visual_labels, n_folds, non_neur_deaths_green, all_patients_visualized)
        train_repr = model.encode(train_data, encoding_window='full_series' if train_labels.ndim == 1 else None)
        test_data = np.concatenate(list(test_data.values()), axis=0)
        test_repr = model.encode(test_data, encoding_window='full_series' if train_labels.ndim == 1 else None)
        all_repr = np.concatenate((train_repr, test_repr), axis=0)
        test_labels_flat = np.concatenate(list(test_labels.values()), axis=0)
        all_labels = np.concatenate((train_labels, test_labels_flat), axis=0)

        visualizations.tsne_plot(all_repr,
                                all_labels,
                                binary_labels,
                                visual_labels,
                                fold,
                                dimension_tsne,
                                non_neur_deaths_green,
                                all_patients_visualized,
                                args.seed)


    ### Saliency maps ###
    if saliency:
        visualizations.saliency(model, device, test_data,
                                example_patient_number,
                                example_epoch_number,
                                num_samples,
                                noise_level,
                                highlight_top_salient_regions,
                                top_salient_regions_range,
                                per_channel,
                                args.seed)


    ### Feature heatmap ###
    if feature_heatmap:
        visualizations.feature_heatmap(test_data,
                                       test_labels,
                                       example_patient_number,
                                       example_epoch_number,
                                       model,
                                       channel_labels=['C3','C4','F7','F8'])




    ##############################################################################
    ### get the average ROC and Confusion Matrix over all folds for binary labels only ###
    if args.eval:
        if binary_labels:
            
            ### Plot ROC Curve, average over all folds with all separate folds ###
            plt.figure(figsize=(8, 6))
            fpr_common = np.linspace(0, 1, 100)
            tpr_interp = []
            for roc in roc_list:
                tpr = np.interp(fpr_common, roc['fpr'], roc['tpr'])
                tpr_interp.append(tpr)
                plt.plot(fpr_common, tpr, color='lightblue', alpha=1, linewidth=1)
            tpr_interp = np.array(tpr_interp)
            mean_tpr = np.mean(tpr_interp, axis=0)
            # std_tpr = np.std(tpr_interp, axis=0) # for ±1 std dev
            mean_auc = auc(fpr_common, mean_tpr)
            
            plt.plot(fpr_common, mean_tpr, label=f"AUC = {mean_auc:.3f}", color="blue")
            # plt.fill_between(fpr_common, mean_tpr - std_tpr, mean_tpr + std_tpr, color="gray", alpha=0.2) # for ±1 std dev
            plt.plot([0, 1], [0, 1], linestyle='--', color='gray', label='Random Guess')
            plt.xlabel('False Positive Rate')
            plt.ylabel('True Positive Rate')
            # plt.title('Average ROC Curves with ±1 Std Dev')
            plt.legend(loc='lower right')
            plt.grid(True)
            plt.tight_layout()
            if not args.nosave:
                plt.savefig(save_location + f"roc_curve_data_seed_{args.seed}.png", dpi=300, bbox_inches='tight')
                plt.close()
            else:
                plt.show()

            print("p1's found in the folds: ", p1_list, "p2's found in the folds: ", p2_list)



        ### Confusion Matrix over all folds ###
        if binary_labels:
            all_classes = [1,'other'] if visual_labels else [0,1]
        else:
            all_classes = [1,2,3,4,5,6,7] if visual_labels else [1,2,3,4,5,6]
        cm = confusion_matrix(ppl_list, ppp_list, labels=[0,1] if visual_labels and binary_labels else all_classes)
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=all_classes, yticklabels=all_classes)
        plt.xlabel('Predicted')
        plt.ylabel('True')
        if not args.nosave:
            plt.savefig(save_location + f"ConfusionMatrix_full_seed_{args.seed}.png", dpi=300, bbox_inches='tight')
            plt.close()
        else:
            plt.show()


    print("Finished.")
