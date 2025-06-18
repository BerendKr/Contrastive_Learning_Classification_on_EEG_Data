import matplotlib.pyplot as plt
import numpy as np
import torch
from matplotlib.colors import ListedColormap
from mpl_toolkits.mplot3d import Axes3D
from sklearn.manifold import TSNE
import seaborn as sns
import matplotlib.gridspec as gridspec
from matplotlib import cm

colors = cm.get_cmap('PuBu', 7)  # Blue-Purple colormap with x discrete colors
Color = colors(5) # set color for all plots

def compute_smoothgrad_saliency(model, input_ts, num_samples=50, noise_level=0.01, return_raw=False):
    """
    Compute SmoothGrad saliency from a TS2Vec encoder.

    Args:
        model: TS2Vec encoder
        input_ts: torch.Tensor of shape (1, T, C)
        num_samples: number of noisy samples
        noise_level: standard deviation of the Gaussian noise as a fraction of input std

    Returns:
        saliency_per_timestep: torch.Tensor of shape (T,) or (T,C)
    """
    if isinstance(input_ts, np.ndarray):
        input_ts = torch.from_numpy(input_ts).float()

    if input_ts.ndim == 2:
        input_ts = input_ts.unsqueeze(0)  # (1, T, C)

    input_ts = input_ts.to(model.device)
    device = input_ts.device

    saliency_accum = torch.zeros_like(input_ts, device=device)
    input_std = input_ts.std().item()

    model._net.eval()  # ensure model is in eval mode

    for _ in range(num_samples):
        noise = torch.randn_like(input_ts) * input_std * noise_level
        noisy_input = (input_ts + noise).clone().detach().requires_grad_(True)

        z = model._net(noisy_input)       # (1, T, R)
        z_mean = z.mean(dim=1)            # (1, R)

        # If you want to use the norm of the output as target, uncomment line below and comment the next block
        summary = z.norm()            # scalar output

        # # If you want to use a random direction vector as target, uncomment block below, otherwise comment it
        # if 'direction' not in locals():
        #     direction = torch.randn_like(z_mean).detach()  # (1, R)
        #     direction = direction / direction.norm()  # Normalize the direction vector
        # summary = torch.sum(z_mean * direction)            # scalar output


        model._net.zero_grad()
        summary.backward()

        saliency_accum += noisy_input.grad.abs()

    
    # Average gradients
    saliency_final = saliency_accum.squeeze(0) / num_samples # shape: (T, C)

    if return_raw:
        return saliency_final.detach().cpu()
    else:
        return saliency_final.mean(1).detach().cpu() # shape (T,)



def plot_saliency_per_channel(input_ts, saliency_raw, highlight_top_salient_regions=True, top_percentage=0.10, per_channel=False):
    """
    Plots saliency map for each channel separately over time. or as overlay on the EEG signal.
    """
    if highlight_top_salient_regions:
        if isinstance(input_ts, torch.Tensor):
            input_ts = input_ts.squeeze().cpu().numpy()  # (T, C)
        if isinstance(saliency_raw, torch.Tensor):
            saliency_raw = saliency_raw.cpu().numpy()

        # input_ts = input_ts[:1000] # Limit to 10 seconds of data
        T, C = input_ts.shape
        time_axis = np.arange(T) / 100  # Assuming a sampling rate of 100 Hz
        fig, axs = plt.subplots(C, 1, figsize=(14, 2.5 * C), sharex=True)

        if not per_channel:
            threshold = np.quantile(saliency_raw.flatten(), 1 - top_percentage)
            threshold_strong = np.quantile(saliency_raw.flatten(), 1 - 0.01)

        channels = {0:'C3' ,1:'C4' ,2:'F7' ,3:'F8'}
        for c in range(C):

            signal = input_ts[:, c]
            saliency = saliency_raw[:, c]

            line1, = axs[c].plot(time_axis, signal, label=f'Channel {channels[c]}', linewidth=0.6, color=Color)  # Use 'Color' if defined
            axs[c].set_ylabel('Amplitude')

            if per_channel:
                threshold = np.quantile(saliency, 1 - top_percentage)
                threshold_strong = np.quantile(saliency, 0.99)
            
            # print("channel: ", c, 'threshold saliency 90%', threshold)
            salient_indices_strong = np.where(saliency >= threshold_strong)[0]
            normal_indices = np.where((saliency >= threshold) & (saliency < threshold_strong))[0]
            # for idx in normal_indices: #  uncomment these 4 lines for highlights
            #     axs[c].axvspan(idx / 100, (idx + 1) / 100, color='yellow', alpha=0.2)
            # for idx in salient_indices_strong:
            #     axs[c].axvspan(idx / 100, (idx + 1) / 100, color='orange', alpha=0.2)
            
            # Create secondary y-axis for saliency
            ax_sal = axs[c].twinx()
            

            # Global saliency max for consistent scaling
            global_saliency_max = np.max(saliency_raw)

            # Plot the saliency on the same time axis
            line2, = ax_sal.plot(time_axis, saliency, color='red', alpha=0.8, linewidth=0.8, label='Saliency')
            ax_sal.set_ylim(0, global_saliency_max)
            # ax_sal.set_yticks([])  # hide y-ticks for cleaner look
            
            # For checking correlation between signal and saliency
            correlation = np.corrcoef(signal, saliency)[0, 1]
            print("Correlation between signal and saliency:", correlation)
            correlation = np.corrcoef(np.absolute(signal), saliency)[0, 1]
            print("Correlation between absolute signal and saliency:", correlation)

            
            axs[c].legend(handles=[line1, line2], loc='upper right')

        axs[-1].set_xlabel('Time (s)')
        global_min = signal.min()
        global_max = signal.max()
        global_absolute = max(abs(global_min), abs(global_max))
        margin = 0.20 * (global_max - global_min)  # margin
        for ax in axs:
            ax.set_ylim(-global_absolute - margin, global_absolute + margin)
        plt.tight_layout(rect=[0, 0.03, 1, 0.95])
        plt.show()

    else:
        if saliency_raw.shape[0] < saliency_raw.shape[1]:  # Make sure shape is (T, C)
            saliency_raw = saliency_raw.T

        T, C = saliency_raw.shape

    return saliency_raw



def saliency(model, device, test_data, example_patient_number=0, example_epoch_number=0, num_samples=50, noise_level=0.1, highlight_top_salient_regions=True, top_salient_regions_range=0.10, per_channel=False, random_seed=42):
    '''
    Computes and visualizes the saliency map for a given patient and epoch.
    Parameters:
    - model: TS2Vec encoder model
    - device: torch device (e.g., 'cuda' or 'cpu')
    - test_data: dictionary containing test data with patient IDs as keys and epochs as values
    - example_patient_number: index of the patient to visualize
    - example_epoch_number: index of the epoch to visualize
    - num_samples: number of noisy samples for SmoothGrad
    - noise_level: standard deviation of the Gaussian noise as a fraction of input std
    - highlight_top_salient_regions: whether to highlight the top salient regions
    - top_salient_regions_range: percentage of top salient regions to highlight
    - per_channel: whether to plot saliency per channel or as an overlay on the EEG signal
    - random_seed: random seed for reproducibility
    '''
    example_patient = list(test_data.keys())[example_patient_number]
    print(f'Computing saliency for patient {example_patient}, epoch {example_epoch_number}')
    example_epoch = test_data[example_patient][example_epoch_number][:,:]  # just one sample (T, C)
    input_ts = torch.tensor(example_epoch, dtype=torch.float32).to(device)  # (1, T, C)
    input_ts = (input_ts - input_ts.mean(dim=0, keepdim=True)) / input_ts.std(dim=0, keepdim=True)
    saliency_map_smooth_raw = compute_smoothgrad_saliency(model, input_ts, num_samples, noise_level, return_raw=True)

    # SmoothGrad saliency per channel (Best one, use for paper)
    plot_saliency_per_channel(input_ts, saliency_map_smooth_raw, highlight_top_salient_regions, top_salient_regions_range, per_channel)



def tsne_plot(all_repr, all_labels, binary_labels, visual_labels, fold, dimension_tsne, non_neur_deaths_green, all_patients_visualized, random_seed):
    '''
    Plots a t-SNE visualization of the learned representations.
    Parameters:
    - train_repr: np.ndarray of shape (N, d), representations for training set
    - train_labels: np.ndarray of shape (N,), labels for training set
    - test_repr: np.ndarray of shape (M, d), representations for test set
    - test_labels: np.ndarray of shape (M,), labels for test set
    - all_repr: np.ndarray of shape (N+M, d), concatenated representations for both sets
    - all_labels: np.ndarray of shape (N+M,), concatenated labels for both sets
    - binary_labels: bool, whether the labels are binary
    - visual_labels: bool, whether to use custom colors for visualization
    - dimension_tsne: int, either 2 or 3 for the t-SNE dimensionality reduction
    - random_seed: int, random seed for reproducibility
    '''
    def merge_dim01(array):
        return array.reshape(array.shape[0]*array.shape[1], *array.shape[2:])

    if all_labels.ndim == 2:
        all_repr = merge_dim01(all_repr)
        all_labels = merge_dim01(all_labels)

    if binary_labels:
        custom_colors = [
            "#0000aa", # blue dark
            "#d81416"  # red
        ]
        if non_neur_deaths_green:
            custom_colors = [ # temporary, uncomment for non neurological deaths (3 different labels)
                "#0000aa", # blue dark
                "#d81416", # red
                "#22bb44"  # green
            ]
    else:
        if visual_labels:
            custom_colors = [
            "#0000aa",  # blue dark
            "#1e90ff",  # blue
            "#00fa9a",  # turquoise
            "#eeba99",  # pink/sand
            "#ffd700",  # gold
            "#c71585",  # violet
            "#d73027"   # deep red
        ]

        else: #PCPC12m labels
            custom_colors = [
                "#0000aa",  # blue dark
                "#1e90ff",  # blue
                "#00fa9a",  # turquoise
                "#eeba99",  # pink/sand
                "#000000",  # black
                "#d73027"   # deep red
            ]
        
    if all_patients_visualized: # overrides other colors if all patients are visualized
        custom_colors = [
            "#696969", "#a9a9a9", "#d3d3d3", "#2f4f4f", "#556b2f", "#6b8e23", "#a0522d", "#a52a2a",
            "#2e8b57", "#191970", "#006400", "#8b0000", "#808000", "#483d8b", "#5f9ea0", "#778899",
            "#008000", "#3cb371", "#bc8f8f", "#663399", "#008080", "#bdb76b", "#cd853f", "#4682b4",
            "#d2691e", "#9acd32", "#20b2aa", "#cd5c5c", "#00008b", "#4b0082", "#32cd32", "#daa520",
            "#8fbc8f", "#800080", "#b03060", "#d2b48c", "#66cdaa", "#9932cc", "#ff0000", "#ff4500",
            "#ff8c00", "#ffa500", "#ffd700", "#ffff00", "#c71585", "#0000cd", "#7cfc00", "#40e0d0",
            "#00ff00", "#ba55d3", "#00fa9a", "#00ff7f", "#4169e1", "#dc143c", "#00ffff", "#00bfff",
            "#9370db", "#0000ff", "#a020f0", "#adff2f", "#ff6347", "#d8bfd8", "#b0c4de", "#ff7f50",
            "#ff00ff", "#1e90ff", "#db7093", "#f0e68c", "#fa8072", "#eee8aa", "#ffff54", "#6495ed",
            "#dda0dd", "#90ee90", "#ff1493", "#7b68ee", "#ffa07a", "#afeeee", "#ee82ee", "#87cefa",
            "#7fffd4", "#ff69b4", "#ffe4c4", "#ffc0cb"
        ]
    
    cmap_custom = ListedColormap(custom_colors)


    if dimension_tsne == 3:
        tsne = TSNE(n_components=3, random_state=random_seed, perplexity=30)
        tsne_results = tsne.fit_transform(all_repr)
        fig = plt.figure(figsize=(10, 8))
        ax = fig.add_subplot(111, projection='3d') #3d
        scatter = ax.scatter(tsne_results[:, 0], tsne_results[:, 1], tsne_results[:, 2], c=all_labels, cmap=cmap_custom, s=5, alpha=0.7)
        # ax.set_title('t-SNE Visualization')
        fig.colorbar(scatter)
        plt.show()

    if dimension_tsne == 2:
        tsne = TSNE(n_components=2, random_state=random_seed, perplexity=30)
        tsne_results = tsne.fit_transform(all_repr)
        plt.figure(figsize=(10, 8))
        scatter = plt.scatter(tsne_results[:, 0], tsne_results[:, 1], c=all_labels.astype(float).tolist(), cmap=cmap_custom, s=5, alpha=0.7)
        # plt.title('t-SNE Visualization')
        plt.colorbar(scatter)
        plt.show()
    


def feature_heatmap(test_data, test_labels, example_patient_number, example_epoch_number, model, channel_labels=['C3', 'C4', 'F7', 'F8']):
    """
    Plots a multichannel EEG time series (stacked vertically) and a heatmap of the top 16 TS2Vec latent dimensions by variance.

    Parameters:
    - test_data: dict of {patient_id: [epochs]} where each epoch is shape (T, C)
    - example_patient_number: index into list(test_data.keys())
    - example_epoch_number: which epoch from that patient to visualize
    - model: TS2Vec-like encoder model with .encode(x, encoding_window=...) returning (1, T, d)
    - channel_labels: optional list of EEG channel names (default: ['C3', 'C4', 'F7', 'F8'])
    """

    # Select and prepare data
    example_patient = list(test_data.keys())[example_patient_number]
    example_epoch = test_data[example_patient][example_epoch_number]  # shape: (T, C)
    x_for_model = np.expand_dims(example_epoch, axis=0)  # (1, T, C) for encoder
    x_for_plot = example_epoch  # (T, C) for plotting
    example_patient_label = test_labels[example_patient]
    print(f'Computing saliency for patient {example_patient}, PCPC label {example_patient_label[0]}, epoch {example_epoch_number}')

    # Encode to get (1, T, d) representation
    repr = model.encode(x_for_model, encoding_window=1) # encoding_window = 1
    print('representation shape:', repr.shape)
    repr = repr.squeeze(0)  # (T, d)

    # Select top 16 latent dimensions by variance
    variances = np.var(repr, axis=0)
    top_indices = np.argsort(variances)[-16:][::-1]
    repr_top = repr[:, top_indices]  # (T, 16)

    T, C = x_for_plot.shape
    time = np.arange(T)
    xlim = (0, T - 1)

    # Layout: C EEG plots + 1 heatmap row
    fig = plt.figure(figsize=(14, 1.2 * C + 4))
    fig.subplots_adjust(left=0.1)  # ensures enough space for consistent y-labels
    gs = gridspec.GridSpec(C + 1, 2, width_ratios=[20, 1], height_ratios=[0.7] * C + [3])

    # EEG channel plots (stacked vertically)
    y_min = np.min(x_for_plot)
    y_max = np.max(x_for_plot)

    for i in range(C):
        ax = plt.subplot(gs[i, 0])
        label = channel_labels[i] if channel_labels else f'Ch {i+1}'
        ax.plot(time, x_for_plot[:, i], color=Color, linewidth=0.5)
        ax.set_ylabel(label, rotation=0, labelpad=30)
        ax.set_xlim(xlim)
        ax.set_ylim(y_min, y_max)  # consistent scaling across all channels
        ax.margins(x=0)
        ax.set_xticks([])
        ax.set_xticklabels([])
        ax.spines['top'].set_visible(False)
        ax.spines['bottom'].set_visible(False)

    # Heatmap of latent features
    ax_hm = plt.subplot(gs[C, 0])
    cbar_ax = plt.subplot(gs[C, 1])
    sns.heatmap(repr_top.T, ax=ax_hm, cmap='viridis', cbar=True, cbar_ax=cbar_ax)
    ax_hm.set_ylabel('Latent Dim (Top 16)')
    ax_hm.set_xlabel('Time step')
    ax_hm.set_xlim(xlim)
    ax_hm.margins(x=0)

    step = 400
    xticks = np.arange(0, T + 1, step)
    ax_hm.set_xticks(xticks)
    ax_hm.set_xticklabels([str(t) for t in xticks])

    plt.subplots_adjust(hspace=0.0, wspace=0.025, bottom=0.08)
    plt.tight_layout()
    plt.show()
