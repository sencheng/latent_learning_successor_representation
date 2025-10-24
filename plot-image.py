import os
import pickle
import matplotlib.pyplot as plt
from matplotlib.collections import PatchCollection
import seaborn as sns
from scipy.stats import sem
from sklearn.metrics.pairwise import cosine_similarity
import numpy as np

def load_data(data_folder):
    """
    Load all .pickle files from the specified folder.

    Parameters
    ----------
    data_folder : str
        Path to the folder containing the .pickle files.

    Returns
    ----------
    data : list
        A list of dictionaries containing the data from each run.
    """
    data = []
    for file_name in sorted(os.listdir(data_folder)):
        if file_name.endswith('.pickle'):
            file_path = os.path.join(data_folder, file_name)
            with open(file_path, 'rb') as handle:
                data.append(pickle.load(handle))
    return data

def plot_observation_similarity(data, output_folder):
    """
    Plot the cosine similarity of one observation to all other observations.

    Parameters
    ----------
    data : dict
        A dictionary containing the observations.
    observation_index : int
        The index of the observation to compare against all others.
    output_folder : str
        Path to the folder where the plot will be saved.
    """
    # Extract observations
    observations = np.array(data['Observations'])

    # Compute the cosine similarity matrix
    similarity_matrix = cosine_similarity(observations)

    # Plot the heatmap
    plt.figure(figsize=(10, 8))
    sns.heatmap(similarity_matrix, cmap='viridis', annot=False, cbar=True)
    plt.title('Cosine Similarity Heatmap of Observations', fontsize=25)
    plt.xlabel('Observation Index', fontsize=25)
    plt.ylabel('Observation Index', fontsize=25)
    # Change the size of the tick labels
    plt.xticks(fontsize=10)
    plt.yticks(fontsize=10)
    # Save the heatmap
    os.makedirs(output_folder, exist_ok=True)
    plot_path = os.path.join(output_folder, 'cosine_similarity_heatmap.png')
    plt.savefig(plot_path)
    print(f'Heatmap saved to {plot_path}')
    plt.show()

    
def plot_sf(paradigm, data_root_folder, trial, output_folder):
    # data_path = data_path +'/run_3.pickle'
    data_path = os.path.join(data_root_folder, paradigm)
    num_states, num_acts = 100, 4
    
    print(f'Loading successor features data for paradigm "{paradigm}" in trial "{trial}"...')
    # Load data
    mean_sr, obs = load_sf_data(data_path, trial)

    C = []
    for act in range(num_acts):
        inverse = np.linalg.inv(np.matmul(obs, obs.T))
        C.append(np.matmul(np.matmul(mean_sr[:num_states, act, :], obs.T), inverse))
    C = np.array(C)

    for act in range(4):
        
        fig, axs = plt.subplots(figsize=(17, 12))
        im = axs.imshow(C[act][:], vmax=2.0)
        # im = axs.imshow(C[act][5].reshape(10,10),vmax=2.0)
        axs.set_title('act: %s'%act)
        axs.set_xlabel('state idx')
        axs.set_ylabel('state idx')

        # Add color bar
        cbar = fig.colorbar(im, ax=axs) # , fraction=0.046, pad=0.04)
        cbar.ax.tick_params(labelsize=16)
        cbar.set_label('Deep SR', fontsize=18)
        # Adjust the y-axis label font size
        axs.figure.axes[-1].yaxis.label.set_size(20)

    cos_sim = cosine_similarity(obs, obs)
    axs.imshow(cos_sim, vmin = 0, vmax = 1.0)

    plt.show()

def plot_cosine(paradigm, data_root_folder, trial):
    '''
    Plot the cosine similarity between different paradigms.
    paradigm : list
        List of paradigms to compare. E.g., ['direct', 'targeted', 'continuous']
    data_root_folder : str
        Path to the root folder containing the data.
    trial : int
        The trial number to analyze.
    '''
    num_states, num_acts = 100, 4
    
    data_path = os.path.join(data_root_folder, paradigm[0])
    SR_direct, observation = load_sf_data(data_path, trial)
    C_direct = []
    for act in range(num_acts):
        inverse = np.linalg.inv(np.matmul(observation, observation.T))
        C_direct.append(np.matmul(np.matmul(SR_direct[:num_states, act, :], observation.T), inverse))
    C_direct = np.array(C_direct)

    data_path = os.path.join(data_root_folder, paradigm[1])
    SR_targeted, observation = load_sf_data(data_path, trial)
    C_targeted = []
    for act in range(num_acts):
        inverse = np.linalg.inv(np.matmul(observation, observation.T))
        C_targeted.append(np.matmul(np.matmul(SR_targeted[:num_states, act, :], observation.T), inverse))
    C_targeted = np.array(C_targeted)
    
    data_path = os.path.join(data_root_folder, paradigm[2])
    SR_continuous, observation = load_sf_data(data_path, trial)
    C_continuous = []
    for act in range(num_acts):
        inverse = np.linalg.inv(np.matmul(observation, observation.T))
        C_continuous.append(np.matmul(np.matmul(SR_continuous[:num_states, act, :], observation.T), inverse))
    C_continuous = np.array(C_continuous)


    for action in range(4):
        
        targeted_similarity_total = []
        continuous_similarity_total = []
        
        similarity = []
        similarity_continuous = []
    
        for state in range(num_states):
            sr_row_direct = C_direct[action][state]
            sr_row_targeted = C_targeted[action][state]
            sr_row_continuous = C_continuous[action][state]
            
            similarity.append(cosine_similarity(sr_row_targeted.reshape(1, -1), sr_row_direct.reshape(1, -1)))
            similarity_continuous.append(cosine_similarity(sr_row_continuous.reshape(1, -1), sr_row_direct.reshape(1, -1)))
    
        targeted_similarity_total.append(similarity)
        continuous_similarity_total.append(similarity_continuous)

        targeted_similarity_total = np.array(targeted_similarity_total)
        targeted_mean = targeted_similarity_total.mean(0)
        targeted_std = sem(targeted_similarity_total)

        continuous_similarity_total = np.array(continuous_similarity_total)
        continuous_mean = continuous_similarity_total.mean(0)
        continuous_std = sem(continuous_similarity_total)

        fig, ax = plt.subplots(figsize=(25, 12))
        plt.errorbar(range(num_states), targeted_mean[0, :, 0], yerr=targeted_std[:, 0, 0], linewidth=2, color = 'darkorange', linestyle='-', markersize=10, marker='^', capsize=10, label='Latent Learning - targeted pre-exposure')
        plt.errorbar(range(num_states), continuous_mean[0, :, 0], yerr=continuous_std[:, 0, 0], linewidth=2, color = 'purple', markersize=10, linestyle='--', marker='o', capsize=10, label='Latent Learning - continuous pre-exposure')
        
        ax.set_xticks(range(0, num_states, 5))
        ax.set_xticklabels(range(0, num_states, 5))
        legend = ax.legend(loc='upper left', shadow=True, fontsize='large')
        plt.ylabel('Similarity Score', fontsize=20, labelpad=20)
        plt.xlabel('State', fontsize=20, labelpad=20)
        

    plt.show()

def load_escape_data(data_folder):
    """
    Load escape data from the specified folder.

    Parameters
    ----------
    data_folder : str
        Path to the folder containing the .pickle files.

    Returns
    ----------
    data : list
        A list of dictionaries containing the escape data from each run.
    """
    all_escapes = []
    for file_name in sorted(os.listdir(data_folder)):
        if file_name.endswith('.pickle'):
            file_path = os.path.join(data_folder, file_name)
            print(f'Loading successor features data from {file_name}...')
            with open(file_path, 'rb') as handle:
                data = pickle.load(handle)
                all_escapes.append([len(x) for x in data['Escape']])
                    
    all_escapes = np.array(all_escapes)
    mean_escape = np.mean(all_escapes, axis=0)
    sem_escape = sem(all_escapes)
                
    return mean_escape, sem_escape

def load_success_data(data_folder):
    """
    Load escape data from the specified folder.

    Parameters
    ----------
    data_folder : str
        Path to the folder containing the .pickle files.

    Returns
    ----------
    data : list
        A list of dictionaries containing the escape data from each run.
    """
    all_escapes = []
    for file_name in sorted(os.listdir(data_folder)):
        if file_name.endswith('.pickle'):
            file_path = os.path.join(data_folder, file_name)
            print(f'Loading successor features data from {file_name}...')
            with open(file_path, 'rb') as handle:
                data = pickle.load(handle)
                
                success = []
                for steps in data['Escape']:
                    if len(steps) < 150:
                        success.append(1)
                    else:
                        success.append(0)

                all_escapes.append(success)
                    
    all_escapes = np.array(all_escapes)
    mean_escape = np.mean(all_escapes, axis=0)
    sem_escape = sem(all_escapes)
                
    return mean_escape, sem_escape

def plot_escape_latency(paradigms, data_root_folder, output_folder, labels, lines):
    """
    Plot the escape latency for each paradigm.

    Parameters
    ----------
    paradigms : list
        List of paradigms to plot.
    data_root_folder : str
        Path to the root folder containing the data.
    output_folder : str
        Path to the folder where the plot will be saved.
    """
    fig, ax = plt.subplots(figsize=(17, 12))

    for paradigm,s_label, line in zip(paradigms, labels, lines):
        data_folder = os.path.join(data_root_folder, paradigm)
        if not os.path.exists(data_folder):
            print(f'Data folder for paradigm "{paradigm}" does not exist. Skipping...')
            continue
        
        print(f'Loading escape latency data for paradigm "{paradigm}"...')
        mean_escape, sem_escape = load_escape_data(data_folder)
        # mean_escape, sem_escape = load_success_data(data_folder)
        
        # Plot the average with standard deviation
        if 'direct' not in paradigm:
            escape_range = np.arange(-50,50)

            ax.plot(escape_range, mean_escape[:], linestyle=line, linewidth=3, label=s_label)
            ax.fill_between(escape_range, mean_escape[:] - sem_escape[:], mean_escape[:] + sem_escape[:], alpha=0.2)

        else:
            escape_range = np.arange(50)
            ax.plot(escape_range, mean_escape, linestyle=line, linewidth=3, label=s_label)
            ax.fill_between(escape_range, mean_escape - sem_escape, mean_escape + sem_escape, alpha=0.2)


    ax.set_xlabel('Trial', labelpad=10)
    ax.set_ylabel('Escape Latency', labelpad=30)
    ax.legend()
    ax.grid(True)
    
    plt.show()

def load_sf_data(data_folder, trial):
    """
    Load successor feature data from the specified folder.

    Parameters
    ----------
    data_folder : str
        Path to the folder containing the .pickle files.

    Returns
    ----------
    data : list
        A list of dictionaries containing the successor feature data from each run.
    """
    all_sfs = []

    for file_name in sorted(os.listdir(data_folder)):
        if file_name.endswith('.pickle') :
            file_path = os.path.join(data_folder, file_name)
            
            
            with open(file_path, 'rb') as handle:
                print(f'Loading escape latency data {file_name}"...')
                data = pickle.load(handle)
                all_sfs.append(data['SRs'][trial])

    observations = data['Observations']                
    all_sfs = np.array(all_sfs)
    mean_sf = np.mean(all_sfs, axis=0)
                
    return mean_sf, observations

if __name__ == '__main__':
    # Parameters
    paradigms = [ 'targeted',  'direct' ]  # Add other paradigms if needed
    data_root_folder = '/media/matheus/Expansion/matheus - latent learning/data/gridworld-image-encoded-vgg16-states99-steps300-oldsettings-reward5-remake'
    # data_root_folder = 'tolman-image-encoded-vgg16-states16-steps500-oldsettings-reward5-nonsequential-random-observation-4x512'
    output_folder = 'plots'

    labels = [ 'targeted pre-exposure', 'direct learning', 'continuous pre-exposure ', 'mistargeted pre-exposure '] 
    lines = ["-",  "--", "-.",  (0, (1, 1)), (0, (3, 1, 1, 1, 1, 1))]
# 
    plot_escape_latency(paradigms, data_root_folder, output_folder, labels, lines)
    plot_sf('direct', data_root_folder, 50, output_folder)
    plot_sf('targeted', data_root_folder, 100, output_folder)
    plot_cosine(paradigms, data_root_folder, 50)
    