import numpy as np
import os, pickle
import matplotlib.pyplot as plt
import matplotlib as mat
from sklearn.metrics.pairwise import cosine_similarity
from matplotlib.patches import Polygon
from matplotlib.collections import PatchCollection
import seaborn as sns
## set the font size of the plots
font = {'family' : 'DejaVu Sans',
        'weight' : 'normal',
        'size'   : 23}
mat.rc('font', **font)

reachable_states = [184, 170, 156, 155, 154, 157, 158, 159, 173, 187, 145, 131, 117, 118, 119, 116, 115, 114, 128, 142, 100,  86,  72,  71,
                             70,  73,  74,  75,  89, 103,  61,  47,  33,  32,  31,  34,  35,  36,  22,   8,  50,  64,  78,  77,  76,  79,  80,  81,
                             67,  53,  95, 109, 123, 124, 125, 122, 121, 120, 106,  92, 134, 148, 162, 161, 160, 163, 164, 165, 151, 137, 179, 193] 

map_tuples = [(x, y) for x, y in enumerate(reachable_states)]
  
def plot(M, scheme, file_name, episode, shape, action_mapping={0: 'left', 1: 'up', 2: 'right', 3: 'down'}):
    number_of_states = shape[0] * shape[1]
    plt.figure(1, figsize=(shape[0], shape[1]))
    # draw grid
    for i in range(shape[0]):
        plt.axhline(i, linewidth=0.0, color='w', alpha=0.0)
    for i in range(shape[1]):
        plt.axvline(i, linewidth=0.0, color='w', alpha=0.0)
    plt.xlim(0, shape[1])
    plt.ylim(0, shape[0])
    plt.title("Tolman \nQ-values %s, Trial %d" % (file_name, episode))
    plt.pcolor(np.ones((shape[0], shape[1]))*0.3, cmap='PRGn', vmin=0.0, vmax=1.0)
    ax = plt.gca()
    P = []

    for e, exp in enumerate(M):
        
        # determine action index
        a = int(e/number_of_states)
        # determine state index
        s = e - number_of_states * a
        # determine origin coordiantes
        x = int(s/shape[1])
        y = s - x * shape[1]
        x = shape[0] - x - 1
        coords = np.array([[x, y]]).astype(float)
        # draw polygon for transition
        if action_mapping[a] == 'left':
            coords = coords + np.array([[1/3, 1/3], [1/2, 0.05], [2/3,1/3]])
        elif action_mapping[a] == 'up':
            coords = coords + np.array([[2/3, 1/3], [0.95, 1/2], [2/3, 2/3]])
        elif action_mapping[a] == 'right':
            coords = coords + np.array([[1/3, 2/3], [2/3, 2/3], [1/2, 0.95]])
        elif action_mapping[a] == 'down':
            coords = coords + np.array([[0.05, 0.5], [1/3, 1/3], [1/3, 2/3]])
        coords = np.flip(coords)
        
        P += [Polygon(coords)]
    
    P = PatchCollection(P)
    P.set_array(M)
    P.set_cmap('hot')
    plt.colorbar(P)
    P.set_clim([0, 0.2])
    ax.add_collection(P)
    plt.xticks([])
    plt.yticks([])
    plt.show()

def NormalizeData(data):
    return (data - np.min(data)) / (np.max(data) - np.min(data))

def map_tolman(exps):
    
    gridworld = np.full((14,14),np.min(exps))
     
    for m in map_tuples:
       i = int(m[1]%14)
       j = int(m[1]/14)
       gridworld[j][i] = exps[m[0]]
    return gridworld

def map_tolman_values(exps):
    
    gridworld = np.full((14,14,4),np.min(exps))
    shape = exps.shape
    # gridworld = gridworld.reshape(14*14,4)
    
    for m in map_tuples:
       i = int(m[1]/14)
       j = m[1]%14
       gridworld[j][i][:] = exps[m[0]][:]

    # gridworld = gridworld.reshape(14,14,4)
    return gridworld

def map_gridworld(exps):
    
    gridworld = np.full((10,10,4),np.min(exps))
    shape = exps.shape
    print(shape)
    for e in range(100):
       i = int(e/10)
       j = e % 10
       gridworld[j][i][:] = exps[e][:]

    return gridworld

# plot escape_latency
def escape_latency(path, simulations, total_steps, schemes, labels, lines):
    
    fig, ax = plt.subplots(figsize=(17, 12))

    for scheme, s_label, line in zip(schemes,labels,lines):
        escape_latency = []

        for epoch in range(simulations):
            file_path = path + '/Trainingdata_%s_%s.pickle' % (scheme, epoch + 1)
            with open(file_path, 'rb') as handle:
                data = pickle.load(handle)
            escape_latency.append([len(x) for x in data['Escape']])
            
        escape_latency = np.asarray(escape_latency)
        
        if 'latent' in scheme :
            trial_range = np.arange(-total_steps/2, total_steps/2)
            ax.plot(trial_range, np.mean(escape_latency, axis=0), linestyle=line, linewidth=3, label=s_label)
            ax.fill_between(trial_range, np.mean(escape_latency, axis=0) -np.std(escape_latency, axis=0), 
                            np.mean(escape_latency, axis=0) + np.std(escape_latency, axis=0),alpha=0.2)

        if 'direct' in scheme:
            trial_range = np.arange(0, total_steps/2)
            ax.plot(trial_range, np.mean(escape_latency, axis=0), linestyle=line, linewidth=3, label=s_label)
            ax.fill_between(trial_range, np.mean(escape_latency, axis=0)-np.std(escape_latency, axis=0), 
                            np.mean(escape_latency, axis=0) + np.std(escape_latency, axis=0),alpha=0.2)

    ax.set_xlabel('Trial', labelpad=10)
    ax.set_ylabel('Escape Latency', labelpad=30)
    plt.legend()
    plt.grid(True)

# plot the successor representation matrix, the cosine similarity or a state successor feature
def successor_representation (env, path, simulations, total_steps, schemes, labels, plot_cosine_similarity=False, plot_individual=False):
    
    fig, ax = plt.subplots(figsize=(17, 12))

    if env == 'gridworld':
        num_states = 100
    elif env == 'tolman':
        num_states = 72
    num_actions = 4
    recording_trial = [50, 100] # The trial to plot the SR matrix. It ranges from 0 to 100 in intervals of 10. Direct learning only supports up to 50.
    action_mapping = {0: 'left', 1: 'up', 2: 'right', 3: 'down'}
    SR_direct = []
    SR_targeted = []
    SR_continuous = []
    for scheme, label in zip(schemes, labels):
        
        SRs = {}
        SR = []
        for e in recording_trial:
            if ('direct' in scheme and e <= 50) or ('latent' in scheme):
                SRs[e] = np.zeros((num_states, num_actions, num_states))
            elif 'direct' in scheme and e > 50:
                break
            
        # extract average SR
        epochs = range(simulations)
        for epoch in epochs:
            data_path = path + '/Trainingdata_%s_%s.pickle' % (scheme, epoch + 1)
            with open(data_path, 'rb') as handle:
                data = pickle.load(handle)
            for e in recording_trial:
                if ('direct' in scheme and e <= 50) or ('latent' in scheme):
                    SRs[e] += data['SRs'][e]
                    SR.append(data['SRs'][e])


        if 'direct' in scheme:
            SR_direct = np.array(SR)
        elif 'targeted' in scheme:
            SR_targeted = np.array(SR)
        elif 'continuous' in scheme:
            SR_continuous = np.array(SR)

    # [simulation][state][action][sr]
    if plot_cosine_similarity: # it runs the last trial value in recording_trial list.
        
        ########### cosine similarity #############
        
        for action in range(4):
            
            targeted_similarity_total = []
            continuous_similarity_total = []
            
            for epoch in range(simulations):
                similarity = []
                similarity_continuous = []
            
                for state in range(num_states):
                    sr_row_direct = SR_direct[epoch][state][action]
                    sr_row_targeted = SR_targeted[epoch][state][action]
                    sr_row_continuous = SR_continuous[epoch][state][action]
                    
                    similarity.append(cosine_similarity(sr_row_targeted.reshape(1, -1), sr_row_direct.reshape(1, -1)))
                    similarity_continuous.append(cosine_similarity(sr_row_continuous.reshape(1, -1), sr_row_direct.reshape(1, -1)))
            
                targeted_similarity_total.append(similarity)
                continuous_similarity_total.append(similarity_continuous)
        
            targeted_similarity_total = np.array(targeted_similarity_total)
            targeted_mean = targeted_similarity_total.mean(0)
            targeted_std = targeted_similarity_total.std(0)

            continuous_similarity_total = np.array(continuous_similarity_total)
            continuous_mean = continuous_similarity_total.mean(0)
            continuous_std = continuous_similarity_total.std(0)

            fig, ax = plt.subplots(figsize=(25, 12))
            plt.errorbar(range(num_states), targeted_mean[:, 0, 0], yerr=targeted_std[:, 0, 0], linewidth=2, color = 'darkorange', linestyle='-', markersize=10, marker='^', capsize=10, label='Latent Learning - targeted pre-exposure')
            plt.errorbar(range(num_states), continuous_mean[:, 0, 0], yerr=continuous_std[:, 0, 0], linewidth=2, color = 'purple', markersize=10, linestyle='--', marker='o', capsize=10, label='Latent Learning - continuous pre-exposure')
            
            ax.set_xticks(range(0, num_states, 5))
            ax.set_xticklabels(range(0, num_states, 5))
            legend = ax.legend(loc='upper left', shadow=True, fontsize='large')
            plt.ylabel('Similarity Score', fontsize=20, labelpad=20)
            plt.xlabel('State', fontsize=20, labelpad=20)
            ax.set_title('Tolman maze, action - %s' % (action_mapping[action]))
          
    else:
        
        # select the trial
        trial = 100 # ranges from 0 to 100 in intervals of 10 -> make sure to add the trial value in recording_trial list above.
        if 'direct' in scheme and trial > 50:
            trial = 50 # ranges from 0 to 50 in intervals of 10 -> make sure to add the trial value in recording_trial list above.
        
        successor_features = SRs[trial]
        action = list(action_mapping.keys())[list(action_mapping.values()).index('up')]     

        if plot_individual:
               
            observed_state = 0 # select the state to look its successor feature
            sf = successor_features[observed_state][action]
            sf = NormalizeData(sf)
            if env == 'gridworld':
                sf = np.array(sf).reshape(10,10)
            elif env == 'tolman':
                sf = map_tolman(sf)
            
            ax = sns.heatmap(np.array(sf), cmap=plt.get_cmap('viridis'), cbar_kws={'label': 'Successor Feature'},  linewidth=0.0, ax=ax, vmin=0.0, vmax=0.2)
            plt.title("%s" % (label))
            
            map_to_heatmap = map_tuples[observed_state]

            if env == 'gridworld':
                state_heat_tuple = ( observed_state % 10, int(observed_state / 10),)
            elif env == 'tolman':
                state_heat_tuple = ( int(map_to_heatmap[1]%14), map_to_heatmap[1]/14, )
            # add a cross to the heatmap
            (p, q) = state_heat_tuple # location of the cross
            plt.scatter(p+0.5, q+0.5, s= 500, marker='X', color = 'darkorange')
                
        else: # it shows the result only to the last element of the schemes list in the main function. Change the element's order in the schemes list to vary the SR results.

            successor_features = NormalizeData(successor_features)   
            successor_matrix = []

            for state in range (num_states):
                successor_matrix.append(successor_features[state][action])

            ax = sns.heatmap(successor_matrix, cmap='viridis', linewidths=0.0, rasterized=True, ax=ax, vmin=0, vmax=0.2)
            
            plt.title("%s, Trial %d" % (label, trial))
            # Adjust the colorbar tick label font size
            cbar = ax.collections[0].colorbar
            cbar.set_label('Deep SR', size=40, labelpad=30)
            
            cbar.ax.tick_params(labelsize=20)

            # Adjust the y-axis label font size
            ax.figure.axes[-1].yaxis.label.set_size(20)

# plot leaned Q-values
def q_values(env, path, schemes, labels, trials):
    
    if env == 'tolman':
        num_states = 72
        grid_size = 14
    elif env == 'gridworld':
        num_states = 100
        grid_size = 10
    num_actions = 4

    recording_trials = [50]

    for label, scheme in zip(labels,schemes):
        fig, ax = plt.subplots(figsize=(15, 12))
        Qs = {}
        Q = []
        for e in recording_trials:
            Qs[e] = np.zeros((num_states, num_actions))
        
        epochs = range(trials)
        for epoch in range(len(epochs)):
            data_path = path + '/Trainingdata_%s_%s.pickle' % (scheme, epoch + 1)
            with open(data_path, 'rb') as handle:
                data = pickle.load(handle)
            for e in recording_trials:
                Qs[e] += data['Qvalues'][e]
                Q.append(data['Qvalues'][e])

        
        # averaging
        for e in recording_trials:
            if 'direct' in scheme and e <= 50:
                Qs[e] /= len(recording_trials)
        
        # select the output trial
        trial = 50
        Q_val = Qs[trial]
        
        if env == 'gridworld':
            Q_val = map_gridworld(Q_val)    
        elif env == 'tolman':
            Q_val = map_tolman_values(Q_val)
        
        mapped_Q = np.array(Q_val).flatten(order='F')
        mapped_Q = NormalizeData(mapped_Q)
        plot(mapped_Q, scheme, label, trial, (grid_size, grid_size, 4))

        # NOTE: To extract Q-values from the SR, load the SR as in 'def successor_representation()'
        # and multiply it by the reward function below:
        
        # model_reward = np.zeros((num_states), dtype=float)
        # model_reward[-1] = 5.0/num_actions
        # Q_from_sr = SRs[episode][:].dot(model_reward)
        # if env == 'gridworld':
        #     Q_from_sr = map_gridworld(Q_from_sr)    
        # elif env == 'tolman':
        #     Q_from_sr = map_tolman_values(Q_from_sr)
        
        # mapped_Q = np.array(Q_from_sr).flatten(order='F')
        # mapped_Q = NormalizeData(mapped_Q)
        # plot(mapped_Q, scheme, label, trial, (grid_size, grid_size, 4))

# plot probabilities of taking the correct action over all actions in a particular trial
def action_probabilities(path, simulations):
    
    num_states = 72 # only for Tolman maze

    scheme = 'latent_targeted'
    recording_trials = [49] # before starting the learning phase
    probs = {}
    for e in recording_trials:
        p = []
        probs[e] = []
    
    for epoch in range(simulations):
        data_path = path + '/Trainingdata_%s_%s.pickle' % (scheme, epoch + 1)
        with open(data_path, 'rb') as handle:
            data = pickle.load(handle)
            for i in range(len(recording_trials)):
                probs[recording_trials[i]].append(np.array(data['SAProbs'][recording_trials[i]]))
    
    np.seterr(invalid='ignore')
    probs_per_trial = []
    for e in recording_trials:
        probs[e] = np.array(probs[e])
        p_shape = probs[e].shape
        for i in range(p_shape[0]):
            vector = []
            for j in range(p_shape[-1]):
                if probs[e][i][0][j] != 0:
                    vector.append(probs[e][i][1][j]/probs[e][i][0][j])
                else:
                    vector.append(0)
            p.append(vector)
        probs_per_trial.append(np.array(p).mean(0))
        
    probs_per_trial = np.array(probs_per_trial)
    probs_targeted = probs_per_trial
   

    scheme = 'latent_continuous'
    probs = {}
    for e in recording_trials:
        probs[e] = []
    for epoch in range(simulations):
        data_path = path + '/Trainingdata_%s_%s.pickle' % (scheme, epoch + 1)
        with open(data_path, 'rb') as handle:
            data = pickle.load(handle)
            for i in range(len(recording_trials)):
                probs[recording_trials[i]].append(np.array(data['SAProbs'][recording_trials[i]]))
    
    probs_per_episode = []
    for e in recording_trials:
        p = []
        probs[e] = np.array(probs[e])
        p_shape = probs[e].shape
        for i in range(p_shape[0]):
            vector = []
            for j in range(p_shape[-1]):
                if probs[e][i][0][j] != 0:
                    vector.append(probs[e][i][1][j]/probs[e][i][0][j])
                else:
                    vector.append(0)
            p.append(vector)
        probs_per_episode.append(np.array(p).mean(0))
        
        
    probs_per_episode = np.array(probs_per_episode)
    probs_continuous = probs_per_episode
    
    # Width of the bars
    bar_width = 0.35
    dead_end = [x for x in range(4,72,5)]
    dead_end.insert(0, 0)
    
    for trial in recording_trials:
        targeted = []
        continuous = []
        d_end = []
        for state in range(num_states):
            if state not in dead_end:

                targeted.append(probs_targeted[recording_trials.index(trial)][state])
                continuous.append(probs_continuous[recording_trials.index(trial)][state])
                d_end.append('')
            
            else:
                d_end.append('Dead-end state')
                targeted.append(0)
                continuous.append(0)

        fig, ax = plt.subplots(1, 1, sharey=True)
        fig.set_size_inches(42, 12)

        # Plotting the bars for data set 1
        ax.bar(np.arange(len(targeted)) - bar_width/2, targeted, bar_width, label='Targeted pre-exposure')
        # Plotting the bars for data set 2
        ax.bar(np.arange(len(continuous)) + bar_width/2, continuous, bar_width, label='Continuous pre-exposure')
        
        for i in range(num_states):
            ax.text(i + 0.5, 0.05 , d_end[i], size=30, rotation=90, rotation_mode='anchor')
        
        ax.text(i + 0.5, 0.05 , "Goal state", size=30, rotation=90, rotation_mode='anchor')
    
        ax.set_xlabel('States', labelpad=20)
        ax.set_ylabel('Matched Actions (%)', labelpad=20)
        ax.set_title('Tolman maze, trial %s' % (trial-49), fontsize=50)
        ax.set_xticks(range(1, num_states+1, 2))
        plt.xticks(fontsize=25)
        plt.yticks(fontsize=25)
        plt.legend()
        plt.rc('legend', fontsize = 25)
        ax.figure.axes[-1].yaxis.label.set_size(30)
        ax.figure.axes[-1].xaxis.label.set_size(30)
        plt.show()

def __main__():
    
    # load latent and direct learning data
    env = 'tolman'
    policy = 'greedy'
    path = './data/'+env+'/'+policy
    simulations = 30
    trials = 100    
    schemes = ['direct', 'latent_targeted', 'latent_continuous']
    labels = ['Direct learning', 'Latent learning - targeted pre-exposure', 'Latent learning - continuous pre-exposure'] 
    total_steps = 100

    lines = ["-",  "--", "-.", (0, (1, 1)), (0, (3, 1, 1, 1, 1, 1))]

    escape_latency(path, simulations, total_steps, schemes, labels, lines)
    # successor_representation (env, path, simulations, total_steps, schemes, labels, plot_cosine_similarity=False, plot_individual=False)
    # q_values(env, path, schemes, labels, simulations)
    # action_probabilities(path, simulations)

    plt.show()

__main__()

