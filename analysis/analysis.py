import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import os
import re
import glob


def plot_data(data, ax, label, color="gray", stats_axis=0):
    # pd.Series(data['value']).rolling(window_size, min_periods=window_size).mean()
    mean = np.mean(data, axis=stats_axis)[:, -1]
    std = np.std(data, axis=stats_axis)[:, -1]
    steps = data[0, :, 0]

    ax.plot(steps, mean, 'k', linewidth=2, label=label, color=color)
    ax.fill_between(steps, mean + std, mean - std, color=color, alpha=0.3)
    ax.axhline(6, color="gray", ls="--")
    return ax

# Linear interpolation between 2 datapoints
def interpolate(pt1, pt2, x):
    x1, y1 = pt1
    x2, y2 = pt2
    y = y1 + (x - x1)*(y2 - y1)/(x2 - x1)
    return y


def merge_by_episodes(data, min_data_axs):
    # Evaluation was done every 20 episodes
    eval_rate = 20
    x_label = np.arange(0, min_data_axs * eval_rate, eval_rate)

    # Truncate to the seed with least episodes
    data = [np.transpose(
               np.vstack((x_label, d[:min_data_axs, -1]))) for d in data]
    data = np.stack(data, axis=0)
    return data


def merge_by_timesteps(data, min_data_axs):
    '''
        data:
            list, each element is a different seed
            data[i].shape = [n_evaluations, 2] (timestep, value)
    '''
    idxs = np.arange(90) * 1000  # 0, 1k, 2k ... 90k
    data_copy = []
    for d in data:
        run_values = np.zeros(shape=(len(idxs), d.shape[-1]))
        for i in range(len(idxs)):
            for d_idx in range(len(d) - 1):
                pt1 = d[d_idx]
                pt2 = d[d_idx+1]
                if(pt1[0] <= idxs[i] and idxs[i] < pt2[0]):
                    run_values[i] = np.array([idxs[i],  # new timestep
                                              interpolate(pt1, pt2, idxs[i])])
        data_copy.append(run_values)
    # data = [d[:min_data_axs] for d in data]
    data = np.stack(data_copy, axis=0)
    return data


# Data is a list
def plot_experiments(data, show=True, save=True,
                     save_name="return", metric="return",
                     save_folder="./analysis/figures/",
                     x_label="timesteps",
                     y_label="Completed tasks"):
    fig, ax = plt.subplots(1, 1, figsize=(10, 6), sharey=True)
    ax.set_title("Evaluation")

    cm = plt.get_cmap('viridis')
    colors = cm(np.linspace(0, 1, len(data)))

    for exp_data, c in zip(data, colors):
        name, data = exp_data
        ax = plot_data(data, ax, label=name, color=c, stats_axis=0)

    ax.set_xlabel(x_label)
    ax.set_ylabel(y_label)
    ax.legend(loc='upper left')
    fig.suptitle("%s" % (metric.title()))

    if not os.path.exists(save_folder):
        os.makedirs(save_folder)
    if(save):
        fig.savefig(os.path.join(save_folder, "%s.png" % save_name), dpi=200)
    if(show):
        plt.show()


# Plot validation data for a single experiment, multiple seeds
def seeds_mean(files, top_row=-1, data_merge_fnc=merge_by_timesteps):
    data = []
    for file_n in files:
        # Skip wall time
        data.append(pd.read_csv(file_n).to_numpy()[:top_row, 1:])
    search_res = re.search(r"\((.*?)\)", files[0])
    if search_res:
        search_res = search_res.group(1)
        n_eval_ep = int(search_res[:-2])  # Remove "ep"
    else:
        n_eval_ep = 10

    min_data_axs = min([d.shape[0] for d in data])
    # Change timesteps by episodes -> x axis will show every n episodes result
    data = data_merge_fnc(data, min_data_axs)

    return data


def plot_eval_and_train(eval_files, train_files, task, top_row=-1,
                        show=True, save=True, save_name="return",
                        metric="return"):
    eval_data, train_data = [], []
    min_val = np.inf
    for evalFile, trainFile in zip(eval_files, train_files):
        # Skip wall time
        eval_data.append(pd.read_csv(evalFile).to_numpy()[:top_row, 1:])
        stats = pd.read_csv(trainFile).to_numpy()[:, 1:]
        train_limit = top_row * len(stats)//100
        if(train_limit < min_val):
            min_val = train_limit
        train_data.append(stats[:train_limit])
    search_res = re.search(r"\((.*?)\)", eval_files[0])
    if search_res:
        search_res = search_res.group(1)
        n_eval_ep = int(search_res[:-2])  # Remove "ep"
    else:
        n_eval_ep = 10

    fig, axs = plt.subplots(1, 2, figsize=(13, 5), sharey=True)
    train_data = [run[:min_val] for run in train_data]
    train_data = np.stack(train_data, axis=0)
    axs[0].set_title("Training")
    axs[0] = plot_data(train_data, axs[0], stats_axis=0)
    axs[0].set_xlabel("Episode")
    axs[0].set_ylabel(metric.title())

    eval_data = np.stack(eval_data, axis=0)
    axs[1].set_title("Evaluation")
    axs[1] = plot_data(eval_data, axs[1], stats_axis=0)
    axs[1].set_xlabel("Timesteps")
    axs[1].set_ylabel("Mean %s over %s episodes" % (metric, n_eval_ep))
    fig.suptitle("%s %s" % (task.title(), metric.title()))

    if not os.path.exists("./results/figures"):
        os.makedirs("./results/figures")
    if(save):
        fig.savefig("./results/figures/%s.png" % save_name, dpi=200)
    if(show):
        plt.show()


def get_mean_and_std(exp_name="slide", metric="return",
                     csv_folder="./results/results_csv/",
                     data_merge_fnc=merge_by_timesteps):
    if(metric == "return"):
        eval_files = glob.glob("%s*%s*eval*return*.csv" % (csv_folder, exp_name))
        # train_files = \
        #   glob.glob("%s*%s*train*return*.csv" % (csv_folder, exp_name))
    elif(metric == "success"):
        eval_files = glob.glob("%s*%s*eval*success*.csv" % (csv_folder, exp_name))
    else:  # episode length
        eval_files = glob.glob("%s*%s*eval*length*.csv" % (csv_folder, exp_name))
        # train_files = \
        #   glob.glob("%s*%s*train*length*.csv" % (csv_folder, exp_name))
        metric = "episode length"
    # assert len(eval_files) == len(train_files)

    experiment_data = seeds_mean(eval_files, data_merge_fnc=data_merge_fnc)
    return experiment_data


def plot_by_timesteps(plot_dict, csv_dir="./results_csv/"):
    # metrics = ["return", "episode length"]
    metrics = ["success"]
    experiments_data = []
    for metric in metrics:
        for exp_name, title in plot_dict.items():
            mean_data = get_mean_and_std(exp_name,
                                         csv_folder=csv_dir,
                                         metric=metric,
                                         data_merge_fnc=merge_by_timesteps)
            experiments_data.append([title, mean_data])
        plot_experiments(experiments_data,
                         show=True,
                         save=True,
                         save_name=metric + "_by_timesteps",
                         save_folder="./analysis/figures/",
                         metric=metric,
                         x_label="timesteps",
                         y_label="Completed tasks")


def plot_by_episodes(plot_dict, csv_dir="./results_csv/"):
    # metrics = ["return", "episode length"]
    metrics = ["success"]
    experiments_data = []
    for metric in metrics:
        min_ep = np.inf
        for exp_name, title in plot_dict.items():
            # n_seeds, n_ep, 2
            mean_data = get_mean_and_std(exp_name,
                                         csv_folder=csv_dir,
                                         metric=metric,
                                         data_merge_fnc=merge_by_episodes)
            experiments_data.append([title, mean_data])
            if(mean_data.shape[1] < min_ep):
                min_ep = mean_data.shape[1]
        # Crop to experiment with least episodes
        experiments_data = [[title, data[:, :min_ep]]
                            for title, data in experiments_data]
        plot_experiments(experiments_data,
                         show=True,
                         save=True,
                         save_name=metric + "_by_episodes",
                         save_folder="./analysis/figures/",
                         metric=metric,
                         x_label="Episodes",
                         y_label="Completed tasks")


if __name__ == "__main__":
    plot_dict = {"master_sparse": "Baseline",
                #  "master_target_affMask_sparse": "Sparse + detected target + affordance mask",
                 "master_target_dense": "Ours"}
                #  "master_target_affMask_dense": "Dense + detected target + affordance mask"}
    plot_by_episodes(plot_dict, csv_dir="./analysis/results_csv/pickup_success/")
    plot_by_timesteps(plot_dict, csv_dir="./analysis/results_csv/pickup_success/")
