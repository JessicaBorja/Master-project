import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import os
import re
import glob


def plot_data(data, ax, stats_axis=0):
    mean = np.mean(data, axis=stats_axis)[:, -1]
    std = np.std(data, axis=stats_axis)[:, -1]
    n_runs = data.shape[0]
    steps = data[0, :, 0]

    cm = plt.get_cmap('viridis')
    colors = cm(np.linspace(0, 1, n_runs))
    # for run, color in zip(data, colors):
    #     ax.plot(steps, run[:, -1], c = color, alpha = 0.7)

    ax.plot(steps, mean, 'k', linewidth=2)
    ax.fill_between(steps, mean + std, mean - std, color="black", alpha=0.3)
    ax.axhline(200, color="gray", ls="--")
    return ax


def plot_eval_data(files, task, top_row=-1, show=True, save=True,
                   save_name="return", metric="return",
                   save_folder="./results/figures/"):
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
    x_label = np.arange(0, min_data_axs*10, 10)
    # Change timesteps by episodes -> x axis will show every n episodes result
    data = [np.transpose(
               np.vstack((x_label, d[:min_data_axs, -1]))) for d in data]

    # data = [d[:min_data_axs] for d in data]
    fig, ax = plt.subplots(1, 1, figsize=(10, 6), sharey=True)
    data = np.stack(data, axis=0)
    ax.set_title("Evaluation")
    ax = plot_data(data, ax, stats_axis=0)
    ax.set_xlabel("Episodes")
    ax.set_ylabel("Mean %s over %s episodes" % (metric, n_eval_ep))
    fig.suptitle("%s %s" % (task.title(), metric.title()))

    if not os.path.exists(save_folder):
        os.makedirs(save_folder)
    if(save):
        fig.savefig(os.path.join(save_folder, "%s.png" % save_name), dpi=200)
    if(show):
        plt.show()


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


def plot_metric(task="slide", top_row=-1, show=True,
                save=True, save_name="return", metric="return",
                csv_folder="./results/results_csv/"):
    save_folder = "./analysis/figures/"
    if not os.path.exists(save_folder):
        os.makedirs(save_folder)
    if(metric == "return"):
        eval_files = glob.glob("%s*%s*eval*return*.csv" % (csv_folder, task))
        # train_files = \
        #   glob.glob("%s*%s*train*return*.csv" % (csv_folder, task))
    else:  # episode length
        eval_files = glob.glob("%s*%s*eval*length*.csv" % (csv_folder, task))
        # train_files = \
        #   glob.glob("%s*%s*train*length*.csv" % (csv_folder, task))
        metric = "episode length"
    # assert len(eval_files) == len(train_files)
    plot_eval_data(eval_files, task, top_row, show, save,
                   save_name, metric=metric, save_folder=save_folder)


def main(plot_dict, csv_dir="./results_csv/"):
    # metrics = ["return", "episode length"]
    metrics = ["return"]
    for k_name, plot_name in plot_dict.items():
        for metric in metrics:
            plot_metric(k_name,
                        show=True,
                        save=True,
                        save_name=plot_name,
                        csv_folder=csv_dir,
                        metric=metric)


if __name__ == "__main__":
    plot_dict = {"master_sparse": "Baseline",
                 "master_target_affMask_sparse": "Sparse + detected target + affordance mask",
                 "master_target_dense": "Dense + detected target ",
                 "master_target_affMask_dense": "Dense + detected target + affordance mask"}
    main(plot_dict, csv_dir="./analysis/results_csv/pickup_bin_mask/")
