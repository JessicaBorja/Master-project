import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import os, re
import glob

jpgFilenamesList = glob.glob('145592*.jpg')


def plot_data(data, ax, stats_axis = 0):
    mean = np.mean(data, axis = stats_axis)[:, -1]
    std = np.std(data, axis = stats_axis)[:, -1]
    n_runs = data.shape[0]
    steps = data[0,:,0]
    
    cm = plt.get_cmap('viridis')
    colors = cm( np.linspace(0, 1, n_runs) )
    # for run, color in zip(data, colors):
    #     ax.plot(steps, run[:, -1], c = color, alpha = 0.7)

    ax.plot(steps, mean, 'k', linewidth=2)
    ax.fill_between(steps, mean + std, mean - std, color ="black" , alpha = 0.3 )
    ax.axhline(0, color="gray", ls = "--" ) 
    return ax

def plot_eval_and_train(eval_files, train_files, task, top_row = -1,\
                         show=True, save=True, save_name = "return", metric = "return"):
    eval_data, train_data = [], []
    min_val = np.inf
    for evalFile, trainFile in zip(eval_files, train_files):
        #Skip wall time
        eval_data.append( pd.read_csv(evalFile).to_numpy()[:top_row, 1:] )
        stats = pd.read_csv(trainFile).to_numpy()[:, 1:]
        train_limit = top_row * len(stats)//100
        if(train_limit < min_val):
            min_val = train_limit
        train_data.append( stats[:train_limit] )
    search_res = re.search(r"\((.*?)\)", eval_files[0])
    if search_res:
        search_res = search_res.group(1)
        n_eval_ep = int( search_res[:-2] )#Remove "ep"
    else:
        n_eval_ep = 10
    
    fig, axs = plt.subplots(1, 2, figsize=(13, 5), sharey=True)
    train_data = [run[:min_val] for run in train_data]
    train_data = np.stack( train_data, axis=0)
    axs[0].set_title("Training")
    axs[0] = plot_data(train_data, axs[0], stats_axis = 0)
    axs[0].set_xlabel("Episode")
    axs[0].set_ylabel(metric.title())

    eval_data = np.stack( eval_data, axis=0)
    axs[1].set_title("Evaluation")
    axs[1] = plot_data(eval_data, axs[1], stats_axis = 0)
    axs[1].set_xlabel("Timesteps")
    axs[1].set_ylabel("Mean %s over %s episodes"%(metric, n_eval_ep) )
    fig.suptitle("%s %s"%(task.title(), metric.title()))

    if not os.path.exists("./results/figures"):
        os.makedirs("./results/figures")
    if(save):
        fig.savefig("./results/figures/%s.png"%save_name, dpi=200)
    if(show):
        plt.show()

def plot_ep_len(task = "slide", top_row = -1, show=True, save=True, save_name = "return"):
    if not os.path.exists("./results/"):
        os.makedirs("./results/")
    csv_folder = "./results/results_csv/"
    eval_files =  glob.glob("%s*%s*eval*length*.csv"%(csv_folder, task))
    train_files =  glob.glob("%s*%s*train*length*.csv"%(csv_folder, task))
    assert len(eval_files) == len(train_files)
    plot_eval_and_train(eval_files, train_files, task, top_row, show, save, save_name, metric = "episode length")
    
def plot_mean_std(task = "slide", top_row = -1, show=True, save=True, save_name = "return"):
    if not os.path.exists("./results/"):
        os.makedirs("./results/")
    csv_folder = "./results/results_csv/"
    eval_files =  glob.glob("%s*%s*eval*return*.csv"%(csv_folder, task))
    train_files =  glob.glob("%s*%s*train*return*.csv"%(csv_folder, task))
    assert len(eval_files) == len(train_files)
    plot_eval_and_train(eval_files, train_files, task, top_row, show, save, save_name, metric = "reward")

def main():
    plot_mean_std("slide", top_row = 60, show=False, save=True, save_name="slide_ep_return_meanstd")
    plot_ep_len("slide", top_row = 60, show=False, save=True, save_name="slide_ep_length_meanstd")
    
    plot_mean_std("drawer", top_row = 70, show=False, save=True, save_name="drawer_ep_return_meanstd")
    plot_mean_std("drawer", top_row = 70, show=False, save=True, save_name="drawer_ep_length_meanstd")

    plot_mean_std("hinge", top_row = 100, show=False, save=True, save_name="hinge_ep_return_meanstd")
    plot_mean_std("hinge", top_row = 100, show=False, save=True, save_name="hinge_ep_length_meanstd")

if __name__ == "__main__":
    main()