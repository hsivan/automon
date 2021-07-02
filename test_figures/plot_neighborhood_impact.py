import matplotlib.pyplot as plt
from matplotlib import rcParams, rcParamsDefault
from test_figures.plot_figures_utils import get_figsize, reformat_large_tick_values
from test_utils import read_config_file
from stats_analysis_utils import get_violations_per_neighborhood_size, get_neighborhood_size_error_bound_connection
import os
import numpy as np
import matplotlib.ticker as tick
import pickle as pkl


def plot_impact_of_neighborhood_size_on_violations_four_error_bounds(error_bounds, data_folders):
    rcParams['pdf.fonttype'] = 42
    rcParams['ps.fonttype'] = 42
    rcParams.update({'legend.fontsize': 5.4})
    rcParams.update({'font.size': 5.8})

    fig, axs = plt.subplots(2, 2, figsize=get_figsize(hf=0.45), sharex=True, sharey=True)

    for i in range(len(error_bounds)):
        neighborhood_sizes, neighborhood_violations_arr, safe_zone_violations_arr = get_violations_per_neighborhood_size(data_folders[i])

        axs[i // 2, i % 2].stackplot(neighborhood_sizes, neighborhood_violations_arr, safe_zone_violations_arr, labels=['neighborhood violations', 'safe zone violations'])

        axs[i // 2, i % 2].annotate(r"$\epsilon$=" + str(error_bounds[i]), xy=(0.4, 0.7), xycoords='axes fraction')

        total_violations = [neighborhood_violations_arr[i] + safe_zone_violations_arr[i] for i in range(len(neighborhood_violations_arr))]
        optimal_neighborhood_size = neighborhood_sizes[np.argmin(total_violations)]
        min_violation = np.min(total_violations)
        axs[i // 2, i % 2].plot(optimal_neighborhood_size, min_violation, 'o', label=r"optimal neighborhood size $r^*$", markersize=2, color='black')
        axs[i // 2, i % 2].spines['right'].set_visible(False)
        axs[i // 2, i % 2].spines['top'].set_visible(False)
        axs[i // 2, i % 2].spines['bottom'].set_linewidth(0.5)
        axs[i // 2, i % 2].spines['left'].set_linewidth(0.5)
        axs[i // 2, i % 2].tick_params(width=0.5)

    axs[1, 0].set_xlabel('neighborhood size $r$')
    axs[1, 1].set_xlabel('neighborhood size $r$')
    axs[0, 0].set_ylabel('#violations')
    axs[1, 0].set_ylabel('#violations')

    axs[0, 0].set_yticks([0, 2500, 5000])
    axs[1, 0].set_yticks([0, 2500, 5000])
    axs[0, 0].yaxis.set_major_formatter(tick.FuncFormatter(reformat_large_tick_values))
    axs[1, 0].yaxis.set_major_formatter(tick.FuncFormatter(reformat_large_tick_values))

    handles, labels = axs[1, 0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="upper center", ncol=3, columnspacing=1.05, bbox_to_anchor=(0.49, 1.04), frameon=False, handletextpad=0.25, handlelength=1.7)

    plt.subplots_adjust(top=0.88, bottom=0.21, left=0.13, right=0.97, hspace=0.22, wspace=0.1)
    fig.savefig("impact_of_neighborhood_on_violations_four_error_bounds.pdf")
    plt.close(fig)

    rcParams.update(rcParamsDefault)


def get_communication_or_violation_per_error_bound_connection(parent_test_folder, y_axis="communication"):
    sub_test_folders = list(filter(lambda x: os.path.isdir(os.path.join(parent_test_folder, x)), os.listdir(parent_test_folder)))
    sub_test_folders.sort()
    test_folders = [parent_test_folder + "/" + sub_folder for sub_folder in sub_test_folders]
    error_bounds = np.zeros(len(test_folders))
    optimal_neighborhood_size_communication = np.zeros(len(test_folders))
    tuned_neighborhood_size_communication = np.zeros(len(test_folders))
    fixed_neighborhood_size_communications = None

    for idx, test_folder in enumerate(test_folders):
        conf = read_config_file(test_folder)
        error_bounds[idx] = conf["error_bound"]

        sub_test_folders = list(filter(lambda x: os.path.isdir(os.path.join(test_folder, x)), os.listdir(test_folder)))
        neighborhood_size_folders = [test_folder + "/" + sub_folder for sub_folder in sub_test_folders]
        neighborhood_sizes = []

        num_fixed_neighborhood_sizes = len(neighborhood_size_folders) - 2
        if fixed_neighborhood_size_communications is None:
            fixed_neighborhood_size_communications = np.zeros((num_fixed_neighborhood_sizes, len(test_folders)))

        for folder in neighborhood_size_folders:

            results_file = open(folder + "/results.txt", 'r')
            results = results_file.read()
            results = results.split("\n")
            if y_axis == "communication":
                num_messages = int([result for result in results if "Total msgs broadcast enabled" in result][-1].split("disabled ")[1])
            else:
                num_messages = int([result for result in results if "Total violations msg counter" in result][-1].split("counter ")[1])

            if "fixed" in folder:
                conf = read_config_file(folder)
                neighborhood_sizes.append(conf['neighborhood_size'])
                fixed_neighborhood_size_index = int(folder.split("/")[-1].split("_")[0])
                fixed_neighborhood_size_communications[fixed_neighborhood_size_index][idx] = num_messages
            elif "optimal_neighborhood" in folder:
                optimal_neighborhood_size_communication[idx] = num_messages
            elif "tuned_neighborhood" in folder:
                tuned_neighborhood_size_communication[idx] = num_messages

    neighborhood_sizes.sort()

    return error_bounds, neighborhood_sizes, optimal_neighborhood_size_communication, tuned_neighborhood_size_communication, fixed_neighborhood_size_communications


def plot_communication_error_bound_connection(parent_test_folder, func_name, ax):
    experiment_folders = list(filter(lambda x: os.path.isdir(os.path.join(parent_test_folder, x)), os.listdir(parent_test_folder)))
    experiment_folders.sort()
    experiment_folders = [parent_test_folder + "/" + sub_folder for sub_folder in experiment_folders]

    optimal_neighborhood_size_communication = []
    tuned_neighborhood_size_communication = []
    fixed_neighborhood_size_communications = []

    for experiment in experiment_folders:
        error_bounds, neighborhood_sizes, optimal_neighborhood_size_comm, tuned_neighborhood_size_comm, fixed_neighborhood_size_comms = get_communication_or_violation_per_error_bound_connection(experiment, y_axis="communication")
        optimal_neighborhood_size_communication.append(optimal_neighborhood_size_comm)
        tuned_neighborhood_size_communication.append(tuned_neighborhood_size_comm)
        fixed_neighborhood_size_communications.append(fixed_neighborhood_size_comms)

    ax.plot(error_bounds, np.mean(optimal_neighborhood_size_communication, axis=0), linestyle='-', marker='.', markersize=6, linewidth=0.7, label=r"optimal $r^*$", markevery=3, color="tab:blue")
    ax.plot(error_bounds, np.mean(tuned_neighborhood_size_communication, axis=0), linestyle='-', marker='x', markersize=4, linewidth=0.7, label=r"tuned $\hat{r}$", markevery=3, color="red")
    ax.fill_between(error_bounds, np.mean(optimal_neighborhood_size_communication, axis=0) - np.std(optimal_neighborhood_size_communication, axis=0),
                    np.mean(optimal_neighborhood_size_communication, axis=0) + np.std(optimal_neighborhood_size_communication, axis=0), alpha=.3, color="tab:blue")
    ax.fill_between(error_bounds, np.mean(tuned_neighborhood_size_communication, axis=0) - np.std(tuned_neighborhood_size_communication, axis=0),
                    np.mean(tuned_neighborhood_size_communication, axis=0) + np.std(tuned_neighborhood_size_communication, axis=0), alpha=.3, color="red")

    fixed_neighborhood_size_communications_mean = np.mean(fixed_neighborhood_size_communications, axis=0)
    fixed_neighborhood_size_communications_std = np.std(fixed_neighborhood_size_communications, axis=0)
    interesting_fixed_neighborhood_size_indices = [0, 1, -1]  # First (too small), second (close to optimal), last (too large)
    markers = ["1", "2", "3"]
    colors = ["tab:green", "tab:orange", "tab:purple"]
    for i, idx in enumerate(interesting_fixed_neighborhood_size_indices):
        ax.plot(error_bounds, fixed_neighborhood_size_communications_mean[idx], linestyle='--', marker=markers[i], markersize=5, linewidth=0.7, label=r"$r=$" + str(neighborhood_sizes[idx]), markevery=3, color=colors[i])
        ax.fill_between(error_bounds, fixed_neighborhood_size_communications_mean[idx] - fixed_neighborhood_size_communications_std[idx],
                        fixed_neighborhood_size_communications_mean[idx] + fixed_neighborhood_size_communications_std[idx], alpha=.3, color=colors[i])

    ax.plot(error_bounds, [10*1000]*len(error_bounds), color="black", linestyle=':', linewidth=0.7)

    ax.set_xlabel(r'error bound $\epsilon$')
    ax.spines['right'].set_visible(False)
    ax.spines['top'].set_visible(False)
    ax.spines['bottom'].set_linewidth(0.5)
    ax.spines['left'].set_linewidth(0.5)
    ax.tick_params(width=0.5)
    ax.yaxis.set_major_formatter(tick.FuncFormatter(reformat_large_tick_values))
    rcParams['axes.titlesize'] = 5.8

    ax.annotate(func_name, xy=(0.4, 0.65), xycoords='axes fraction', xytext=(20, 13),
                textcoords='offset pixels', fontsize=5.8,
                horizontalalignment='right', verticalalignment='bottom')

    mean_communication_error_percent_optimal_tuned = np.mean(np.abs(np.mean(optimal_neighborhood_size_communication, axis=0) - np.mean(tuned_neighborhood_size_communication, axis=0)) / np.mean(optimal_neighborhood_size_communication, axis=0) * 100)
    mean_communication_error_percent_optimal_fixed_small_r = np.mean(np.abs(np.mean(optimal_neighborhood_size_communication, axis=0) - fixed_neighborhood_size_communications_mean[0]) / np.mean(optimal_neighborhood_size_communication, axis=0) * 100)
    mean_communication_error_percent_optimal_fixed_medium_r = np.mean(np.abs(np.mean(optimal_neighborhood_size_communication, axis=0) - fixed_neighborhood_size_communications_mean[1]) / np.mean(optimal_neighborhood_size_communication, axis=0) * 100)
    mean_communication_error_percent_optimal_fixed_large_r = np.mean(np.abs(np.mean(optimal_neighborhood_size_communication, axis=0) - fixed_neighborhood_size_communications_mean[2]) / np.mean(optimal_neighborhood_size_communication, axis=0) * 100)
    print("Function", func_name, "error tuned:", mean_communication_error_percent_optimal_tuned,
          "error small fixed r:", mean_communication_error_percent_optimal_fixed_small_r,
          "error med fixed r:", mean_communication_error_percent_optimal_fixed_medium_r,
          "error large fixed r:", mean_communication_error_percent_optimal_fixed_large_r)


def plot_communication_or_violation_error_bound_connection_rozenbrock_mlp_2(parent_test_folder_rozenbrock, parent_test_folder_mlp_2):
    rcParams['pdf.fonttype'] = 42
    rcParams['ps.fonttype'] = 42
    rcParams.update({'legend.fontsize': 5.4})
    rcParams.update({'font.size': 5.8})

    fig, axs = plt.subplots(1, 2, figsize=get_figsize(hf=0.35))

    plot_communication_error_bound_connection(parent_test_folder_rozenbrock, "Rozenbrock", axs[0])
    plot_communication_error_bound_connection(parent_test_folder_mlp_2, "MLP-2", axs[1])

    axs[0].set_ylabel('#messages')
    axs[1].set_yticks([0, 2000, 4000, 6000, 8000, 10000, 12000])
    axs[1].set_xticks([0, 0.1, 0.2, 0.3])

    axs[0].annotate('Centralization', xy=(1, 0), xycoords='axes fraction', xytext=(-5, 23), textcoords='offset pixels',
                    horizontalalignment='right', verticalalignment='bottom', fontsize=5.2)

    axs[1].annotate('Centralization', xy=(1, 0), xycoords='axes fraction', xytext=(-5, 35), textcoords='offset pixels',
                    horizontalalignment='right', verticalalignment='bottom', fontsize=5.2)

    handles, labels = axs[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="lower center", ncol=6, columnspacing=0.8, frameon=False, bbox_to_anchor=(0.5, -0.04))
    plt.subplots_adjust(top=0.96, bottom=0.39, left=0.12, right=0.99, wspace=0.17)
    fig.savefig("neighborhood_impact_on_communication_error_bound_connection.pdf")
    plt.close(fig)
    rcParams.update(rcParamsDefault)


def plot_neighborhood_size_error_bound_connection(parent_test_folder, func_name, ax):
    parent_test_folder_suffix = parent_test_folder.split('/')[-1]

    if parent_test_folder_suffix not in os.listdir("."):
        # Create folder with with aggregated data for the figure in pickle files
        os.mkdir(parent_test_folder_suffix)

        experiment_folders = list(filter(lambda x: os.path.isdir(os.path.join(parent_test_folder, x)), os.listdir(parent_test_folder)))
        experiment_folders.sort()
        experiment_folders = [parent_test_folder + "/" + sub_folder for sub_folder in experiment_folders]

        optimal_neighborhood_sizes_experiments = []
        tuned_neighborhood_sizes_experiments = []
        for experiment in experiment_folders:
            error_bounds, optimal_neighborhood_sizes, tuned_neighborhood_sizes = get_neighborhood_size_error_bound_connection(experiment)
            optimal_neighborhood_sizes_experiments.append(optimal_neighborhood_sizes)
            tuned_neighborhood_sizes_experiments.append(tuned_neighborhood_sizes)

        pkl.dump(error_bounds, open(parent_test_folder_suffix + "/" + func_name + "_error_bounds.pkl", 'wb'))
        pkl.dump(optimal_neighborhood_sizes_experiments, open(parent_test_folder_suffix + "/" + func_name + "_optimal_neighborhood_sizes_experiments.pkl", 'wb'))
        pkl.dump(tuned_neighborhood_sizes_experiments, open(parent_test_folder_suffix + "/" + func_name + "_tuned_neighborhood_sizes_experiments.pkl", 'wb'))

    error_bounds = pkl.load(open(parent_test_folder_suffix + "/" + func_name + "_error_bounds.pkl", 'rb'))
    optimal_neighborhood_sizes_experiments = pkl.load(open(parent_test_folder_suffix + "/" + func_name + "_optimal_neighborhood_sizes_experiments.pkl", 'rb'))
    tuned_neighborhood_sizes_experiments = pkl.load(open(parent_test_folder_suffix + "/" + func_name + "_tuned_neighborhood_sizes_experiments.pkl", 'rb'))

    optimal_neighborhood_sizes_mean = np.mean(optimal_neighborhood_sizes_experiments, axis=0)
    optimal_neighborhood_sizes_std = np.std(optimal_neighborhood_sizes_experiments, axis=0)
    tunes_neighborhood_sizes_mean = np.mean(tuned_neighborhood_sizes_experiments, axis=0)
    tuned_neighborhood_sizes_std = np.std(tuned_neighborhood_sizes_experiments, axis=0)

    dilution = 1
    if func_name == "MLP-2":
        dilution = 2

    ax.plot(error_bounds[::dilution], optimal_neighborhood_sizes_mean[::dilution], linestyle='-', marker='.', markersize=4, linewidth=0.5, label=r"optimal $r^*$")
    ax.fill_between(error_bounds[::dilution], optimal_neighborhood_sizes_mean[::dilution] - optimal_neighborhood_sizes_std[::dilution], optimal_neighborhood_sizes_mean[::dilution] + optimal_neighborhood_sizes_std[::dilution], alpha=.3)
    ax.set_xlabel(r'error bound $\epsilon$')

    ax.spines['right'].set_visible(False)
    ax.spines['top'].set_visible(False)
    ax.spines['bottom'].set_linewidth(0.5)
    ax.spines['left'].set_linewidth(0.5)
    ax.tick_params(width=0.5)
    ax.set_ylim(bottom=0)
    if func_name == "Rozenbrock":
        ax.set_ylim(top=0.21)

    ax.errorbar(error_bounds[::dilution], tunes_neighborhood_sizes_mean[::dilution], tuned_neighborhood_sizes_std[::dilution], fmt='x', marker="x", markersize=2.3, markeredgewidth=0.5, solid_capstyle='projecting', capsize=2, elinewidth=0.4, capthick=0.4, color='red', label=r"tuned $\hat{r}$")
    ax.legend(loc="lower right", ncol=1, frameon=False, handletextpad=0.29, bbox_to_anchor=(1.08, -0.05), labelspacing=0.1)

    ax.annotate(func_name, xy=(0.4, 0.74), xycoords='axes fraction', xytext=(20, 13),
                textcoords='offset pixels', fontsize=5.8,
                horizontalalignment='right', verticalalignment='bottom')

    mean_neighborhood_size_error_percent_optimal_tuned = np.mean(np.abs(optimal_neighborhood_sizes_mean - tunes_neighborhood_sizes_mean) / optimal_neighborhood_sizes_mean * 100)
    print("Function", func_name, "mean neighborhood size error tuned:", mean_neighborhood_size_error_percent_optimal_tuned)
    mean_wrt_optimal_std = np.mean(np.abs(optimal_neighborhood_sizes_mean - tunes_neighborhood_sizes_mean) / optimal_neighborhood_sizes_std)
    max_wrt_optimal_std = np.max(np.abs(optimal_neighborhood_sizes_mean - tunes_neighborhood_sizes_mean) / optimal_neighborhood_sizes_std)
    print("Function", func_name, "mean_wrt_optimal_std:", mean_wrt_optimal_std, ", max_wrt_optimal_std:", max_wrt_optimal_std)


def plot_neighborhood_size_error_bound_connection_avg_rozenbrock_mlp_2(parent_test_folder_rozenbrock, parent_test_folder_mlp_2):
    rcParams['pdf.fonttype'] = 42
    rcParams['ps.fonttype'] = 42
    rcParams.update({'legend.fontsize': 5.4})
    rcParams.update({'font.size': 5.8})

    fig, axs = plt.subplots(1, 2, figsize=get_figsize(hf=0.35))

    plot_neighborhood_size_error_bound_connection(parent_test_folder_rozenbrock, "Rozenbrock", axs[0])
    plot_neighborhood_size_error_bound_connection(parent_test_folder_mlp_2, "MLP-2", axs[1])

    axs[0].set_ylabel('best size $r$')
    axs[1].set_yticks([0, 0.5, 1])

    plt.subplots_adjust(top=0.94, bottom=0.26, left=0.11, right=0.99, wspace=0.22)
    fig.savefig("optimal_and_tuned_neighborhood.pdf")
    plt.close(fig)
    rcParams.update(rcParamsDefault)


if __name__ == "__main__":
    neighborhood_size_error_bound_connection_rozenbrock_data_folder = "../test_results/results_optimal_and_tuned_neighborhood_rozenbrock_2021-04-05_08-48-07"
    neighborhood_size_error_bound_connection_mlp_data_folder = "../test_results/results_optimal_and_tuned_neighborhood_mlp_2_2021-04-05_15-45-13"

    # Figure 2 in the paper: figure with 4 specific error bounds for Rozenbrock
    error_bounds = [0.05, 0.15, 0.25, 0.95]
    test_folder = neighborhood_size_error_bound_connection_rozenbrock_data_folder + "/0/"
    data_folder_error_bound_0_075 = test_folder + "thresh_0_05/"
    data_folder_error_bound_0_1 = test_folder + "thresh_0_15000000000000002/"
    data_folder_error_bound_0_15 = test_folder + "thresh_0_25000000000000006/"
    data_folder_error_bound_0_8 = test_folder + "thresh_0_9500000000000002/"
    data_folders = [data_folder_error_bound_0_075, data_folder_error_bound_0_1, data_folder_error_bound_0_15, data_folder_error_bound_0_8]
    plot_impact_of_neighborhood_size_on_violations_four_error_bounds(error_bounds, data_folders)

    # Figure 7 in the paper: optimal neighborhood and tuned neighborhood
    plot_neighborhood_size_error_bound_connection_avg_rozenbrock_mlp_2(neighborhood_size_error_bound_connection_rozenbrock_data_folder, neighborhood_size_error_bound_connection_mlp_data_folder)

    # Figure 8 in the paper: 5 lines - optimal neighborhood, tuned neighborhood, and 3 constant neighborhood sizes
    data_folder_rozenbrock = "../test_results/results_comm_neighborhood_rozen_2021-04-10_10-13-41/"
    data_folder_tiny_mlp = "../test_results/results_comm_neighborhood_mlp_2_2021-04-10_12-27-37/"
    plot_communication_or_violation_error_bound_connection_rozenbrock_mlp_2(data_folder_rozenbrock, data_folder_tiny_mlp)
