import matplotlib.pyplot as plt
from matplotlib import rcParams, rcParamsDefault
import numpy as np
from tests.visualization.utils import get_figsize, reformat_large_tick_values
from automon.test_utils import read_config_file
import os
import matplotlib.ticker as tick


def plot_monitoring_f_and_error_bound_around_it(kld_test_folder, inner_product_test_folder, mlp_40_test_folder,
                                                mlp_2_test_folder, quadratic_test_folder, dnn_intrusion_detection_test_folder):
    rcParams['pdf.fonttype'] = 42
    rcParams['ps.fonttype'] = 42
    rcParams.update({'legend.fontsize': 5.4})
    rcParams.update({'font.size': 5.8})

    test_folders = [dnn_intrusion_detection_test_folder,
                    kld_test_folder,
                    mlp_40_test_folder,
                    mlp_2_test_folder,
                    quadratic_test_folder,
                    inner_product_test_folder]
    func_names = ["DNN", "KLD", "MLP-40", "MLP-2", "Quadratic", "Inner Prod."]

    fig, axs = plt.subplots(3, 2, figsize=get_figsize(hf=0.6*0.83))
    axs = [axs[0, 0], axs[0, 1], axs[1, 0], axs[1, 1], axs[2, 0], axs[2, 1]]

    for idx, test_folder in enumerate(test_folders):
        conf = read_config_file(test_folder)
        error_bound = conf["error_bound"]

        real_function_value_file_suffix = "real_function_value.csv"
        real_function_value_files = [f for f in os.listdir(test_folder) if f.endswith(real_function_value_file_suffix)]
        real_function_value = np.genfromtxt(test_folder + "/" + real_function_value_files[0])  # Need only one, as they are all the same
        dilution_factor = 1
        if len(real_function_value) > 10000:
            dilution_factor = 100
        data_len = len(real_function_value)
        axs[idx].plot(np.arange(data_len), real_function_value, label=r'$f \pm T$', color="black", linewidth=0.5)
        axs[idx].fill_between(np.arange(0, data_len, dilution_factor),
                              real_function_value[::dilution_factor] - error_bound,
                              real_function_value[::dilution_factor] + error_bound, facecolor='tab:blue', alpha=0.3)
        axs[idx].set_ylabel(func_names[idx])
        axs[idx].spines['right'].set_visible(False)
        axs[idx].spines['top'].set_visible(False)
        axs[idx].spines['bottom'].set_linewidth(0.5)
        axs[idx].spines['left'].set_linewidth(0.5)
        axs[idx].tick_params(width=0.5)
        axs[idx].yaxis.set_label_coords(-0.24, 0.5)
        if len(real_function_value) > 10000:
            axs[idx].xaxis.set_major_formatter(tick.FuncFormatter(reformat_large_tick_values))

    axs[len(func_names) - 1].set_xlabel('rounds', labelpad=2)
    axs[len(func_names) - 2].set_xlabel('rounds', labelpad=2)
    plt.subplots_adjust(top=0.99, bottom=0.17, left=0.115, right=0.985, hspace=0.65, wspace=0.4)
    fig.savefig("function_values_and_error_bound_around_it.pdf")

    rcParams.update(rcParamsDefault)


if __name__ == "__main__":
    # Figure 3

    # real_function_value.csv and config.txt files for MLP-40 and Rozenbrock were taken from the following experiment folders
    # mlp_40_test_folder = "../test_results/results_test_dimension_impact_mlp_2021-10-10_13-40-24/dimension_40"
    # rozenbrock_folder = "../test_results/results_comm_neighborhood_rozen_2021-09-20_15-53-41/0/thresh_0_05/0_fixed_neighborhood_0_05"

    kld_test_folder = "../../datasets/air_quality"
    inner_product_test_folder = "../../datasets/inner_product"
    mlp_2_test_folder = "../../datasets/MLP_2"
    quadratic_test_folder = "../../datasets/quadratic"
    dnn_intrusion_detection_test_folder = "../../datasets/intrusion_detection"
    mlp_40_test_folder = "./function_values/results_mlp_40_2021-10-10_13-40-24"
    rozenbrock_folder = "./function_values/results_rozenbrock_2021-04-10_10-13-41"

    plot_monitoring_f_and_error_bound_around_it(kld_test_folder, inner_product_test_folder,
                                                mlp_40_test_folder, mlp_2_test_folder,
                                                quadratic_test_folder, dnn_intrusion_detection_test_folder)
