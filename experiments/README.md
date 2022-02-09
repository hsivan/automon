# Reproduce experimental results

## Reproduce experiments using a single script
We provide a single script that runs all the experiments and generates the papers figures.
The script can be run as a standalone, and in that case it also downloads the project source code,
or it can be run as part of a cloned project.

To run the script as a standalone, download only the file [`reproduce_experiments.py`](../reproduce_experiments.py) and run it.
To run the script from within a cloned project, first clone the project,
`git clone https://github.com/hsivan/automon`,
and then run the script: `python <automon_root>/reproduce_experiments.py`.

To run the distributed experiments on AWS, in addition to the simulation experiments,
use the `--aws` flag when running the script. Read [this](../aws_experiments/README.md) for more information.

## Reproduce experiments manually

Before running the experiments, download AutoMon's source code:
```bash
git clone https://github.com/hsivan/automon
```
Let `automon_root` be the root folder of the project on your local computer.

Download the external datasets by running: `python <automon_root>/experiments/reproduce.py --dd`.
The script downloads the two external datasets and does the following:
1. [Air Quality](https://archive.ics.uci.edu/ml/datasets/Beijing+Multi-Site+Air-Quality+Data):
download `PRSA2017_Data_20130301-20170228.zip` from [here](https://archive.ics.uci.edu/ml/machine-learning-databases/00501/) and unzip it.
Put the 12 `PRSA_Data_XXX_20130301-20170228.csv` files in `<automon_root>/datasets/air_quality` folder.
2. [Intrusion Detection](http://kdd.ics.uci.edu/databases/kddcup99/kddcup99.html):
download `kddcup.data_10_percent.gz` and `corrected.gz` from [here](http://kdd.ics.uci.edu/databases/kddcup99/kddcup99.html) and decompress.
Put the decompressed `kddcup.data_10_percent` and `corrected` files in `<automon_root>/datasets/
_detection` folder.

Run all the experiments from within the `experiments` folder:
```bash
export PYTHONPATH=$PYTHONPATH:<automon_root>
cd <automon_root>/experiments
```

## Error-Communication Tradeoff (Sec. 4.3)
To reproduce _Error-Communication Tradeoff_ results run:
```bash
python test_max_error_vs_communication_inner_product.py
python test_max_error_vs_communication_quadratic.py
python test_max_error_vs_communication_kld_air_quality.py
python test_max_error_vs_communication_dnn_intrusion_detection.py
```
You will find the output files and figures in `<automon_root>/experiments/test_results/results_test_max_error_vs_communication_inner_product_xxx` folders.
To generate figures that summarize the four experiments, replace the xxx in the following command with the suffixes of the result folders, and run:
```bash
python visualization/plot_error_communication_tradeoff.py test_results \
    results_test_max_error_vs_communication_inner_product_xxx \
    results_test_max_error_vs_communication_quadratic_xxx \
    results_test_max_error_vs_communication_dnn_intrusion_detection_xxx \
    results_test_max_error_vs_communication_kld__air_quality_xxx
```

## Scalability: Dimensions, Nodes, Runtime (Sec. 4.4)
To reproduce _Scalability to Dimensionality_ results run:
```bash
python test_dimension_impact_inner_product.py
python test_dimension_impact_kld_air_quality.py
python test_dimension_impact_mlp.py
```
These generate the output folders `<automon_root>/experiments/test_results/results_test_dimension_impact_xxx`.
To generate figures that summarize the three experiments, replace the xxx in the following command with the suffixes of the result folders, and run:
```bash
python visualization/plot_dimensions_stats.py test_results \
    results_test_dimension_impact_inner_product_xxx \
    results_test_dimension_impact_kld_air_quality_xxx \
    results_test_dimension_impact_mlp_xxx
```
The script generates four figures:
1. dimension_communication.pdf shows the total number of messages in each run for different functions and input dimensions.
2. dimension_coordinator_runtime.pdf shows the average time for the full sync of an AutoMon coordinator.
3. dimension_node_runtime.pdf shows the average time a node takes to check a single data update for each dimension.
4. dimension_node_runtime_in_parts.pdf shows the average time it takes the node to complete different tasks during the data update process.

To reproduce _Scalability to Number of Nodes_ results run:
```bash
python test_num_nodes_impact_inner_product.py
python test_num_nodes_impact_mlp_40.py
```
These generate the output folders `<automon_root>/experiments/test_results/results_test_num_nodes_impact_xxx`.
To generate figures that summarize the two experiments, replace the xxx in the following command with the suffixes of the result folders, and run:
```bash
python visualization/plot_num_nodes_impact.py test_results \
    results_test_dimension_impact_inner_product_xxx \
    results_test_dimension_impact_mlp_40_xxx
```
The script generates the figure num_nodes_vs_communication.pdf that illustrates how the number of messages grows with
the number of AutoMon nodes.

## Impact of Neighborhood Size Tuning (Sec. 4.5)
To reproduce _Neighborhood Size Tuning_ results run:
```bash
python test_optimal_and_tuned_neighborhood_rozenbrock.py
python test_optimal_and_tuned_neighborhood_mlp_2.py
```
These generate the output folders `<automon_root>/experiments/test_results/results_optimal_and_tuned_neighborhood_xxx` where you will find the
figure neighborhood_size_error_bound_connection_avg.pdf and other output files.

Replace the xxx in the following commands with the suffixes of the result folders, and run:
```bash
python test_neighborhood_impact_on_communication_rozenbrock.py test_results/results_optimal_and_tuned_neighborhood_rozenbrock_xxx
python test_neighborhood_impact_on_communication_mlp_2.py test_results/results_optimal_and_tuned_neighborhood_mlp_2_xxx
```
You will find the output files and figures in `<automon_root>/experiments/test_results/results_comm_neighborhood_xxx` folders.
To generate figures that summarize the four experiments, replace the xxx in the following command with the suffixes of the result folders, and run:
```bash
python visualization/plot_neighborhood_impact.py test_results \
    results_optimal_and_tuned_neighborhood_rozenbrock_xxx \
    results_optimal_and_tuned_neighborhood_mlp_2_xxx \
    results_comm_neighborhood_rozenbrock_xxx \
    results_comm_neighborhood_mlp_2_xxx
```

## Impact of ADCD, Slack, and Lazy Sync (Sec. 4.6)
To reproduce the _Ablation Study_ results run:
```bash
python test_ablation_study_quadratic_inverse.py
python test_ablation_study_mlp_2.py
```
You will find the output files and figures in `<automon_root>/experiments/test_results/results_ablation_study_xxx` folders.
To generate figures that summarize the two experiments, replace the xxx in the following command with the suffixes of the result folders, and run:
```bash
python visualization/plot_monitoring_stats_ablation_study.py test_results \
    results_ablation_study_quadratic_inverse_xxx \
    results_ablation_study_mlp_2_xxx
```
