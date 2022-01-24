"""
Requirements:
(1) Python 3
(2) Docker engine (https://docs.docker.com/engine/install)

The script uses only libraries from the Python standard library, which prevents the need to install external packages.
"""
import urllib.request
import zipfile
import shutil
import os
import sys
import gzip
import subprocess
from pathlib import Path
from timeit import default_timer as timer
import argparse


def verify_requirements():
    """
    Verifies the requirements for running the script - Python 3 and docker engine installed
    :return:
    """
    if sys.version_info.major != 3:
        print("Running this script requires Python 3")
        exit(1)
    result = subprocess.run('docker version', shell=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
    if '\'docker\' is not recognized as an internal or external command' in result.stderr.decode():
        print("Running this script requires docker engine. See https://docs.docker.com/engine/install.")
        exit(1)


def download_repository():
    """
    Checks if the script as part of AutoMon cloned project or a standalone.
    If the script is part of a cloned project there is no need to download the source code.
    Otherwise, downloads AutoMon's code from GitHub.
    :return: the location of the source code (the project_root)
    """
    script_abs_dir = os.path.abspath(__file__).replace("reproduce.py", "")
    if os.path.isfile(script_abs_dir + "../requirements.txt"):
        print("The reproduce.py script is part of a cloned AutoMon project. No need to download AutoMon's source code.")
        project_root = '..'
        return project_root

    print("The reproduce.py script is a standalone. Downloading AutoMon's source code.")
    zipped_project = 'automon-main.zip'
    project_root = './' + zipped_project.replace(".zip", "")
    if os.path.isdir(project_root):
        # Source code already exists. No need to download again.
        return project_root
    urllib.request.urlretrieve('https://github.com/hsivan/automon/archive/refs/heads/main.zip', zipped_project)
    with zipfile.ZipFile(zipped_project, 'r') as zip_ref:
        zip_ref.extractall()
    os.remove(zipped_project)

    print("Downloaded the project to", project_root)
    return project_root


def download_air_quality_dataset(project_root):
    """
    Downloads the Air Quality dataset and copies it to the dataset's folder in the project folder
    :return:
    """
    # Check if the dataset already exists
    if os.path.isdir(project_root + '/datasets/air_quality/') and len([f for f in os.listdir(project_root + '/datasets/air_quality/') if "PRSA_Data" in f]) == 12:
        return
    zipped_dataset = 'PRSA2017_Data_20130301-20170228.zip'
    inner_dataset_folder = "PRSA_Data_20130301-20170228"
    # Download the file zipped_dataset, save it locally, extract it and copy the csv files to dataset_root
    urllib.request.urlretrieve('https://archive.ics.uci.edu/ml/machine-learning-databases/00501/' + zipped_dataset, zipped_dataset)
    with zipfile.ZipFile(zipped_dataset, 'r') as zip_ref:
        zip_ref.extractall()
    for f in os.listdir(inner_dataset_folder):
        shutil.copyfile(inner_dataset_folder + "/" + f, project_root + '/datasets/air_quality/' + f)
    os.remove(zipped_dataset)
    shutil.rmtree(inner_dataset_folder)


def download_intrusion_detection_dataset(project_root):
    """
    Downloads the Intrusion Detection dataset and copies it to the dataset's folder in the project folder
    :return:
    """
    gz_files = ['kddcup.data_10_percent.gz', 'corrected.gz']
    for gz_file in gz_files:
        # Check if the file already exists
        if os.path.isfile(project_root + '/datasets/intrusion_detection/' + gz_file.replace(".gz", "")):
            continue
        urllib.request.urlretrieve('http://kdd.ics.uci.edu/databases/kddcup99/' + gz_file, gz_file)
        with gzip.open(gz_file, 'rb') as f_in:
            with open(project_root + '/datasets/intrusion_detection/' + gz_file.replace(".gz", ""), 'wb') as f_out:
                shutil.copyfileobj(f_in, f_out)
        os.remove(gz_file)


def download_external_datasets(project_root):
    """
    Downloads the Air Quality and the Intrusion Detection datasets
    :param project_root:
    :return:
    """
    download_air_quality_dataset(project_root)
    download_intrusion_detection_dataset(project_root)


def execute_shell_command(cmd, stdout_verification=None, b_stderr_verification=False):
    result = subprocess.run(cmd, shell=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
    print("Executed command:", cmd)
    print("STDOUT:")
    print(result.stdout)
    print("STDERR:")
    print(result.stderr)

    if stdout_verification is not None:
        if stdout_verification not in result.stdout.decode():
            print("Verification string", stdout_verification, "not in stdout")
            exit(1)
    if b_stderr_verification:
        if result.stderr != b'':
            print("stderr is not empty.")
            exit(1)


def execute_shell_command_with_live_output(cmd):
    """
    Executes shell command with live output of STD, so it would not be boring :)
    :param cmd: the command to execute
    :return:
    """
    process = subprocess.Popen(cmd, shell=True, stdout=subprocess.PIPE, stderr=subprocess.STDOUT)
    print("Executed command:", cmd)
    while True:
        output = process.stdout.readline()
        if process.poll() is not None:
            break
        if output:
            print(output.strip().decode('utf-8'))
    rc = process.poll()
    if rc != 0:
        print("RC not 0 (RC=" + rc + ") for command:", cmd)
        exit(1)
    return rc


def edit_dockerignore():
    """
    Edits the .dockerignore file so it will not ignore the experiments folder.
    :return:
    """
    with open('./.dockerignore', 'r') as f:
        content = f.read()
        content = content.replace("experiments", "experiments/test_results")
    with open('./.dockerignore', 'w') as f:
        f.write(content)


def revert_dockerignore_changes():
    """
    Reverts changes made to the .dockerignore file by the edit_dockerignore() function.
    :return:
    """
    with open('./.dockerignore', 'r') as f:
        content = f.read()
        content = content.replace("experiments/test_results", "experiments")
    with open('./.dockerignore', 'w') as f:
        f.write(content)


def get_latest_test_folder(result_dir, filter_str):
    """
    Returns the latest, most updated, test folder in result_dir that contains filter_str in its name.
    If no such folder exists, returns None.
    :param result_dir:
    :param filter_str:
    :return:
    """
    paths = sorted(Path(result_dir).iterdir(), key=os.path.getmtime)
    if len(paths) == 0:
        return None
    test_folders = [str(f) for f in paths if filter_str in str(f)]
    if len(test_folders) == 0:
        return None
    return test_folders[-1].split('/')[-1]


def run_experiment(local_result_dir, docker_run_command_prefix, functions, test_name_prefix, result_folder_prefix, file_to_verify_execution, args=None):
    """
    Receives a list of functions and runs a given experiment for each function in the list.
    :param local_result_dir: path of the folder where the experiment's output folder is mapped to on local computer
    :param docker_run_command_prefix: docker run command prefix, which includes mapping of result folder and docker image name
    :param functions: the functions included in the experiment (e.g. inner product, kld, etc.). Running an experiment for each function
    :param test_name_prefix: the prefix of the experiment script
    :param result_folder_prefix: the prefix of the experiment result folder
    :param file_to_verify_execution: the existence of the file indicates if the experiment have been executed successfully already
    :param args: if not None, this is a list of string, each of them is an argument to one function's experiment
    :return:
    """
    test_folders = []
    for i, function in enumerate(functions):
        test_folder = get_latest_test_folder(local_result_dir, result_folder_prefix + function)
        if test_folder and os.path.isfile(local_result_dir + "/" + test_folder + "/" + file_to_verify_execution):
            print("Found existing test folder for " + function + ": " + test_folder + ". Skipping.")
        else:
            cmd = docker_run_command_prefix + test_name_prefix + function + '.py'
            if args:
                cmd += ' ' + args[i]
            start = timer()
            execute_shell_command_with_live_output(cmd)
            end = timer()
            print("The experiment", test_name_prefix + function + '.py', "took: ", str(end - start), " seconds")
            test_folder = get_latest_test_folder(local_result_dir, result_folder_prefix + function)
        assert test_folder
        assert os.path.isfile(local_result_dir + "/" + test_folder + "/" + file_to_verify_execution)
        test_folders.append(test_folder)
    return test_folders


def generate_figures(docker_result_dir, docker_run_command_prefix, test_folders, plot_script):
    """
    Generates figures from the experiment results.
    :param docker_result_dir: path of the folder where the experiment's output folder is created on the docker
    :param docker_run_command_prefix: docker run command prefix, which includes mapping of result folder and docker image name
    :param test_folders: the output folders of the experiment
    :param plot_script: the script to run that generates the figures
    :return:
    """
    cmd = docker_run_command_prefix + 'visualization/' + plot_script + ' ' + docker_result_dir + " " + " ".join(test_folders)
    execute_shell_command_with_live_output(cmd)


def run_error_communication_tradeoff_experiment(local_result_dir, docker_result_dir, docker_run_command_prefix):
    """
    Runs Error-Communication Tradeoff experiment and generates Figures 5 and 6 (Sec. 4.3 in the paper).
    The result folders and figures could be found in local_result_dir (<project_root>/test_results)
    :param local_result_dir: path of the folder where the experiment's output folder is mapped to on local computer
    :param docker_result_dir: path of the folder where the experiment's output folder is created on the docker
    :param docker_run_command_prefix: docker run command prefix, which includes mapping of result folder and docker image name
    :return:
    """
    print("Executing Error-Communication Tradeoff experiment")
    test_name_prefix = "test_max_error_vs_communication_"
    result_folder_prefix = "results_" + test_name_prefix
    functions = ["inner_product", "quadratic", "dnn_intrusion_detection", "kld_air_quality"]

    test_folders = run_experiment(local_result_dir, docker_run_command_prefix, functions, test_name_prefix, result_folder_prefix, "max_error_vs_communication.pdf")
    generate_figures(docker_result_dir, docker_run_command_prefix, test_folders, 'plot_error_communication_tradeoff.py')

    print("Successfully executed Error-Communication Tradeoff experiment")


def run_scalability_to_dimensionality_experiment(local_result_dir, docker_result_dir, docker_run_command_prefix):
    """
    Runs Scalability to Dimensionality experiment and generates Figure 7 (a) and more (Sec. 4.4 in the paper).
    The result folders and figures could be found in local_result_dir (<project_root>/test_results)
    :param local_result_dir: path of the folder where the experiment's output folder is mapped to on local computer
    :param docker_result_dir: path of the folder where the experiment's output folder is created on the docker
    :param docker_run_command_prefix: docker run command prefix, which includes mapping of result folder and docker image name
    :return:
    """
    print("Executing Scalability to Dimensionality experiment")
    test_name_prefix = "test_dimension_impact_"
    result_folder_prefix = "results_" + test_name_prefix
    functions = ["inner_product", "kld_air_quality", "mlp"]

    test_folders = run_experiment(local_result_dir, docker_run_command_prefix, functions, test_name_prefix, result_folder_prefix, "dimension_200/results.txt")
    generate_figures(docker_result_dir, docker_run_command_prefix, test_folders, 'plot_dimensions_stats.py')

    print("Successfully executed Scalability to Dimensionality experiment")


def run_scalability_to_number_of_nodes_experiment(local_result_dir, docker_result_dir, docker_run_command_prefix):
    """
    Runs Scalability to Number of Nodes experiment and generates Figure 7 (b) (Sec. 4.4 in the paper).
    The result folders and figures could be found in local_result_dir (<project_root>/test_results)
    :param local_result_dir: path of the folder where the experiment's output folder is mapped to on local computer
    :param docker_result_dir: path of the folder where the experiment's output folder is created on the docker
    :param docker_run_command_prefix: docker run command prefix, which includes mapping of result folder and docker image name
    :return:
    """
    print("Executing Scalability to Number of Nodes experiment")
    test_name_prefix = "test_num_nodes_impact_"
    result_folder_prefix = "results_" + test_name_prefix
    functions = ["inner_product", "mlp_40"]

    test_folders = run_experiment(local_result_dir, docker_run_command_prefix, functions, test_name_prefix, result_folder_prefix, "num_nodes_vs_communication.pdf")
    generate_figures(docker_result_dir, docker_run_command_prefix, test_folders, 'plot_num_nodes_impact.py')

    print("Successfully executed Scalability to Number of Nodes experiment")


def run_neighborhood_size_tuning_experiment(local_result_dir, docker_result_dir, docker_run_command_prefix):
    """
    Runs Neighborhood Size Tuning experiment and generates Figures 3, 8, 9 (Sec. 3.6 and 4.5 in the paper).
    The result folders and figures could be found in local_result_dir (<project_root>/test_results)
    :param local_result_dir: path of the folder where the experiment's output folder is mapped to on local computer
    :param docker_result_dir: path of the folder where the experiment's output folder is created on the docker
    :param docker_run_command_prefix: docker run command prefix, which includes mapping of result folder and docker image name
    :return:
    """
    print("Executing Neighborhood Size Tuning experiment")
    test_name_prefix = "test_optimal_and_tuned_neighborhood_"
    result_folder_prefix = "results_optimal_and_tuned_neighborhood_"
    functions = ["rozenbrock", "mlp_2"]

    test_folders = run_experiment(local_result_dir, docker_run_command_prefix, functions, test_name_prefix, result_folder_prefix, "neighborhood_size_error_bound_connection_avg.pdf")

    test_name_prefix = "test_neighborhood_impact_on_communication_"
    result_folder_prefix = "results_comm_neighborhood_"

    args = [docker_result_dir + "/" + f for f in test_folders]
    test_folders += run_experiment(local_result_dir, docker_run_command_prefix, functions, test_name_prefix, result_folder_prefix, "neighborhood_impact_on_communication_error_bound_connection.pdf", args)

    generate_figures(docker_result_dir, docker_run_command_prefix, test_folders, 'plot_neighborhood_impact.py')

    print("Successfully executed Neighborhood Size Tuning experiment")


def run_ablation_study_experiment(local_result_dir, docker_result_dir, docker_run_command_prefix):
    """
    Runs Ablation Study experiment and generates Figures 10 (Sec. 4.6 in the paper).
    The result folders and figures could be found in local_result_dir (<project_root>/test_results)
    :param local_result_dir: path of the folder where the experiment's output folder is mapped to on local computer
    :param docker_result_dir: path of the folder where the experiment's output folder is created on the docker
    :param docker_run_command_prefix: docker run command prefix, which includes mapping of result folder and docker image name
    :return:
    """
    print("Executing Ablation Study experiment")
    test_name_prefix = "test_ablation_study_"
    result_folder_prefix = "results_ablation_study_"
    functions = ["quadratic_inverse", "mlp_2"]

    test_folders = run_experiment(local_result_dir, docker_run_command_prefix, functions, test_name_prefix, result_folder_prefix, "results.txt")
    generate_figures(docker_result_dir, docker_run_command_prefix, test_folders, 'plot_monitoring_stats_ablation_study.py')

    print("Successfully executed Ablation Study experiment")


if __name__ == "__main__":

    parser = argparse.ArgumentParser()
    parser.add_argument("--dd", dest="b_download_dataset", help="if --dd is specified, the script only downloads the external datasets", action='store_true')
    args = parser.parse_args()

    if not args.b_download_dataset:
        verify_requirements()
    project_root = download_repository()
    download_external_datasets(project_root)
    print("Downloaded external datasets")
    if args.b_download_dataset:
        exit(0)

    try:
        # Build docker image with .dockerignore which does not ignore the experiments folder
        os.chdir(project_root)
        edit_dockerignore()
        execute_shell_command_with_live_output('sudo docker build -f experiments/experiments.Dockerfile -t automon_experiment .')
        print("Successfully built docker image for the experiments")

        # Run the experiments
        docker_result_dir = '/app/experiments/test_results'
        local_result_dir = os.getcwd() + "/test_results"
        docker_run_command_prefix = 'sudo docker run -v ' + local_result_dir + ':' + docker_result_dir + ' --rm automon_experiment python /app/experiments/'

        print("Experiment results are written to:", local_result_dir)

        # Run Error-Communication Tradeoff experiment and generates Figures 5 and 6 (Sec. 4.3 in the paper)
        run_error_communication_tradeoff_experiment(local_result_dir, docker_result_dir, docker_run_command_prefix)
        run_scalability_to_dimensionality_experiment(local_result_dir, docker_result_dir, docker_run_command_prefix)
        run_scalability_to_number_of_nodes_experiment(local_result_dir, docker_result_dir, docker_run_command_prefix)
        run_neighborhood_size_tuning_experiment(local_result_dir, docker_result_dir, docker_run_command_prefix)
        run_ablation_study_experiment(local_result_dir, docker_result_dir, docker_run_command_prefix)

        # TODO: Add option to run the AWS experiments.
        # TODO: build the paper from its source files with the new figures

    finally:
        revert_dockerignore_changes()
