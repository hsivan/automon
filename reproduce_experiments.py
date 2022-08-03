"""
Reproduce AutoMon's experiments. The script downloads the code from https://github.com/hsivan/automon, downloads the
external datasets, runs the experiments, generates the paper's figures, and finally compiles the paper's Latex source
with the newly generated figures.
Run this script on Linux (Ubuntu 18.04 or later).

Requirements:
(1) Python 3
(2) Docker engine (see https://docs.docker.com/engine/install/ubuntu/, and make sure running 'sudo docker run hello-world' works as expected)

The script uses only libraries from the Python standard library, which prevents the need to install external packages.
It installs TexLive for the compilation of the paper's Latex source files.

Note: in case the script is called with the --aws options it installs AWS cli and configures it.
After running the simulations, it runs the distributed experiments on AWS.
It requires the user to have an AWS account; after opening the account, the user must create AWS IAM user with
AdministratorAccess permissions, download the csv file new_user_credentials.csv that contains the key ID and the secret
key, and place the new_user_credentials.csv file in project_root/aws_experiments/new_user_credentials.csv.
After completing these steps, re-run this script.
Running the AWS experiments would cost a few hundred dollars!
The user must monitor the ECS tasks and EC2 instances, and manually shutdown any dangling tasks/instances in case of
failures.
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
import datetime
import argparse
from argparse import RawTextHelpFormatter


def print_to_std_and_file(str_to_log):
    print(str_to_log)
    with open(log_file, 'a') as f:
        test_timestamp = datetime.datetime.now().strftime('%Y-%m-%d_%H-%M-%S')
        f.write(test_timestamp + ": " + str_to_log + "\n")


def verify_requirements():
    """
    Verifies the requirements for running the script - Python 3 and docker engine installed
    :return:
    """
    if sys.version_info.major != 3:
        print("Running this script requires Python 3")
        sys.exit(1)
    result = subprocess.run('docker version', shell=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
    if '\'docker\' is not recognized as an internal or external command' in result.stderr.decode():
        print_to_std_and_file("Running this script requires docker engine. See https://docs.docker.com/engine/install.")
        sys.exit(1)


def download_repository():
    """
    Checks if the script is part of AutoMon's cloned project or a standalone.
    If the script is part of a cloned project there is no need to download the source code.
    Otherwise, downloads AutoMon's code from GitHub.
    :return: the location of the source code (the project_root)
    """
    script_abs_dir = os.path.abspath(__file__).replace("reproduce_experiments.py", "")
    if os.path.isfile(script_abs_dir + "./requirements.txt"):
        print_to_std_and_file("The reproduce_experiments.py script is part of a cloned AutoMon project. No need to download AutoMon's source code.")
        project_root = '..'
        return project_root

    print_to_std_and_file("The reproduce_experiments.py script is a standalone. Downloading AutoMon's source code.")
    zipped_project = 'automon-main.zip'
    project_root = script_abs_dir + '/' + zipped_project.replace(".zip", "")
    if os.path.isdir(project_root):
        # Source code already exists. No need to download again.
        return project_root
    urllib.request.urlretrieve('https://github.com/hsivan/automon/archive/refs/heads/main.zip', zipped_project)
    with zipfile.ZipFile(zipped_project, 'r') as zip_ref:
        zip_ref.extractall()
    os.remove(zipped_project)

    print_to_std_and_file("Downloaded the project to " + project_root)
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
    target_file_names = ['kddcup.data_10_percent_corrected', 'corrected']
    for i, gz_file in enumerate(gz_files):
        # Check if the file already exists
        if os.path.isfile(project_root + '/datasets/intrusion_detection/' + gz_file.replace(".gz", "")):
            continue
        urllib.request.urlretrieve('http://kdd.ics.uci.edu/databases/kddcup99/' + gz_file, gz_file)
        with gzip.open(gz_file, 'rb') as f_in:
            with open(project_root + '/datasets/intrusion_detection/' + target_file_names[i], 'wb') as f_out:
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
    print_to_std_and_file("Executed command: " + cmd)
    print("STDOUT:")
    print(result.stdout)
    print("STDERR:")
    print(result.stderr)

    if stdout_verification is not None:
        if stdout_verification not in result.stdout.decode():
            print_to_std_and_file("Verification string " + stdout_verification + " not in stdout")
            raise Exception
    if b_stderr_verification:
        if result.stderr != b'':
            print_to_std_and_file("stderr is not empty: " + result.stderr)
            raise Exception


def execute_shell_command_with_live_output(cmd):
    """
    Executes shell command with live output of STD, so it would not be boring :)
    :param cmd: the command to execute
    :return:
    """
    process = subprocess.Popen(cmd, shell=True, stdout=subprocess.PIPE, stderr=subprocess.STDOUT)
    print_to_std_and_file("Executed command: " + cmd)
    while True:
        output = process.stdout.readline()
        if process.poll() is not None:
            break
        if output:
            print(output.strip().decode('utf-8'))
    rc = process.poll()
    if rc != 0:
        print_to_std_and_file("RC not 0 (RC=" + str(rc) + ") for command: " + cmd)
        raise Exception
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
            print_to_std_and_file("Found existing test folder for " + function + ": " + test_folder + ". Skipping.")
        else:
            cmd = docker_run_command_prefix + test_name_prefix + function + '.py'
            if args:
                cmd += ' ' + args[i]
            start = timer()
            execute_shell_command_with_live_output(cmd)
            end = timer()
            print_to_std_and_file('The experiment ' + test_name_prefix + function + '.py took: ' + str(end - start) + ' seconds')
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
    print_to_std_and_file("Executing Error-Communication Tradeoff experiment")
    test_name_prefix = "test_max_error_vs_communication_"
    result_folder_prefix = "results_" + test_name_prefix
    functions = ["inner_product", "quadratic", "dnn_intrusion_detection", "kld_air_quality"]

    test_folders = run_experiment(local_result_dir, docker_run_command_prefix, functions, test_name_prefix, result_folder_prefix, "max_error_vs_communication.pdf")
    generate_figures(docker_result_dir, docker_run_command_prefix, test_folders, 'plot_error_communication_tradeoff.py')

    print_to_std_and_file("Successfully executed Error-Communication Tradeoff experiment")


def run_scalability_to_dimensionality_experiment(local_result_dir, docker_result_dir, docker_run_command_prefix):
    """
    Runs Scalability to Dimensionality experiment and generates Figure 7 (a) and more (Sec. 4.4 in the paper).
    The result folders and figures could be found in local_result_dir (<project_root>/test_results)
    :param local_result_dir: path of the folder where the experiment's output folder is mapped to on local computer
    :param docker_result_dir: path of the folder where the experiment's output folder is created on the docker
    :param docker_run_command_prefix: docker run command prefix, which includes mapping of result folder and docker image name
    :return:
    """
    print_to_std_and_file("Executing Scalability to Dimensionality experiment")
    test_name_prefix = "test_dimension_impact_"
    result_folder_prefix = "results_" + test_name_prefix
    functions = ["inner_product", "kld_air_quality", "mlp"]

    test_folders = run_experiment(local_result_dir, docker_run_command_prefix, functions, test_name_prefix, result_folder_prefix, "dimension_200/results.txt")
    generate_figures(docker_result_dir, docker_run_command_prefix, test_folders, 'plot_dimensions_stats.py')

    print_to_std_and_file("Successfully executed Scalability to Dimensionality experiment")


def run_scalability_to_number_of_nodes_experiment(local_result_dir, docker_result_dir, docker_run_command_prefix):
    """
    Runs Scalability to Number of Nodes experiment and generates Figure 7 (b) (Sec. 4.4 in the paper).
    The result folders and figures could be found in local_result_dir (<project_root>/test_results)
    :param local_result_dir: path of the folder where the experiment's output folder is mapped to on local computer
    :param docker_result_dir: path of the folder where the experiment's output folder is created on the docker
    :param docker_run_command_prefix: docker run command prefix, which includes mapping of result folder and docker image name
    :return:
    """
    print_to_std_and_file("Executing Scalability to Number of Nodes experiment")
    test_name_prefix = "test_num_nodes_impact_"
    result_folder_prefix = "results_" + test_name_prefix
    functions = ["inner_product", "mlp_40"]

    test_folders = run_experiment(local_result_dir, docker_run_command_prefix, functions, test_name_prefix, result_folder_prefix, "num_nodes_vs_communication.pdf")
    generate_figures(docker_result_dir, docker_run_command_prefix, test_folders, 'plot_num_nodes_impact.py')

    print_to_std_and_file("Successfully executed Scalability to Number of Nodes experiment")


def run_neighborhood_size_tuning_experiment(local_result_dir, docker_result_dir, docker_run_command_prefix):
    """
    Runs Neighborhood Size Tuning experiment and generates Figures 3, 8, 9 (Sec. 3.6 and 4.5 in the paper).
    The result folders and figures could be found in local_result_dir (<project_root>/test_results)
    :param local_result_dir: path of the folder where the experiment's output folder is mapped to on local computer
    :param docker_result_dir: path of the folder where the experiment's output folder is created on the docker
    :param docker_run_command_prefix: docker run command prefix, which includes mapping of result folder and docker image name
    :return:
    """
    print_to_std_and_file("Executing Neighborhood Size Tuning experiment")
    test_name_prefix = "test_optimal_and_tuned_neighborhood_"
    result_folder_prefix = "results_optimal_and_tuned_neighborhood_"
    functions = ["rozenbrock", "mlp_2"]

    test_folders = run_experiment(local_result_dir, docker_run_command_prefix, functions, test_name_prefix, result_folder_prefix, "neighborhood_size_error_bound_connection_avg.pdf")

    test_name_prefix = "test_neighborhood_impact_on_communication_"
    result_folder_prefix = "results_comm_neighborhood_"

    args = [docker_result_dir + "/" + f for f in test_folders]
    test_folders += run_experiment(local_result_dir, docker_run_command_prefix, functions, test_name_prefix, result_folder_prefix, "neighborhood_impact_on_communication_error_bound_connection.pdf", args)

    generate_figures(docker_result_dir, docker_run_command_prefix, test_folders, 'plot_neighborhood_impact.py')

    print_to_std_and_file("Successfully executed Neighborhood Size Tuning experiment")


def run_ablation_study_experiment(local_result_dir, docker_result_dir, docker_run_command_prefix):
    """
    Runs Ablation Study experiment and generates Figures 10 (Sec. 4.6 in the paper).
    The result folders and figures could be found in local_result_dir (<project_root>/test_results)
    :param local_result_dir: path of the folder where the experiment's output folder is mapped to on local computer
    :param docker_result_dir: path of the folder where the experiment's output folder is created on the docker
    :param docker_run_command_prefix: docker run command prefix, which includes mapping of result folder and docker image name
    :return:
    """
    print_to_std_and_file("Executing Ablation Study experiment")
    test_name_prefix = "test_ablation_study_"
    result_folder_prefix = "results_ablation_study_"
    functions = ["quadratic_inverse", "mlp_2"]

    test_folders = run_experiment(local_result_dir, docker_run_command_prefix, functions, test_name_prefix, result_folder_prefix, "results.txt")
    generate_figures(docker_result_dir, docker_run_command_prefix, test_folders, 'plot_monitoring_stats_ablation_study.py')

    print_to_std_and_file("Successfully executed Ablation Study experiment")


def run_aws_experiment(node_type, coordinator_aws_instance_type, local_result_dir, b_centralized):
    if b_centralized:
        node_name = 'centralization_' + node_type
        b_centralization = ' --centralized'
    else:
        node_name = node_type
        b_centralization = ''

    test_folder = get_latest_test_folder(local_result_dir, "max_error_vs_comm_" + node_name + "_aws/")
    if test_folder:
        print_to_std_and_file("Found existing local AWS test folder for " + node_name + ": " + test_folder + ". Skipping.")
        return

    start = timer()
    # Run the AWS deploy script inside a docker container to avoid the need to install boto3, etc. Use --block flag so the docker waits until it finds the results in AWS S3.
    execute_shell_command_with_live_output('sudo docker run --rm automon_aws_experiment python /app/aws_experiments/deploy_aws_experiment.py --node_type ' + node_type + ' --coordinator_aws_instance_type ' + coordinator_aws_instance_type + ' --block' + b_centralization)

    # Collect the results from S3
    test_timestamp = datetime.datetime.now().strftime('%Y-%m-%d_%H-%M-%S')
    test_folder = os.path.join(local_result_dir, "max_error_vs_comm_" + node_name + "_aws_" + test_timestamp)
    os.makedirs(test_folder)
    # This command requires AWS cli installed and configured
    execute_shell_command_with_live_output('aws s3 cp s3://automon-experiment-results/max_error_vs_comm_' + node_name + '_aws ' + test_folder + ' --recursive')
    end = timer()
    print_to_std_and_file('The distributed experiment' + node_name + ' took: ' + str(end - start) + ' seconds')


def generate_aws_figures(local_result_dir, docker_result_dir, docker_run_command_prefix):
    # Plot figures
    test_folders = [get_latest_test_folder(local_result_dir, "results_test_max_error_vs_communication_inner_product"),
                    get_latest_test_folder(local_result_dir, "results_test_max_error_vs_communication_quadratic"),
                    get_latest_test_folder(local_result_dir, "results_test_max_error_vs_communication_kld_air_quality"),
                    get_latest_test_folder(local_result_dir, "results_test_max_error_vs_communication_dnn_intrusion_detection"),
                    get_latest_test_folder(local_result_dir, "max_error_vs_comm_inner_product_aws"),
                    get_latest_test_folder(local_result_dir, "max_error_vs_comm_quadratic_aws"),
                    get_latest_test_folder(local_result_dir, "max_error_vs_comm_kld_aws"),
                    get_latest_test_folder(local_result_dir, "max_error_vs_comm_dnn_aws"),
                    get_latest_test_folder(local_result_dir, "max_error_vs_comm_centralization_inner_product_aws"),
                    get_latest_test_folder(local_result_dir, "max_error_vs_comm_centralization_quadratic_aws"),
                    get_latest_test_folder(local_result_dir, "max_error_vs_comm_centralization_kld_aws"),
                    get_latest_test_folder(local_result_dir, "max_error_vs_comm_centralization_dnn_aws")]

    for test_folder in test_folders:
        # Make sure that all test folders were found
        assert test_folder

    generate_figures(docker_result_dir, docker_run_command_prefix, test_folders, 'plot_aws_stats.py')


def install_aws_cli():
    try:
        execute_shell_command('aws --version', stdout_verification="aws-cli")
    except:
        pass
    else:
        # AWS cli is already installed
        return
    execute_shell_command_with_live_output('sudo apt-get -y install unzip')
    execute_shell_command_with_live_output('curl "https://awscli.amazonaws.com/awscli-exe-linux-x86_64.zip" -o "awscliv2.zip"')
    execute_shell_command_with_live_output('unzip awscliv2.zip')
    execute_shell_command_with_live_output('sudo ./aws/install')
    execute_shell_command('aws --version', stdout_verification="aws-cli")


def configure_aws_cli(region='us-east-2'):
    # Get the access key and secret access key from the new_user_credentials.csv file, without using pandas, boto3, etc.
    with open('./aws_experiments/new_user_credentials.csv', 'r') as f:
        credentials = f.read()
        access_key_id = credentials.split('link')[1].split(',')[2]
        secret_access_key = credentials.split('link')[1].split(',')[3]

    execute_shell_command_with_live_output('aws configure set aws_access_key_id ' + access_key_id)
    execute_shell_command_with_live_output('aws configure set aws_secret_access_key ' + secret_access_key)
    execute_shell_command_with_live_output('aws configure set region ' + region)
    execute_shell_command_with_live_output('aws configure set output json')
    execute_shell_command('aws configure get region', stdout_verification=region)  # Verify configuration worked


def build_and_push_docker_image_to_aws_ecr():
    execute_shell_command_with_live_output('sudo docker build -f aws_experiments/awstest.Dockerfile  -t automon_aws_experiment .')
    print_to_std_and_file("Successfully built docker image for the AWS experiments")
    # Get the AWS account id from the new_user_credentials.csv file, without using pandas, boto3, etc.
    with open('./aws_experiments/new_user_credentials.csv', 'r') as f:
        credentials = f.read()
        account_id = credentials.split('https://')[1].split('.signin')[0]
    # These two commands require AWS cli installed and configured
    execute_shell_command_with_live_output('aws ecr describe-repositories --repository-names automon || aws ecr create-repository --repository-name automon')
    execute_shell_command_with_live_output('aws ecr get-login-password --region us-east-2 | sudo docker login --username AWS --password-stdin ' + account_id + '.dkr.ecr.us-east-2.amazonaws.com/automon')
    print_to_std_and_file("Successfully obtained ECR login password")
    execute_shell_command_with_live_output('sudo docker tag automon_aws_experiment ' + account_id + '.dkr.ecr.us-east-2.amazonaws.com/automon')
    print_to_std_and_file("Successfully tagged docker image")
    execute_shell_command_with_live_output('sudo docker push ' + account_id + '.dkr.ecr.us-east-2.amazonaws.com/automon')
    print_to_std_and_file("Successfully pushed docker image to ECR")


def run_aws_experiments(local_result_dir, docker_result_dir, docker_run_command_prefix):
    install_aws_cli()
    configure_aws_cli()
    build_and_push_docker_image_to_aws_ecr()

    # Run AutoMon distributed experiments
    run_aws_experiment('inner_product', 'ec2', local_result_dir, b_centralized=False)
    run_aws_experiment('quadratic', 'ec2', local_result_dir, b_centralized=False)
    run_aws_experiment('kld', 'ec2', local_result_dir, b_centralized=False)
    run_aws_experiment('dnn', 'ec2', local_result_dir, b_centralized=False)

    # Run the distributed centralized experiments
    run_aws_experiment('inner_product', 'ec2', local_result_dir, b_centralized=True)
    run_aws_experiment('quadratic', 'ec2', local_result_dir, b_centralized=True)
    run_aws_experiment('kld', 'ec2', local_result_dir, b_centralized=True)
    run_aws_experiment('dnn', 'ec2', local_result_dir, b_centralized=True)

    # Plot figures
    generate_aws_figures(local_result_dir, docker_result_dir, docker_run_command_prefix)


def compile_reproduced_main_pdf():
    """
    Installs TexLive for the compilation of the paper's Latex source files and builds the paper from Latex source
    files with the new figures.
    :param
    :return:
    """
    execute_shell_command_with_live_output('sudo apt-get install -y texlive-latex-base texlive-fonts-recommended texlive-fonts-extra texlive-latex-extra texlive-science')

    # Copy figures from local_result_dir to project_root/docs/latex_src/figures and report of missing figures
    figure_list = {"Figure 3": "impact_of_neighborhood_on_violations_three_error_bounds.pdf",
                   "Figure 5": "max_error_vs_communication.pdf",
                   "Figure 6": "percent_error_kld_and_dnn.pdf",
                   "Figure 7 (a)": "dimension_communication.pdf",
                   "Figure 7 (b)": "num_nodes_vs_communication.pdf",
                   "Figure 8": "neighborhood_impact_on_communication_error_bound_connection.pdf",
                   "Figure 9 (a)": "monitoring_stats_quadratic_inverse.pdf",
                   "Figure 9 (b)": "monitoring_stats_barchart_mlp_2.pdf",
                   "Figure 10 (top)": "max_error_vs_transfer_volume.pdf",
                   "Figure 10 (bottom)": "communication_automon_vs_network.pdf"}
    for figure_name, figure_file in figure_list.items():
        if not os.path.isfile(local_result_dir + "/" + figure_file):
            print_to_std_and_file("Note: " + figure_name + " (" + figure_file + ") wasn't found in " + local_result_dir + ". Using the original figure from the paper.")
        else:
            shutil.copyfile(local_result_dir + "/" + figure_file, project_root + "/docs/latex_src/figures/" + figure_file)
            print_to_std_and_file("Replaced " + figure_name + " (" + figure_file + ") with the reproduced one.")

    os.chdir(project_root + "/docs/latex_src")

    execute_shell_command('pdflatex main.tex')
    execute_shell_command('bibtex main.aux')
    execute_shell_command('pdflatex main.tex')
    execute_shell_command('pdflatex main.tex', stdout_verification="main.pdf")

    shutil.copyfile("main.pdf", reproduced_main_pfd_file)
    print_to_std_and_file("The reproduced paper, containing the new figures based on these experiments, is in " + reproduced_main_pfd_file)


if __name__ == "__main__":

    parser = argparse.ArgumentParser(description="Reproduce AutoMon's experiments. The script downloads the code from https://github.com/hsivan/automon, downloads the external datasets, and runs the experiments.\n"
                                                 "Run this script on Linux (Ubuntu 18.04 or later).\n\n"
                                                 "Requirements:\n"
                                                 "(1) Python 3\n"
                                                 "(2) Docker engine (see https://docs.docker.com/engine/install/ubuntu, and make sure running 'sudo docker run hello-world' works as expected)\n\n"
                                                 "The script uses only libraries from the Python standard library, which prevents the need to install external packages.", formatter_class=RawTextHelpFormatter)

    parser.add_argument("--dd", dest="b_download_dataset", help="if --dd is specified, the script only downloads the repository and external datasets and exits (without running any experiments)", action='store_true')
    parser.add_argument("--aws", dest="b_aws_experiments", help="if --aws is specified, also run AWS experiments (in addition to the simulation experiments).\n"
                                                                "Note: in case the script is called with the --aws options it installs AWS cli and configures it. After running\n"
                                                                "the simulations, it runs the distributed experiments on AWS.\n"
                                                                "It requires the user to have an AWS account; after opening the account, the user must create AWS IAM user with\n"
                                                                "AdministratorAccess permissions, download the csv file new_user_credentials.csv that contains the key ID and the secret\n"
                                                                "key, and place the new_user_credentials.csv file in project_root/aws_experiments/new_user_credentials.csv.\n"
                                                                "After completing these steps, re-run this script.\n"
                                                                "Running the AWS experiments would cost a few hundred dollars!\n"
                                                                "The user must monitor the ECS tasks and EC2 instances, and manually shutdown any dangling tasks/instances in case of failures.", action='store_true')
    args = parser.parse_args()

    log_file = os.path.abspath(__file__).replace("reproduce_experiments.py", "reproduce_experiments.log")
    reproduced_main_pfd_file = os.path.abspath(__file__).replace("reproduce_experiments.py", "main.log")
    print_to_std_and_file("======================== Reproduce AutoMon's Experiments ========================")
    print("The script log is at", log_file)

    project_root = download_repository()
    download_external_datasets(project_root)
    print_to_std_and_file("Downloaded external datasets")
    if args.b_download_dataset:
        sys.exit()

    verify_requirements()
    os.chdir(project_root)

    if args.b_aws_experiments:
        # Include the following only here, after the source code was downloaded
        # Verify the new_user_credentials.csv exists
        if not os.path.isfile('aws_experiments/new_user_credentials.csv'):
            print_to_std_and_file("To run AWS experiments, you must have an AWS account. After opening the account, create AWS IAM user with "
                                  "AdministratorAccess permissions and download the csv file new_user_credentials.csv that contains the key ID and the secret key. "
                                  "Place the new_user_credentials.csv file in " + project_root + "/aws_experiments/new_user_credentials.csv and re-run this script.\n"
                                  "Note: AWS cli will be installed on your computer and will be configured!")
            sys.exit(1)

    try:
        # Build docker image with .dockerignore which does not ignore the experiments folder
        edit_dockerignore()
        execute_shell_command_with_live_output('sudo docker build -f experiments/experiments.Dockerfile -t automon_experiment .')
        print_to_std_and_file("Successfully built docker image for the experiments")

        # Run the experiments
        docker_result_dir = '/app/experiments/test_results'
        local_result_dir = os.getcwd() + "/test_results"
        try:
            os.makedirs(local_result_dir)
        except FileExistsError:
            pass
        docker_run_command_prefix = 'sudo docker run -v ' + local_result_dir + ':' + docker_result_dir + ' --rm automon_experiment python /app/experiments/'

        print_to_std_and_file("Experiment results are written to: " + local_result_dir)

        # TODO: write to the log the approximated time for each experiment
        run_error_communication_tradeoff_experiment(local_result_dir, docker_result_dir, docker_run_command_prefix)
        run_scalability_to_dimensionality_experiment(local_result_dir, docker_result_dir, docker_run_command_prefix)
        run_scalability_to_number_of_nodes_experiment(local_result_dir, docker_result_dir, docker_run_command_prefix)
        run_neighborhood_size_tuning_experiment(local_result_dir, docker_result_dir, docker_run_command_prefix)
        run_ablation_study_experiment(local_result_dir, docker_result_dir, docker_run_command_prefix)

        if args.b_aws_experiments:
            run_aws_experiments(local_result_dir, docker_result_dir, docker_run_command_prefix)

        # Build the paper from Latex source files with the new figures
        compile_reproduced_main_pdf()

    finally:
        os.chdir(project_root)
        revert_dockerignore_changes()
