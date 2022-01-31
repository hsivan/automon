"""
Requirements:
(1) Python 3
(2) Docker engine (https://docs.docker.com/engine/install)

The script uses only libraries from the Python standard library, which prevents the need to install external packages.
"""
import os
import sys
from timeit import default_timer as timer
import datetime
import argparse
from experiments.reproduce import verify_requirements, download_repository, download_external_datasets, \
    execute_shell_command_with_live_output, get_latest_test_folder, generate_figures, edit_dockerignore, \
    revert_dockerignore_changes, execute_shell_command


def run_experiment(node_type, coordinator_aws_instance_type, local_result_dir, b_centralized):
    if b_centralized:
        node_name = 'centralization_' + node_type
        b_centralization = ' --centralized'
    else:
        node_name = node_type
        b_centralization = ''

    test_folder = get_latest_test_folder(local_result_dir, "max_error_vs_comm_" + node_name + "_aws/")
    if test_folder:
        print("Found existing local AWS test folder for " + node_name + ": " + test_folder + ". Skipping.")
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
    print("The distributed experiment", node_name, "took: ", str(end - start), " seconds")


def generate_aws_figures(local_result_dir, docker_result_dir, docker_run_command_prefix):
    # Plot figures
    test_folders = []

    test_folders.append(get_latest_test_folder(local_result_dir, "results_test_max_error_vs_communication_inner_product"))
    test_folders.append(get_latest_test_folder(local_result_dir, "results_test_max_error_vs_communication_quadratic"))
    test_folders.append(get_latest_test_folder(local_result_dir, "results_test_max_error_vs_communication_kld_air_quality"))
    test_folders.append(get_latest_test_folder(local_result_dir, "results_test_max_error_vs_communication_dnn_intrusion_detection"))

    test_folders.append(get_latest_test_folder(local_result_dir, "max_error_vs_comm_inner_product_aws"))
    test_folders.append(get_latest_test_folder(local_result_dir, "max_error_vs_comm_quadratic_aws"))
    test_folders.append(get_latest_test_folder(local_result_dir, "max_error_vs_comm_kld_aws"))
    test_folders.append(get_latest_test_folder(local_result_dir, "max_error_vs_comm_dnn_aws"))

    test_folders.append(get_latest_test_folder(local_result_dir, "max_error_vs_comm_centralization_inner_product_aws"))
    test_folders.append(get_latest_test_folder(local_result_dir, "max_error_vs_comm_centralization_quadratic_aws"))
    test_folders.append(get_latest_test_folder(local_result_dir, "max_error_vs_comm_centralization_kld_aws"))
    test_folders.append(get_latest_test_folder(local_result_dir, "max_error_vs_comm_centralization_dnn_aws"))

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


def configure_aws_cli(region='us-west-2'):
    # Get the access key and secret access key from the new_user_credentials.csv file, without using pandas, boto3, etc.
    with open('./aws_experiments/new_user_credentials.csv', 'r') as f:
        credentials = f.read()
        access_key_id = credentials.split('https://')[1].split(',')[2]
        secret_access_key = credentials.split('https://')[1].split(',')[3]

    execute_shell_command_with_live_output('aws configure set aws_access_key_id ' + access_key_id)
    execute_shell_command_with_live_output('aws configure set aws_secret_access_key ' + secret_access_key)
    execute_shell_command_with_live_output('aws configure set region ' + region)
    execute_shell_command_with_live_output('aws configure set output json')
    execute_shell_command('aws configure get region', stdout_verification=region)  # Verify configuration worked


def build_and_push_docker_image_to_ecr():
    execute_shell_command_with_live_output('sudo docker build -f aws_experiments/awstest.Dockerfile  -t automon_aws_experiment .')
    print("Successfully built docker image for the AWS experiments")
    # Get the AWS account id from the new_user_credentials.csv file, without using pandas, boto3, etc.
    with open('./aws_experiments/new_user_credentials.csv', 'r') as f:
        credentials = f.read()
        account_id = credentials.split('https://')[1].split('.signin')[0]
    # These two commands require AWS cli installed and configured
    execute_shell_command_with_live_output('aws ecr describe-repositories --repository-names automon || aws ecr create-repository --repository-name automon')
    execute_shell_command_with_live_output('aws ecr get-login-password --region us-east-2 | sudo docker login --username AWS --password-stdin ' + account_id + '.dkr.ecr.us-east-2.amazonaws.com/automon')
    print("Successfully obtained ECR login password")
    execute_shell_command_with_live_output('sudo docker tag automon_aws_experiment ' + account_id + '.dkr.ecr.us-east-2.amazonaws.com/automon')
    print("Successfully tagged docker image")
    execute_shell_command_with_live_output('sudo docker push ' + account_id + '.dkr.ecr.us-east-2.amazonaws.com/automon')
    print("Successfully pushed docker image to ECR")


def run_aws_experiments():
    install_aws_cli()
    configure_aws_cli()
    build_and_push_docker_image_to_ecr()

    # Run the experiments
    docker_result_dir = '/app/experiments/test_results'
    local_result_dir = os.getcwd() + "/test_results"
    docker_run_command_prefix = 'sudo docker run -v ' + local_result_dir + ':' + docker_result_dir + ' --rm automon_experiment python /app/experiments/'

    run_experiment('inner_product', 'ec2', local_result_dir, b_centralized=False)
    run_experiment('quadratic', 'ec2', local_result_dir, b_centralized=False)
    run_experiment('kld', 'ec2', local_result_dir, b_centralized=False)
    run_experiment('dnn', 'ec2', local_result_dir, b_centralized=False)

    # Run the distributed centralized experiments
    run_experiment('inner_product', 'ec2', local_result_dir, b_centralized=True)
    run_experiment('quadratic', 'ec2', local_result_dir, b_centralized=True)
    run_experiment('kld', 'ec2', local_result_dir, b_centralized=True)
    run_experiment('dnn', 'ec2', local_result_dir, b_centralized=True)

    # Plot figures
    generate_aws_figures(local_result_dir, docker_result_dir, docker_run_command_prefix)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--dd", dest="b_download_dataset", help="if --dd is specified, the script only downloads the external datasets", action='store_true')
    args = parser.parse_args()

    project_root = download_repository()
    download_external_datasets(project_root)
    print("Downloaded external datasets")
    if args.b_download_dataset:
        sys.exit()

    verify_requirements()
    os.chdir(project_root)

    # Verify the new_user_credentials.csv exists
    if not os.path.isfile('aws_experiments/new_user_credentials.csv'):
        print("To run AWS experiments, you must have an AWS account. After opening the account, create AWS IAM user with "
              "AdministratorAccess permissions and download the csv file new_user_credentials.csv that contains the key ID and the secret key. "
              "Place the new_user_credentials.csv file in " + project_root + "/aws_experiments/new_user_credentials.csv and re-run this script.\n"
              "Note: AWS cli will be installed on your computer and will be configured!")
        sys.exit(1)

    try:
        # Build docker image with .dockerignore which does not ignore the experiments folder. This image is used to plot
        # the figures after all the experiments finish.
        edit_dockerignore()
        execute_shell_command_with_live_output('sudo docker build -f experiments/experiments.Dockerfile -t automon_experiment .')
        print("Successfully built docker image for the experiments")

        run_aws_experiments()

    finally:
        revert_dockerignore_changes()
