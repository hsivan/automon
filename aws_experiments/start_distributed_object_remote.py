"""
This script is the entry point of the docker file awstest.Dockerfile (the script that the docker container runs).
The script starts Nethogs and runs one of the distributed_xxx.py scripts (e.g. distributed_inner_product.py).
It uses environment variables provided to the docker container to decide with experiment to run, whether this is a
coordinator or a node instance, the node index, the coordinator IP, etc.
It waits until the distributed_xxx.py script finishes and then collects network statistics.
It copies the result folder to S3 bucket and if needed terminates the instance (self termination).
"""
import os
import shutil
import subprocess
from aws_experiments.utils import stop_and_terminate_ec2_instance
from aws_experiments.utils import copy_result_folder_to_s3

# sudo apt-get install -y nethogs
# sudo nethogs -t -v2 &> nethogs_out.txt


def get_test_result_folder(experiment_name):
    # Get the test output folder (last created folder in test_results)
    test_folders = ['./test_results/' + folder for folder in os.listdir('./test_results/') if experiment_name in folder]
    test_folders.sort()
    test_folder = test_folders[-1]
    print("The test folder is:", test_folder)
    return test_folder


def parser_nethogs_log_file(test_folder, proc_pid):
    shutil.copyfile("./nethogs_out.txt", test_folder + "/nethogs_out.txt")
    nethogs_stats = {}
    nethogs_stats_by_port = {}
    with open(test_folder + "/nethogs_out.txt") as file:
        lines = file.readlines()
        for line in lines:
            if "python/" in line or ":6400-" in line:
                sent = line.split()[1]
                received = line.split()[2]
                if "python/" in line:
                    pid = line.split("/")[1]
                    nethogs_stats[int(pid)] = (int(float(sent)), int(float(received)))
                else:
                    nethogs_stats_by_port[proc_pid] = (int(float(sent)), int(float(received)))
    if len(nethogs_stats) == 0:
        nethogs_stats = nethogs_stats_by_port
    print(nethogs_stats)
    return nethogs_stats


def compare_automon_and_nethogs_stats(test_folder, nethogs_stats, proc_pid, node_idx):
    # Parse AutoMon results file and compare the sent and received data to that of Nethogs
    nethogs_vs_automon_file = open(test_folder + "/nethogs_vs_automon.txt", "w")
    log_file_name = experiment_name + ".log"
    if int(node_idx) == -1:
        log_file_name = "results.txt"
    with open(test_folder + "/" + log_file_name) as file:
        lines = file.readlines()
        num_received_bytes = int([line for line in lines if "Bytes received" in line][0].split()[-1])
        num_sent_bytes = int([line for line in lines if "Bytes sent" in line][0].split()[-1])
        nethogs_num_sent_bytes, nethogs_num_received_bytes = nethogs_stats[proc_pid]
        received_diff = nethogs_num_received_bytes - num_received_bytes
        sent_diff = nethogs_num_sent_bytes - num_sent_bytes
        print("num_received_bytes", num_received_bytes, "num_sent_bytes", num_sent_bytes, "nethogs_num_received_bytes", nethogs_num_received_bytes, "nethogs_num_sent_bytes", nethogs_num_sent_bytes)
        nethogs_vs_automon_file.write("num_received_bytes " + str(num_received_bytes) + " num_sent_bytes " + str(num_sent_bytes) + " nethogs_num_received_bytes " + str(nethogs_num_received_bytes) + " nethogs_num_sent_bytes " + str(nethogs_num_sent_bytes) + "\n")
        print("received_diff", received_diff, "sent_diff", sent_diff)
        nethogs_vs_automon_file.write("received_diff " + str(received_diff) + " sent_diff " + str(sent_diff) + "\n")
        if num_received_bytes > 0 and num_sent_bytes > 0:
            print("received_diff_percent", received_diff * 100.0 / num_received_bytes, "sent_diff_percent", sent_diff * 100.0 / num_sent_bytes)
            nethogs_vs_automon_file.write("received_diff_percent " + str(received_diff * 100.0 / num_received_bytes) + " sent_diff_percent " + str(sent_diff * 100.0 / num_sent_bytes))
    nethogs_vs_automon_file.close()


def log_cpu_info(test_folder):
    # Get CPU info
    cpu_info = (subprocess.check_output("lscpu", shell=True).strip()).decode()
    with open(test_folder + "/cpu_info.txt", "w") as f:
        f.write(cpu_info)


if __name__ == "__main__":
    try:
        # Read environment variables
        node_idx = os.environ['NODE_IDX']  # -1 for the coordinator and 0 to num_nodes-1 for a node
        node_type = os.environ['NODE_TYPE']  # Could be inner_product / kld / quadratic / dnn
        if node_idx == -1:
            host = '0.0.0.0'
            error_bound = os.environ['ERROR_BOUND']  # The error bound
        else:
            host = os.environ['HOST']  # Coordinator IP
            error_bound = os.getenv('ERROR_BOUND', None)  # If error bound is given to node it is used in the name of the result folder (to distinguish between experiments with different error bounds).
        if node_type == "inner_product":
            experiment_name = "distributed_inner_product"
        elif node_type == "kld":
            experiment_name = "distributed_kld_air_quality"
        elif node_type == "dnn":
            experiment_name = "distributed_dnn_intrusion_detection"
        else:
            experiment_name = "distributed_quadratic"

        # Start Nethogs
        outfile = open('nethogs_out.txt', 'w')
        nethogs_proc = subprocess.Popen(["nethogs", "-t", "-v2", "-d", "5"], stdout=outfile)  # trace mode with view mode of total bytes and refresh rate of 5 seconds

        # Start AutoMon experiment
        try:
            dir_path = os.path.abspath(os.path.dirname(__file__))
            print("This script folder path:", dir_path)
            if error_bound is not None:
                obj_proc = subprocess.Popen(["python", os.path.join(dir_path, experiment_name+".py"), "--node_idx", node_idx, "--host", host, "--error_bound", error_bound])  # host is the coordinator IP
            else:
                obj_proc = subprocess.Popen(["python", os.path.join(dir_path, experiment_name + ".py"), "--node_idx", node_idx, "--host", host])  # host is the coordinator IP
            print("Object PID is:", obj_proc.pid)
            proc_pid = obj_proc.pid
            obj_proc.communicate()
        except KeyboardInterrupt:
            obj_proc.kill()

        # Stop Nethogs
        os.system("pkill -9 -P " + str(nethogs_proc.pid))
        outfile.close()

        test_folder = get_test_result_folder(experiment_name)

        try:
            nethogs_stats = parser_nethogs_log_file(test_folder, proc_pid)
            compare_automon_and_nethogs_stats(test_folder, nethogs_stats, proc_pid, node_idx)
            log_cpu_info(test_folder)
        except Exception as err:
            print(err)

        copy_result_folder_to_s3(test_folder, node_type, node_idx, error_bound)

    finally:
        # If this instance is an EC2 instance, then INSTANCE_ID is defined and this instance can be self terminated.
        stop_and_terminate_ec2_instance()
