import os
import subprocess
from aws_experiments.start_distributed_object_remote import parser_nethogs_log_file, get_test_result_folder, log_cpu_info
from aws_experiments.utils import copy_result_folder_to_s3, stop_and_terminate_ec2_instance

# sudo apt-get install -y nethogs
# sudo nethogs -t -v2 &> nethogs_out.txt


if __name__ == "__main__":
    try:
        node_idx = os.getenv('NODE_IDX', '-1')  # NODE_idx by default is -1 (for the coordinator)
        node_type = os.getenv('NODE_TYPE', 'inner_product')  # Could be inner_product / kld / quadratic / dnn
        if node_idx == -1:
            host = '0.0.0.0'
        else:
            host = os.getenv('HOST', '132.68.36.202')  # Coordinator IP (Ninja1 by default)

        experiment_name = node_type + "_centralization"

        outfile = open('./nethogs_out.txt', 'w')
        nethogs_proc = subprocess.Popen(["nethogs", "-t", "-v2", "-d", "5"], stdout=outfile)  # trace mode with view mode of total bytes and refresh rate of 5 seconds

        dir_path = os.path.abspath(os.path.dirname(__file__))
        print("This script folder path:", dir_path)

        try:
            proc = subprocess.Popen(["python", os.path.join(dir_path, "distributed_centralization.py"), "--node_idx", node_idx, "--host", host, "--type", node_type])
            print("Proc PID is:", proc.pid)
            proc_pid = proc.pid
            proc.communicate()
        except KeyboardInterrupt:
            proc.kill()

        os.system("pkill -9 -P " + str(nethogs_proc.pid))
        outfile.close()

        # Get the test output folder (last created folder in test_results)
        test_folder = get_test_result_folder(experiment_name)

        try:
            nethogs_stats = parser_nethogs_log_file(test_folder, proc_pid)
            nethogs_num_sent_bytes, nethogs_num_received_bytes = nethogs_stats[proc_pid]
            with open(test_folder + "/nethogs.txt", "w") as f:
                print("Node", node_idx, "nethogs_num_received_bytes", nethogs_num_received_bytes, "nethogs_num_sent_bytes", nethogs_num_sent_bytes)
                f.write("Node " + str(node_idx) + " nethogs_num_received_bytes " + str(nethogs_num_received_bytes) + " nethogs_num_sent_bytes " + str(nethogs_num_sent_bytes) + "\n")
            log_cpu_info(test_folder)
        except Exception as err:
            print(err)

        copy_result_folder_to_s3(test_folder, node_type, node_idx, None)

    finally:
        # If this instance is an EC2 instance, then INSTANCE_ID is defined and this instance can be self terminated.
        stop_and_terminate_ec2_instance()
