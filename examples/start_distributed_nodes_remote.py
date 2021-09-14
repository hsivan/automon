import os
import shutil
import subprocess
import time
import datetime

# sudo apt-get install -y nethogs
# sudo nethogs -t -v2 &> nethogs_out.txt

num_nodes = 10
procs = []
proc_pids = []
experiment_name = "distributed_inner_product"

outfile = open('nethogs_out.txt', 'w')
nethogs_proc = subprocess.Popen(["sudo", "nethogs", "-t", "-v2", "-d", "5"], stdout=outfile)  # trace mode with view mode of total bytes and refresh rate of 5 seconds

dir_path = os.path.abspath(os.path.dirname(__file__))
print("This script folder path:", dir_path)

try:
    for i in range(num_nodes):
        node_proc = subprocess.Popen(["python", os.path.join(dir_path, experiment_name+".py"), "--node_idx", str(i), "--host", "132.68.36.202"])  # Coordinator IP (coordinator run on ninja1 132.68.36.202)
        print("Node idx", i, "PID is:", node_proc.pid)
        procs.append(node_proc)
        proc_pids.append(node_proc.pid)
        time.sleep(3)

    for proc in procs:
        proc.communicate()
except KeyboardInterrupt:
    for proc in procs:
        proc.kill()

os.system("sudo pkill -9 -P " + str(nethogs_proc.pid))
outfile.close()

# Get the test output folder (last created folder in test_results)
node_test_folders = ['./test_results/' + folder for folder in os.listdir('./test_results/') if experiment_name in folder]
node_test_folders.sort()
node_test_folders = node_test_folders[-num_nodes:]
print(node_test_folders)

test_timestamp = datetime.datetime.now().strftime('%Y-%m-%d_%H-%M-%S')
test_folder_name = os.path.join(os.getcwd(), 'test_results/results_' + experiment_name.replace("distributed", "dist") + "_" + test_timestamp)
os.mkdir(test_folder_name)

# Parse Nethogs log file
shutil.copyfile("./nethogs_out.txt", test_folder_name + "/nethogs_out.txt")
dict_pid_to_stats = {}
with open(test_folder_name + "/nethogs_out.txt") as file:
    lines = file.readlines()
    for line in lines:
        if "python/" in line:
            pid = line.split("/")[1]
            sent = line.split()[1]
            received = line.split()[2]
            dict_pid_to_stats[int(pid)] = (int(float(sent)), int(float(received)))
print(dict_pid_to_stats)

# Parse AutoMon node log files and compare the sent and received data to that of Nethogs
nethogs_vs_automon_file = open(test_folder_name + "/nethogs_vs_automon.txt", "w")
for folder in node_test_folders:
    with open(folder + "/" + experiment_name + ".log") as file:
        lines = file.readlines()
        received_data_line = [line for line in lines if "Bytes received" in line][0].split()
        sent_data_line = [line for line in lines if "Bytes sent" in line][0].split()
        node_idx = int(received_data_line[4])
        num_received_bytes = int(received_data_line[-1])
        num_sent_bytes = int(sent_data_line[-1])
        proc_pid = proc_pids[node_idx]
        nethogs_num_sent_bytes, nethogs_num_received_bytes = dict_pid_to_stats[proc_pid]
        received_diff = nethogs_num_received_bytes - num_received_bytes
        sent_diff = nethogs_num_sent_bytes - num_sent_bytes
        print("Node", node_idx, "num_received_bytes", num_received_bytes, "num_sent_bytes", num_sent_bytes, "nethogs_num_received_bytes", nethogs_num_received_bytes, "nethogs_num_sent_bytes", nethogs_num_sent_bytes)
        print("Node", node_idx, "received_diff", received_diff, "sent_diff", sent_diff)
        print("Node", node_idx, "received_diff_percent", received_diff * 100.0 / num_received_bytes, "sent_diff_percent", sent_diff * 100.0 / num_sent_bytes)
        nethogs_vs_automon_file.write("Node " + str(node_idx) + " num_received_bytes " + str(num_received_bytes) + " num_sent_bytes " + str(num_sent_bytes) + " nethogs_num_received_bytes " + str(nethogs_num_received_bytes) + " nethogs_num_sent_bytes " + str(nethogs_num_sent_bytes) + "\n")
        nethogs_vs_automon_file.write("Node " + str(node_idx) + " received_diff " + str(received_diff) + " sent_diff " + str(sent_diff) + "\n")
        nethogs_vs_automon_file.write("Node " + str(node_idx) + " received_diff_percent " + str(received_diff * 100.0 / num_received_bytes) + " sent_diff_percent " + str(sent_diff * 100.0 / num_sent_bytes) + "\n")
    shutil.move(folder, test_folder_name)

nethogs_vs_automon_file.close()
