import os
import shutil
import subprocess
import time
import datetime

# sudo apt-get install -y nethogs
# sudo nethogs -t -v2 &> nethogs_out.txt

num_nodes = 1  # TODO: set to 1 to run the coordinator or to NUM_NODES to run the nodes
procs = []
proc_pids = []
experiment_type = "inner_product"

outfile = open('./nethogs_out.txt', 'w')
nethogs_proc = subprocess.Popen(["sudo", "nethogs", "-t", "-v2", "-d", "5"], stdout=outfile)  # trace mode with view mode of total bytes and refresh rate of 5 seconds

dir_path = os.path.abspath(os.path.dirname(__file__))
print("This script folder path:", dir_path)

try:
    for i in range(num_nodes):
        if num_nodes == 1:
            node_proc = subprocess.Popen(["python", os.path.join(dir_path, "distributed_centralization.py"), "--node_idx", "-1", "--host", "0.0.0.0", "--type", experiment_type])
            print("Coordinator PID is:", node_proc.pid)
        else:
            node_proc = subprocess.Popen(["python", os.path.join(dir_path, "distributed_centralization.py"), "--node_idx", str(i), "--host", "132.68.36.202", "--type", experiment_type])  # Coordinator IP (coordinator run on ninja1 132.68.36.202)
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
node_test_folders = ['./test_results/' + folder for folder in os.listdir('./test_results/') if experiment_type in folder and "centralization" in folder]
node_test_folders.sort()
node_test_folders = node_test_folders[-num_nodes:]
print(node_test_folders)

test_timestamp = datetime.datetime.now().strftime('%Y-%m-%d_%H-%M-%S')
test_folder_name = os.path.join(os.getcwd(), 'test_results/results_dist_centralization_' + experiment_type + "_" + test_timestamp)
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

nethogs_file = open(test_folder_name + "/nethogs.txt", "w")
for node_idx, proc_pid in enumerate(proc_pids):
    nethogs_num_sent_bytes, nethogs_num_received_bytes = dict_pid_to_stats[proc_pid]
    print("Node", node_idx, "nethogs_num_received_bytes", nethogs_num_received_bytes, "nethogs_num_sent_bytes", nethogs_num_sent_bytes)
    nethogs_file.write("Node " + str(node_idx) + " nethogs_num_received_bytes " + str(nethogs_num_received_bytes) + " nethogs_num_sent_bytes " + str(nethogs_num_sent_bytes) + "\n")
nethogs_file.close()

for folder in node_test_folders:
    shutil.move(folder, test_folder_name)
