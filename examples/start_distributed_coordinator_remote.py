import os
import shutil
import subprocess

# sudo apt-get install -y nethogs
# sudo nethogs -t -v2 &> nethogs_out.txt

proc_pid = None
experiment_name = "distributed_inner_product"

outfile = open('nethogs_out.txt', 'w')
nethogs_proc = subprocess.Popen(["sudo", "nethogs", "-t", "-v2", "-d", "5"], stdout=outfile)  # trace mode with view mode of total bytes and refresh rate of 5 seconds

dir_path = os.path.abspath(os.path.dirname(__file__))
print("This script folder path:", dir_path)

try:
    coordinator_proc = subprocess.Popen(["python", os.path.join(dir_path, experiment_name+".py"), "--node_idx", "-1"])
    print("Coordinator PID is:", coordinator_proc.pid)
    proc_pid = coordinator_proc.pid
    coordinator_proc.communicate()
except KeyboardInterrupt:
    coordinator_proc.kill()

os.system("sudo pkill -9 -P " + str(nethogs_proc.pid))
outfile.close()

# Get the test output folder (last created folder in test_results)
test_folders = ['./test_results/' + folder for folder in os.listdir('./test_results/') if experiment_name in folder]
test_folders.sort()
test_folder = test_folders[-1]
print(test_folder)

# Parse Nethogs log file
shutil.copyfile("./nethogs_out.txt", test_folder + "/nethogs_out.txt")
dict_pid_to_stats = {}
with open(test_folder + "/nethogs_out.txt") as file:
    lines = file.readlines()
    for line in lines:
        if "python/" in line:
            pid = line.split("/")[1]
            sent = line.split()[1]
            received = line.split()[2]
            dict_pid_to_stats[int(pid)] = (int(float(sent)), int(float(received)))
print(dict_pid_to_stats)

# Parse AutoMon coordinator log files and compare the sent and received data to that of Nethogs
nethogs_vs_automon_file = open(test_folder + "/nethogs_vs_automon.txt", "w")
with open(test_folder + "/" + experiment_name + ".log") as file:
    lines = file.readlines()
    received_data_line = [line for line in lines if "Bytes received" in line][0].split()
    sent_data_line = [line for line in lines if "Bytes sent" in line][0].split()
    num_received_bytes = int(received_data_line[-1])
    num_sent_bytes = int(sent_data_line[-1])
    nethogs_num_sent_bytes, nethogs_num_received_bytes = dict_pid_to_stats[proc_pid]
    received_diff = nethogs_num_received_bytes - num_received_bytes
    sent_diff = nethogs_num_sent_bytes - num_sent_bytes
    print("Coordinator num_received_bytes", num_received_bytes, "num_sent_bytes", num_sent_bytes, "nethogs_num_received_bytes", nethogs_num_received_bytes, "nethogs_num_sent_bytes", nethogs_num_sent_bytes)
    print("Coordinator received_diff", received_diff, "sent_diff", sent_diff)
    print("Coordinator received_diff_percent", received_diff * 100.0 / num_received_bytes, "sent_diff_percent", sent_diff * 100.0 / num_sent_bytes)
    nethogs_vs_automon_file.write("Coordinator num_received_bytes " + str(num_received_bytes) + " num_sent_bytes " + str(num_sent_bytes) + " nethogs_num_received_bytes " + str(nethogs_num_received_bytes) + " nethogs_num_sent_bytes " + str(nethogs_num_sent_bytes) + "\n")
    nethogs_vs_automon_file.write("Coordinator received_diff " + str(received_diff) + " sent_diff " + str(sent_diff) + "\n")
    nethogs_vs_automon_file.write("Coordinator received_diff_percent " + str(received_diff * 100.0 / num_received_bytes) + " sent_diff_percent " + str(sent_diff * 100.0 / num_sent_bytes))

nethogs_vs_automon_file.close()
