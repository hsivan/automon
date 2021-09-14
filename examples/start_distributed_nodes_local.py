import subprocess
import time
import os

num_nodes = 10
procs = []
experiment_name = "distributed_inner_product"

dir_path = os.path.abspath(os.path.dirname(__file__))
print("This script folder path:", dir_path)

try:
    for i in range(num_nodes):
        node_proc = subprocess.Popen(["python", os.path.join(dir_path, experiment_name+".py"), "--node_idx", str(i), "--host", "127.0.0.1"])
        procs.append(node_proc)
        time.sleep(3)

    for proc in procs:
        proc.communicate()
except KeyboardInterrupt:
    for proc in procs:
        proc.kill()
