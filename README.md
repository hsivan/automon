# AutoMon

AutoMon is a library for evaluating a mathematical function over the average of multiple vectors that are distributed over multiple nodes of a system (sometimes known formally as distributed functional monioring over continous distributed streams).

Consider a distributed system with a single coordinator node (e.g., a server) and n nodes (e.g., remote workers, sensors),
and where every node i holds a dynamic local data vector x_i that is constantly being updated from its local data stream.
Suppose we want to evaluate some arbitrary real multivariate function f of the average vector of all these local vectors.
In other words, we want to maintain an estimate of f(x), where x is the average of all the x_i vectrors.

AutoMon is an easy-to-use algorithmic building block for automatically approximating f(x) over time, without having to send all the local updates to x_i to a centralized location.
Given Python code to compute f(x) from x as well as the desired approximation error, AutoMon will automatically provides communication-efficient distributed monitoring of the function approximation, without requiring any manual mathematical analysis by the developer.

For more information, see our SIGMOD 2022 paper, [AutoMon: Automatic Distributed Monitoring for Arbitrary
Multivariate Functions](https://assaf.net.technion.ac.il/files/2021/12/SIGMOD2022_AutoMon_revision.pdf).

## Installation

AutoMon is written in pure Python.
Use the following instructions to install a
binary package with `pip`, or to download AutoMon's source code.
We support installing `automon` package on Linux (Ubuntu 18.04 or later) and
Windows (10 or later) platforms, while Windows support is only partial (see Features section below).
**The installation requires Python >=3.7, <3.11**.

To download AutoMon's source code run:
```bash
git clone https://github.com/hsivan/automon
```
Let `automon_root` be the root folder of the project on your local computer, for example `/home/username/automon`.

To install AutoMon run:
```bash
pip install git+https://github.com/hsivan/automon.git
```
Or, download the code and then run:
```bash
pip install <automon_root>
```

## Features

### Lightweight design: a library, not a framework
AutoMon is designed to be integrated easily into distributed applications. 
It focuses on managing the distributed approximation algorithm, and does not impose (nor use) any specific underlying messaging fabric.
Instead, the library has a simple and easy to use API, relying on the application to pass messages between the nodes and the coordinator.
AutoMon focuses on the mathematical and algorithmic aspects, leaving developers to focus on application and systems aspects.

### Communication-efficient and adaptive
AutoMon often uses far fewer messages than simply uploading all data updates to a centralize location.
Moreover, unlike frequently used periodic approaches (e.g., only send one update every T times), AutoMon adapts to the data, function, and desired approximation error. 
This means that AutoMon can incurr no communication in periods of quiesence (where the data does not change by much), yet quickly detect and update the approximation in the face of sudden changes.
On the other hand, periodic approaches can be wasteful during quiesence and result in large approximation errors when data changes quickly.

### <a name="choice"></a>Choice of automatic differentiation tool
AutoMon uses automatic differentiation tool to derive local constraints to the nodes.
These local constraints are used locally by a node to monitor the node's local vector, and communication
between the node and the coordinator is required only when the constraints are violated. 

AutoMon supports two automatic differentiation tools: [JAX](https://github.com/google/jax) and [Autograd](https://github.com/HIPS/autograd).
Since JAX is not fully supported on Windows, we use
Autograd as the default automatic differentiation tool when running on Windows, and
JAX when running on Linux.
It is possible to force using Autograd by adding at the beginning of an experiment:
```python
import os
os.environ['AUTO_GRAD_TOOL'] = 'AutoGrad'
```
### Testbed for approaches based on the Geometric Monitoring Protocol
AutoMon adopts the geometric monitoring (GM) protocol for continuous threshold monitoring,
which has been widely adopted by distributed monitoring methods.

The implementation separates between the code of the basic GM protocol and the code of 
a specific monitoring technique that adopts this protocol (such as AutoMon).
The GM protocol code is implemented in `automon/coordinator_common.py` and `automon/node_common.py`.
The code of a specific monitoring technique is in a subpackage named after the technique.
For example, AutoMon in under `automon/automon`.
This design enables developers to easily add new monitoring techniques, add features to existing techniques, or to
enrich the basic protocol.

AutoMon also provide implementations of other monitoring techniques that were used as baselines in our paper.
For the CB technique we provide the implementation of cosine similarity and inner product monitoring, based on
[Lazerson et al., 2016](https://dl.acm.org/doi/pdf/10.1145/3226113).
For the GM technique we provide the implementation of entropy monitoring, based on
[Gabel et al., 2017](https://dl.acm.org/doi/pdf/10.1145/3097983.3098092), and of variance monitoring,
based on [Gabel et al., 2014](https://ieeexplore.ieee.org/stamp/stamp.jsp?tp=&arnumber=6877240).
(Note that while AutoMon can monitor any arbitrary function given as a code, the other techniques have a tailored
implementation per function.)


## Usage example

The following example shows how to monitor the inner product function with AutoMon.
In this example, the data for the nodes is randomly generated - a new sample is randomly sampled every second until the
user terminates the node.
For the communication between the coordinator and nodes we use [PyZMQ](https://github.com/zeromq/pyzmq); however,
this can be replaced with any other messaging library.

The first step is to define the function.
For example, to monitor the inner product function, define the function in a file, and call it e.g. `function_def.py`:
```python
import jax.numpy as np  # For Windows, import autograd.numpy instead of jax.numpy

def func_inner_product(x):
    return np.matmul(x[:x.shape[0] // 2], x[x.shape[0] // 2:])
```

Next, copy the following code into a second file, called for example `coordinator.py`, and run it.
This code will initialize and run a coordinator instance that waits for 4 nodes.
The number of nodes and the listening port of the coordinator can be modified in this script.
```python
from automon import AutomonCoordinator
from automon.zmq_socket_utils import init_server_socket, get_next_node_message, send_message_to_node
from function_def import func_inner_product
import logging
logging.getLogger('automon').setLevel(logging.INFO)

coordinator = AutomonCoordinator(num_nodes=4, func_to_monitor=func_inner_product, error_bound=2.0, d=40)
# Open a server socket. Wait for all nodes to connect and send 'start' signal to all nodes to start their data loop.
server_socket = init_server_socket(port=6400, num_nodes=4)

while True:
    msg = get_next_node_message(server_socket)
    replies = coordinator.parse_message(msg)
    for node_idx, reply in replies:
        send_message_to_node(server_socket, node_idx, reply)
```

Lastly, use the following code to initiate and run **the 4 nodes, one at a time**.
A node can be run on any computer or device with internet access.
Make sure the `host` is set to the IP of the coordinator machine and the `port` is set to the one defined in the coordinator script.
Don't forget to update `NODE_IDX` for every new instance.
```python
import numpy as np
from timeit import default_timer as timer
from automon import AutomonNode
from automon.zmq_socket_utils import init_client_socket
from function_def import func_inner_product
import logging
logging.getLogger('automon').setLevel(logging.INFO)

def time_to_wait_for_next_sample_milliseconds(start_time, num_received_samples):
    return (num_received_samples - (timer() - start_time)) * 1000

NODE_IDX = 0  # Change the node index for different nodes
node = AutomonNode(idx=NODE_IDX, func_to_monitor=func_inner_product, d=40)
# Open a client socket and connect to the server socket. Wait for 'start' message from the server.
client_socket = init_client_socket(NODE_IDX, host='127.0.0.1', port=6400)

# Wait for a message from the coordinator (local data requests or local constraint updates) and send the reply to the coordinator.
# Read new data samples every 1 second and update the node local vector. Report violations to the coordinator.
start = timer()
num_data_samples = 0
while True:
    if time_to_wait_for_next_sample_milliseconds(start, num_data_samples) <= 0:
        # Time to read the next data sample
        data = np.random.normal(loc=1, scale=0.1, size=(40,))
        message_violation = node.update_data(data)
        if message_violation:
            client_socket.send(message_violation)
        num_data_samples += 1
    event = client_socket.poll(timeout=time_to_wait_for_next_sample_milliseconds(start, num_data_samples))
    if event != 0:
        # Received a message from the coordinator before the timeout has reached
        message = client_socket.recv()
        reply = node.parse_message(message)
        if reply:
            client_socket.send(reply)
```
After all the nodes and the coordinator are initiated, the experiment begins automatically.

More examples can be found in the `examples` folder.

## Run as a docker container
We provide Dockerfile to support building the project as a docker image.
To build the docker image you must first install docker engine and docker cli.
After installing these, run the command to build the docker image from within `<automon_root>`:
```
sudo docker build -t automon .
```
This docker image could be used to run an AutoMon node or coordinator.
The following example shows how to run the basic example above, of monitoring the inner product function with 4 nodes,
using containers.
To make it easier to test this example, we make it possible to run the 4 containers on a single machine by using a user-defined network
for the 4 containers. We, therefore, must first create the user-defined network:
```
sudo docker network create automonnet
```
Run the docker container as coordinator:
```
sudo docker run --net automonnet --name automonserver --rm automon python /app/examples/simple_automon_coordinator.py
```
Run the docker container as node 0:
```
sudo docker run --net automonnet --env HOST=automonserver --env NODE_IDX=0 --rm automon python /app/examples/simple_automon_node.py
```
Run the other 3 nodes similarly, while changing `NODE_IDX` value accordingly.

To run the same example on different machines, simply remove `--net automonnet` and `--name automonserver` from the commands and change `HOST`
value in the node command to the IP of the coordinator machine.

## Reproduce paper's experimental results

We provide detailed instructions for reproducing the experiments from our [paper](https://assaf.net.technion.ac.il/files/2021/12/SIGMOD2022_AutoMon_revision.pdf).
The full [instructions](experiments/README.md) and scripts are in the `experiments` folder.


## Distributed experiment on a real-world WAN
We include code for a series of cross-region experiments on AWS.
The [instructions](examples/aws_utils/README.md) for running these experiments, as well as the scripts are in the `examples/aws_utils` folder.


## Citing AutoMon

If AutoMon has been useful for your research and you would like to cite it in an academic
publication, please use the following Bibtex entry:
```bibtex
@inproceedings{sivan_automon_2022,
  author    = {Sivan, Hadar and Gabel, Moshe and Schuster, Assaf},
  title     = {{AutoMon}: Automatic Distributed Monitoring for Arbitrary Multivariate Functions},
  year      = {2022},
  series    = {SIGMOD '22},
  booktitle = {Proceedings of the 2022 {ACM} {SIGMOD} International Conference on Management of Data},
  note      = {to appear}
}
```
