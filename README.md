# AutoMon

AutoMon library for distributed functional monitoring.

AutoMon is an easy-to-use algorithmic building block for automatically approximating arbitrary real
multivariate functions over distributed data streams.
Consider a distributed system with a single coordinator node and $n$ nodes, where each node $i$ holds a dynamic local
data vector $x^i$ computed from its local data stream.
Let $f$ be an arbitrary real multivariate function $f : \mathbb{R}^d →\mathbb{R}$ of
the average vector of local data $\bar{x} = \frac{1}{n} \sum_{i=1}^{n} x_i$.
Given $f$ expressed as code in Python and an approximation
error bound, AutoMon automatically provides communication-efficient distributed monitoring of the function approximation,
without requiring any manual analysis by the user.
For more information regarding AutoMon see [AutoMon: Automatic Distributed Monitoring for Arbitrary
Multivariate Functions](https://assaf.net.technion.ac.il/files/2021/12/SIGMOD2022_AutoMon_revision.pdf).

## Installation

AutoMon is written in pure Python.
Use the following instructions to install a
binary package with `pip`, or to download AutoMon's source code.
We support installing `automon` package on Linux (Ubuntu 18.04 or later) and
Windows (10 or later) platforms, while Windows support is only partial (see Features section below). 

To download AutoMon's source code run:
```bash
git clone https://github.com/hsivan/automon
```
Let `automon_root` be the root folder of the project on your local computer, for example `/home/username/automon`.

To install AutoMon run:
```bash
pip install git+https://github.com/hsivan/automon
```
Or, download the code and then run:
```bash
pip install <automon_root>
```

## Features

### Lightweight design &ndash; a library, not a framework
AutoMon is designed to be integrated easily into distributed applications. 
It focuses on managing the distributed approximation algorithm, and does not impose (nor use) any specific underlying messaging fabric.

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

### The basic geometric monitoring protocol
AutoMon adopts the geometric monitoring (GM) protocol for continuous threshold monitoring in a distributed system,
which has been widely adopted by distributed monitoring methods.
The GM protocol comprises two basic parts: the coordinator algorithm and the node algorithm.
Each node receives local data and updates its dynamic local vector.
A node is responsible for monitoring the local constraints, reporting violation of these constraints
to the coordinator, and receiving updated constraints from the coordinator.
The coordinator is responsible for resolving violations of the local constraints by distributing updated local
constraints to nodes.
The GM protocol does not define the derivation of the local constraints, and this is done differently by any monitoring technique.

The implementation in this project separates between the code of the basic GM protocol and the code of 
a specific monitoring technique that adopts this protocol (such as AutoMon).
The GM protocol code is implemented in `automon/coordinator_common.py` and `automon/node_common.py`.
The code of a specific monitoring technique is in a subpackage named after the technique.
For example, AutoMon in under `automon/automon`.
This design enables developers to easily add new monitoring techniques, add features to existing techniques, or to
enrich the basic protocol.

We also provide some implementations of other monitoring techniques that were used as baselines in our paper.
For the CB technique we provide the implementation of cosine similarity and inner product monitoring, based on
[Lazerson et al., 2016](https://dl.acm.org/doi/pdf/10.1145/3226113).
For the GM technique we provide the implementation of entropy monitoring, based on
[Gabel et al., 2017](https://dl.acm.org/doi/pdf/10.1145/3097983.3098092), and of variance monitoring,
based on [Gabel et al., 2014](https://ieeexplore.ieee.org/stamp/stamp.jsp?tp=&arnumber=6877240).
Note that while AutoMon can monitor any arbitrary function given as a code, the other techniques have a tailored
implementation per function.


## Usage example

The following example shows how to monitor the inner product function with AutoMon.
In this example, the data for the nodes is randomly generated - a new sample is randomly sampled every second until the
user terminates the node.
For the communication between the coordinator and nodes we use [PyZMQ](https://github.com/zeromq/pyzmq); however,
this can be replaced with any other messaging library.

The first step is to define the function.
For example, we define the inner product function in a file called `function_def.py`:
```python
import jax.numpy as np  # the user could use autograd.numpy instead of JAX

def func_inner_product(x):
    return np.matmul(x[:x.shape[0] // 2], x[x.shape[0] // 2:])
```

Next, initiate and run the coordinator on a designated server.
You could change the listening port of the coordinator.
```python
import sys
import logging
from automon.automon.coordinator_automon import CoordinatorAutoMon
from automon.automon.node_common_automon import NodeCommonAutoMon
from automon.utils_zmq_sockets import init_server_socket, get_next_node_message, send_message_to_node
from function_def import func_inner_product
logging.basicConfig(stream=sys.stdout, level=logging.INFO)

# Create a dummy node for the coordinator that uses it in the process of resolving violations.
verifier = NodeCommonAutoMon(idx=-1, x0_len=40, func_to_monitor=func_inner_product)
coordinator = CoordinatorAutoMon(verifier, num_nodes=4, error_bound=2.0)
# Open a server socket. Wait for all nodes to connect and send 'start' signal to all nodes to start their data loop.
server_socket = init_server_socket(port=6400, num_nodes=4)

while True:
    msg = get_next_node_message(server_socket)
    replies = coordinator.parse_message(msg)
    for node_idx, reply in replies:
        send_message_to_node(server_socket, node_idx, reply)
```

Lastly, initiate and run a node. The node can run on any computer or device with internet access.
Make sure the `host` and `port` are set to the IP and port of the coordinator.
```python
import sys
import logging
from timeit import default_timer as timer
import numpy as np
from automon.automon.node_common_automon import NodeCommonAutoMon
from automon.messages_common import prepare_message_data_update
from automon.utils_zmq_sockets import init_client_socket
from function_def import func_inner_product
logging.basicConfig(stream=sys.stdout, level=logging.INFO)

def time_to_wait_for_next_sample_milliseconds(start_time, num_received_samples):
    return (num_received_samples - (timer() - start_time)) * 1000

NODE_IDX = 0  # Change the node index for different nodes
node = NodeCommonAutoMon(idx=NODE_IDX, x0_len=40, func_to_monitor=func_inner_product)
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
        message_data_update = prepare_message_data_update(NODE_IDX, data)
        message_violation = node.parse_message(message_data_update)
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
Initiate all 3 other nodes similarly.
Don't forget to update the node_idx for every new instance.
After all the nodes and the coordinator are initiated, the experiment begins automatically.

More examples can be found in the `examples` folder.

## Run as a docker container
We provide Dockerfile to support building the project as a docker image.
To build the docker image you must first install docker engine and docker cli.
After installing these, run the command to build the docker image:
```
sudo docker build -t automon <automon_root>
```
To run the docker container in a coordinator mode:
```
sudo docker run -p 6400:6400 --env NODE_IDX=-1 --env NODE_TYPE=inner_product --env ERROR_BOUND=0.3 -it --rm automon
```
and in a node 0 mode:
```
sudo docker run --env HOST=192.68.36.202 --env NODE_IDX=0 --env NODE_TYPE=inner_product -it --rm automon
```
If setting the container environment variable `S3_WRITE` to `1` (`--env S3_WRITE=1`), the results are written to AWS S3 bucket named `automon-experiment-results`.
Otherwise, it is advisable to run the container with `-v /home/ubuntu/test_results:<automon_root>/test_results`, so the results are written to the computer filesystem.

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
@inproceedings{automon,
  author    = {Sivan, Hadar and Gabel, Moshe and Schuster, Assaf},
  title     = {AutoMon: Automatic Distributed Monitoring for Arbitrary Multivariate Functions},
  year      = {2022},
  pages     = {TODO-TODO},
  numpages  = {TODO},
  url       = {TODO},
  doi       = {TODO},
  isbn      = {TODO},
  series    = {SIGMOD '22}
  booktitle = {Proceedings of the 2022 ACM SIGMOD International Conference on Management of Data}
}