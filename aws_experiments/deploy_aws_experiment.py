"""
This script deploys all the AWS instances required for a single distributed experiment and runs the docker container
on the instances.
The experiment is one of inner_product / quadratic / kld / dnn.
The user could choose whether to run the coordinator on a strong EC2 instance, or on an ECS Fargate instance.
"""
import sys
import time
import boto3
import json
import os
import argparse
from aws_experiments.aws_ec2_coordinator import run_coordinator_on_ec2_instance
from aws_experiments.utils import read_credentials_file, num_completed_experiments


def get_service_ips(ecs_client, cluster, tasks, region):
    tasks_detail = ecs_client.describe_tasks(cluster=cluster, tasks=tasks)

    enis = []
    for task in tasks_detail.get("tasks", []):
        for attachment in task.get("attachments", []):
            for detail in attachment.get("details", []):
                if detail.get("name") == "networkInterfaceId":
                    enis.append(detail.get("value"))

    ips = []
    for eni in enis:
        eni_resource = session.resource("ec2", region).NetworkInterface(eni)
        ips.append(eni_resource.association_attribute.get("PublicIp"))

    return ips


def run_task(ecs_client, automon_task, cluster_name, node_idx, host, node_type, error_bound, subnet_id, sg_id, command):
    task_name = "node-" + str(node_idx) + "_" + cluster_name
    if node_idx == -1:
        task_name = "coordinator_" + cluster_name

    response = ecs_client.list_task_definitions(familyPrefix=task_name)

    for task_definition in response["taskDefinitionArns"]:
        # De-register old task definitions
        print("Deregister existing task definition", task_name)
        _ = ecs_client.deregister_task_definition(taskDefinition=task_definition)

    # Register the new task definition
    automon_task["family"] = task_name
    automon_task["containerDefinitions"][0]["logConfiguration"]["options"]["awslogs-stream-prefix"] = task_name
    if task_name == "coordinator":
        automon_task["cpu"] = "4096"  # 4 vCPU
        automon_task["memory"] = "16384"  # 16 GB
    else:
        automon_task["cpu"] = "1024"  # 1 vCPU
        automon_task["memory"] = "4096"  # 4 GB
    _ = ecs_client.register_task_definition(**automon_task)

    # Could also override LS_LATENCY (lazy sync latency) and FS_LATENCY (full sync latency) according to the function and network latency
    response = ecs_client.run_task(
        taskDefinition=task_name,
        cluster=cluster_name,
        launchType='FARGATE',
        count=1,
        networkConfiguration={
            'awsvpcConfiguration': {
                'subnets': [
                    subnet_id,
                ],
                'securityGroups': [sg_id],
                'assignPublicIp': 'ENABLED'
            }
        },
        overrides={
            'containerOverrides': [
                {
                    'name': automon_task["containerDefinitions"][0]["name"],
                    'command': command.split(),
                    'environment': [
                        {
                            'name': 'NODE_IDX',
                            'value': str(node_idx)
                        },
                        {
                            'name': 'HOST',
                            'value': host
                        },
                        {
                            'name': 'NODE_TYPE',
                            'value': node_type
                        },
                        {
                            'name': 'ERROR_BOUND',
                            'value': str(error_bound)
                        }
                    ],
                }
            ]
        }
    )
    if len(response['tasks']) == 0:
        # Something went wrong
        print("Error: node", node_idx, "run_task() failure reason:", response['failures'][0]['reason'])
        print("Stop coordinator and nodes manually")
        raise Exception


def get_default_security_group(region):
    ec2_client = session.client('ec2', region_name=region)
    response = ec2_client.describe_security_groups()
    for sg in response['SecurityGroups']:
        if sg['Description'] == 'default VPC security group':
            sg_id = sg['GroupId']
    # _ = ec2_client.describe_vpcs()
    response = ec2_client.describe_subnets()
    subnet_id = response['Subnets'][0]['SubnetId']
    print('Found sg and subnet for region', region, ': sg_id:', sg_id, 'subnet_id:', subnet_id)
    return sg_id, subnet_id


def create_log_group(automon_task, node_type, error_bound):
    log_group_name = node_type.replace("_", "-") + "_" + str(error_bound).replace(".", "-")
    automon_task["containerDefinitions"][0]["logConfiguration"]["options"]["awslogs-group"] = log_group_name
    logs_client = session.client('logs', region_name=automon_task["containerDefinitions"][0]["logConfiguration"]["options"]["awslogs-region"])
    response = logs_client.describe_log_groups()
    for log_group in response["logGroups"]:
        if log_group["logGroupName"] == log_group_name:
            # Log group is already exist
            print("Found log group", log_group_name)
            return
    logs_client.create_log_group(logGroupName=log_group_name)


def run_coordinator_on_ecs_fargate(coordinator_region, node_type, error_bound, automon_task, command):
    ecs_client = session.client('ecs', region_name=coordinator_region)
    if error_bound:
        cluster_name = node_type.replace("_", "-") + "_" + str(error_bound).replace(".", "-") + "_" + coordinator_region
    else:
        cluster_name = node_type.replace("_", "-") + "_" + str(error_bound).replace(".", "-") + "_" + coordinator_region
    print("Cluster:", cluster_name)
    _ = ecs_client.create_cluster(clusterName=cluster_name)  # Use different cluster for every experiment
    sg_id, subnet_id = get_default_security_group(coordinator_region)
    run_task(ecs_client, automon_task, cluster_name, -1, '0.0.0.0', node_type, error_bound, subnet_id, sg_id, command)
    time.sleep(10)  # Wait until the task obtains its public IP
    tasks = ecs_client.list_tasks(cluster=cluster_name)
    ips = get_service_ips(ecs_client, cluster_name, tasks['taskArns'], coordinator_region)
    coordinator_ip = ips[0]
    print("Coordinator public IP:", coordinator_ip)
    return coordinator_ip


def run_nodes_on_ecs_fargate(nodes_region, node_type, error_bound, automon_task, coordinator_ip, command):
    ecs_client = session.client('ecs', region_name=nodes_region)
    cluster_name = node_type.replace("_", "-") + "_" + str(error_bound).replace(".", "-") + "_" + nodes_region
    print("Cluster:", cluster_name)
    _ = ecs_client.create_cluster(clusterName=cluster_name)  # Use different cluster for every experiment
    sg_id, subnet_id = get_default_security_group(nodes_region)
    for node_idx in range(NUM_NODES):
        run_task(ecs_client, automon_task, cluster_name, node_idx, coordinator_ip, node_type, error_bound, subnet_id, sg_id, command)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--node_type", type=str, dest="node_type", help="node type. Could be one of: inner_product / quadratic / kld / dnn", default='inner_product')
    parser.add_argument("--coordinator_aws_instance_type", type=str, dest="coordinator_aws_instance_type",
                        help="coordinator AWS instance type. Could be one of: ec2 / fargate. Use ECS Fargate coordinator for cases that do not require strong coordinator "
                             "(e.g., inner_product), or use strong EC2 coordinator.",
                        default='ec2')
    parser.add_argument("--centralized", dest="b_centralized",
                        help="if --centralized is specified, a centralization (not AutoMon) experiment ie deployed",
                        action='store_true')
    parser.add_argument("--block", dest="b_block",
                        help="if --block is specified, the script first checks if the experiment output files are already exist in AWS S3. If not, if runs the experiment "
                             "and waits until it finds the output files in S3.",
                        action='store_true')
    args = parser.parse_args()

    if args.b_centralized:
        command = 'python /app/aws_experiments/start_distributed_centralization_object_remote.py'
        node_name = 'centralization_' + args.node_type

        # Centralized node simply sends all its data to the coordinator. It is not an adaptive algorithm, hence not impacted by error bound.
        # Therefore, we use a single dummy error_bound in the list of error_bounds.
        error_bounds = ['centralization']  # Used for cluster name and log group name
    else:
        command = 'python /app/aws_experiments/start_distributed_object_remote.py'
        node_name = args.node_type

        # These values were taken from the test_max_error_vs_communication_xxx experiments
        if args.node_type == "inner_product":
            error_bounds = [0.02, 0.05, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8]
        if args.node_type == "quadratic":
            error_bounds = [0.015, 0.02, 0.03, 0.04, 0.05, 0.08, 0.1, 1.0]
        if args.node_type == "kld":
            # Use only 8 error bounds to have less than 100 ECS instances in the same region, as Amazon has 100 ECS instance limit for a single region (8 error
            # bounds x 12 nodes = 96 ECS instance, plus 8 EC2/ECS coordinator instances in a different region). The original list is:
            # [0.003, 0.004, 0.005, 0.01, 0.02, 0.04, 0.06, 0.08, 0.1, 0.12, 0.14]
            error_bounds = [0.005, 0.01, 0.02, 0.04, 0.06, 0.08, 0.1, 0.14]
        if args.node_type == "dnn":
            # Use only 6 error bounds to save time and money. The original list is:
            # [0.001, 0.002, 0.0027, 0.003, 0.005, 0.007, 0.01, 0.016, 0.025, 0.05]
            error_bounds = [0.002, 0.003, 0.005, 0.007, 0.016, 0.05]

    if args.node_type == "inner_product":
        NUM_NODES = 10
    if args.node_type == "quadratic":
        NUM_NODES = 10
    if args.node_type == "kld":
        NUM_NODES = 12
    if args.node_type == "dnn":
        NUM_NODES = 9

    if args.b_block:
        num_experiments = num_completed_experiments(node_name)
        if num_experiments >= len(error_bounds):
            print("Found existing remote AWS S3 test folder for " + node_name + ". Skipping.")
            sys.exit()

    username, access_key_id, secret_access_key = read_credentials_file()
    session = boto3.session.Session(aws_access_key_id=access_key_id, aws_secret_access_key=secret_access_key)
    account_id = session.client('sts').get_caller_identity().get('Account')
    with open(os.path.abspath(os.path.dirname(__file__)) + "/automon_aws_task.json", 'r') as f:
        automon_task = f.read()
    automon_task = automon_task.replace("<account_id>", account_id)
    automon_task = json.loads(automon_task)

    for error_bound in error_bounds:
        create_log_group(automon_task, args.node_type, error_bound)

        # Coordinator
        coordinator_region = "us-west-2"
        if args.coordinator_aws_instance_type == "fargate":
            coordinator_ip = run_coordinator_on_ecs_fargate(coordinator_region, args.node_type, error_bound, automon_task, command)  # Using ECS Fargate coordinator is limited to 4 vCPU and 16GB of memory on an Intel Xeon CPU at 2.2–2.5 GHz.
        else:
            coordinator_ip = run_coordinator_on_ec2_instance(coordinator_region, args.node_type, error_bound, command)  # Run the coordinator on EC2 c5.4xlarge instance (16 vCPU and 32GB of memory on an Intel Xeon CPU at 3.4–3.9 GHz).

        # Nodes
        nodes_region = "us-east-2"
        run_nodes_on_ecs_fargate(nodes_region, args.node_type, error_bound, automon_task, coordinator_ip, command)

    if args.b_block:
        # Wait for the experiment to finish by checking the result folders in S3, and then collect the result from S3.
        while True:
            num_experiments = num_completed_experiments(node_name)
            if num_experiments >= len(error_bounds):
                break
            print("AWS experiment", node_name, "is still running. Checking status again in 5 minutes.")
            sys.stdout.flush()
            time.sleep(5 * 60)
        # Wait one more minute to let all S3 writes to finish
        time.sleep(60)
