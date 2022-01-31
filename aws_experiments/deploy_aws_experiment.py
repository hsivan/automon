import time
import boto3
import json
import os


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
        eni_resource = boto3.resource("ec2", region).NetworkInterface(eni)
        ips.append(eni_resource.association_attribute.get("PublicIp"))

    return ips


def run_task(ecs_client, automon_task, cluster_name, node_idx, host, node_type, error_bound, subnet_id, sg_id):
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
    ec2_client = boto3.client('ec2', region_name=region)
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
    logs_client = boto3.client('logs', region_name=automon_task["containerDefinitions"][0]["logConfiguration"]["options"]["awslogs-region"])
    response = logs_client.describe_log_groups()
    for log_group in response["logGroups"]:
        if log_group["logGroupName"] == log_group_name:
            # Log group is already exist
            print("Found log group", log_group_name)
            return
    logs_client.create_log_group(logGroupName=log_group_name)


def run_coordinator_on_ecs_fargate(coordinator_region, node_type, error_bound, automon_task):
    ecs_client = boto3.client('ecs', region_name=coordinator_region)
    cluster_name = node_type.replace("_", "-") + "_" + str(error_bound).replace(".", "-") + "_" + coordinator_region
    print("Cluster:", cluster_name)
    _ = ecs_client.create_cluster(clusterName=cluster_name)  # Use different cluster for every experiment
    sg_id, subnet_id = get_default_security_group(coordinator_region)
    run_task(ecs_client, automon_task, cluster_name, -1, '0.0.0.0', node_type, error_bound, subnet_id, sg_id)
    time.sleep(10)  # Wait until the task obtains its public IP
    tasks = ecs_client.list_tasks(cluster=cluster_name)
    ips = get_service_ips(ecs_client, cluster_name, tasks['taskArns'], coordinator_region)
    coordinator_ip = ips[0]
    print("Coordinator public IP:", coordinator_ip)
    return coordinator_ip


def run_nodes_on_ecs_fargate(nodes_region, node_type, error_bound, automon_task, coordinator_ip):
    ecs_client = boto3.client('ecs', region_name=nodes_region)
    cluster_name = node_type.replace("_", "-") + "_" + str(error_bound).replace(".", "-") + "_" + nodes_region
    print("Cluster:", cluster_name)
    _ = ecs_client.create_cluster(clusterName=cluster_name)  # Use different cluster for every experiment
    sg_id, subnet_id = get_default_security_group(nodes_region)
    for node_idx in range(NUM_NODES):
        run_task(ecs_client, automon_task, cluster_name, node_idx, coordinator_ip, node_type, error_bound, subnet_id, sg_id)


if __name__ == "__main__":
    NODE_TYPE = "inner_product"  # Could be one of: inner_product / quadratic / kld / dnn

    # These values were taken from the test_max_error_vs_communication_xxx experiments
    if NODE_TYPE == "inner_product":
        error_bounds = [0.3]
        NUM_NODES = 10
    if NODE_TYPE == "quadratic":
        error_bounds = [0.015, 0.02, 0.03, 0.04, 0.05, 0.08, 0.1, 1.0]
        NUM_NODES = 10
    if NODE_TYPE == "kld":
        error_bounds = [0.003, 0.004, 0.005, 0.01, 0.02, 0.04, 0.06, 0.08, 0.1, 0.12, 0.14]
        NUM_NODES = 12
    if NODE_TYPE == "dnn":
        error_bounds = [0.002, 0.003, 0.005, 0.007, 0.016, 0.05]
        NUM_NODES = 9

    account_id = boto3.client('sts').get_caller_identity().get('Account')
    with open(os.path.abspath(os.path.dirname(__file__)) + "/automon_aws_task.json", 'r') as f:
        automon_task = f.read()
    automon_task = automon_task.replace("<account_id>", account_id)
    automon_task = json.loads(automon_task)

    for error_bound in error_bounds:
        create_log_group(automon_task, NODE_TYPE, error_bound)

        # Coordinator
        coordinator_region = "us-west-2"
        # TODO: could use either ECS coordinator for cases that do not require strong coordinator (e.g., inner_product), or use strong EC2 coordinator.
        coordinator_ip = run_coordinator_on_ecs_fargate(coordinator_region, NODE_TYPE, error_bound, automon_task)  # Using ECS Fargate coordinator is limited to 4 vCPU and 16GB of memory on an Intel Xeon CPU at 2.2–2.5 GHz.
        # coordinator_ip = run_coordinator_on_ec2_instance(coordinator_region, NODE_TYPE, error_bound)  # Run the coordinator on EC2 c5.4xlarge instance (16 vCPU and 32GB of memory on an Intel Xeon CPU at 3.4–3.9 GHz).

        # Nodes
        nodes_region = "us-east-2"
        run_nodes_on_ecs_fargate(nodes_region, NODE_TYPE, error_bound, automon_task, coordinator_ip)
