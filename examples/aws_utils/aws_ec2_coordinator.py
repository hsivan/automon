import time
import boto3
import paramiko
import importlib.resources as pkg_resources
from botocore.client import ClientError
from examples.aws_utils.utils import read_credentials_file, create_iam_role


def create_keypair(ec2_client, keypair_name):
    # Search for existing *.pem file (private key) or create one if not found.
    keypair_file_name = keypair_name + '.pem'
    try:
        pkg_resources.open_text('examples.aws_utils', keypair_file_name)
    except ModuleNotFoundError:
        # Create new key pair
        response = ec2_client.create_key_pair(KeyName=keypair_name)
        with open(keypair_name + '.pem', 'w') as private_key_file:
            private_key_file.write(response['KeyMaterial'])
            private_key_file.close()
    key = paramiko.RSAKey.from_private_key_file(keypair_file_name)
    ssh_client = paramiko.SSHClient()
    ssh_client.set_missing_host_key_policy(paramiko.AutoAddPolicy())
    return key, ssh_client


def create_instance(ec2_client, keypair_name):
    instances = ec2_client.run_instances(
        ImageId="ami-09889d8d54f9e0a0e",  # Ubuntu Server 18.04 LTS (HVM), SSD Volume Type - ami-09889d8d54f9e0a0e (64-bit x86) / ami-09c7c5f2666edbd1b (64-bit Arm) (free tier eligible)
        MinCount=1,
        MaxCount=1,
        InstanceType="c5.4xlarge",
        KeyName=keypair_name
    )
    instance_id = instances["Instances"][0]["InstanceId"]
    print("EC2 instance ID:", instance_id)
    return instance_id


def get_public_ip(ec2_client, instance_id):
    reservations = ec2_client.describe_instances(InstanceIds=[instance_id]).get("Reservations")

    for reservation in reservations:
        for instance in reservation['Instances']:
            instance_public_ip = instance.get("PublicIpAddress")
            print("EC2 instance public IP:", instance_public_ip)
    return instance_public_ip


def execute_ssh_command(ssh_client, cmd, stdout_verification=None, b_stderr_verification=False):
    stdin, stdout, stderr = ssh_client.exec_command(cmd)
    print("Executed command:", cmd)
    print("STDOUT:")
    stdout_str = stdout.read()
    print(stdout_str)
    print("STDERR:")
    stderr_str = stderr.read()
    print(stderr_str)

    if stdout_verification is not None:
        if stdout_verification not in stdout_str.decode():
            print("Verification string", stdout_verification, "not in stdout")
            exit(1)
    if b_stderr_verification:
        if stderr_str != b'':
            print("stderr is not empty.")
            exit(1)


def attach_cloudwatch_policy_to_ec2_instance(region, ec2_client, instance_id):
    iam_client = boto3.client('iam', region_name=region)

    instance_profile_name = 'AutomonInstanceProfile'
    try:
        response = iam_client.create_instance_profile(InstanceProfileName=instance_profile_name)
        print(response)
        time.sleep(10)
    except iam_client.exceptions.EntityAlreadyExistsException as e:
        print(e)

    role_name = 'AutomonEc2CloudWatchRole'
    create_iam_role(role_name)

    try:
        response = iam_client.add_role_to_instance_profile(InstanceProfileName=instance_profile_name, RoleName=role_name)
        print(response)
    except iam_client.exceptions.LimitExceededException as e:
        print(e)

    response = iam_client.get_instance_profile(InstanceProfileName=instance_profile_name)
    print(response)
    instance_profile = response['InstanceProfile']
    try:
        response = ec2_client.associate_iam_instance_profile(
            IamInstanceProfile={
                'Arn': instance_profile['Arn'],
                'Name': instance_profile['InstanceProfileName']
            },
            InstanceId=instance_id
        )
        print(response)
        time.sleep(20)
    except ClientError as e:
        print(e)


def run_coordinator_on_ec2_instance(region='us-west-2', node_type='inner_product', error_bound=0.05):
    ec2_client = boto3.client('ec2', region_name=region)
    keypair_name = "aws_automon_private_key"

    try:
        # Create the instance and connect/ssh to it
        key, ssh_client = create_keypair(ec2_client, keypair_name)
        instance_id = create_instance(ec2_client, keypair_name)
        time.sleep(20)  # Wait until the instance obtains its public IP
        instance_public_ip = get_public_ip(ec2_client, instance_id)
        ssh_client.connect(hostname=instance_public_ip, username="ubuntu", pkey=key)

        # Install docker engine and cli
        execute_ssh_command(ssh_client, 'sudo apt-get install -y apt-transport-https ca-certificates curl gnupg lsb-release')
        execute_ssh_command(ssh_client, 'curl -fsSL https://download.docker.com/linux/ubuntu/gpg | sudo gpg --dearmor -o /usr/share/keyrings/docker-archive-keyring.gpg')
        execute_ssh_command(ssh_client, 'echo "deb [arch=amd64 signed-by=/usr/share/keyrings/docker-archive-keyring.gpg] https://download.docker.com/linux/ubuntu $(lsb_release -cs) stable" | sudo tee /etc/apt/sources.list.d/docker.list > /dev/null')
        execute_ssh_command(ssh_client, 'sudo apt-get -y update')
        execute_ssh_command(ssh_client, 'sudo apt-get install -y docker-ce docker-ce-cli containerd.io')
        execute_ssh_command(ssh_client, 'sudo docker run hello-world', stdout_verification="This message shows that your installation appears to be working correctly")

        # Install aws cli
        execute_ssh_command(ssh_client, 'sudo apt-get -y install unzip')
        execute_ssh_command(ssh_client, 'curl "https://awscli.amazonaws.com/awscli-exe-linux-x86_64.zip" -o "awscliv2.zip"')
        execute_ssh_command(ssh_client, 'unzip awscliv2.zip')
        execute_ssh_command(ssh_client, 'sudo ./aws/install')
        execute_ssh_command(ssh_client, 'aws --version', stdout_verification="aws-cli")

        # Configure aws cli
        account_id = boto3.client('sts').get_caller_identity().get('Account')
        username, access_key_id, secret_access_key = read_credentials_file()
        execute_ssh_command(ssh_client, 'aws configure set aws_access_key_id ' + access_key_id)
        execute_ssh_command(ssh_client, 'aws configure set aws_secret_access_key ' + secret_access_key)
        execute_ssh_command(ssh_client, 'aws configure set region ' + region)
        execute_ssh_command(ssh_client, 'aws configure set output json')
        execute_ssh_command(ssh_client, 'aws configure get region', stdout_verification=region)  # Verify configuration worked

        # Pull docker images from ECR
        execute_ssh_command(ssh_client, 'aws ecr get-login-password --region us-east-2 | sudo docker login --username AWS --password-stdin ' + account_id + '.dkr.ecr.us-east-2.amazonaws.com/automon', stdout_verification="Login Succeeded")
        execute_ssh_command(ssh_client, 'sudo docker pull ' + account_id + '.dkr.ecr.us-east-2.amazonaws.com/automon', stdout_verification="Downloaded newer image")

        attach_cloudwatch_policy_to_ec2_instance(region, ec2_client, instance_id)

        # Run the docker image
        s3_write = 1
        # Use the container ID as log stream name (awslogs-stream is not defined as to not override the log stream)
        docker_run_cmd = 'sudo docker run -p 6400:6400' + \
                         ' --env HOST=0.0.0.0' + \
                         ' --env NODE_IDX=-1' + \
                         ' --env NODE_TYPE=' + node_type + \
                         ' --env ERROR_BOUND=' + str(error_bound) + \
                         ' --env S3_WRITE=' + str(s3_write) + \
                         ' --env INSTANCE_ID=' + instance_id + \
                         ' --log-driver=awslogs' + \
                         ' --log-opt awslogs-region=us-east-2' + \
                         ' --log-opt awslogs-group=' + node_type.replace("_", "-") + "_" + str(error_bound).replace(".", "-") + \
                         ' --log-opt awslogs-create-group=true' + \
                         ' -d -it --rm ' + account_id + '.dkr.ecr.us-east-2.amazonaws.com/automon'
        execute_ssh_command(ssh_client, docker_run_cmd, b_stderr_verification=True)

        ssh_client.close()
    except Exception as e:
        print(e)
        exit(1)
    return instance_public_ip


if __name__ == "__main__":
    run_coordinator_on_ec2_instance()
