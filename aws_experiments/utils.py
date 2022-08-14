import json
import time
import boto3
import pandas as pd
import os
from botocore.client import ClientError


# Not used
def create_iam_user_and_access_key(user_name, session):
    iam_client = session.client('iam')
    try:
        response = iam_client.create_user(UserName=user_name)
        print(response)
        response = iam_client.attach_user_policy(UserName=user_name, PolicyArn='arn:aws:iam::aws:policy/AdministratorAccess')
        print(response)
    except iam_client.exceptions.EntityAlreadyExistsException:
        print("User", user_name, "already exists")

    response = iam_client.create_access_key(UserName=user_name)
    print(response)
    username = response['AccessKey']['UserName']
    access_key_id = response['AccessKey']['AccessKeyId']
    secret_access_key = response['AccessKey']['SecretAccessKey']
    return username, access_key_id, secret_access_key


# Not used
def delete_access_key(user_name, access_key_id, session):
    iam_client = session.client('iam')
    response = iam_client.delete_access_key(UserName=user_name, AccessKeyId=access_key_id)
    print(response)


def read_credentials_file():
    credentials_file = os.path.abspath(os.path.dirname(__file__)) + "/new_user_credentials.csv"
    df = pd.read_csv(credentials_file)
    username = df['User name'][0]
    access_key_id = df['Access key ID'][0]
    secret_access_key = df['Secret access key'][0]
    return username, access_key_id, secret_access_key


def create_iam_role(region, role_name, session):
    iam_client = session.client('iam', region_name=region)

    role_policy = {
        "Version": "2012-10-17",
        "Statement": [
            {
                "Sid": "",
                "Effect": "Allow",
                "Principal": {
                    "Service": "lambda.amazonaws.com"
                },
                "Action": "sts:AssumeRole"
            },
            {
                "Sid": "",
                "Effect": "Allow",
                "Principal": {
                    "Service": "ec2.amazonaws.com"
                },
                "Action": "sts:AssumeRole"
            },
            {
                "Sid": "",
                "Effect": "Allow",
                "Principal": {
                    "Service": "events.amazonaws.com"
                },
                "Action": "sts:AssumeRole"
            }
        ]
    }
    try:
        response = iam_client.create_role(RoleName=role_name, AssumeRolePolicyDocument=json.dumps(role_policy))
        print(response)
        response = iam_client.attach_role_policy(RoleName=role_name, PolicyArn='arn:aws:iam::aws:policy/AdministratorAccess')
        print(response)
        time.sleep(10)
    except iam_client.exceptions.EntityAlreadyExistsException as e:
        print(e)
    role = iam_client.get_role(RoleName=role_name)
    return role


# Self termination by the EC2 instance with credentials installed (no need for session)
def stop_and_terminate_ec2_instance():
    try:
        ec2_instance_id = os.environ['INSTANCE_ID']
        region = os.environ['REGION']
    except KeyError:
        print("INSTANCE_ID is not defined. This is an ECS instance. No need to terminate.")
        return

    ec2_client = boto3.client('ec2', region_name=region)
    response = ec2_client.terminate_instances(InstanceIds=[ec2_instance_id])
    print(response)


def get_s3_bucket():
    bucket_name = "automon-experiment-results"
    username, access_key_id, secret_access_key = read_credentials_file()

    resource_args = dict()
    resource_args['verify'] = False
    resource_args['use_ssl'] = False
    resource_args['aws_access_key_id'] = access_key_id
    resource_args['aws_secret_access_key'] = secret_access_key

    s3_resource = boto3.resource('s3', **resource_args)

    # Check if bucket exists
    try:
        s3_resource.meta.client.head_bucket(Bucket=bucket_name)
    except ClientError:
        print("Bucket", bucket_name, "doesn't exist. Created.")
        s3_resource.create_bucket(Bucket=bucket_name)

    s3_bucket = s3_resource.Bucket(bucket_name)
    return s3_bucket


def copy_result_folder_to_s3(test_folder, node_type, node_idx, error_bound):
    try:
        print("Before writing to s3")
        b_write_to_s3 = os.environ['S3_WRITE']
        if b_write_to_s3 != '1':
            return
        if error_bound:
            if int(node_idx) == -1:
                dest_folder = 'max_error_vs_comm_' + node_type + "_aws/" + "error_bound_" + str(error_bound) + "/coordinator"
            else:
                dest_folder = 'max_error_vs_comm_' + node_type + "_aws/" + "error_bound_" + str(error_bound) + "/nodes/node_" + str(node_idx)
        else:  # No error bound, hence this is a distributed centralization experiment
            if int(node_idx) == -1:
                dest_folder = 'max_error_vs_comm_centralization_' + node_type + "_aws/" + "coordinator"
            else:
                dest_folder = 'max_error_vs_comm_centralization_' + node_type + "_aws/" + "nodes/node_" + str(node_idx)

        bucket = get_s3_bucket()

        for file in os.listdir(test_folder):
            if "data_file.txt" in file or "nethogs_out.txt" in file or "node_schedule_file.txt" in file:
                # Theses are relatively large files (specifically for KLD and DNN), which are not needed for result
                # analysis (we use nethogs_vs_automon.txt which summarizes the network results instead).
                continue
            print("About to write " + os.path.join(test_folder, file) + " to " + dest_folder + '/' + file)
            bucket.upload_file(os.path.join(test_folder, file), dest_folder + '/' + file)
        print("After writing to s3")

    except KeyError:
        print("S3_WRITE is not defined")
    except Exception as err:
        print(err)


def num_completed_experiments(node_type):
    s3_bucket = get_s3_bucket()
    files_and_folders = s3_bucket.objects.all()
    files_and_folders = [f.key for f in files_and_folders if node_type in f.key and 'coordinator' in f.key and 'nethogs' in f.key]  # nethogs_vs_automon.txt for automon and nethogs.txt for centralization
    return len(files_and_folders)


def get_default_security_group(region, session):
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


def create_ingress_rule(region, session, security_group_id):
    """
    Creates a security group ingress rule with the specified configuration.
    """
    vpc_client = session.client("ec2", region_name=region)
    try:
        response = vpc_client.authorize_security_group_ingress(
            GroupId=security_group_id,
            IpPermissions=[{
                'IpProtocol': 'tcp',
                'FromPort': 10,
                'ToPort': 65535,
                'IpRanges': [{
                    'CidrIp': '0.0.0.0/0'
                }]
            }])
        print("Added inbound rule to sg", security_group_id, "response:\n", response)

    except ClientError as e:
        if e.response["Error"]["Code"] == "InvalidPermission.Duplicate":
            print("Inbound rule for TCP port-range 10-65535 already exists.")
        else:
            print('Could not create ingress security group rule.')
            raise
