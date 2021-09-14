import time
import boto3


def lambda_handler(event, context):
    ec2_instance_id = event.get("ec2_instance_id")
    region = event.get("region")
    print("About to stop EC2 instance " + ec2_instance_id)
    ec2_client = boto3.client('ec2', region_name=region)
    response = ec2_client.stop_instances(InstanceIds=[ec2_instance_id], Force=True)
    print(response)
    time.sleep(10)
    print("About to terminate EC2 instance " + ec2_instance_id)
    response = ec2_client.terminate_instances(InstanceIds=[ec2_instance_id])
    print(response)
