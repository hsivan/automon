import json
import time
import boto3
import pandas as pd
import os


# Not used
def create_iam_user_and_access_key(user_name):
    iam_client = boto3.client('iam')
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
def delete_access_key(user_name, access_key_id):
    iam_client = boto3.client('iam')
    response = iam_client.delete_access_key(UserName=user_name, AccessKeyId=access_key_id)
    print(response)


def read_credentials_file():
    credentials_file = os.path.abspath(os.path.dirname(__file__)) + "/new_user_credentials.csv"
    df = pd.read_csv(credentials_file)
    username = df['User name'][0]
    access_key_id = df['Access key ID'][0]
    secret_access_key = df['Secret access key'][0]
    return username, access_key_id, secret_access_key


def create_iam_role(role_name):
    iam_client = boto3.client('iam')

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
