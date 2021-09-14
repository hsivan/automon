import json
import os
import boto3
import importlib.resources as pkg_resources
from zipfile import ZipFile
import io
from examples.aws_utils.utils import create_iam_role


def make_zip_file_bytes():
    buf = io.BytesIO()
    with pkg_resources.path('examples.aws_utils', 'lambda_function_stop_and_terminate_ec2.py') as full_path:
        with ZipFile(buf, 'w') as z:
            z.write(full_path, "lambda_function_stop_and_terminate_ec2.py")
    return buf.getvalue()


def stop_and_terminate_ec2_instance(region='us-west-2'):
    try:
        ec2_instance_id = os.environ['INSTANCE_ID']
    except KeyError:
        print("INSTANCE_ID is not defined. This is an ECS instance. No need to terminate.")
        return

    lambda_client = boto3.client('lambda', region_name=region)
    role_name = 'AutomonLambdaFunctionRole'
    role = create_iam_role(role_name)

    lambda_func_name = "stopAndTerminateEc2Lambda"
    try:
        response = lambda_client.create_function(
            FunctionName=lambda_func_name,
            Runtime='python3.8',
            Role=role['Role']['Arn'],
            Handler='lambda_function_stop_and_terminate_ec2.lambda_handler',
            Code=dict(ZipFile=make_zip_file_bytes()),
            Timeout=300
        )
        print(response)
    except lambda_client.exceptions.ResourceConflictException as e:
        print(e)

    payload = {"ec2_instance_id": ec2_instance_id, "region": region}
    response = lambda_client.invoke(
        FunctionName=lambda_func_name,
        InvocationType='Event',
        Payload=json.dumps(payload)
    )
    print(response)


if __name__ == "__main__":
    os.environ['INSTANCE_ID'] = 'i-0a9d48cb2df919512'
    stop_and_terminate_ec2_instance()
