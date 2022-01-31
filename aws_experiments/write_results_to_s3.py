import os
import boto3
from botocore.client import ClientError
from aws_experiments.utils import read_credentials_file


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
        if int(node_idx) == -1:
            dest_folder = 'max_error_vs_comm_' + node_type + "/" + "error_bound_" + str(error_bound) + "/coordinator"
        else:
            dest_folder = 'max_error_vs_comm_' + node_type + "/" + "error_bound_" + str(error_bound) + "/nodes/node_" + str(node_idx)

        bucket = get_s3_bucket()

        for file in os.listdir(test_folder):
            print("About to write " + os.path.join(test_folder, file) + " to " + dest_folder + '/' + file)
            bucket.upload_file(os.path.join(test_folder, file), dest_folder + '/' + file)
        print("After writing to s3")

    except KeyError:
        print("S3_WRITE is not defined")
    except Exception as err:
        print(err)
