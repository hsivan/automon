"""
Cleanup the following AWS resources, which were allocated by our distributed experiments:
1. S3 bucket called automon-experiment-results
2. ECR repository named automon
3. CloudWatch log groups whose name contains inner-product, quadratic, kld, and dnn
4. ECS clusters whose name contains inner-product, quadratic, kld, and dnn
5. EC2 instances whose KeyName contains automon

Requirements:
Assuming AWS cli is already installed and configured before running this script, by previous run of the
reproduce_experiment.py script.
The default region must be us-east-2 (as configured by reproduce_experiment.py).
"""
import subprocess


def execute_shell_command_with_live_output(cmd):
    """
    Executes shell command with live output of STD, so it would not be boring :)
    :param cmd: the command to execute
    :return:
    """
    process = subprocess.Popen(cmd, shell=True, stdout=subprocess.PIPE, stderr=subprocess.STDOUT)
    print("Executed command: " + cmd)
    while True:
        output = process.stdout.readline()
        if process.poll() is not None:
            break
        if output:
            print(output.strip().decode('utf-8'))
    rc = process.poll()
    if rc != 0:
        print("RC not 0 (RC=" + str(rc) + ") for command: " + cmd)
        raise Exception
    return rc


if __name__ == "__main__":
    node_types = ["inner-product", "quadratic", "kld", "dnn"]

    # Delete S3 bucket called automon-experiment-results
    try:
        execute_shell_command_with_live_output('aws s3 rb s3://automon-experiment-results --force')
    except Exception:
        pass

    # Delete ECR repository named automon
    try:
        execute_shell_command_with_live_output('aws ecr delete-repository --repository-name automon --force')
    except Exception:
        pass

    # Delete CloudWatch log groups whose name contains inner-product, quadratic, kld, and dnn
    for node_type in node_types:
        try:
            command = "aws logs describe-log-groups --query 'logGroups[*].logGroupName' --output table | awk '{print $2}' | grep " + node_type + " | while read x; do  echo \"deleting $x\" ; aws logs delete-log-group --log-group-name $x; done"
            execute_shell_command_with_live_output(command)
        except Exception:
            pass

    # Delete ECS clusters whose name contains inner-product, quadratic, kld, and dnn
    for node_type in node_types:
        try:
            command = "aws ecs list-clusters --query 'clusterArns[*]' --output table | awk '{print $2}' | grep " + node_type + " | while read x; do  echo \"deleting $x\" ; aws ecs delete-cluster --cluster $x; done"
            execute_shell_command_with_live_output(command)
        except Exception:
            pass

    # Terminate EC2 instances from us-west-2 region whose KeyName contains automon
    try:
        command = "aws ec2 describe-instances --region us-west-2 --query 'Reservations[*].Instances[*].{Key:KeyName,Instance:InstanceId}' --output text | grep automon | awk '{print $1}' | while read x; do  echo \"terminating $x\" ; aws ec2 terminate-instances --region us-west-2 --instance-ids $x; done"
        execute_shell_command_with_live_output(command)
    except Exception:
        pass
