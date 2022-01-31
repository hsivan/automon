# Distributed experiment on a real-world WAN
We include code for a series of cross-region experiments on AWS using two clusters:
one cluster is located in US-West (Oregon) region and is comprised of a single coordinator using 16 virtual CPUs and 32GB of memory;
the other one is located in US-East (Ohio) region and includes all the node tasks, each of them with 1 virtual CPU and 4GB of memory.

To run these distributed experiments you will need an AWS account, docker engine, docker cli, and aws cli.
After having these tools installed and configured follow these steps:
1. Download AutoMon's source code and the external datasets (see [download instructions](../experiments/README.md)). Set `PYTHONPATH`: `export PYTHONPATH=$PYTHONPATH:<automon_root>`
2. Create AWS IAM user with  AdministratorAccess permissions and download the csv file `new_user_credentials.csv` that contains the key ID and the secret key.
3. Place the `new_user_credentials.csv` file in `<automon_root>/aws_experiments` folder.
4. Build AutoMon's docker image, push it to AWS ECR, and start the experiment:
```bash
sudo docker build -f aws_experiments/awstest.Dockerfile  -t automon .
aws ecr get-login-password --region us-east-2 | sudo docker login --username AWS --password-stdin <your_AWS_account_number>.dkr.ecr.us-east-2.amazonaws.com/automon
sudo docker tag automon <your_AWS_account_number>.dkr.ecr.us-east-2.amazonaws.com/automon
sudo docker push <your_AWS_account_number>.dkr.ecr.us-east-2.amazonaws.com/automon
python <automon_root>/aws_experiments/deploy_aws_experiment.py
```
You can follow the status of the tasks in ECS console and watch the logs in CloudWatch console.
After the experiment finishes the results are written to S3 bucket named `automon-experiment-results`.

There are two different options to run the experiment.
The first option is to run the coordinator and nodes as ECS Fargate tasks.
The following figure demonstrates the system structure:

![](../docs/ecs_coordinator.png)

ECS Fargate task is limited to 4 vCPU and 16GB of memory on an Intel Xeon CPU at 2.2–2.5 GHz.
Therefore, this option is suitable for cases that do not require strong coordinator (e.g., inner product monitoring). 
The second option is to run the coordinator on an EC2 instance:

![](../docs/ec2_coordinator.png)

We use EC2 c5.4xlarge instance (16 vCPU and 32GB of memory on an Intel Xeon CPU at 3.4–3.9 GHz).
This option is suitable for cases that require heavy computations (e.g. DNN monitoring).
