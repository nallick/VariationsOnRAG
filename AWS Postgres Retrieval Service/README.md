## Using EC2 for building
Create small Amazon Linux x86_64 instance (t2.small) with 2 CPU, 2GB RAM, 32GB disk.
Currently cpuinfo fails (perhaps in PyTorch) on ARM because it can't parse /sys/devices/system/cpu/possible and /sys/devices/system/cpu/present.

ssh -i id_ed25519_aws_ec2.pem ec2-user@ec2-3-84-29-225.compute-1.amazonaws.com  
\[ec2-user\]&#xFF04; ```mkdir postgres_retrieval_service```  
\[ec2-user\]&#xFF04; ```python3 -m ensurepip --upgrade```  
\[ec2-user\]&#xFF04; ```python3 -m pip install --upgrade pip```  # which pip -> ~/.local/bin/pip  

To install Docker see:  
https://docs.aws.amazon.com/AmazonECS/latest/developerguide/create-container-image.html  
```
sudo yum update -y
sudo yum install docker
sudo service docker start
sudo usermod -a -G docker ec2-user
docker info # reboot machine if this returns an error
```

\[ec2-user\]&#xFF04; <copy/paste AWS credientials in environment> # export AWS_ACCESS_KEY_ID= ...

```
scp -r -i id_ed25519_aws_ec2.pem \* ec2-user@ec2-3-84-29-225.compute-1.amazonaws.com:postgres_retrieval_service  
scp -i id_ed25519_aws_ec2.pem .env ec2-user@ec2-3-84-29-225.compute-1.amazonaws.com:postgres_retrieval_service
```

## Building
```
docker build -t postgres_retrieval_service:latest .
```

## Testing
```
docker run -p 9000:8080 postgres_retrieval_service:latest
curl -XPOST "http://localhost:9000/2015-03-31/functions/function/invocations" -d '{"body":"Does WWT celebrate Juneteenth?"}'
```

## Deployment
```
aws ecr get-login-password --region us-east-1 | docker login --username AWS --password-stdin 397717753551.dkr.ecr.us-east-1.amazonaws.com  
docker tag postgres_retrieval_service:latest 397717753551.dkr.ecr.us-east-1.amazonaws.com/rag/postgres_retrieval_service:latest  
docker push 397717753551.dkr.ecr.us-east-1.amazonaws.com/rag/postgres_retrieval_service:latest  
```
