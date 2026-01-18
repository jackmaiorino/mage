@echo off
setlocal

echo.
echo Deploying GPU-Accelerated Mage RL Training
echo.
echo Cost Breakdown Estimate:
echo - Control Plane: $72/month (~$2.40/day)
echo - 1x g4dn.xlarge (on-demand): ~$100/month (~$3.33/day) 
echo - 2x g4dn.large spots: ~$40/month (~$1.33/day)
echo - EFS storage: ~$10/month (~$0.33/day)
echo - Total: ~$7-15/day (with GPU acceleration!)
echo.

echo Checking for prerequisites...

REM Check Docker
docker version >nul 2>&1
if errorlevel 1 (
    echo [ERROR] Docker is not running. Please start Docker Desktop and try again.
    exit /b 1
)
echo  - Docker Desktop is running.

REM Check AWS CLI
aws --version >nul 2>&1
if errorlevel 1 (
    echo [ERROR] AWS CLI not found. Please install it and try again.
    exit /b 1
)
echo  - AWS CLI is available.

REM Check eksctl
eksctl version >nul 2>&1
if errorlevel 1 (
    echo [ERROR] eksctl not found. Please install it and try again.
    exit /b 1
)
echo  - eksctl is available.

REM Check kubectl
kubectl version --client >nul 2>&1
if errorlevel 1 (
    echo [ERROR] kubectl not found. Please install it and try again.
    exit /b 1
)
echo  - kubectl is available.

REM ===== Config =====
set CLUSTER_NAME=mage-rl-gpu
set AWS_REGION=us-east-1
set NAMESPACE=mage-rl
set SA_NAME=s3-backup-sa
set BUCKET_NAME=mage-rl-model-bucket
set ROLE_NAME=mage-rl-s3-backup-role
set POLICY_NAME=mage-rl-s3-backup-policy

echo.
echo Creating EKS cluster with GPU support...
eksctl create cluster ^
  --name %CLUSTER_NAME% ^
  --region %AWS_REGION% ^
  --version 1.30 ^
  --nodegroup-name standard-workers ^
  --node-type t3.medium ^
  --nodes 1 ^
  --nodes-min 1 ^
  --nodes-max 2 ^
  --managed ^
  --tags "project=mage-rl-gpu"

if errorlevel 1 (
    echo [ERROR] Failed to create EKS cluster
    exit /b 1
)

echo.
echo Creating GPU node group (g4dn instances with NVIDIA T4)...
eksctl create nodegroup ^
  --cluster %CLUSTER_NAME% ^
  --region %AWS_REGION% ^
  --name gpu-workers ^
  --node-type g4dn.xlarge ^
  --nodes 1 ^
  --nodes-min 0 ^
  --nodes-max 3 ^
  --managed ^
  --node-labels "accelerator=nvidia-tesla-t4" ^
  --tags "project=mage-rl-gpu,gpu=enabled"

if errorlevel 1 (
    echo [WARNING] GPU nodegroup creation failed. Check capacity/quotas in AWS Console.
    echo [WARNING] Common issues: insufficient g4dn.xlarge capacity, EC2 quota limits, subnet capacity
    echo Continuing with deployment - you can add GPU nodes later...
)

echo.
echo Installing NVIDIA Device Plugin for Kubernetes...
kubectl apply -f https://raw.githubusercontent.com/NVIDIA/k8s-device-plugin/v0.14.1/nvidia-device-plugin.yml

echo.
echo Configuring EFS storage...

REM Get VPC ID
for /f "tokens=*" %%i in ('aws eks describe-cluster --name %CLUSTER_NAME% --region %AWS_REGION% --query "cluster.resourcesVpcConfig.vpcId" --output text') do set VPC_ID=%%i
echo - Using VPC_ID: %VPC_ID%

REM Create security group
echo - Creating new security group 'efs-sg-gpu'...
for /f "tokens=*" %%i in ('aws ec2 create-security-group --group-name efs-sg-gpu --description "EFS access for Mage RL GPU" --vpc-id %VPC_ID% --query "GroupId" --output text') do set SG_ID=%%i

REM Add security group rules
aws ec2 authorize-security-group-ingress --group-id %SG_ID% --protocol tcp --port 2049 --source-group %SG_ID%

REM Get worker node security group
for /f "tokens=*" %%i in ('aws eks describe-cluster --name %CLUSTER_NAME% --region %AWS_REGION% --query "cluster.resourcesVpcConfig.clusterSecurityGroupId" --output text') do set WORKER_SG=%%i
aws ec2 authorize-security-group-ingress --group-id %SG_ID% --protocol tcp --port 2049 --source-group %WORKER_SG%

REM Create EFS filesystem
echo - Creating new EFS filesystem '%CLUSTER_NAME%'...
for /f "tokens=*" %%i in ('aws efs create-file-system --creation-token %CLUSTER_NAME% --performance-mode generalPurpose --tags "Key=Name,Value=%CLUSTER_NAME%" --query "FileSystemId" --output text') do set EFS_ID=%%i

echo - EFS_ID: %EFS_ID%

REM Get subnets
for /f "tokens=*" %%i in ('aws ec2 describe-subnets --filters "Name=vpc-id,Values=%VPC_ID%" --query "Subnets[*].SubnetId" --output text') do set SUBNETS=%%i

REM Create mount targets
if "%EFS_ID%"=="" (
    echo [ERROR] EFS filesystem creation failed. Check AWS Console.
    exit /b 1
)
for %%s in (%SUBNETS%) do (
    echo - Creating mount target for subnet %%s
    aws efs create-mount-target --file-system-id %EFS_ID% --subnet-id %%s --security-groups %SG_ID%
)

REM Update kubeconfig
aws eks update-kubeconfig --region %AWS_REGION% --name %CLUSTER_NAME%

echo.
echo Installing EFS CSI driver...
kubectl apply -k "github.com/kubernetes-sigs/aws-efs-csi-driver/deploy/kubernetes/overlays/stable/?ref=master"

echo.
echo Verifying cluster is ACTIVE before proceeding...
for /f "tokens=*" %%i in ('aws eks describe-cluster --name %CLUSTER_NAME% --region %AWS_REGION% --query "cluster.status" --output text') do set CLUSTER_STATUS=%%i
if /I not "%CLUSTER_STATUS%"=="ACTIVE" (
  echo [ERROR] Cluster status is %CLUSTER_STATUS%. Please wait until ACTIVE and re-run.
  exit /b 1
)

echo.
echo Creating ECR repository...
aws ecr create-repository --repository-name mage-rl-gpu --region %AWS_REGION% 2>nul

echo.
echo Resolving AWS account/region and logging into ECR...
for /f "tokens=*" %%i in ('aws sts get-caller-identity --query Account --output text') do set AWS_ACCOUNT_ID=%%i
aws ecr get-login-password --region %AWS_REGION% | docker login --username AWS --password-stdin %AWS_ACCOUNT_ID%.dkr.ecr.%AWS_REGION%.amazonaws.com

REM ===== IRSA and S3 backup role/policy setup =====


echo.
echo Ensuring OIDC provider (IRSA) is associated to the cluster...
eksctl utils associate-iam-oidc-provider --cluster %CLUSTER_NAME% --region %AWS_REGION% --approve

echo Creating/ensuring S3 access policy for backups...
echo { > %TEMP%\mage-rl-s3-policy.json
echo   "Version": "2012-10-17", >> %TEMP%\mage-rl-s3-policy.json
echo   "Statement": [ >> %TEMP%\mage-rl-s3-policy.json
echo     { "Effect":"Allow", "Action":["s3:ListBucket"], "Resource":["arn:aws:s3:::%BUCKET_NAME%"] }, >> %TEMP%\mage-rl-s3-policy.json
echo     { "Effect":"Allow", "Action":["s3:PutObject","s3:GetObject","s3:DeleteObject","s3:ListBucketMultipartUploads","s3:AbortMultipartUpload"], "Resource":["arn:aws:s3:::%BUCKET_NAME%/*"] } >> %TEMP%\mage-rl-s3-policy.json
echo   ] >> %TEMP%\mage-rl-s3-policy.json
echo } >> %TEMP%\mage-rl-s3-policy.json
for /f "tokens=*" %%i in ('aws iam list-policies --scope Local --query "Policies[?PolicyName=='%POLICY_NAME%'].Arn" --output text') do set POLICY_ARN=%%i
if "%POLICY_ARN%"=="" (
  for /f "tokens=*" %%i in ('aws iam create-policy --policy-name %POLICY_NAME% --policy-document file://%TEMP%\mage-rl-s3-policy.json --query Policy.Arn --output text') do set POLICY_ARN=%%i
)

echo Creating/annotating IAM ServiceAccount for backup CronJob...
eksctl create iamserviceaccount ^
  --cluster %CLUSTER_NAME% ^
  --region %AWS_REGION% ^
  --name %SA_NAME% ^
  --namespace %NAMESPACE% ^
  --role-name %ROLE_NAME% ^
  --attach-policy-arn %POLICY_ARN% ^
  --override-existing-serviceaccounts ^
  --approve

set ROLE_ARN=arn:aws:iam::%AWS_ACCOUNT_ID%:role/%ROLE_NAME%

echo.
echo Building and pushing GPU-optimized Docker image (immutable tag)...
for /f "tokens=*" %%i in ('git rev-parse --short HEAD') do set GIT_SHA=%%i
set IMAGE_TAG=%GIT_SHA%
docker build -f Dockerfile.gpu-optimized -t mage-rl-gpu:%IMAGE_TAG% .
docker tag mage-rl-gpu:%IMAGE_TAG% %AWS_ACCOUNT_ID%.dkr.ecr.%AWS_REGION%.amazonaws.com/mage-rl-gpu:%IMAGE_TAG%
docker push %AWS_ACCOUNT_ID%.dkr.ecr.%AWS_REGION%.amazonaws.com/mage-rl-gpu:%IMAGE_TAG%

echo.
echo Deploying GPU-accelerated configuration...

REM Apply Kubernetes manifests
kubectl apply -f k8s/namespace.yaml

REM Create EFS dynamic provisioning (StorageClass + PVC)
echo apiVersion: storage.k8s.io/v1 > k8s/storage-gpu.yaml
echo kind: StorageClass >> k8s/storage-gpu.yaml
echo metadata: >> k8s/storage-gpu.yaml
echo   name: efs-sc >> k8s/storage-gpu.yaml
echo provisioner: efs.csi.aws.com >> k8s/storage-gpu.yaml
echo reclaimPolicy: Retain >> k8s/storage-gpu.yaml
echo volumeBindingMode: WaitForFirstConsumer >> k8s/storage-gpu.yaml
echo --- >> k8s/storage-gpu.yaml
echo apiVersion: v1 >> k8s/storage-gpu.yaml
echo kind: PersistentVolumeClaim >> k8s/storage-gpu.yaml
echo metadata: >> k8s/storage-gpu.yaml
echo   name: efs-pvc >> k8s/storage-gpu.yaml
echo   namespace: mage-rl >> k8s/storage-gpu.yaml
echo spec: >> k8s/storage-gpu.yaml
echo   accessModes: >> k8s/storage-gpu.yaml
echo     - ReadWriteMany >> k8s/storage-gpu.yaml
echo   resources: >> k8s/storage-gpu.yaml
echo     requests: >> k8s/storage-gpu.yaml
echo       storage: 100Gi >> k8s/storage-gpu.yaml
echo   storageClassName: efs-sc >> k8s/storage-gpu.yaml

kubectl apply -f k8s/storage-gpu.yaml
kubectl apply -f k8s/configmap.yaml
kubectl apply -f k8s/monitoring.yaml

REM Prepare backup CronJob manifest with resolved values
powershell -Command "$c = Get-Content k8s/backup-cronjob.yaml; $c = $c -replace 'REPLACE_BUCKET_NAME','%BUCKET_NAME%'; $c = $c -replace 'us-east-1','%AWS_REGION%'; $c = $c -replace 'arn:aws:iam::REPLACE_ACCOUNT_ID:role/REPLACE_IAM_ROLE_FOR_S3','%ROLE_ARN%'; Set-Content k8s/backup-cronjob.yaml $c"

REM Update image in gpu deployment with immutable tag
powershell -Command "$content = Get-Content k8s/gpu-deployment.yaml; $content = $content -replace '915449292132.dkr.ecr.us-east-1.amazonaws.com/mage-rl-gpu:latest', '%AWS_ACCOUNT_ID%.dkr.ecr.%AWS_REGION%.amazonaws.com/mage-rl-gpu:%IMAGE_TAG%'; Set-Content k8s/gpu-deployment.yaml $content"

kubectl apply -f k8s/gpu-deployment.yaml
kubectl apply -f k8s/hpa.yaml
kubectl apply -f k8s/backup-cronjob.yaml
kubectl apply -f k8s/network-policy.yaml

REM ===== Secure runtime values: public IP and Grafana admin secret =====
for /f "tokens=*" %%i in ('powershell -Command "(Invoke-RestMethod -Uri https://api.ipify.org).ToString()"') do set PUBLIC_IP=%%i
set PUBLIC_CIDR=%PUBLIC_IP%/32
set GRAFANA_ADMIN_USER=admin
if "%GRAFANA_ADMIN_PASS%"=="" set /p GRAFANA_ADMIN_PASS=Enter Grafana admin password (will not be saved in repo): 

REM Create/patch grafana admin secret
powershell -Command "kubectl -n mage-rl delete secret grafana-admin --ignore-not-found | Out-Null; kubectl -n mage-rl create secret generic grafana-admin --from-literal=password='%GRAFANA_ADMIN_PASS%' | Out-Null"

REM Apply IP allowlist to Ingress and keep basic auth placeholder if used
powershell -Command "$c = Get-Content k8s/grafana-ingress.yaml; $c = $c -replace 'REPLACE_IP_CIDR','%PUBLIC_CIDR%'; Set-Content k8s/grafana-ingress.yaml $c"
kubectl apply -f k8s/grafana-ingress.yaml

REM Also restrict Service LoadBalancer to your IP
kubectl -n mage-rl patch svc grafana -p "{\"spec\":{\"loadBalancerSourceRanges\":[\"%PUBLIC_CIDR%\"]}}"

echo.
echo Waiting for rollouts to complete...
kubectl rollout status deployment/mage-rl-gpu-learner -n mage-rl --timeout=300s
kubectl rollout status deployment/mage-rl-gpu-workers -n mage-rl --timeout=300s

echo.
echo GPU deployment complete!
echo.
echo Monitoring Commands:
echo kubectl get pods -n mage-rl -w
echo kubectl logs -f deployment/mage-rl-gpu-learner -n mage-rl
echo kubectl describe nodes
echo nvidia-smi (on worker nodes)
echo.

echo Setup complete! Estimated daily cost: $7-15 (with GPU acceleration)