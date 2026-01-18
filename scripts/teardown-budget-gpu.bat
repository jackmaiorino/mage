@echo off
setlocal

REM ===== Config =====
set CLUSTER_NAME=mage-rl-gpu
set AWS_REGION=us-east-1
set NAMESPACE=mage-rl
set ECR_REPO=mage-rl-gpu
set DELETE_ECR_REPO=false
set IRSA_ROLE_NAME=mage-rl-s3-backup-role
set IRSA_POLICY_NAME=mage-rl-s3-backup-policy
set EFS_SG_NAME=efs-sg-gpu

echo.
echo Teardown: %CLUSTER_NAME% in %AWS_REGION%

REM ===== Pre-checks =====
aws --version >nul 2>&1 || (echo [ERROR] AWS CLI missing && exit /b 1)
eksctl version >nul 2>&1 || (echo [ERROR] eksctl missing && exit /b 1)
kubectl version --client >nul 2>&1 || (echo [ERROR] kubectl missing && exit /b 1)

for /f "tokens=*" %%i in ('aws sts get-caller-identity --query Account --output text') do set AWS_ACCOUNT_ID=%%i
if "%AWS_ACCOUNT_ID%"=="" (echo [ERROR] Not logged into AWS && exit /b 1)

REM ===== 1) Delete external LBs early (Ingress/Service) =====
echo.
echo [1/8] Deleting Ingress/Services to release external load balancers...
kubectl -n %NAMESPACE% delete ingress grafana --ignore-not-found
kubectl -n %NAMESPACE% delete svc grafana --ignore-not-found

REM ===== 2) Delete namespace (removes deployments, PVCs, HPAs, etc.) =====
echo.
echo [2/8] Deleting namespace %NAMESPACE% (this removes workloads, PVCs, HPAs, ConfigMaps)...
kubectl delete ns %NAMESPACE% --ignore-not-found
kubectl wait --for=delete ns/%NAMESPACE% --timeout=180s >nul 2>&1

REM Delete cluster-scoped StorageClass created by deploy script
echo.
echo [2b/8] Deleting StorageClass efs-sc (cluster-scoped)...
kubectl delete storageclass efs-sc --ignore-not-found

REM ===== 3) Delete IRSA service account, role, and policy =====
echo.
echo [3/8] Deleting IRSA service account (and CloudFormation stack if managed by eksctl)...
eksctl delete iamserviceaccount --cluster %CLUSTER_NAME% --region %AWS_REGION% --namespace %NAMESPACE% --name s3-backup-sa --wait >nul 2>&1

echo [3b/8] Deleting IRSA IAM role and policy (if present)...
for /f "tokens=*" %%i in ('aws iam list-policies --scope Local --query "Policies[?PolicyName=='%IRSA_POLICY_NAME%'].Arn" --output text') do set IRSA_POLICY_ARN=%%i
aws iam detach-role-policy --role-name %IRSA_ROLE_NAME% --policy-arn arn:aws:iam::%AWS_ACCOUNT_ID%:policy/%IRSA_POLICY_NAME% >nul 2>&1
aws iam delete-role --role-name %IRSA_ROLE_NAME% >nul 2>&1
if not "%IRSA_POLICY_ARN%"=="" (
  for /f "tokens=*" %%v in ('aws iam list-policy-versions --policy-arn %IRSA_POLICY_ARN% --query "Versions[?IsDefaultVersion==`false`].VersionId" --output text') do aws iam delete-policy-version --policy-arn %IRSA_POLICY_ARN% --version-id %%v >nul 2>&1
  aws iam delete-policy --policy-arn %IRSA_POLICY_ARN% >nul 2>&1
)

REM ===== 4) Delete GPU nodegroup (managed) =====
echo.
echo [4/8] Deleting GPU node group 'gpu-workers'...
eksctl delete nodegroup --cluster %CLUSTER_NAME% --region %AWS_REGION% --name gpu-workers --wait >nul 2>&1

REM ===== 5) Delete EKS cluster (removes standard-workers too) =====
echo.
echo [5/8] Deleting EKS cluster %CLUSTER_NAME% (this can take 10-15 minutes)...
eksctl delete cluster --name %CLUSTER_NAME% --region %AWS_REGION% --wait

REM ===== 6) Delete EFS filesystem, mount targets, and security group =====
echo.
echo [6/8] Deleting EFS filesystem and mount targets...
for /f "tokens=*" %%i in ('aws efs describe-file-systems --creation-token %CLUSTER_NAME% --query "FileSystems[0].FileSystemId" --output text 2^>nul') do set EFS_ID=%%i
if not "%EFS_ID%"=="" (
  for /f "tokens=*" %%m in ('aws efs describe-mount-targets --file-system-id %EFS_ID% --query "MountTargets[].MountTargetId" --output text') do (
    echo   - Deleting mount target %%m
    aws efs delete-mount-target --mount-target-id %%m >nul 2>&1
  )
  REM wait a bit for mount targets to disappear
  timeout /t 10 >nul
  aws efs delete-file-system --file-system-id %EFS_ID% >nul 2>&1
) else (
  echo   - No EFS filesystem found via creation-token %CLUSTER_NAME%
)

echo [6b/8] Deleting EFS security group (if present)...
for /f "tokens=*" %%g in ('aws ec2 describe-security-groups --filters Name=group-name,Values=%EFS_SG_NAME% --query "SecurityGroups[0].GroupId" --output text 2^>nul') do set EFS_SG_ID=%%g
if not "%EFS_SG_ID%"=="" (
  aws ec2 delete-security-group --group-id %EFS_SG_ID% >nul 2>&1
)

REM ===== 7) Optionally delete ECR repository (with images) =====
if /I "%DELETE_ECR_REPO%"=="true" (
  echo.
  echo [7/8] Deleting ECR repository %ECR_REPO% (force, includes all images)...
  aws ecr delete-repository --repository-name %ECR_REPO% --region %AWS_REGION% --force >nul 2>&1
)

REM ===== 8) Final cleanup =====
echo.
echo [8/8] Final cleanup: removing local generated files...
if exist k8s\storage-gpu.yaml del /q k8s\storage-gpu.yaml

echo.
echo Teardown initiated. Some deletions (ELBs/Cluster) take time; you can monitor in AWS Console.
echo Done.

endlocal

