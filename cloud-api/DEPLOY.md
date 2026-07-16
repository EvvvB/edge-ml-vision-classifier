# AWS deployment runbook

Single EC2 instance in `us-west-1` running the API and Postgres via Docker
Compose, images stored in S3, credentials via an instance IAM role, reachable
at an Elastic IP over HTTP. No Terraform — every resource is created with an
explicit CLI command below so the environment can be rebuilt from this file.

Placeholders to substitute throughout: `YOUR_BUCKET`, `YOUR_IP` (your home IP
for SSH), `ELASTIC_IP`, and IDs printed by earlier commands (`sg-…`, `i-…`).

## 0. One-time local setup

Install the AWS CLI (`brew install awscli`), then create an access key for
your IAM user in the AWS console (IAM → Users → Security credentials →
Create access key → "CLI") and run:

```bash
aws configure   # paste the key, secret, region us-west-1, output json
aws sts get-caller-identity   # sanity check
```

## 1. S3 bucket

If reusing an existing bucket, skip creation but still block public access:

```bash
aws s3api create-bucket --bucket YOUR_BUCKET --region us-west-1 \
  --create-bucket-configuration LocationConstraint=us-west-1
aws s3api put-public-access-block --bucket YOUR_BUCKET \
  --public-access-block-configuration \
  BlockPublicAcls=true,IgnorePublicAcls=true,BlockPublicPolicy=true,RestrictPublicBuckets=true
```

## 2. IAM role for the instance

```bash
aws iam create-role --role-name vision-api-ec2 \
  --assume-role-policy-document '{
    "Version": "2012-10-17",
    "Statement": [{
      "Effect": "Allow",
      "Principal": {"Service": "ec2.amazonaws.com"},
      "Action": "sts:AssumeRole"
    }]
  }'

aws iam put-role-policy --role-name vision-api-ec2 \
  --policy-name vision-detections-s3 \
  --policy-document '{
    "Version": "2012-10-17",
    "Statement": [
      {
        "Effect": "Allow",
        "Action": ["s3:PutObject", "s3:GetObject"],
        "Resource": "arn:aws:s3:::YOUR_BUCKET/*"
      },
      {
        "Effect": "Allow",
        "Action": ["s3:ListBucket"],
        "Resource": "arn:aws:s3:::YOUR_BUCKET"
      }
    ]
  }'

aws iam create-instance-profile --instance-profile-name vision-api-ec2
aws iam add-role-to-instance-profile \
  --instance-profile-name vision-api-ec2 --role-name vision-api-ec2
```

## 3. Security group

HTTP open to the world (the Pi needs it), SSH only from your IP:

```bash
aws ec2 create-security-group --group-name vision-api \
  --description "Edge ML vision API"          # note the sg- id it prints

aws ec2 authorize-security-group-ingress --group-id sg-XXXX \
  --protocol tcp --port 80 --cidr 0.0.0.0/0
aws ec2 authorize-security-group-ingress --group-id sg-XXXX \
  --protocol tcp --port 22 --cidr YOUR_IP/32
```

## 4. Key pair and instance

```bash
aws ec2 create-key-pair --key-name vision-api \
  --query 'KeyMaterial' --output text > ~/.ssh/vision-api.pem
chmod 600 ~/.ssh/vision-api.pem

# Latest Ubuntu 24.04 LTS arm64 AMI in us-west-1
AMI=$(aws ec2 describe-images --owners 099720109477 \
  --filters 'Name=name,Values=ubuntu/images/hvm-ssd-gp3/ubuntu-noble-24.04-arm64-server-*' \
  --query 'sort_by(Images,&CreationDate)[-1].ImageId' --output text)

aws ec2 run-instances \
  --image-id "$AMI" \
  --instance-type t4g.micro \
  --key-name vision-api \
  --security-group-ids sg-XXXX \
  --iam-instance-profile Name=vision-api-ec2 \
  --metadata-options HttpTokens=required,HttpPutResponseHopLimit=2 \
  --block-device-mappings 'DeviceName=/dev/sda1,Ebs={VolumeSize=20,VolumeType=gp3}' \
  --tag-specifications 'ResourceType=instance,Tags=[{Key=Name,Value=vision-api}]'
```

`HttpPutResponseHopLimit=2` matters: with the default of 1, code inside a
Docker container cannot reach the instance metadata service, so boto3 would
find no credentials.

## 5. Elastic IP

```bash
aws ec2 allocate-address                       # note the AllocationId
aws ec2 associate-address --instance-id i-XXXX --allocation-id eipalloc-XXXX
```

This IP is permanently yours until released. Adding a domain later is just an
A record pointing at it plus nginx/Let's Encrypt on the instance.

## 6. Install Docker on the instance

```bash
ssh -i ~/.ssh/vision-api.pem ubuntu@ELASTIC_IP

sudo apt-get update
sudo apt-get install -y docker.io docker-compose-v2
sudo usermod -aG docker ubuntu
exit   # log out/in so the group change takes effect
```

## 7. Deploy the app

From your machine, build the frontend (its output lands in
`cloud-api/static/`, which the API container serves at `/`), then copy the
service to the instance (rsync respects the excludes; .env files are never
copied):

```bash
(cd frontend && npm ci && npm run build)

rsync -av -e "ssh -i ~/.ssh/vision-api.pem" \
  --exclude '.venv' --exclude '__pycache__' --exclude '.pytest_cache' \
  --exclude '.env*' \
  cloud-api/ ubuntu@ELASTIC_IP:~/cloud-api/
```

On the instance:

```bash
cd ~/cloud-api
cp .env.production.example .env.production
nano .env.production   # set POSTGRES_PASSWORD (twice: alone and inside the URL),
                       # CLOUD_S3_BUCKET, and check AWS_REGION=us-west-1

docker compose --env-file .env.production -f compose.prod.yaml up -d --build
```

## 8. Verify

```bash
curl http://ELASTIC_IP/health          # {"ok":true}, no auth required
curl http://ELASTIC_IP/ready           # checks Postgres and S3, no auth required

# /detections endpoints require the X-API-Key header matching CLOUD_API_KEY
# in .env.production (missing/wrong key returns 401).
curl -X POST http://ELASTIC_IP/detections \
  -H "X-API-Key: $CLOUD_API_KEY" \
  -F 'image=@/path/to/test.jpg;type=image/jpeg' \
  -F 'metadata={"device_id":"test","label":"cat","confidence":0.9,"captured_at":"2026-07-15T12:00:00-07:00"}'

curl -H "X-API-Key: $CLOUD_API_KEY" http://ELASTIC_IP/detections
```

Then point the Raspberry Pi uploader at `http://ELASTIC_IP`.

The dashboard is at `http://ELASTIC_IP/` — it asks for the API key
(`CLOUD_API_KEY` from `.env.production`) on first visit and stores it in the
browser.

## Redeploying after code changes

```bash
(cd frontend && npm run build)   # only needed if frontend/ changed
rsync -av -e "ssh -i ~/.ssh/vision-api.pem" \
  --exclude '.venv' --exclude '__pycache__' --exclude '.pytest_cache' \
  --exclude '.env*' \
  cloud-api/ ubuntu@ELASTIC_IP:~/cloud-api/
ssh -i ~/.ssh/vision-api.pem ubuntu@ELASTIC_IP \
  'cd ~/cloud-api && docker compose --env-file .env.production -f compose.prod.yaml up -d --build'
```

## Operations notes

- Logs: `docker compose --env-file .env.production -f compose.prod.yaml logs -f api`
- Postgres data lives in the `postgres-data` Docker volume on the instance's
  EBS disk; it survives `docker compose down` but not instance termination.
  For backups: `docker compose --env-file .env.production -f compose.prod.yaml exec postgres pg_dump -U vision vision_classifier > backup.sql`
- Set a billing alarm: AWS console → Billing → Budgets → create a monthly
  budget with an email alert.
- Rough monthly cost: t4g.micro ~$6, EBS 20 GB ~$2, Elastic IP (attached)
  free, S3 pennies at this scale.

## Deployed resources (2026-07-15)

| Resource | Value |
| --- | --- |
| Account | 365733422629 |
| Region | us-west-1 |
| Elastic IP | 54.193.80.15 |
| Instance | i-061dade88294cbbc1 (t4g.micro, Ubuntu 24.04 arm64) |
| Security group | sg-09e123632e19765f0 (80 open; 22 from home IP only) |
| IAM role / instance profile | vision-api-ec2 |
| S3 bucket | vision-detection-365733422629-us-west-1-an |
| SSH | `ssh -i ~/.ssh/vision-api.pem ubuntu@54.193.80.15` |

If your home IP changes, re-allow SSH with:

```bash
aws ec2 authorize-security-group-ingress --group-id sg-09e123632e19765f0 \
  --protocol tcp --port 22 --cidr $(curl -s https://checkip.amazonaws.com)/32
```
