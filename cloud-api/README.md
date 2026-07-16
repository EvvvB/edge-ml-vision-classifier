# Edge ML Vision Cloud API

This service accepts image detections at `POST /detections`, stores the image in S3, and stores metadata in PostgreSQL on the EC2 instance.

## Local development

Local development uses PostgreSQL for metadata and MinIO as an S3-compatible
object store. The included credentials work only with these local containers.

Requirements:

- Python 3.11 or newer
- Docker with the Compose plugin

Set up the application:

```bash
cd cloud-api
cp .env.example .env
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements-dev.txt
```

Start PostgreSQL and MinIO:

```bash
./bin/compose up -d
```

Run the API:

```bash
fastapi dev app/main.py --host 0.0.0.0 --port 8000
```

Useful local URLs:

- API documentation: http://127.0.0.1:8000/docs
- API liveness: http://127.0.0.1:8000/health
- API dependency readiness: http://127.0.0.1:8000/ready
- MinIO console: http://127.0.0.1:9001

Sign in to the MinIO console with `minioadmin` / `minioadmin`. Uploaded images
appear in the `vision-detections` bucket.

Run the automated tests:

```bash
pytest
```

Stop the local services without deleting their data:

```bash
./bin/compose down
```

## Send an image and metadata

```bash
curl -X POST http://127.0.0.1:8000/detections \
  -F 'image=@/path/to/image.jpg;type=image/jpeg' \
  -F 'metadata={"device_id":"pi-01","label":"cat","confidence":0.94,"captured_at":"2026-07-02T20:15:00-07:00"}'
```

Successful responses include the generated `image_id`, S3 bucket, S3 key, and `s3://` URL.

When `CLOUD_API_KEY` is set (always in production), every `/detections`
endpoint requires a matching `X-API-Key` header and returns 401 without it.
`/health` and `/ready` stay unauthenticated. Leave `CLOUD_API_KEY` unset for
local development to disable the check.

## Read stored metadata

```bash
curl http://127.0.0.1:8000/detections
curl http://127.0.0.1:8000/detections/{image_id}
curl 'http://127.0.0.1:8000/detections?device_id=pi-01&limit=20'
```

## Browser clients

The browser does not create an S3 object definition. Send a `multipart/form-data`
request containing an `image` file and a JSON string in `metadata`:

```javascript
const form = new FormData();
form.append("image", imageFile);
form.append("metadata", JSON.stringify({
  device_id: "browser-01",
  captured_at: new Date().toISOString(),
}));

const response = await fetch("http://127.0.0.1:8000/detections", {
  method: "POST",
  body: form,
});
```

FastAPI generates the object ID and S3 key, uploads the image, and records the
metadata. A future direct-to-S3 browser flow would use backend-generated
presigned URLs instead of exposing S3 credentials to the browser.

## Production AWS configuration

Remove the local endpoint settings and configure:

```bash
export CLOUD_DATABASE_URL='postgresql://vision:password@database-host/vision_classifier'
export CLOUD_S3_BUCKET='your-detection-image-bucket'
export CLOUD_S3_PREFIX='detections'
export AWS_REGION='us-west-2'
```

On EC2, prefer an instance IAM role with `s3:PutObject` and `s3:ListBucket`
permission for the configured bucket. Do not place AWS secret keys in browser
code or commit them to `.env`.

The app creates the `detections` table and indexes on startup if they do not
exist.
