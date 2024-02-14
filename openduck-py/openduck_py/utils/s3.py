import os
import re
import aioboto3
import boto3
from pathlib import Path

AWS_ACCESS_KEY_ID = os.environ.get("AWS_ACCESS_KEY_ID")
AWS_SECRET_ACCESS_KEY = os.environ.get("AWS_SECRET_ACCESS_KEY")


def async_session():
    session = aioboto3.Session(
        aws_secret_access_key=AWS_SECRET_ACCESS_KEY,
        aws_access_key_id=AWS_ACCESS_KEY_ID,
    )
    return session


async def upload_to_s3_bucket(fp, bucket, filename):
    session = aioboto3.Session()
    async with session.client("s3") as s3:
        await s3.upload_fileobj(fp, bucket, filename)
    return filename


def s3_client():
    session = boto3.Session(
        aws_access_key_id=AWS_ACCESS_KEY_ID,
        aws_secret_access_key=AWS_SECRET_ACCESS_KEY,
    )
    _client = session.client("s3")
    return _client


def upload_fileobj(object_name, bucket, fo):
    _client = s3_client()
    _client.upload_fileobj(fo, bucket, object_name)


def delete_object(bucket, object):
    _client = s3_client()
    _client.delete_object(Bucket=bucket, Key=object)


def download_file(object_name, bucket, path=None):
    safe_name = re.sub(r"[^.0-9a-zA-Z_-]", "_", object_name)
    if path is None:
        path = str(Path("/tmp") / Path(safe_name))
    if Path(path).exists() and Path(path).stat().st_size > 0:
        return path
    if os.path.dirname(path):
        os.makedirs(os.path.dirname(path), exist_ok=True)
    _client = s3_client()
    with open(path, "wb") as f:
        _client.download_fileobj(bucket, object_name, f)
    assert Path(path).exists(), f"{path} does not exist"
    return path
