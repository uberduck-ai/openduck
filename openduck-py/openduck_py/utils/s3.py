import os
import aioboto3

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
