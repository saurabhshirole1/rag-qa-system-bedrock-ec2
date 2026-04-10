import sys
import os
parent_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.insert(0, parent_dir)
from dotenv import load_dotenv
load_dotenv(os.path.join(parent_dir, ".env")) 
import boto3
from botocore.exceptions import ClientError
from config import AWS_REGION, S3_BUCKET_NAME, S3_PREFIX


def create_s3_bucket():
    """
    Creates an S3 bucket in your AWS account.
    S3 buckets are like folders in the cloud where you store files.
    """
    
    s3 = boto3.client("s3", region_name=AWS_REGION)
    
    try:
        if AWS_REGION == "us-east-1":
            s3.create_bucket(Bucket=S3_BUCKET_NAME)
        else:
            s3.create_bucket(
                Bucket=S3_BUCKET_NAME,
                CreateBucketConfiguration={"LocationConstraint": AWS_REGION}
            )
        
        print(f"Bucket created: {S3_BUCKET_NAME}")
        
        s3.put_public_access_block(
            Bucket=S3_BUCKET_NAME,
            PublicAccessBlockConfiguration={
                "BlockPublicAcls": True,         
                "IgnorePublicAcls": True,        
                "BlockPublicPolicy": True,        
                "RestrictPublicBuckets": True     
            }
        )
        print("Public access blocked (documents are private)")
        
    except ClientError as e:
        if e.response["Error"]["Code"] == "BucketAlreadyOwnedByYou":
           
            print(f"Bucket already exists: {S3_BUCKET_NAME}")
        else:
            
            print(f"Error: {e}")
            raise


def upload_sample_documents():
    """
    Uploads all files from the local 'documents/' folder to S3.
    These are the files Bedrock will read and index.
    """
    
    s3 = boto3.client("s3", region_name=AWS_REGION)
    
    local_docs_path = "documents"
    
    for file_name in os.listdir(local_docs_path):
        
        local_file = os.path.join(local_docs_path, file_name)
        
        s3_key = S3_PREFIX + file_name
        
        print(f"Uploading {file_name}...")
        
        s3.upload_file(
            Filename=local_file,       
            Bucket=S3_BUCKET_NAME,     
            Key=s3_key                
        )
        
        print(f"Done: s3://{S3_BUCKET_NAME}/{s3_key}")

if __name__ == "__main__":
    print("=== S3 Setup Script ===")
    create_s3_bucket()
    upload_sample_documents()
    print("Setup complete! Now create your Bedrock Knowledge Base.")