# SECTION 1: IMPORTS
import os
import logging
import boto3
from botocore.exceptions import ClientError
from dotenv import load_dotenv

from config import (
    AWS_REGION,        
    KNOWLEDGE_BASE_ID, 
    MODEL_ARN,         
    NUMBER_OF_RESULTS, 
    S3_BUCKET_NAME,    
    S3_PREFIX          
)


# SECTION 2: INITIALIZATION
load_dotenv()
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# HELPER FUNCTION: clean_nova_response()


def clean_nova_response(raw_text: str) -> str:
    """
    Amazon Nova Lite sometimes includes its internal "thinking" steps
    in the response like:
        Action: GlobalDataSource.search(query="...")
        Observation: ...

    This function removes those lines and returns only the
    clean human-readable answer.

    Args:
        raw_text (str): The raw response text from Nova Lite

    Returns:
        str: Cleaned answer text with action lines removed
    """

    # Split the full response into individual lines
    lines = raw_text.split("\n")
    skip_prefixes = [
        "Action:",       
        "Observation:",  
        "Thought:",      
        "Tool:",         
        "Input:",        
    ]

    cleaned_lines = []
    for line in lines:
        should_skip = any(
            line.strip().startswith(prefix)
            for prefix in skip_prefixes
        )
        if not should_skip:
            cleaned_lines.append(line)
    cleaned_answer = "\n".join(cleaned_lines).strip()
    if not cleaned_answer:
        cleaned_answer = (
            "I searched your documents but could not find a specific answer. "
            "Please make sure your documents are uploaded and synced in "
            "the Bedrock Knowledge Base console."
        )

    return cleaned_answer

# FUNCTION 1: get_bedrock_client()

def get_bedrock_client():
    """
    Creates and returns a boto3 client for Amazon Bedrock Agent Runtime.

    WHY "bedrock-agent-runtime"?
    There are two Bedrock services:
      - "bedrock"               → for listing/configuring models
      - "bedrock-agent-runtime" → for RUNNING queries using Knowledge Bases ✅

    HOW AUTHENTICATION WORKS:
    boto3 automatically looks for credentials in this order:
      1. Environment variables loaded from your .env file by load_dotenv()
      2. ~/.aws/credentials file (if you ran "aws configure")
      3. IAM Role attached to EC2 (used when deployed on AWS server)

    Returns:
        boto3 client object ready to make Bedrock API calls
    """
    try:
        client = boto3.client(
            service_name="bedrock-agent-runtime", 
            region_name=AWS_REGION                 
        )
        logger.info("Bedrock client created successfully")
        return client

    except Exception as e:
        logger.error(f"Failed to create Bedrock client: {e}")
        raise  

# FUNCTION 2: query_knowledge_base()

def query_knowledge_base(question: str) -> dict:
    """
    This is the core RAG function. It:
      1. Takes the user's question as plain text
      2. Sends it to Amazon Bedrock Knowledge Base
      3. Bedrock converts the question into a vector (numbers)
      4. Bedrock searches the vector database for matching document chunks
      5. The matching chunks + question are sent to Nova Lite AI
      6. Nova Lite generates a grounded answer based on your documents
      7. We clean the response and extract citations
      8. Return everything as a clean Python dictionary

    Args:
        question (str): The user's natural language question
                        e.g. "What is supervised learning?"

    Returns:
        dict: {
            "success":   True or False,
            "answer":    "Supervised learning is...",
            "citations": [
                {"source": "s3://bucket/docs/file.docx", "excerpt": "..."},
            ]
        }
    """

    # Step 1: Get the Bedrock connection object
    client = get_bedrock_client()

    try:
        logger.info(f"Querying Knowledge Base with: {question}")
        # Step 2: Call the Bedrock API to retrieve relevant documents and generate an answer

        response = client.retrieve_and_generate(
            input={
                "text": question  # the user's question
            },
            retrieveAndGenerateConfiguration={
                "type": "KNOWLEDGE_BASE",
                "knowledgeBaseConfiguration": {
                    "knowledgeBaseId": KNOWLEDGE_BASE_ID,

                    "modelArn": MODEL_ARN
                }
            }
        )

        # Step 3: Extract the raw answer text from the response
        raw_answer = response["output"]["text"]
        logger.info("Raw answer received from Bedrock")

        # Step 4: Clean the Nova Lite response
        answer_text = clean_nova_response(raw_answer)
        logger.info("Answer cleaned successfully")

        # Step 5: Extract citations
        citations = []

        for citation in response.get("citations", []):
            for reference in citation.get("retrievedReferences", []):

                source_uri = reference["location"]["s3Location"]["uri"]

                excerpt = reference["content"]["text"]
                excerpt_preview = (
                    excerpt[:300] + "..."
                    if len(excerpt) > 300
                    else excerpt
                )

                citations.append({
                    "source":  source_uri,      
                    "excerpt": excerpt_preview  
                })

        # Step 6: Remove duplicate citations
        seen_sources     = set()   
        unique_citations = []

        for cite in citations:
            if cite["source"] not in seen_sources:
                seen_sources.add(cite["source"])
                unique_citations.append(cite)

        # Step 7: Return the final clean result
        return {
            "success":   True,
            "answer":    answer_text,      # cleaned answer from Nova Lite
            "citations": unique_citations  # list of source documents used
        }

    except ClientError as e:
        error_code    = e.response["Error"]["Code"]
        error_message = e.response["Error"]["Message"]
        logger.error(f"AWS ClientError [{error_code}]: {error_message}")

        return {
            "success":   False,
            "answer":    f"AWS Error: {error_message}",
            "citations": []
        }

    except Exception as e:
        logger.error(f"Unexpected error: {e}")

        return {
            "success":   False,
            "answer":    f"An unexpected error occurred: {str(e)}",
            "citations": []
        }

# FUNCTION 3: upload_document_to_s3()

def upload_document_to_s3(file_path: str, file_name: str) -> bool:
    """
    Uploads a local file to your S3 bucket so Bedrock can index it.

    IMPORTANT: After uploading, you must go to the AWS Bedrock console
    and click "Sync" on your Knowledge Base. Without syncing, Bedrock
    does not know the new file exists and cannot answer questions from it.

    Args:
        file_path (str): Full path to the file on your computer
                         Windows example: "C:/Users/You/AppData/Local/Temp/file.pdf"
                         Linux example:   "/tmp/file.pdf"

        file_name (str): What to name the file inside S3
                         Example: "ml_questions.docx"

    Returns:
        bool: True if upload succeeded, False if it failed
    """
    s3_client = boto3.client("s3", region_name=AWS_REGION)
    s3_key = S3_PREFIX + file_name

    try:
        s3_client.upload_file(
            Filename=file_path,  
            Bucket=S3_BUCKET_NAME, 
            Key=s3_key             
        )

        logger.info(
            f"Successfully uploaded '{file_name}' "
            f"to s3://{S3_BUCKET_NAME}/{s3_key}"
        )
        return True  

    except ClientError as e:
        logger.error(f"Failed to upload '{file_name}' to S3: {e}")
        return False  # upload failed