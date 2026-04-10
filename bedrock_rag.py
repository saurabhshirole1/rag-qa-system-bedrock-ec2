# ============================================================
# FILE: bedrock_rag.py
# PURPOSE: This file contains ALL the logic for:
#   1. Connecting to Amazon Bedrock
#   2. Sending a question to the Knowledge Base
#   3. Retrieving relevant document chunks (the "R" in RAG)
#   4. Getting the AI to generate a cited answer (the "G" in RAG)
#   5. Uploading documents to S3
#   6. Returning the answer + citations back to the Streamlit UI
# ============================================================


# ============================================================
# SECTION 1: IMPORTS
# ============================================================

import os
# os = used to read environment variables
# Example: os.getenv("AWS_ACCESS_KEY_ID") reads your key from .env

import logging
# logging = prints structured messages to the terminal
# Better than print() because it shows timestamps and severity levels

import boto3
# boto3 = official AWS SDK for Python
# This is how Python talks to AWS services like S3, Bedrock, EC2

from botocore.exceptions import ClientError
# ClientError = the specific error boto3 raises when AWS rejects a request
# Example causes: wrong permissions, wrong region, invalid parameters

from dotenv import load_dotenv
# load_dotenv = reads your .env file and loads the keys as environment variables
# Without this, boto3 cannot find your AWS credentials

from config import (
    AWS_REGION,        # "us-east-1" — which AWS data center to use
    KNOWLEDGE_BASE_ID, # the ID of your Bedrock Knowledge Base e.g. "ABCD1234EF"
    MODEL_ARN,         # the model ARN — now pointing to Amazon Nova Lite
    NUMBER_OF_RESULTS, # how many document chunks to retrieve per question
    S3_BUCKET_NAME,    # your S3 bucket name e.g. "my-company-knowledgebase-2024"
    S3_PREFIX          # folder inside bucket e.g. "documents/"
)


# ============================================================
# SECTION 2: INITIALIZATION
# Runs once when this file is imported by app.py
# ============================================================

# Load .env file so AWS credentials are available to boto3
# This must happen BEFORE any boto3 calls
load_dotenv()

# Set up the logger for this file
# logging.basicConfig sets the minimum level to show (INFO and above)
# INFO = normal messages, WARNING = something odd, ERROR = something failed
logging.basicConfig(level=logging.INFO)

# Create a logger specifically named "bedrock_rag"
# So terminal messages look like: INFO:bedrock_rag:your message here
logger = logging.getLogger(__name__)
# __name__ automatically becomes "bedrock_rag" (the filename without .py)


# ============================================================
# HELPER FUNCTION: clean_nova_response()
# Cleans raw Nova Lite output — removes internal action text
# ============================================================

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

    # These prefixes indicate Nova Lite's internal processing steps
    # We want to SKIP these lines — they are not part of the real answer
    skip_prefixes = [
        "Action:",       # e.g. Action: GlobalDataSource.search(...)
        "Observation:",  # e.g. Observation: Found 3 results
        "Thought:",      # e.g. Thought: I should search for...
        "Tool:",         # e.g. Tool: search
        "Input:",        # e.g. Input: {"query": "..."}
    ]

    # Loop through each line and keep only "clean" lines
    cleaned_lines = []
    for line in lines:
        # Check if this line starts with any of the skip prefixes
        should_skip = any(
            line.strip().startswith(prefix)
            for prefix in skip_prefixes
        )
        # Only keep the line if it is NOT an internal action line
        if not should_skip:
            cleaned_lines.append(line)

    # Rejoin the kept lines back into a single string
    # .strip() removes any leading/trailing blank lines
    cleaned_answer = "\n".join(cleaned_lines).strip()

    # If after cleaning nothing remains, return a helpful fallback message
    # This can happen if Nova returned ONLY action lines with no real answer
    if not cleaned_answer:
        cleaned_answer = (
            "I searched your documents but could not find a specific answer. "
            "Please make sure your documents are uploaded and synced in "
            "the Bedrock Knowledge Base console."
        )

    return cleaned_answer


# ============================================================
# FUNCTION 1: get_bedrock_client()
# Creates a connection to Amazon Bedrock
# ============================================================

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
        # boto3.client() creates a connection to the specified AWS service
        client = boto3.client(
            service_name="bedrock-agent-runtime", # which AWS service
            region_name=AWS_REGION                 # which region e.g. "us-east-1"
        )
        logger.info("Bedrock client created successfully")
        return client

    except Exception as e:
        # If connection fails, log the error and stop execution
        logger.error(f"Failed to create Bedrock client: {e}")
        raise  # re-raise so the calling function knows it failed


# ============================================================
# FUNCTION 2: query_knowledge_base()
# The MAIN function — takes a question, returns an AI answer
# ============================================================

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

        # Step 2: Call retrieve_and_generate()
        # This single API call does BOTH retrieve AND generate:
        #
        #   RETRIEVE → Bedrock embeds your question into a vector,
        #              searches the vector DB, finds top matching chunks
        #
        #   GENERATE → Those chunks + your question are sent to Nova Lite,
        #              which writes a natural language answer
        #
        # KEY NOTE: We do NOT include "retrievalConfiguration" here
        # because the current Bedrock API version does not support it
        # inside knowledgeBaseConfiguration — it caused errors before.

        response = client.retrieve_and_generate(
            input={
                "text": question  # the user's question
            },
            retrieveAndGenerateConfiguration={
                "type": "KNOWLEDGE_BASE",
                # type = "KNOWLEDGE_BASE" tells Bedrock to use your KB
                # instead of calling the model directly without context

                "knowledgeBaseConfiguration": {
                    "knowledgeBaseId": KNOWLEDGE_BASE_ID,
                    # KNOWLEDGE_BASE_ID = the ID from AWS Bedrock console
                    # looks like "A1B2C3D4E5" — set in config.py

                    "modelArn": MODEL_ARN
                    # MODEL_ARN = which AI model writes the final answer
                    # Now set to Amazon Nova Lite in config.py:
                    # "arn:aws:bedrock:us-east-1::foundation-model/amazon.nova-lite-v1:0"
                }
            }
        )

        # Step 3: Extract the raw answer text from the response
        # AWS returns a large nested dictionary
        # The answer text lives at response["output"]["text"]
        raw_answer = response["output"]["text"]
        logger.info("Raw answer received from Bedrock")

        # Step 4: Clean the Nova Lite response
        # Nova Lite sometimes puts internal action steps in the answer
        # like "Action: GlobalDataSource.search(...)"
        # Our clean_nova_response() function removes those lines
        answer_text = clean_nova_response(raw_answer)
        logger.info("Answer cleaned successfully")

        # Step 5: Extract citations
        # Bedrock tells us WHICH document chunks it used to form the answer
        # This is called "grounding" — proof that the answer came from YOUR documents
        citations = []

        # response["citations"] is a list of citation objects
        for citation in response.get("citations", []):

            # Each citation has "retrievedReferences" = the actual document chunks
            for reference in citation.get("retrievedReferences", []):

                # The S3 URI = full path to the source document
                # e.g. "s3://my-company-knowledgebase-2024/documents/ml_questions.docx"
                source_uri = reference["location"]["s3Location"]["uri"]

                # The actual text snippet that was retrieved from the document
                excerpt = reference["content"]["text"]

                # Only show first 300 characters as a preview
                # Full chunk text can be very long
                excerpt_preview = (
                    excerpt[:300] + "..."
                    if len(excerpt) > 300
                    else excerpt
                )

                citations.append({
                    "source":  source_uri,      # which file the answer came from
                    "excerpt": excerpt_preview  # what part of the file was used
                })

        # Step 6: Remove duplicate citations
        # Same document can match multiple times if multiple chunks were retrieved
        # We only want to show each document ONCE in the citations list
        seen_sources     = set()   # set() only stores unique values
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
        # ClientError = AWS rejected the API request
        # Most common causes:
        #   - Wrong KNOWLEDGE_BASE_ID in config.py
        #   - Model not enabled in Bedrock console (go enable Nova Lite)
        #   - IAM user missing AmazonBedrockFullAccess permission
        #   - Wrong AWS region in config.py
        error_code    = e.response["Error"]["Code"]
        error_message = e.response["Error"]["Message"]
        logger.error(f"AWS ClientError [{error_code}]: {error_message}")

        return {
            "success":   False,
            "answer":    f"AWS Error: {error_message}",
            "citations": []
        }

    except Exception as e:
        # Catches any other unexpected Python errors
        # e.g. network timeout, malformed response, etc.
        logger.error(f"Unexpected error: {e}")

        return {
            "success":   False,
            "answer":    f"An unexpected error occurred: {str(e)}",
            "citations": []
        }


# ============================================================
# FUNCTION 3: upload_document_to_s3()
# Uploads a file from your computer to the S3 bucket
# ============================================================

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

    # Create an S3 client to talk to the S3 storage service
    # This is separate from the Bedrock client above
    s3_client = boto3.client("s3", region_name=AWS_REGION)

    # Build the full S3 key (path inside the bucket)
    # S3_PREFIX = "documents/"
    # So if file_name = "ml_questions.docx"
    # Then s3_key   = "documents/ml_questions.docx"
    # Full S3 path  = s3://my-company-knowledgebase-2024/documents/ml_questions.docx
    s3_key = S3_PREFIX + file_name

    try:
        # upload_file() copies a local file up to S3
        s3_client.upload_file(
            Filename=file_path,    # local path on your computer
            Bucket=S3_BUCKET_NAME, # which S3 bucket to upload to
            Key=s3_key             # where inside the bucket to put it
        )

        logger.info(
            f"Successfully uploaded '{file_name}' "
            f"to s3://{S3_BUCKET_NAME}/{s3_key}"
        )
        return True  # upload worked

    except ClientError as e:
        # Common reasons upload fails:
        #   - IAM user doesn't have AmazonS3FullAccess permission
        #   - S3_BUCKET_NAME in config.py is wrong
        #   - The file at file_path doesn't exist
        logger.error(f"Failed to upload '{file_name}' to S3: {e}")
        return False  # upload failed