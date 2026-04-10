# Enterprise Knowledge Base Q&A System
### RAG Implementation using Amazon Bedrock

![Python](https://img.shields.io/badge/Python-3.9+-blue.svg)
![Streamlit](https://img.shields.io/badge/Streamlit-1.32.0-red.svg)
![AWS](https://img.shields.io/badge/AWS-Bedrock-orange.svg)
![License](https://img.shields.io/badge/License-MIT-green.svg)

---

## Table of Contents

- [Project Overview](#project-overview)
- [What is RAG?](#what-is-rag)
- [Tech Stack](#tech-stack)
- [Project Structure](#project-structure)
- [Architecture Diagram](#architecture-diagram)
- [Prerequisites](#prerequisites)
- [AWS Setup](#aws-setup)
- [Local Installation](#local-installation)
- [Configuration](#configuration)
- [Running the App Locally](#running-the-app-locally)
- [Uploading Documents](#uploading-documents)
- [Deploying on AWS EC2](#deploying-on-aws-ec2)
- [How to Use the App](#how-to-use-the-app)
- [File Descriptions](#file-descriptions)
- [Common Errors and Fixes](#common-errors-and-fixes)
- [Project Workflow](#project-workflow)

---

## Project Overview

This project is a **Retrieval-Augmented Generation (RAG)** application built using **Amazon Bedrock Knowledge Bases**. It allows users to ask natural language questions on internal company documents and receive accurate, citation-backed answers powered by Amazon Nova Lite AI model.

**Key Features:**
- Ask questions in plain English about your company documents
- Get AI-generated answers grounded in your actual documents
- View citations showing exactly which document was used
- Upload new documents directly from the web interface
- Deployed on AWS EC2 for 24/7 availability

---

## What is RAG?

**RAG = Retrieval-Augmented Generation**

Instead of asking an AI to answer from its training memory, RAG first **retrieves** relevant chunks from your own documents, then **gives those chunks to the AI** as context so it answers accurately with citations.

```
User Question
     ↓
Convert question to vector (embedding)
     ↓
Search vector database for similar document chunks
     ↓
Send matching chunks + question to AI model
     ↓
AI generates grounded answer with citations
     ↓
User sees accurate, cited answer
```

**Without RAG:** AI answers from general training data → may hallucinate or give wrong info

**With RAG:** AI answers from YOUR documents → accurate, grounded, cited answers

---

## Tech Stack

| Technology | Purpose |
|---|---|
| **Python 3.9+** | Core programming language |
| **Streamlit** | Web application framework (no HTML/CSS needed) |
| **Amazon Bedrock** | Managed AI service — Knowledge Bases + Nova Lite model |
| **Amazon S3** | Cloud storage for company documents |
| **Amazon OpenSearch Serverless** | Vector database for semantic search |
| **Amazon Titan Embeddings G1** | Converts text to vectors for similarity search |
| **Amazon Nova Lite** | LLM that generates the final cited answer |
| **AWS EC2** | Cloud server to deploy and host the application |
| **AWS IAM** | Identity and access management — secure permissions |
| **boto3** | Official AWS SDK for Python |
| **python-dotenv** | Loads secret keys from .env file |

---

## Project Structure

```
enterprise-rag-system/
│
├── app.py                    # Streamlit web application (main UI file)
├── bedrock_rag.py            # Core RAG logic — connects to Bedrock
├── config.py                 # All settings and constants in one place
├── requirements.txt          # Python package dependencies
├── .env                      # Secret AWS keys (never commit to Git!)
├── .gitignore                # Files to exclude from Git
│
├── documents/                # Local folder for your documents
│   ├── hr_policy.txt
│   ├── it_security_policy.txt
│   └── machine_learning_basics.txt
│
├── scripts/
│   ├── setup_s3.py           # One-time script to create S3 bucket and upload docs
│   └── deploy_ec2.sh         # Shell script to set up EC2 server
│
└── .streamlit/
    └── config.toml           # Streamlit appearance and server settings
```

---

## Architecture Diagram
```
┌─────────────┐        Question         ┌──────────────────┐
│    User     │ ─────────────────────▶ │  Streamlit App   │
│  (Browser)  │                        │   (EC2 Server)   │
└─────────────┘ ◀───────────────────── └────────┬─────────┘
                        Answer + Citations      │
                                               │ boto3 API call
                                               ▼
                                   ┌──────────────────────┐
                                   │   Amazon Bedrock     │
                                   │   Knowledge Base     │
                                   └────────┬─────────────┘
                                            │
                     ┌──────────────────────┼──────────────────────┐
                     ▼                      ▼                      ▼
             ┌──────────────┐      ┌──────────────┐      ┌──────────────┐
             │   Amazon S3  │      │ OpenSearch   │      │ Amazon Nova  │
             │  (Documents) │      │  Serverless  │      │    Lite      │
             └──────────────┘      │  (Vectors)   │      │  (Answer)    │
                                   └──────────────┘      └──────────────┘
```
---

## Prerequisites

Before starting, make sure you have the following:

- Python 3.9 or higher installed on your computer
- An AWS account (free tier works — credit card required for Bedrock)
- AWS CLI installed (`pip install awscli`)
- Basic knowledge of terminal/command line

---

## AWS Setup

### Step 1 — Enable Bedrock Model Access

Go to: https://console.aws.amazon.com/bedrock/home?region=us-east-1#/modelaccess

1. Click **"Manage model access"**
2. Enable these models:
   - Amazon Titan Embeddings G1 - Text
   - Amazon Nova Lite
3. Click **"Save changes"**
4. Wait 1-2 minutes for approval

### Step 2 — Create IAM User with Permissions

Go to: https://console.aws.amazon.com/iam/home#/users/create

1. Create user named `rag-dev`
2. Attach these policies directly:
   - `AmazonBedrockFullAccess`
   - `AmazonS3FullAccess`
   - `AmazonOpenSearchServiceFullAccess`
3. Go to **Security credentials → Create access key**
4. Choose **"Local code"** → Create
5. **Copy both keys immediately** — you cannot see the Secret Key again!

### Step 3 — Create S3 Bucket

Run after completing local installation:
```bash
python scripts/setup_s3.py
```

### Step 4 — Create Bedrock Knowledge Base

Go to: https://console.aws.amazon.com/bedrock/home?region=us-east-1#/knowledge-bases

1. Click **"Create knowledge base"**
2. Name: `company-knowledge-base`
3. IAM role: Create and use new service role
4. Data source: Amazon S3 → `s3://YOUR-BUCKET-NAME/documents/`
5. Embeddings model: Amazon Titan Embeddings G1 - Text
6. Vector store: Quick create new vector store (OpenSearch Serverless)
7. Click **"Create knowledge base"** — wait 3-5 minutes
8. After creation, click **"Sync"** to index your documents
9. Copy the **Knowledge Base ID** (looks like `ABCD1234EF`)

---

## Local Installation

```bash
# Step 1 — Clone or navigate to project folder
cd C:\Innomatics\enterprise-rag-system

# Step 2 — Create and activate virtual environment
python -m venv venv

# Windows
venv\Scripts\activate

# Mac/Linux
source venv/bin/activate

# Step 3 — Install all dependencies
pip install -r requirements.txt
```

---

## Configuration

### 1. Set up your `.env` file

Create a file named `.env` in the project root:

```env
AWS_ACCESS_KEY_ID=AKIAIOSFODNN7EXAMPLE
AWS_SECRET_ACCESS_KEY=wJalrXUtnFEMI/K7MDENG/bPxRfiCYEXAMPLEKEY
AWS_DEFAULT_REGION=us-east-1
```

> ⚠️ **Never commit this file to Git!** Add `.env` to your `.gitignore`

### 2. Update `config.py`

```python
# AWS Settings
AWS_REGION = "us-east-1"

# S3 Settings
S3_BUCKET_NAME = "your-bucket-name-here"    # your actual bucket name
S3_PREFIX = "documents/"

# Bedrock Settings
KNOWLEDGE_BASE_ID = "ABCD1234EF"            # paste your KB ID here

# Model — Amazon Nova Lite
MODEL_ARN = "arn:aws:bedrock:us-east-1::foundation-model/amazon.nova-lite-v1:0"

NUMBER_OF_RESULTS = 5
```

---

## Running the App Locally

```bash
# Make sure virtual environment is active
# Windows:
venv\Scripts\activate

# Run the app
streamlit run app.py
```

App opens automatically at: **http://localhost:8501**

---

## Uploading Documents

### Method A — Via the Streamlit App (recommended)

1. Open the app at `http://localhost:8501`
2. In the **left sidebar**, find "Upload Documents"
3. Click **"Browse files"** → select your PDF, TXT, or DOCX
4. Click **"Upload to S3"**
5. Go to AWS Bedrock console → click **Sync**
6. Wait for status: **Completed**

### Method B — Via Script (for bulk upload)

1. Put your files in the `documents/` folder
2. Run:
```bash
python scripts/setup_s3.py
```
3. Go to Bedrock console → click **Sync**

### Supported File Types

| Format | Extension |
|--------|-----------|
| PDF | `.pdf` |
| Plain Text | `.txt` |
| Word Document | `.docx` |
| HTML | `.html` |
| Markdown | `.md` |
| CSV | `.csv` |

> ⚠️ Scanned PDFs (images of documents) do NOT work. Use text-based PDFs only.

### Important — Always Sync After Uploading

S3 and Bedrock are separate services. Uploading to S3 alone is not enough. You must click **Sync** in the Bedrock console after every upload so Bedrock re-indexes the new documents.

---

## Deploying on AWS EC2

### Step 1 — Create IAM Role for EC2

Go to: https://console.aws.amazon.com/iam/home#/roles/create

1. Trusted entity: **AWS service → EC2**
2. Add policies:
   - `AmazonBedrockFullAccess`
   - `AmazonS3FullAccess`
   - `AmazonOpenSearchServiceFullAccess`
3. Role name: `rag-ec2-role`

### Step 2 — Launch EC2 Instance

Go to: https://console.aws.amazon.com/ec2/home#LaunchInstances

| Setting | Value |
|---------|-------|
| Name | `rag-app-server` |
| AMI | Amazon Linux 2023 (free tier) |
| Instance type | `t2.micro` (free tier) |
| Key pair | Create new → `rag-key` → download `.pem` file |
| Security group | Allow SSH (22) + Custom TCP port **8501** from 0.0.0.0/0 |
| IAM instance profile | `rag-ec2-role` |

### Step 3 — SSH into EC2

```bash
# Windows PowerShell — fix key permissions first
icacls rag-key.pem /inheritance:r /grant:r "%USERNAME%:R"

# Connect
ssh -i rag-key.pem ec2-user@YOUR_EC2_PUBLIC_IP
```

### Step 4 — Install and Run on EC2

```bash
# Update server
sudo yum update -y

# Install Python and pip
sudo yum install python3 python3-pip git -y

# Install Python packages
pip3 install streamlit boto3 python-dotenv botocore
```

### Step 5 — Copy Project Files to EC2

Run this from your **local machine** (not SSH):

```bash
scp -i rag-key.pem -r C:\Innomatics\enterprise-rag-system ec2-user@YOUR_EC2_IP:~/
```

### Step 6 — Start the App

```bash
# Inside EC2 SSH terminal
cd ~/enterprise-rag-system

# Run in background (stays running after SSH closes)
nohup streamlit run app.py \
  --server.port 8501 \
  --server.address 0.0.0.0 \
  > app.log 2>&1 &
```

### Step 7 — Access the Live App

Open browser and go to:
```
http://YOUR_EC2_PUBLIC_IP:8501
```

Share this URL with anyone — they can access it from anywhere!

### Useful EC2 Commands

```bash
# Check if app is running
ps aux | grep streamlit

# View live logs
tail -f ~/enterprise-rag-system/app.log

# Stop the app
pkill -f streamlit

# Restart after code update
pkill -f streamlit
cd ~/enterprise-rag-system
nohup streamlit run app.py --server.port 8501 --server.address 0.0.0.0 > app.log 2>&1 &
```

---

## How to Use the App

1. Open the app URL in your browser
2. In the **chat box at the bottom**, type your question
3. Press **Enter** — wait 3-8 seconds
4. View the AI-generated answer
5. Click **"View Sources"** to see which documents were used
6. To upload new documents, use the **left sidebar → Upload Documents**

**Example questions to ask:**
```
What is the leave policy for new employees?
What is overfitting in machine learning?
What are the password requirements?
Explain supervised learning.
What is the work from home policy?
```

---

## File Descriptions

| File | Purpose |
|------|---------|
| `app.py` | Streamlit UI — all the web interface code. Chat input, sidebar, upload widget, displays answers and citations |
| `bedrock_rag.py` | Core logic — creates Bedrock client, calls retrieve_and_generate API, extracts citations, uploads to S3 |
| `config.py` | Central config — all constants like KB ID, model ARN, bucket name. Change settings here only |
| `requirements.txt` | Lists all pip packages. Run `pip install -r requirements.txt` to install all at once |
| `.env` | Your secret AWS keys. Never share or commit this file |
| `.streamlit/config.toml` | Streamlit appearance settings — colors, port number, server options |
| `scripts/setup_s3.py` | One-time setup script — creates S3 bucket and uploads documents folder |
| `scripts/deploy_ec2.sh` | EC2 setup commands — install dependencies and run the app |

---

## Common Errors and Fixes

| Error | Cause | Fix |
|-------|-------|-----|
| `NoCredentialsError` | .env not loading | Make sure `load_dotenv()` is called before boto3 in `bedrock_rag.py` |
| `ResourceNotFoundException` | Wrong KB ID | Copy correct Knowledge Base ID from AWS Bedrock console into `config.py` |
| `AccessDeniedException` | Missing IAM permissions | Add `AmazonBedrockFullAccess` policy to your IAM user |
| `INVALID_PAYMENT_INSTRUMENT` | No credit card on AWS account | Add payment method at AWS Billing console |
| `ModuleNotFoundError: config` | Wrong Python path | Add `sys.path.insert(0, parent_dir)` at top of scripts |
| `Action: GlobalDataSource.search...` shown | Nova Lite internal output | Use `clean_nova_response()` function to strip action lines |
| "No citations found" | Documents not synced | Go to Bedrock console → click Sync → wait for Completed |
| App works locally but not on EC2 | Port 8501 not open | Add inbound rule for port 8501 in EC2 Security Group |

---

## Project Workflow

```
                    ┌─────────────────────────────────┐
                    │         SETUP (one time)         │
                    └─────────────────────────────────┘
                                    │
              ┌─────────────────────┼──────────────────────┐
              ▼                     ▼                      ▼
       Create S3 Bucket      Create Bedrock KB      Launch EC2
       Upload Documents       Sync Documents        Deploy App
              │                     │                      │
              └─────────────────────┴──────────────────────┘
                                    │
                    ┌───────────────▼─────────────────┐
                    │    App is LIVE at EC2_IP:8501   │
                    └───────────────┬─────────────────┘
                                    │
                    ┌───────────────▼──────────────────┐
                    │      EVERY USER QUESTION         │
                    │                                  │
                    │  User types question in browser  │
                    │           ↓                      │
                    │  Streamlit sends to bedrock_rag  │
                    │           ↓                      │
                    │  Bedrock embeds question         │
                    │           ↓                      │
                    │  OpenSearch finds top chunks     │
                    │           ↓                      │
                    │  Nova Lite generates answer      │
                    │           ↓                      │
                    │  Answer + citations displayed    │
                    └──────────────────────────────────┘
                                    │
                    ┌───────────────▼─────────────────┐
                    │    ADDING NEW DOCUMENTS         │
                    │                                 │
                    │  Upload file to S3              │
                    │           ↓                     │
                    │  Click Sync in Bedrock console  │
                    │           ↓                     │
                    │  Wait for "Completed"           │
                    │           ↓                     │
                    │  New doc is searchable          │
                    └─────────────────────────────────┘
```

---

## Author

**Internship Project — Innomatics Research Labs**

Built with Amazon Bedrock, Streamlit, and Python.
