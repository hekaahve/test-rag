import os

from azure.identity import DefaultAzureCredential
from azure.storage.blob import BlobServiceClient
from dotenv import load_dotenv

load_dotenv()


class BlobService:
    def __init__(self):
        account_url = os.getenv("BLOB_ACCOUNT_URL")
        self.container_name = os.getenv("CONTAINER_NAME")
        if not account_url:
            raise ValueError("BLOB_ACCOUNT_URL is missing")
        if not self.container_name:
            raise ValueError("CONTAINER_NAME is missing")
  
        self.blob_service_client = BlobServiceClient(
            account_url=account_url,
            credential=DefaultAzureCredential(),
        )

    def list_blobs_flat(self):
        client = self.blob_service_client.get_container_client(self.container_name)
        documents = []
        for blob in client.list_blobs():
            blob_data = client.get_blob_client(blob.name).download_blob()
            content = blob_data.readall().decode("utf-8")
            documents.extend(line.strip() for line in content.splitlines() if line.strip())
        return documents



