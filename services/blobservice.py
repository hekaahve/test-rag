import os

from azure.identity import DefaultAzureCredential
from azure.storage.blob import BlobServiceClient
from dotenv import load_dotenv

load_dotenv()


class BlobService:
    account_url = os.getenv("BLOB_ACCOUNT_URL")
    container_name = os.getenv("CONTAINER_NAME")
    credential = DefaultAzureCredential()

    def __init__(self):
        account_url = self.account_url
        if not account_url:
            raise ValueError("BLOB_ACCOUNT_URL is missing")
  
        self.blob_service_client = BlobServiceClient(
            account_url=account_url,
            credential=self.credential,
        )

    def list_blobs_flat(self):
        client = self.blob_service_client.get_container_client(self.container_name)

        blob_list = client.list_blobs()
        for blob in blob_list:
            blob_data = client.get_blob_client(blob.name).download_blob()
            content = blob_data.readall().decode("utf-8")
            return content.splitlines()  # Return as a list of lines
       

