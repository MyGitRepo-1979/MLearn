def read_data_azure_blobstorage():
    import pandas as pd
    from azure.storage.blob import BlobServiceClient
    import json
    from io import StringIO
    import os

    # Azure Blob Storage connection details

    account_url = os.getenv("AZURE_ACCOUNT_URL")
    container_name = "pod-usage"
    blob_name = "aks_pod_application_usage.json"
    sas_token = os.getenv("AZURE_SAS_TOKEN") 

    # Create BlobServiceClient with SAS token
    blob_service_client = BlobServiceClient(account_url=account_url, credential=sas_token)
    container_client = blob_service_client.get_container_client(container_name)
    blob_client = container_client.get_blob_client(blob_name)


    # Download blob content as text
    blob_data = blob_client.download_blob().readall()
    json_str = blob_data.decode('utf-8')

    # Load JSON to DataFrame
    # If the JSON is a list of records:
    df = pd.read_json(StringIO(json_str))

    # If the JSON is a dict with a key containing the records, e.g., {"data": [...]}
    # data_dict = json.loads(json_str)
    # df = pd.DataFrame(data_dict["data"])

    print(df.head())

if __name__ == "__main__":
    read_data_azure_blobstorage()