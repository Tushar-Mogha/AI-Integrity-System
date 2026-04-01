from huggingface_hub import HfApi
import os

api = HfApi()

token = os.environ.get("HF_TOKEN")

api.create_repo(
    repo_id="Tushar101/module1-roberta",
    exist_ok=True,
    token=token
)

api.upload_folder(
    folder_path="models/module1_roberta",
    repo_id="Tushar101/module1-roberta",
    token=token
)

print("Model uploaded successfully!")