from huggingface_hub import HfApi

api = HfApi()

api.upload_folder(
    folder_path="runs/robomme/zoo_n1d6_pool_hamlet_sequential/checkpoint-60000",
    repo_id="baongocdao/zoo_n1d6_hamlet_sequential",
    repo_type="model",
)