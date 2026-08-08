from huggingface_hub import HfApi

api = HfApi()

api.upload_folder(
    folder_path="runs/robomme/zoo_n1d6_pool_short_mem_no_bucket",
    repo_id="baongocdao/zoo_n1d6_pool_short_mem_no_bucket",
    repo_type="model",
)