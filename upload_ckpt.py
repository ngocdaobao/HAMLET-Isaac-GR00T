from huggingface_hub import HfApi

api = HfApi()

api.upload_folder(
    folder_path="runs/robomme/robomme/zoo_n1d6_pool_2short_mem/checkpoint-60000",
    repo_id="baongocdao/zoo_n1d6_pool_short_and_long_mem",
    repo_type="model",
)