from huggingface_hub import HfApi

api = HfApi()

api.upload_folder(
    folder_path="runs/robomme/zoo_n1d6_pool_short_and_long_mem_no_delta_gate/checkpoint-60000",
    repo_id="baongocdao/zoo_n1d6_short_and_long_mem_no_delta_gate",
    repo_type="model",
)