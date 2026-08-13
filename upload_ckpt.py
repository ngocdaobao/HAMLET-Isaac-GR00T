from huggingface_hub import HfApi

api = HfApi()

api.upload_folder(
    folder_path="runs/robomme/zoo_n1d6_fix_dataloader_lr_5e_5/checkpoint-60000",
    repo_id="baongocdao/zoo_n1d6_fix_dataloader_lr_5e_5",
    repo_type="model",
)