CUDA_VISIBLE_DEVICES=1
GR00T_MEM_DEBUG=1 MODEL_PATH=runs/robomme/zoo_n1d6_pool/checkpoint-60000 ROBOMME_PYTHON=robomme_benchmark/.venv/bin/python bash run_scripts/robomme/eval_n1d6_robomme_2.sh
