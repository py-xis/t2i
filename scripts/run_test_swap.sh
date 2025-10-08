PYTHONPATH=$PWD accelerate launch --num_processes 4 src/explore_sdxl_layers.py --experiment blocks
wait

PYTHONPATH=$PWD accelerate launch --num_processes 4 src/explore_sdxl_layers.py --experiment transformers
wait

PYTHONPATH=$PWD accelerate launch --num_processes 4 src/explore_sdxl_layers.py --experiment layers
wait

PYTHONPATH=$PWD accelerate launch --num_processes 4 src/explore_sdxl_layers.py --experiment 2layers
wait

PYTHONPATH=$PWD accelerate launch --num_processes 4 src/explore_sdxl_layers.py --experiment causal