
export HF_DATASETS_TRUST_REMOTE_CODE=true
export HF_ALLOW_CODE_EVAL=1

DIRPATH="$(cd -P -- "$(dirname -- "${BASH_SOURCE[0]}")/.." >/dev/null 2>&1 && pwd)"
MODEL_PATH='/data/arpit/hf_cache/hub/models--GSAI-ML--LLaDA-8B-Instruct/snapshots/08b83a6feb34df1a6011b80c3c00c7563e963b07' 
W_BIT=8
A_BIT=8

# model_path: the path to the pretrained model
# block_size: the block size for DuQuant rotation
# max_rotation_step: the greedy rotation step for DuQuant quantization
# --swc: the ratio of weight clipping
# --lac: the ratio of activation clipping
# we use asymmetric per-channel quantization for weights and per-tensor quantization for activations in DuQuant

python $DIRPATH/DuQuant/generate_act_scale_shift.py --model $MODEL_PATH

# general tasks
# --tasks hellaswag,piqa,winogrande,arc_easy,arc_challenge
python $DIRPATH/DuQuant/main.py \
    --block_size 128 \
    --max_rotation_step 256 \
    --epochs 0 \
    --wbits $W_BIT \
    --abits $A_BIT \
    --model $MODEL_PATH \
    --quant_method duquant \
    --alpha 0.5 \
    --smooth \
    --lac 0.9 \
    --swc 0.8 \
    --task piqa \
    --limit 10 \
    --net llada \
    --multigpu


# # MMLU
# python $DIRPATH/DuQuant/main.py \
#     --block_size 128 \
#     --max_rotation_step 256 \
#     --epochs 0 \
#     --wbits $W_BIT \
#     --abits $A_BIT \
#     --model $MODEL_PATH \
#     --quant_method duquant \
#     --alpha 0.5 \
#     --smooth \
#     --lac 0.9 \
#     --swc 0.8 \
#     --task mmlu --num_fewshot 5 --mc_num 1 \


# # GSM8K
# python $DIRPATH/DuQuant/main.py \
#     --block_size 128 \
#     --max_rotation_step 256 \
#     --epochs 0 \
#     --wbits $W_BIT \
#     --abits $A_BIT \
#     --model $MODEL_PATH \
#     --quant_method duquant \
#     --alpha 0.5 \
#     --smooth \
#     --lac 0.9 \
#     --swc 0.8 \
#     --task gsm8k --gen_length 256 --steps 256 --block_length 32 --num_fewshot 4 \


# # MATH
# python $DIRPATH/DuQuant/main.py \
#     --block_size 128 \
#     --max_rotation_step 256 \
#     --epochs 0 \
#     --wbits $W_BIT \
#     --abits $A_BIT \
#     --model $MODEL_PATH \
#     --quant_method duquant \
#     --alpha 0.5 \
#     --smooth \
#     --lac 0.9 \
#     --swc 0.8 \
#     --tasks minerva_math  --num_fewshot 0 --gen_length 256 --steps 256 --block_length 64 \


# # HumanEval
# python $DIRPATH/DuQuant/main.py \
#     --block_size 128 \
#     --max_rotation_step 256 \
#     --epochs 0 \
#     --wbits $W_BIT \
#     --abits $A_BIT \
#     --model $MODEL_PATH \
#     --quant_method duquant \
#     --alpha 0.5 \
#     --smooth \
#     --lac 0.9 \
#     --swc 0.8 \
#     --task humaneval --gen_length 512 --steps 512 --block_length 32 --num_fewshot 0 \


# # MBPP
# python $DIRPATH/DuQuant/main.py \
#     --block_size 128 \
#     --max_rotation_step 256 \
#     --epochs 0 \
#     --wbits $W_BIT \
#     --abits $A_BIT \
#     --model $MODEL_PATH \
#     --quant_method duquant \
#     --alpha 0.5 \
#     --smooth \
#     --lac 0.9 \
#     --swc 0.8 \
#     --task mbpp --gen_length 512 --steps 512 --block_length 32 --num_fewshot 3 \