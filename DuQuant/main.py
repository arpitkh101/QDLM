import os
import sys
import random
import numpy as np
# from models.LMClass import LMClass
import torch
import time
from datautils import get_loaders
from lm_eval import evaluator
from pprint import pprint
from parallel_utils import map_layers_to_multi_gpus, get_lowest_occupied_gpu, forward_hook_wrapper
import torch.nn as nn
from quantize.duquant import duquant
from tqdm import tqdm
import utils
from pathlib import Path
from categories import subcategories, categories
from models.modelling_fastdllm import Fast_dLLM_QwenForCausalLM, Fast_dLLM_QwenConfig


torch.backends.cudnn.benchmark = True

net_choices = [
    "llama-7b",
    "llama-13b",
    "llama-30b",
    "llama-65b",
    "Llama-2-7b",
    "Llama-2-13b",
    "Llama-2-70b",
    "Llama-3-8b",
    "Llama-3-70b",
    "Vicuna-1.5-7b",
    "Vicuna-1.5-13b",
    "mistral-7b",
    "llada",
    "fastdllm"
]

def move_to_device(lm, args, logger):
    class_name = lm.model.__class__.__name__.lower()
    if args.multigpu:
        if lm.model.__class__.__name__ == "LLaDAModelLM":
            map_layers_to_multi_gpus(lm.model.model.transformer.blocks)
            input_device = lm.model.model.transformer.blocks[0].device
            output_device = lm.model.model.transformer.blocks[-1].device
            assert input_device == output_device
            lm.model.model.transformer.wte.to(input_device)
            lm.model.model.transformer.ln_f.to(output_device)
            lm.model.model.transformer.ff_out.to(output_device)
            lm.model.model.transformer.wte.register_forward_pre_hook(forward_hook_wrapper(input_device), with_kwargs=True)
        elif 'fast_dllm' in class_name or 'fastdllm' in class_name:
            map_layers_to_multi_gpus(lm.model.model.layers)
            input_device = lm.model.model.layers[0].device
            output_device = lm.model.model.layers[-1].device
            assert input_device == output_device
            lm._device = input_device
            lm.model.model.embed_tokens.to(input_device)
            lm.model.model.norm.to(output_device)
            lm.model.model.rotary_emb.to(output_device)
        else:
            map_layers_to_multi_gpus(lm.model.model.layers)
            input_device = lm.model.model.layers[0].device
            output_device = lm.model.model.layers[-1].device
            assert input_device == output_device
            lm._device = input_device
            lm.model.model.embed_tokens.to(input_device)
            lm.model.model.norm.to(output_device)
            lm.model.lm_head.to(output_device)
    else:
        lm.model = lm.model.to(lm.device)
        # Debug: verify layers after moving
        class_name = lm.model.__class__.__name__.lower()
        if 'fast_dllm' in class_name or 'fastdllm' in class_name:
            print(f"DEBUG move_to_device: lm.device={lm.device}")
            print(f"DEBUG move_to_device: Layer 0 type = {type(lm.model.model.layers[0])}")
            print(f"DEBUG move_to_device: Layer 0 device = {next(lm.model.model.layers[0].parameters()).device}")


def test_output(lm, args):
    with torch.cuda.amp.autocast():
        prompt = "Recent advances in neural language modeling have highlighted the importance of scaling laws. In this work, we present a new theoretical framework that—"
        from transformers import AutoTokenizer
        tokenizer = AutoTokenizer.from_pretrained(args.model, trust_remote_code=True)

        # m = [{"role": "user", "content": prompt}, ]
        # prompt = tokenizer.apply_chat_template(m, add_generation_prompt=True, tokenize=False)

        input_ids = tokenizer(prompt)['input_ids']
        input_ids = torch.tensor(input_ids).cuda().unsqueeze(0)

        class_name = lm.model.__class__.__name__.lower()
        if 'llada' in class_name:
            out = lm.generate(input_ids, steps=128, gen_length=128, block_length=32, temperature=0., cfg_scale=0., remasking='low_confidence')
        elif 'dream' in class_name and 'base' in args.model_path.lower():
            # model_args = dict(
            #     diffusion_steps=args.diffusion_steps, max_new_tokens=args.max_new_tokens, mc_num=args.mc_num
            # )
            attn_mask = input_ids.ne(tokenizer.pad_token_id)
            out = lm.generate(input_ids, attention_mask=attn_mask, diffusion_steps=512, max_new_tokens=32)
        elif 'fast_dllm' in class_name or 'fastdllm' in class_name:
            out = lm.generate(input_ids, max_new_tokens=32)
        else:
            out = lm.generate(input_ids, max_new_tokens=32)

        
        print(tokenizer.batch_decode(out[:, input_ids.shape[1]:], skip_special_tokens=True)[0])

        
        # exit(0)

@torch.no_grad()
def evaluate(lm, args, logger):
    results = {}

    if args.eval_mtbench:
    # eval quantized model on MMLU
        from mtbench_generate import run_eval, reorg_answer_file
        from fastchat.utils import str_to_torch_dtype
        for num_few_shots in [0, 5]:
            save_dir = os.path.join(args.output_dir, "mmlu", f"{num_few_shots}-shot" )
        model_id = args.net + f"_w{args.wbits}a{args.abits}"
        print("model_id: ", model_id)

        if args.num_gpus_total // args.num_gpus_per_model > 1:
            import ray

            ray.init()

        question_file = f"data/{args.bench_name}/question.jsonl"
        if args.answer_file:
            answer_file = args.answer_file
        else:
            answer_file = f"data/{args.bench_name}/model_answer/{model_id}.jsonl"

        print(f"Output to {answer_file}")
        
        print(lm.model.generate(lm.tokenizer('Hello, ', return_tensors="pt").input_ids.to(lm._device),max_length=3))
        run_eval(lm.model, 
                 lm.tokenizer, 
                 model_id,
                question_file=question_file,
                question_begin=args.question_begin,
                question_end=args.question_end,
                answer_file=answer_file,
                max_new_token=args.max_new_token,
                num_choices=args.num_choices,
                num_gpus_per_model=args.num_gpus_per_model,
                num_gpus_total=args.num_gpus_total,
                max_gpu_memory=args.max_gpu_memory,
                dtype=str_to_torch_dtype(args.dtype),
                revision=args.revision,
                )

        reorg_answer_file(answer_file)
        
        assert 0

    if args.eval_ppl:
        # for dataset in ["wikitext2", "ptb", "c4","ptb-new",'c4-new']:
        for dataset in ["wikitext2", 'c4']:
            cache_testloader = f'{args.cache_dir}/testloader_{args.model_family}_{dataset}_all.cache'
            if os.path.exists(cache_testloader):
                testloader = torch.load(cache_testloader)
                logger.info(f"load calibration from {cache_testloader}")
            else:
                dataloader, testloader = get_loaders(
                    dataset,
                    seed=args.seed,
                    model=args.model,
                    seqlen=lm.seqlen,
                )
                torch.save(testloader, cache_testloader)
            if "c4" in dataset:
                testenc = testloader
            else:
                testenc = testloader.input_ids

            nsamples = testenc.numel() // lm.seqlen
            use_cache = lm.model.config.use_cache
            lm.model.config.use_cache = False
            lm.model.eval()
            nlls = []
            for i in tqdm(range(nsamples)):
                batch = testenc[:, (i * lm.seqlen) : ((i + 1) * lm.seqlen)].to(lm.device)
                outputs = lm.model.model(batch)
                hidden_states = outputs[0]
                logits = lm.model.lm_head(hidden_states)
                shift_logits = logits[:, :-1, :]
                shift_labels = testenc[:, (i * lm.seqlen) : ((i + 1) * lm.seqlen)][
                    :, 1:
                ].to(lm.model.lm_head.weight.device)
                loss_fct = nn.CrossEntropyLoss()
                loss = loss_fct(
                    shift_logits.view(-1, shift_logits.size(-1)),
                    shift_labels.view(-1),
                )
                neg_log_likelihood = loss.float() * lm.seqlen
                nlls.append(neg_log_likelihood)
                if i == args.limit:
                    break
            ppl = torch.exp(torch.stack(nlls).sum() / (nsamples * lm.seqlen))
            logger.info(f'{dataset} : {ppl.item()}')
            lm.model.config.use_cache = use_cache
            results[dataset] = ppl.item()


    if args.eval_mmlu:
    # eval quantized model on MMLU
        from mmlu_eval import run_mmlu_eval
        for num_few_shots in [0, 5]:
            save_dir = os.path.join(args.output_dir, "mmlu", f"{num_few_shots}-shot" )
            run_mmlu_eval(lm.model, lm.tokenizer, args.net, num_few_shots, args.mmlu_data_dir, save_dir)


    test_output(lm,args)

    if args.tasks != "":
        class_name = lm.model.__class__.__name__.lower()
        model_args = {}
        if 'llada' in class_name:
        # if lm.model.__class__.__name__ == "LLaDAModelLM":
            model_args = dict(
                steps=args.steps, gen_length=args.gen_length, block_length=args.block_length, temperature=0., cfg_scale=0., remasking='low_confidence', mc_num=args.mc_num
            )
        elif 'dream' in class_name and 'base' in args.model_path.lower():
            model_args = dict(
                diffusion_steps=args.diffusion_steps, max_new_tokens=args.max_new_tokens, mc_num=args.mc_num
            )
            
        args.tasks = args.tasks.split(',')
        # Skip lm_eval for Fast-dLLM (requires custom evaluation harness)
        if 'fast_dllm' in class_name or 'fastdllm' in class_name:
            logger.info("Skipping lm_eval for Fast-dLLM - quantization completed successfully")
            return results
        with torch.cuda.amp.autocast():
            t_results = evaluator.simple_evaluate(
                lm,
                tasks=args.tasks,
                num_fewshot=args.num_fewshot,
                limit=None if args.limit == -1 else args.limit,
                model_args=model_args,
                confirm_run_unsafe_code=True
            )
        results.update(t_results)
        # print(results.keys())
        print(args)
        print(results['results'])
        print(args.wbits, args.abits)
        logger.info(results['results'])
        # pprint(results)
        # for test of MMLU
        if 'hendrycksTest' in args.tasks:
            all_cors = []
            all_cors_norm = []
            subcat_cors = {subcat: [] for subcat_lists in subcategories.values() for subcat in subcat_lists}
            cat_cors = {cat: [] for cat in categories}
            cat_cors_norm = {cat: [] for cat in categories}
            for key in t_results['results'].keys():
                if not 'hendrycksTest' in key:
                    continue
                subject = key.split('-')[-1]
                cors = t_results['results'][key]['acc']
                cors_norm = t_results['results'][key]['acc_norm']
                subcats = subcategories[subject]
                for subcat in subcats:
                    subcat_cors[subcat].append(cors)
                    for key in categories.keys():
                        if subcat in categories[key]:
                            cat_cors[key].append(cors)
                            cat_cors_norm[key].append(cors_norm)
                    all_cors.append(cors)
                    all_cors_norm.append(cors_norm)
                    
            for cat in cat_cors:
                cat_acc = np.mean(cat_cors[cat])
                logger.info("Average accuracy {:.4f} - {}".format(cat_acc, cat))
            weighted_acc = np.mean(all_cors)
            logger.info("Average accuracy: {:.4f}".format(weighted_acc))               

    return results


def main():
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("--model", type=str, help="model name of model path")
    parser.add_argument("--cache_dir", default="./cache", type=str, help="cache dir of dataset, leading to faster debug")
    parser.add_argument("--output_dir", default="./duquant_log/", type=str, help="direction of logging file")
    parser.add_argument("--save_dir", default=None, type=str, help="direction for saving fake quantization model")
    parser.add_argument("--resume", type=str, default=None)
    parser.add_argument("--calib_dataset",type=str,default="wikitext2",
        choices=["wikitext2", "ptb", "c4", "mix","pile"],
        help="Where to extract calibration data from.",
    )
    parser.add_argument('--test_dataset', type=str, default='wikitext2', help='dataset for testing')
    parser.add_argument("--nsamples", type=int, default=128, help="Number of calibration data samples.")
    parser.add_argument("--batch_size", type=int, default=1, help="batch size.")
    parser.add_argument("--seed", type=int, default=2, help="Seed for sampling the calibration data.")
    # split by comma, e.g. "wikitext2,ptb,hellaswag"
    parser.add_argument("--tasks", default="", type=str, help="Tasks to evaluate on, split by comma. If empty, will evaluate on all tasks.")
    parser.add_argument("--eval_ppl", action="store_true")
    parser.add_argument("--num_fewshot", type=int, default=0)
    parser.add_argument("--wbits", type=int, default=4)
    parser.add_argument("--abits", type=int, default=16)
    parser.add_argument("--group_size", type=int, default=None)
    parser.add_argument("--alpha", type=float, default=0.5)
    parser.add_argument("--let_alpha", type=float, default=0.8)
    parser.add_argument("--act_group_size", type=int, default=None)
    parser.add_argument("--let_lr", type=float, default=5e-3)
    parser.add_argument("--smooth_lr", type=float, default=1e-4)
    parser.add_argument("--lwc_lr", type=float, default=1e-2)
    parser.add_argument("--wd", type=float, default=0)
    parser.add_argument("--epochs", type=int, default=10)
    parser.add_argument("--smooth_epochs", type=int, default=0)
    parser.add_argument("--smooth",default=False, action="store_true")
    parser.add_argument("--let",default=False, action="store_true",help="activate learnable equivalent transformation")
    parser.add_argument("--lwc",default=False, action="store_true",help="activate learnable weight clipping")
    parser.add_argument("--aug_loss", default=False, action="store_true", help="calculate additional loss with same input")
    parser.add_argument("--symmetric",default=False, action="store_true", help="symmetric quantization")
    parser.add_argument("--a_dynamic_method", type=str, default="per_token", choices=["per_token"])
    parser.add_argument("--w_dynamic_method", type=str, default="per_channel", choices=["per_channel"])
    parser.add_argument("--limit", type=int, default=-1)
    parser.add_argument("--multigpu", action="store_true", help="at eval, map model to multiple gpus")
    parser.add_argument("--deactive_amp", action="store_true", help="deactivate AMP when 8<=bits<16")
    parser.add_argument(
        "--attn_implementation",
        type=str, required=False, default="eager",
        choices=["eager", "sdpa", "flash_attention_2"],
        help="attention implementation that the model works with",
    )
    parser.add_argument("--net", type=str, default=None, choices=net_choices)
    parser.add_argument("--act-scales", type=str, default=None)
    parser.add_argument("--act-shifts", type=str, default=None)

    # DuQuant
    parser.add_argument("--max_rotation_step", type=int, default=256, help="max steps for rotation transformation")
    parser.add_argument("--permutation_times", type=int, default=1, help="times of permutation transformation")
    parser.add_argument("--lac", type=float, default=None, help="activation clipping ratio")
    parser.add_argument("--swc", type=float, default=None, help="weight clipping ratio, enable withou lwc")
    parser.add_argument("--block_size", type=int, default=128, help="block size for rotation matrices")

    # MMLU
    parser.add_argument("--mmlu_data_dir", default="./mmlu/data", type=str, help="direction of mmlu dataset")
    parser.add_argument("--eval_mmlu", action="store_true", help="evaluate on MMLU")
    
    # MTBench
    parser.add_argument("--eval_mtbench", action="store_true", help="evaluate on MTBench")
    parser.add_argument(
        "--bench-name",
        type=str,
        default="mt_bench",
        help="The name of the benchmark question set.",
    )
    parser.add_argument(
        "--question-begin",
        type=int,
        help="A debug option. The begin index of questions.",
    )
    parser.add_argument(
        "--question-end", type=int, help="A debug option. The end index of questions."
    )
    parser.add_argument("--answer-file", type=str, help="The output answer file.")
    parser.add_argument(
        "--max-new-token",
        type=int,
        default=1024,
        help="The maximum number of new generated tokens.",
    )
    parser.add_argument(
        "--num-choices",
        type=int,
        default=1,
        help="How many completion choices to generate.",
    )
    parser.add_argument(
        "--num-gpus-per-model",
        type=int,
        default=1,
        help="The number of GPUs per model.",
    )
    parser.add_argument(
        "--num-gpus-total", type=int, default=1, help="The total number of GPUs."
    )
    parser.add_argument(
        "--max-gpu-memory",
        type=str,
        help="Maxmum GPU memory used for model weights per GPU.",
    )
    parser.add_argument(
        "--dtype",
        type=str,
        choices=["float32", "float16", "bfloat16"],
        help="Override the default dtype. If not set, it will use float16 on GPU and float32 on CPU.",
        default=None,
    )
    parser.add_argument(
        "--revision",
        type=str,
        default="main",
        help="The model revision to load.",
    )
    parser.add_argument(
        "--quant_method",
        type=str,
        default=None,
        choices=["duquant", "hadamard", None],
        help="The quantization method to use.",
    )
    parser.add_argument(
        "--steps",
        type=int,
        default=1024,
        help="The number of steps to run the model for. Default is 128.",
    )
    parser.add_argument(
        "--mc_num",
        type=int,
        default=128,
        help="The number of Monte Carlo samples to use for evaluation. Default is 1.",
    )
    parser.add_argument(
        "--gen_length",
        type=int,
        default=1024,
        help="The number of tokens to generate. Default is 1024.",
    )
    parser.add_argument(
        "--block_length",
        type=int,
        default=1024,
        help="The number of tokens to generate in each block. Default is 1024.",
    )
    parser.add_argument(
        "--max_new_tokens",
        type=int,
        default=128,
        help="The number of tokens to generate in each block. Default is 1024.",
    )
    parser.add_argument(
        "--diffusion_steps",
        type=int,
        default=512,
        help="The number of tokens to generate in each block. Default is 1024.",
    )
    parser.add_argument(
        "--get_wa", action="store_true", help="get hadamard weight and activation for DuQuant"
    )
    args = parser.parse_args()
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed(args.seed)
        
    if args.epochs > 0:
        assert args.lwc or args.let
        
    if (args.wbits<16 and args.wbits>=8) or (args.abits<16 and args.abits>=8):
        args.deactive_amp = True

    # args.quant_method = "duquant"
    # args.quant_method = None

    # init logger
    args.output_dir = os.path.join(args.output_dir, f"{args.model.split('/')[-1]}_w{args.wbits}a{args.abits}")
    if args.output_dir:
        Path(args.output_dir).mkdir(parents=True, exist_ok=True)
    if args.cache_dir:
        Path(args.cache_dir).mkdir(parents=True, exist_ok=True)
    if args.save_dir:
        Path(args.save_dir).mkdir(parents=True, exist_ok=True)
    output_dir = Path(args.output_dir)
    logger = utils.create_logger(output_dir)
    logger.info(args)
    # import ipdb; ipdb.set_trace()
    # load model
    if args.net is None:
        args.net = args.model.split('/')[-1]
    # assert args.net in net_choices
    args.model_family = args.net.split('-')[0]
    # lm = LMClass(args)
    # lm = 
    from transformers import AutoConfig, AutoTokenizer, AutoModelForCausalLM
    # from lm_eval.models.huggingface import HFLM
    from lm_eval.api.registry import get_model

    # model_cls = get_model('llada_dist')
    class_name = args.model.lower()

    if 'llada' in class_name:
        model_cls = get_model('llada_dist')
        model_args = dict(
            steps=args.steps, gen_length=args.gen_length, block_length=args.block_length, temperature=0., cfg_scale=0., remasking='low_confidence', mc_num=args.mc_num, batch_size=args.batch_size
        )
        lm = model_cls(model_path=args.model, **model_args)
    elif 'dream' in class_name and 'base' in args.model.lower():
        model_cls = get_model('dream_base')
        model_args = dict(
            diffusion_steps=args.diffusion_steps, max_new_tokens=args.max_new_tokens, mc_num=args.mc_num, batch_size=args.batch_size
        )
        lm = model_cls(pretrained=args.model, **model_args)
    elif 'fast_dllm' in class_name or 'fastdllm' in class_name:
        config = Fast_dLLM_QwenConfig.from_pretrained(args.model)
        model = Fast_dLLM_QwenForCausalLM.from_pretrained(args.model, config=config, torch_dtype=torch.float16).cuda()
        # Create a wrapper to match lm interface expected by DuQuant
        class FastdLLMWrapper:
            def __init__(self, model, model_path):
                self.model = model  # This is Fast_dLLM_QwenForCausalLM
                self._device = next(model.parameters()).device
                self.seqlen = 2048
                self.tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)
                self.mask_id = 151665  # Fast-dLLM mask token ID
                self.bd_size = config.bd_size if hasattr(config, 'bd_size') else 32
            @property
            def device(self):
                return self._device
            def generate(self, input_ids, max_new_tokens=32, **kwargs):
                """Generate using diffusion-based block decoding (simplified version)"""
                block_size = self.bd_size
                threshold = 0.9
                
                # Pad input to block boundary
                prompt_len = input_ids.shape[1]
                pad_len = block_size - (prompt_len % block_size)
                if pad_len < block_size:
                    input_ids = torch.cat([input_ids, 
                        torch.full((1, pad_len), self.mask_id, device=self._device, dtype=torch.long)], dim=1)
                
                # Add one block of mask tokens for generation
                num_gen_blocks = (max_new_tokens + block_size - 1) // block_size
                print(f"DEBUG: Starting block 0/{num_gen_blocks}. max_new_tokens={max_new_tokens}")
                
                with torch.no_grad():
                    for block_idx in range(num_gen_blocks):
                        # Add a new block of masks
                        x_t = torch.cat([input_ids, 
                            torch.full((1, block_size), self.mask_id, device=self._device, dtype=torch.long)], dim=1)
                        
                        # Iteratively unmask tokens in this block
                        for step in range(block_size):
                            mask_idx = (x_t[:, -block_size:] == self.mask_id)
                            if mask_idx.sum() == 0:
                                break
                            
                            # Forward pass with FULL input to maintain context
                            outputs = self.model(input_ids=x_t, use_cache=False)
                            logits = outputs.logits
                            # Take logits for the last block only (for unmasking decision)
                            block_logits = logits[:, -block_size:, :]
                            block_logits = torch.cat([block_logits[:, :1, :], block_logits[:, :-1, :]], dim=1)
                            
                            # Sample from logits
                            probs = torch.softmax(block_logits, dim=-1)
                            next_tokens = probs.argmax(dim=-1)
                            
                            # Get confidence
                            max_probs = probs.max(dim=-1).values
                            max_probs = torch.where(mask_idx, max_probs, -torch.inf)
                            
                            # Unmask most confident position
                            best_pos = max_probs.argmax(dim=-1)
                            x_t[:, -block_size + best_pos.item()] = next_tokens[:, best_pos.item()]
                        
                        input_ids = x_t
                
                return input_ids
        lm = FastdLLMWrapper(model, args.model)
    else:
        raise NotImplementedError
        
    args.model_path = args.model

    lm.seqlen = 2048
    lm.model.eval()
    for param in lm.model.parameters():
        param.requires_grad = False

    args.weight_quant_params = {
        "n_bits": args.wbits,
        "per_channel_axes": [0],
        "symmetric": args.symmetric,
        "dynamic_method": args.w_dynamic_method,
        "group_size": args.group_size,
        "lwc":args.lwc,
        "swc":args.swc,
        "quant_method": args.quant_method,
        "block_size": args.block_size,
        "max_rotation_step": args.max_rotation_step,
        "permutation_times": args.permutation_times,
    }
    args.act_quant_params = {
        "n_bits":  args.abits,
        "per_channel_axes": [],
        "symmetric": False,
        "lac":args.lac,
        "act_group_size": args.act_group_size,
        "dynamic_method": args.a_dynamic_method,
        "quant_method": args.quant_method,
        "block_size": args.block_size,
        "max_rotation_step": args.max_rotation_step,
        "permutation_times": args.permutation_times,
    }
    args.q_quant_params = {
        "n_bits": args.abits,
        "per_channel_axes": [],
        "symmetric": False,
        "dynamic_method": args.a_dynamic_method,
        "quant_method": args.quant_method,
        "block_size": args.block_size,
        "max_rotation_step": args.max_rotation_step,
    }
    args.k_quant_params = {
        "n_bits": args.abits,
        "per_channel_axes": [],
        "symmetric": False,
        "dynamic_method": args.a_dynamic_method,
        "quant_method": args.quant_method,
        "block_size": args.block_size,
    }
    args.v_quant_params = {
        "n_bits": args.abits,
        "per_channel_axes": [],
        "symmetric": False,
        "dynamic_method": args.a_dynamic_method,
    }
    args.p_quant_params = {
        "n_bits": 16,
        "metric": "fix0to1",
    }
    gpu_id = 0

    
    FILE_PATH = os.path.abspath(__file__)
    BASE_DIR = os.path.dirname(FILE_PATH)
    if args.act_scales is None:
        args.act_scales = f'{BASE_DIR}/act_scales/{args.net}.pt'
    if args.act_shifts is None:
        args.act_shifts = f'{BASE_DIR}/act_shifts/{args.net}.pt'
    
    rot_path = f'{BASE_DIR}/Rot.pkl'
    if not os.path.exists(rot_path) and args.quant_method == "duquant":
        import get_rot
        get_rot.main()
    

    # quantization
    if args.wbits < 16 or args.abits <16:
        logger.info("=== start quantization ===")
        tick = time.time()     
        # load calibration dataset
        cache_dataloader = f'{args.cache_dir}/dataloader_{args.model_family}_{args.calib_dataset}_{args.nsamples}.cache'
        if os.path.exists(cache_dataloader):
            dataloader = torch.load(cache_dataloader)
            logger.info(f"load calibration from {cache_dataloader}")
        else:
            dataloader, _ = get_loaders(
                args.calib_dataset,
                nsamples=args.nsamples,
                seed=args.seed,
                model=args.model,
                seqlen=lm.seqlen,
            )
            torch.save(dataloader, cache_dataloader)    
        act_scales = None
        act_shifts = None
        if args.smooth:
            print(args.act_scales)
            print(args.act_shifts)
            act_scales = torch.load(args.act_scales)
            act_shifts = torch.load(args.act_shifts)
        duquant(
            lm,
            args,
            dataloader,
            act_scales,
            act_shifts,
            logger,
        )
        logger.info(time.time() - tick)
    
    move_to_device(lm, args,logger)
    evaluate(lm, args,logger)


if __name__ == "__main__":
    print(sys.argv)
    main()
