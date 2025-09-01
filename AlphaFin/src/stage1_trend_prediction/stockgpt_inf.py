import argparse
import logging
import torch
import sys
import os
import json
import re
from tqdm import tqdm

from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    BitsAndBytesConfig          ### === 新增 ===
)

sys.path.append(
    os.path.abspath(os.path.join(os.path.dirname(__file__), os.path.pardir)))
from peft import PeftModel

logger = logging.getLogger(__name__)


def parse_args():
    parser = argparse.ArgumentParser(description="Eval the finetued SFT model")
    parser.add_argument(
        "--model_name_or_path",
        type=str,
        help="Path to baseline model",
        required=True,
    )
    parser.add_argument(
        "--lora_name_or_path",
        type=str,
        default="",
        help="Path to pretrained model"
    )
    parser.add_argument(
        "--data_path",
        type=str,
        help="Path to evaluation dataset",
        required=True,
    )
    parser.add_argument(
        "--output_path",
        type=str,
        help="Path to save output results",
        required=True,
    )
    parser.add_argument(                     ### === 新增 ===
        "--load_in_4bit",
        action="store_true",
        help="Load model in 4-bit to save VRAM"
    )

    args = parser.parse_args()
    return args


def ChatGLMPrompt(instruction, inputs):
    return f"[Round 1]\n\n问：{instruction}\n{inputs}\n\n答："


def prompt_eval(args, model, tokenizer, data):
    fw = open(args.output_path, 'a+')

    for prompt in data:
        print("========== Prompt =========")
        print(f"\n{prompt['instruction']}\n{prompt['input']}\n")

        print("========== Ground Truth =========")
        print(f"\n{prompt['output']}\n")

        print(f'========== StockGPT =========')
        input_text = ChatGLMPrompt(prompt['instruction'], prompt['input'])

        # ChatGLM 自带对话接口
        output, history = model.chat(tokenizer, input_text, history=[])

        print(f'\n{output}\n')

        print("==================== Instance End =============================\n\n")

        result = prompt
        result.update({"StockGPT": output})
        fw.write(json.dumps(result, ensure_ascii=False) + '\n')


def main():
    args = parse_args()

    # === 构造量化配置（如指定 --load_in_4bit） ==========================
    quant_cfg = None
    if args.load_in_4bit:
        quant_cfg = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_compute_dtype=torch.float16
        )
    # =====================================================================

    tokenizer = AutoTokenizer.from_pretrained(
        args.model_name_or_path,
        trust_remote_code=True
    )

    model = AutoModelForCausalLM.from_pretrained(
        args.model_name_or_path,
        trust_remote_code=True,
        device_map="auto",          ### === 关键：自动映射到 GPU ===
        torch_dtype=torch.float16,  ### 半精度
        quantization_config=quant_cfg
    )

    if args.lora_name_or_path and os.path.exists(args.lora_name_or_path):
        model = PeftModel.from_pretrained(model, args.lora_name_or_path)

    model.eval()                    ### === 与 .chat 接口配合 ===

    # ---------------------------------------------------------------------
    with open(args.data_path, 'r', encoding='utf-8') as f:
        prompts = json.load(f)

    print('Loading dataset')
    data = []
    for example in tqdm(prompts):
        line = example['input']
        name_ptn = r'这是以(\S+?)（([0-9]+?)）.*'
        date_ptn = r'.*在([0-9]{4}-[0-9]{2}-[0-9]{2} 00:00:00)日期发布的研究报告'
        name_tmp = re.findall(name_ptn, line)
        stock_name, stock_code = name_tmp[0]
        date = re.findall(date_ptn, line)[0]
        data.append({
            "instruction": example['instruction'],
            'input': example['input'],
            'output': example['output'],
            'stock_name': stock_name,
            'stock_code': stock_code,
            'date': date
        })

    prompt_eval(args, model, tokenizer, data)


if __name__ == "__main__":
    main()
