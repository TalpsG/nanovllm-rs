import os
from nanovllm import LLM, SamplingParams
from transformers import AutoTokenizer
import nanovllm.utils.profile as profile


def main():
    path = os.path.expanduser("~/huggingface/Qwen3-0.6B/")
    tokenizer = AutoTokenizer.from_pretrained(path)
    llm = LLM(path, enforce_eager=True, tensor_parallel_size=1)

    # use cli args as max_tokens 
    max_tokens = os.environ["MAX_TOKENS"] if "MAX_TOKENS" in os.environ else 16

    sampling_params = SamplingParams(temperature=0.6, max_tokens=int(max_tokens))
    prompts = [
        "introduce yourself",
        "list all prime numbers within 100",
    ]
    prompts = [
        tokenizer.apply_chat_template(
            [{"role": "user", "content": prompt}],
            tokenize=False,
            add_generation_prompt=True,
        )
        for prompt in prompts
    ]
    outputs = llm.generate(prompts, sampling_params)

    for prompt, output in zip(prompts, outputs):
        print("\n")
        print(f"Prompt: {prompt!r}")
        print(f"Completion: {output['text']!r}")

    # avg min max decode time
    print(f"Total Decode Time: {sum(profile._DECODE_COST)}")
    print(f"Average Decode Time: {sum(profile._DECODE_COST)/len(profile._DECODE_COST)}")
    print(f"Decode Count: {len(profile._DECODE_COST)}")
    print(f"Min Decode Time: {min(profile._DECODE_COST)}")
    print(f"Max Decode Time: {max(profile._DECODE_COST)}")
    # prefill time
    print(f"Total Prefill Time: {sum(profile._PREFILL_COST)}")
    print(f"Average Prefill Time: {sum(profile._PREFILL_COST)/len(profile._PREFILL_COST)}")
    print(f"Prefill Count: {len(profile._PREFILL_COST)}")
    # decode attention time
    print(f"Total Decode Attention Time: {sum(profile._DECODE_ATTN_COST)}")
    print(f"Average Decode Attention Time: {sum(profile._DECODE_ATTN_COST)/len(profile._DECODE_ATTN_COST)}")
    print(f"Decode Attention Count: {len(profile._DECODE_ATTN_COST)}")
    print(f"Min Decode Attention Time: {min(profile._DECODE_ATTN_COST)}")
    print(f"Max Decode Attention Time: {max(profile._DECODE_ATTN_COST)}")
    # store kvcache time
    print(f"Total Store KVCACHE Time: {sum(profile._STORE_KVCACHE_COST)}")
    print(f"Average Store KVCACHE Time: {sum(profile._STORE_KVCACHE_COST)/len(profile._STORE_KVCACHE_COST)}")
    print(f"Store KVCACHE Count: {len(profile._STORE_KVCACHE_COST)}")
    print(f"Min Store KVCACHE Time: {min(profile._STORE_KVCACHE_COST)}")
    print(f"Max Store KVCACHE Time: {max(profile._STORE_KVCACHE_COST)}")


if __name__ == "__main__":
    main()
