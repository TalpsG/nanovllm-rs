# nanovllm-rs

Rust 版本的 Nano-vLLM 原型，目前专注于配置解析与后续推理管线的搭建。

## 配置读取

`Config::from_model_dir` 会读取 Hugging Face 模型目录下的 `config.json`，并根据其中的 `max_position_embeddings`、`eos_token_id` 等字段自动调整推理配置。默认值与 Python 版本保持一致，同时会校验如下约束：

- `kvcache_block_size` 需要是 256 的倍数。
- `tensor_parallel_size` 必须位于 `[1, 8]`。
- `max_num_batched_tokens` 不得小于 `max_model_len`。

### 示例

```rust
use nanovllm_rs::utils::config::Config;

fn main() -> anyhow::Result<()> {
    let config = Config::from_model_dir("/path/to/Qwen3-0.6B")?;
    println!("model dir: {:?}", config.model);
    println!("max model len: {}", config.max_model_len);
    println!("eos token: {}", config.eos);
    Ok(())
}
```

## 下一步

- 将 `Config` 接入 CLI / 推理引擎初始化流程。
- 结合 `sampling_params` 模块补全 `LLMEngine` 所需的运行时上下文。
- 为常用模型预先验证配置覆盖率，补充单元测试。
