# nanovllm-rs 设计文档

## 1. 项目目标与范围

- **最终目标**：基于 Rust + Candle 重写 Python 版本的 `nanovllm` 推理引擎，提供轻量、高效、易于部署的 LLM 推理 Runtime。
- **首期特性**：支持单机单 GPU / MPS 运行 Qwen 系列（Decoder-only）模型，具备批处理调度、KV Cache 复用、基础采样策略。
- **暂不覆盖**：模型训练、跨机集群、CPU 高吞吐优化、完整 Hugging Face 生态兼容。后续可以增量演进。

## 2. 总体架构

```
nanovllm-rs/
├── src/
│   ├── main.rs                # CLI / 示例入口
│   ├── config.rs              # 配置解析与校验
│   ├── sampling.rs            # 采样参数与策略实现
│   ├── sequence.rs            # 序列状态机、Block 索引
│   ├── block.rs               # Block / BlockManager 管理 KV Cache
│   ├── scheduler/
│   │   └── mod.rs             # 预填充 / 解码调度逻辑
│   ├── engine/
│   │   ├── mod.rs             # 面向用户的 LLM 引擎 API
│   │   ├── runner.rs          # 调用 Candle 模型的执行器
│   │   └── worker.rs          # （可选）多进程 / 多 GPU 支持
│   ├── model/
│   │   ├── mod.rs             # 模型抽象接口与工厂
│   │   ├── qwen3.rs           # Qwen3 Candle 实现
│   │   └── layers/            # Rotary、Attention、MLP 等子层
│   ├── utils/
│   │   ├── loader.rs          # 权重加载与张量并行切分
│   │   ├── context.rs         # 运行时上下文（cuSeqlen、slot mapping）
│   │   └── metrics.rs         # 性能统计与日志
│   └── tests/
│       └── integration.rs     # 集成测试 / golden case
├── scripts/
│   └── convert_weights.py     # PyTorch -> Candle safetensors 转换
└── DESIGN.md
```

## 3. 核心模块职责

### 3.1 配置（`config.rs`）
- 对应 Python `Config`，字段包括 `model_path`、`max_num_batched_tokens`、`max_num_seqs`、`max_model_len`、`tensor_parallel_size` 等。
- 使用 `serde` + `toml`/`clap` 进行配置文件与命令行解析。
- 负责路径校验、数值上下限校验、根据模型最大位置编码调整 `max_model_len`。

### 3.2 采样（`sampling.rs`）
- 定义 `SamplingParams`（温度、top-k、top-p、max_tokens、ignore_eos 等）。
- 基于 Candle Tensor + `rand` 实现温度缩放、top-k/top-p 采样。
- 提供可复现模式（可配置随机种子）。

### 3.3 序列管理（`sequence.rs`）
- 还原 Python `Sequence` 结构：记录 `seq_id`、状态、prompt / completion token、Block 表、缓存命中情况。
- 提供 `block(i)`、`append_token` 等方法，方便调度器和执行器使用。
- 如需要跨线程传输，可派生 `serde::{Serialize, Deserialize}`。

### 3.4 KV Cache 块管理（`block.rs`）
- `Block` 保存 `ref_count`、滚动 hash、缓存 token。
- `BlockManager` 用 `VecDeque` 管理可用块，结合 `xxhash-rust` 计算滚动 hash，实现 Prefix Cache 命中。
- 提供 `allocate` / `deallocate` / `can_append` / `may_append` 与 Python 语义对齐。
- 多线程安全需求可用 `parking_lot::Mutex` 或 `RwLock` 包装。

### 3.5 调度器（`scheduler::mod.rs`）
- 维护 `waiting` / `running` 双队列，优先执行预填充批次。
- 调用 `BlockManager` 决定是否可调度，必要时执行抢占（preempt）。
- 输出 `Vec<Arc<Sequence>>` 与当前阶段枚举 `SchedulePhase::{Prefill, Decode}`。

### 3.6 引擎 API（`engine::mod.rs`）
- `LLMEngine::new(config)` 初始化 tokenizer、调度器、执行线程（根据 `tensor_parallel_size`）。
- `add_request`：接受字符串或已 token 化输入，创建 `Sequence` 并压入调度器。
- `step`：调度 -> 执行 -> 采样 -> 回写结果。
- `generate`：封装多轮 `step`，支持 `tqdm` 风格的进度展示或指标回调。
- Tokenizer 建议使用 `tokenizers` crate（加载 Hugging Face `tokenizer.json`）。

### 3.7 模型执行器（`engine::runner.rs`）
- 持有 Candle 模型与设备、KV Cache 张量。
- 实现 `prepare_prefill` / `prepare_decode`：构造输入 token、position、更改 slot mapping。
- 调用 Candle 模型得到 logits；rank0 进行采样。
- 预热（warmup）并根据显存使用情况计算 `num_kvcache_blocks`。
- 可选开启 CUDA Graph 捕获以降低 decode 时的 kernel launch overhead。

### 3.8 多进程 / 张量并行（`engine::worker.rs`）
- 首次版本支持单 GPU；通过特性开关 `tensor-parallel` 引入多进程。
- Candle `candle-nccl` 提供 NCCL 接口，可仿照 Python 版本的共享内存 / 事件同步实现。
- 需要时可使用 `ipc-channel` 或 `shmem-ipc` 传递指令。

### 3.9 模型实现（`model::qwen3.rs`）
- 基于 Candle `candle-transformers` 或自定义层实现 Qwen3。
- 包含嵌入层、旋转位置编码（RoPE）、多查询注意力（MQA/GQA）、SwiGLU MLP。
- 暴露统一 Trait，如 `trait CandleCausalLM { fn forward(&mut self, input_ids: &Tensor, positions: &Tensor, ctx: &RunContext) -> Result<Tensor>; fn logits(&self, hidden_states: &Tensor) -> Result<Tensor>; }`。

### 3.10 工具模块（`utils/`）
- `loader.rs`：读取 safetensors，必要时按头维度切分做张量并行。
- `context.rs`：维护 decode 阶段需要的上下文（cuSeqlen、slot mapping、block tables），结合 Candle pinned memory。
- `metrics.rs`：封装 `tracing`/`indicatif`，记录吞吐、延迟等指标。

## 4. Candle 集成策略

1. **依赖建议**
   - `candle-core`, `candle-nn`, `candle-transformers`
   - `candle-cuda`（可选特性 `cuda`）和 `candle-nccl`（特性 `tensor-parallel`）
   - `tokenizers`, `anyhow`, `serde`, `serde_json`, `toml`, `clap`, `xxhash-rust`, `parking_lot`, `tracing`, `indicatif`

2. **设备选择**
   - 默认尝试 `Device::new_cuda(rank)`；若无 CUDA 则回退到 `Device::new_metal(0)`（macOS）或 `Device::Cpu`。
   - 提供 CLI 参数 `--device cuda:0` / `--device cpu`。

3. **内存与 KV Cache 规划**
   - 预热后读取显存使用，按剩余显存估算 `num_kvcache_blocks`：
     - `block_bytes = 2 * num_layers * block_size * num_kv_heads * head_dim * dtype_size`
     - `num_blocks = ((total * util_ratio) - used - peak + current) / block_bytes`
   - 使用 Candle `Tensor::zeros((2, num_layers, num_blocks, block_size, num_kv_heads, head_dim), device)` 分配。

4. **CUDA Graph 捕获**
   - Candle 对 CUDA Graph 支持有限，应通过特性 `cuda-graphs` 守护。
   - 捕获 decode 常用 batch 大小（1,2,4,8,16..），存入缓存，运行时按 batch 选择最小可用图。

5. **Tokenizer 与 I/O**
   - 读取 `config.model/tokenizer.json` 生成 `Tokenizer`。
   - `LLMEngine::generate` 返回 `Vec<GeneratedOutput>`，包含文本与 token id，便于测试对比。

## 5. 执行流程

1. 用户调用 `LLMEngine::generate(prompts, sampling_params)`。
2. Tokenizer 将输入转为 token，封装成 `Sequence`，推入调度器等待队列。
3. 调度器分配序列到预填充或解码批次，并预留 / 复用 KV Cache Block。
4. `ModelRunner::run` 构造 Candle 输入张量、上下文信息，并执行模型推理。
5. rank0 使用 `Sampler` 得到下一 token；调度器写回序列状态。
6. 序列完成后输出解码文本，未完成序列继续进入下一轮调度。

## 6. 实施里程碑

| 里程碑 | 目标 | 说明 |
| ------ | ---- | ---- |
| **M1 脚手架** | 完成配置、日志、单序列推理闭环（无调度） | 先实现固定 prompt -> 单次解码，验证权重加载与 Candle 模型可用 |
| **M2 调度与 KV Cache** | `Sequence`、`BlockManager`、调度器、prefill 批处理 | 编写单元测试确保缓存复用和抢占逻辑正确 |
| **M3 解码与采样** | 完整 decode 循环、温度/top-k/top-p 采样、吞吐统计 | 与 Python 版本对齐小样本输出 |
| **M4 性能优化** | CUDA Graph、Pinned Memory、异步 Host->Device 拷贝 | 记录 QPS/latency 指标，找出瓶颈 |
| **M5 张量并行** | 多 GPU worker，NCCL 通信，权重切分 | 引入 `tensor-parallel` feature gate，提供回退策略 |
| **M6 服务化** | CLI / HTTP / gRPC 服务，流式输出 | 使用 `axum`/`tonic` 等框架，集成监控指标 |

## 7. 测试与工具链

- **单元测试**：`cargo test` 覆盖 BlockManager、调度器、采样器等核心逻辑。
- **集成测试**：构建 Python 对照脚本，比较固定 prompt 输出的 token/文本差异。
- **基准测试**：使用 `criterion` 或自定义 benchmark 统计 prefill / decode 吞吐。
- **CI 流程**：`cargo fmt`, `cargo clippy --all-targets --all-features`, `cargo test`；按需增加 CUDA 环境 matrix。
- **调试工具**：`tracing` + `tracing-subscriber` 打日志，枚举 `--trace` 选项。

## 8. 风险与假设

- Candle 中 Qwen3 相关算子是否完备（RoPE、GQA），若缺失需自实现或等待 upstream。
- `candle-nccl` 与 CUDA Graph 在当前版本的稳定性；必要时提供特性开关。
- 权重转换依赖 safetensors；若原始权重只提供 PyTorch checkpoint，需提前编写转换脚本。

## 9. 下一步建议

1. 在 `Cargo.toml` 添加上述依赖及特性开关。
2. 先实现 `config.rs`、`sequence.rs`、`block.rs` 并补充单元测试，确保基础数据结构正确。
3. 编写简单 CLI（`cargo run -- --model ./qwen --prompt "你好"`）验证模型加载与单序列推理。
4. 与 Python 版本比对小样本输出，逐步扩展调度、并行与服务化能力。
