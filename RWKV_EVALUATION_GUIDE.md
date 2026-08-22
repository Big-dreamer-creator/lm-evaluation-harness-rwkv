# RWKV7 Evaluation Guide

本仓库在 lm-evaluation-harness 中提供 `rwkv7-http` 后端。评估进程只通过
HTTP 访问独立的 `vllm-rwkv` 服务，不导入推理引擎、CUDA 扩展或模型权重。
因此同一套评估代码可用于 RWKV7 的 0.1B、0.4B、1.5B、2.9B、7.2B 和
13.3B 权重，差异只体现在服务端权重、上下文长度和 served model 名称。

## 1. 环境边界

评估端和推理端必须使用各自仓库内的 uv 环境：

```text
lm-evaluation-harness-rwkv/.venv   # 任务、指标、结果与 HTTP 客户端
vllm-rwkv/.venv-rwkv              # CUDA、RWKV7 权重与 HTTP 服务
```

不要让两个仓库共用虚拟环境，也不要在评估仓库中安装普通 vLLM 来替代
`vllm-rwkv`。

## 2. 安装评估端

```bash
git clone https://github.com/Big-dreamer-creator/lm-evaluation-harness-rwkv.git
cd lm-evaluation-harness-rwkv
uv venv --python 3.12 .venv
uv sync --extra api --extra ruler
```

按实际 benchmark 安装额外依赖。例如需要全部任务依赖时使用：

```bash
uv sync --extra api --extra tasks
```

所有评估命令均从本仓库根目录执行，并通过 `uv run --no-sync` 使用当前
`.venv`。

当前总配置公开 15 个 benchmark family：RACE、DROP、XQuAD、BABILong、
InfiniteBench、RULER、WMDP、CRUXEval、Inverse Scaling Prize、
Model-Written Evals、HumanEval-Infilling、MuTual、MC-TACO、Discrim-Eval 和
Winogender。PALOMA 因数据权限仍阻塞，不计入已支持清单。

## 3. 安装 vllm-rwkv

在另一个目录克隆并构建专用推理后端：

```bash
git clone https://github.com/rwkv-rs/vllm-rwkv.git
cd vllm-rwkv
uv venv --python 3.12 .venv-rwkv
uv pip install --python .venv-rwkv/bin/python -r requirements/rwkv.txt
VLLM_BUILD_PROFILE=rwkv \
VLLM_TARGET_DEVICE=cuda \
uv pip install --python .venv-rwkv/bin/python \
  --no-deps --no-build-isolation --editable .
```

以上命令对应 `vllm-rwkv` 仓库的
`docs/getting_started/installation/rwkv.md`。构建要求 NVIDIA CUDA 环境；
具体 CUDA、PyTorch 和编译器版本以所使用的 `vllm-rwkv` revision 为准。

## 4. 准备权重与模板

RWKV7 Release 权重来自：

- <https://huggingface.co/BlinkDL/rwkv7-g1/tree/main>
- <https://huggingface.co/rwkv-rs/rwkv7-g1-st>

原始权重名必须保留完整版本信息：

```text
{arch_version}-{data_version}-{param_size}-{release_date}-ctx{ctx_len}.pth
```

例如：

```text
rwkv7-g1h-7.2b-20260710-ctx10240.pth
```

`vllm-rwkv` 可从标准文件名识别 0.1B、0.4B、1.5B、2.9B、7.2B 和
13.3B 的结构。模型目录还应提供官方 `chat_template.jinja`。不要为了某个
benchmark 修改该模板；模板选择通过配置中的
`rwkv_profile.prompt_template` 完成。

## 5. 启动推理服务

下面的占位符必须替换为同一权重的真实值：

- `<checkpoint>`：完整 `.pth` 路径。
- `<model-name>`：完整权重版本名，不含 `.pth`。
- `<chat-template>`：该权重对应的官方 `chat_template.jinja`。
- `<context-length>`：权重文件名中的上下文长度。

```bash
cd /path/to/vllm-rwkv
export VLLM_RWKV7_WKV_MODE=fp32io16
export VLLM_USE_V2_MODEL_RUNNER=1
export VLLM_USE_RAPID_SAMPLER=1

uv run --no-sync --python .venv-rwkv/bin/python vllm serve \
  <checkpoint> \
  --host 127.0.0.1 \
  --port 8000 \
  --served-model-name <model-name> \
  --chat-template <chat-template> \
  --max-model-len <context-length> \
  --max-num-seqs 4 \
  --max-num-batched-tokens <context-length> \
  --gpu-memory-utilization 0.92 \
  --enable-tokenizer-info-endpoint
```

`--enable-tokenizer-info-endpoint` 是必需项。评估端通过 `/tokenizer_info`
取得服务正在使用的官方模板，并通过 `/tokenize`、`/detokenize` 和
`/v1/completions` 完成全部通信。若服务暴露到非本机网络，应使用访问控制，
因为 tokenizer-info 接口会公开模板内容。

启动后进行协议检查：

```bash
curl -fsS http://127.0.0.1:8000/health
curl -fsS http://127.0.0.1:8000/v1/models
curl -fsS http://127.0.0.1:8000/tokenizer_info
```

`/v1/models` 返回的 ID 必须与后续配置中的 `model_name` 完全相同。

## 6. 创建评估配置

`configs/eval/lm_eval.toml` 是已经验证的 1.5B preset，采用与 Helicopter
LightEval 相同的版本化、严格字段配置方式。评估其他权重时，复制其结构并修改
以下配置；不要复用旧权重的 model 名、上下文长度或结果目录。

```toml
schema_version = 1
backend = "rwkv7-http"

model_name = "<model-name>"
base_url = "http://127.0.0.1:8000/v1/completions"
benchmarks = [
  "<first-benchmark-name>",
  "<second-benchmark-name>",
]
output_dir = "results/<model-name>"

max_length = <context-length>
batch_size = 1
num_concurrent = 5
device = "cpu"

apply_chat_template = true
fewshot_as_multiturn = true
log_samples = true
seed = [0, 1234, 1234, 1234]

[rwkv_profile]
prompt_template = "assistant"
generation_prompt = "fake_think"
sampling_mode = "profile"
wkv_mode = "fp32io16"
```

加载器会把这些字段转换为 lm-eval 的 `model`、`tasks`、`model_args`、
`output_path` 和 `metadata`。`benchmarks` 数组只写普通 benchmark 名称，
数组顺序就是运行顺序；当模型是 `rwkv7-http` 时，任务管理器会根据 RWKV
适配 task YAML 的声明自动解析到对应的内部 task。`model_name`、
`rwkv_profile.generation_prompt`、`rwkv_profile.prompt_template` 同时生成执行参数
和结果元数据，避免重复字段分叉；`rwkv_profile.wkv_mode` 只记录服务端模式，
仍需与 vllm-rwkv 启动日志核对。
未知字段、重复 benchmark、非法 URL、非正数并发或上下文长度会在连接模型前报错。

`num_concurrent` 应略大于服务的有效推理并发量，使服务允许少量排队但不空载。
显存不足时先降低服务端 `--max-num-seqs`，再同步降低评估端并发，不要通过缩短
权重版本名或混用结果目录来规避配置管理。

## 7. Prompt 与采样模式

`rwkv_profile.prompt_template` 仅接受官方模板中的三种模式：

- `assistant`
- `bot`
- `function_calling`

`rwkv_profile.generation_prompt` 支持：

- `open_think`：temperature 0.96、top_p 0.76、top_k 32、
  presence_penalty 1.0、frequency_penalty 0.1、penalty_decay 0.988。
- `fake_think`：temperature 1.0、top_p 0.28、top_k 32。

`rwkv_profile.sampling_mode = "profile"` 使用上述 RWKV 官方解码参数；
`rwkv_profile.sampling_mode = "task"` 保留 task YAML 的 temperature、top_p 等设置。
只有在 benchmark 协议明确要求自己的生成参数时才使用 `task`。判分、过滤器、
stop 条件和 `max_gen_toks` 仍归 task YAML 所有。具体边界如下：

- 总 TOML：模型版本、普通 benchmark 清单及顺序、RWKV profile 和运行资源参数。
- task YAML：prompt、答案抽取/过滤、metric、stop 和 `max_gen_toks`。
- Python runner：只加载配置、选择模型适配器并执行，不包含 benchmark 分支或策略。

## 8. 运行与恢复

先列出和校验任务：

```bash
uv run --no-sync python -m lm_eval ls tasks
uv run --no-sync python -m lm_eval validate --tasks <task-or-group>
```

先执行 smoke run，仅验证协议，不把分数视为正式结果：

```bash
uv run --no-sync python -m lm_eval run \
  --config configs/eval/lm_eval.toml \
  --limit 10
```

正式运行时去掉 `--limit`：

```bash
uv run --no-sync python -m lm_eval run \
  --config configs/eval/lm_eval.toml
```

如需使用响应缓存，在 TOML 中设置 `use_cache`。lm-eval 默认不复用随机采样结果；
长任务需要在中断后严格恢复同一批已采样输出时，可显式启用：

```bash
export LMEVAL_CACHE_SAMPLED_GENERATIONS=1
```

该开关会固定已缓存的随机输出，不适合需要多次独立采样的实验。缓存目录应位于
Linux 本地文件系统；WSL 下将 SQLite 缓存放在 `/mnt/c` 或 `/mnt/e` 可能导致
评估进程阻塞在挂载盘 I/O。

## 9. 结果与正确性检查

正式结果至少保留：

- 完整 `model_name`、benchmark、样本数和 metric。
- `cot_mode`、`prompt_template`、`wkv_mode` 和上下文长度。
- `log_samples = true` 产生的输入、原始输出、reference 和逐样本 metric。
- `config.truncation` 中的完成数、截断数和 `truncation_rate`。

截断率定义为 finish reason 为 `length` 的样本数除以有 finish reason 的总样本数。
发现异常低分时，先检查 bad cases、模板、stop 条件和截断率，再判断模型能力。

正式测评前还应确认：

1. 服务日志显示 `fp32io16`、正确 checkpoint 和完整 served model 名。
2. `/tokenizer_info` 返回非空的官方 chat template。
3. 请求实际到达 `/v1/completions`，GPU 没有因评估端并发不足而空载。
4. 同参数量 Qwen3.5 使用各自正确后端和同一 benchmark 协议作为独立基线。
5. 结果目录不与其他权重、smoke run 或旧协议结果混用。

## 10. 常见错误

- `rwkv7-http requires tokenizer_backend=remote`：不要给评估端配置本地 tokenizer。
- `/tokenizer_info` 缺失：服务启动时补上
  `--enable-tokenizer-info-endpoint`。
- 模型 ID 不匹配：让 TOML 的 `model_name` 与 `--served-model-name` 完全一致。
- 请求超过上下文：让服务端 `--max-model-len` 与评估端 `max_length` 匹配，
  并检查 benchmark 的输入与输出 token 上限。
- 采样参数不生效：检查 `rwkv_profile.sampling_mode` 是 `profile` 还是
  `task`。
- GPU 空载但评估未结束：检查 SQLite/Hugging Face 缓存是否位于 Windows 挂载盘，
  以及数据集是否在离线模式下反复查询网络元数据。

本仓库只新增 LightEval 和 EvalScope 尚未收录的 benchmark。新增或调整任务时，
应先核对这两个框架的当前可运行注册表，再按 lm-eval 的 task YAML、filter 和 metric
边界实现，不应把模型专用 prompt 或判分逻辑写入 HTTP 后端。
