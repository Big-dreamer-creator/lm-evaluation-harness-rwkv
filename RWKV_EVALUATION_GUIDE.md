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
max_num_seqs = <served --max-num-seqs>
device = "cpu"

apply_chat_template = true
fewshot_as_multiturn = true
log_samples = true
seed = [0, 1234, 1234, 1234]

[rwkv_profile]
prompt_template = "assistant"
generation_prompt = "fake_think"
wkv_mode = "fp32io16"
```

加载器会把这些字段转换为 lm-eval 的 `model`、`tasks`、`model_args`、
`output_path` 和 `metadata`。`benchmarks` 数组只写普通 benchmark 名称，
数组顺序是配置顺序（运行顺序另见 10.4 节：基准按剩余 rollout 数从小到大
顺序执行，配置顺序只作并列时的稳定次序）；这些名称本身就是原生
lm-eval-harness 的 task/group/tag
名称，由 `TaskManager` 直接解析，不存在额外的适配层。`model_name`、
`rwkv_profile.generation_prompt`、`rwkv_profile.prompt_template` 同时生成执行参数
和结果元数据，避免重复字段分叉；`rwkv_profile.wkv_mode` 只记录服务端模式，
仍需与 vllm-rwkv 启动日志核对。
未知字段、重复 benchmark、非法 URL、非正数并发或上下文长度会在连接模型前报错。

`max_num_seqs` 只声明推理端自身的并发容量（与服务端启动参数 `--max-num-seqs`
同值），评估端不再单独手工填写请求并发数：请求并发预算直接取自该声明值，
且 benchmark 严格一次只跑一个，预算不会被成倍放大（与 lighteval-rwkv 的
`concurrent_requests = sum(max_num_seqs)` 一致）。显存不足时先降低服务端
`--max-num-seqs`，再把这里的 `max_num_seqs` 改成同一个值，不要通过缩短
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

采样参数（temperature、top_p、top_k、penalties）固定使用上述 RWKV 官方解码
参数，不可被 task YAML 覆盖。判分、过滤器、stop 条件和 `max_gen_toks` 仍归
task YAML 所有。具体边界如下：

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

## 10. 远程 HTTP 后端（`rwkv7-remote`）

除了自建 `vllm-rwkv` 的 `rwkv7-http` 后端外，仓库还提供 `rwkv7-remote`，用于
直接对接标准 OpenAI 格式的托管 RWKV 服务（例如 `https://api.rwkv.rs`）。两者
互不影响：`rwkv7-remote` 是独立新增的 backend，不改动 `rwkv7-http` 的行为。

`rwkv7-remote` 只使用 `/v1/completions`，并假定该端点原生支持标准 OpenAI 的
`echo`+`logprobs`（用于 loglikelihood/多选任务评分）。这一假设已通过对
`api.rwkv.rs` 的真实请求验证：`drop`（generate_until）和 `race`
（loglikelihood/多选，需配合下文的 `tokenizer_name`）均已实测跑通。

### 10.1 仅生成类任务（`generate_until`，如 DROP）

不设置 `tokenizer_name` 时，prompt 按任务模板渲染后的原始文本直接发送，不做
客户端聊天模板渲染，也不需要本地或 HF Hub 上的官方 tokenizer 文件：

```toml
schema_version = 1
backend = "rwkv7-remote"

model_name = "rwkv7-g1j-1.5b"
base_url = "https://api.rwkv.rs/v1/completions"
benchmarks = ["drop"]
output_dir = "results/rwkv7-g1j-1.5b-remote"

max_length = 16384
apply_chat_template = false
fewshot_as_multiturn = false

[rwkv_profile]
prompt_template = "assistant"
generation_prompt = "fake_think"
wkv_mode = "fp32io16"
```

要点：

- `apply_chat_template` 必须设为 `false`：未设置 `tokenizer_name` 时
  `rwkv7-remote` 没有接入任何 tokenizer/chat_template 来源，无法渲染聊天
  模板。
- `rwkv_profile` 三个字段仍为必填，但只写入结果 `metadata`
  （`wkv_mode`、`cot_mode`、`prompt_template`）作为审计记录，不会被发送给
  模型。
- 该配置下 `rwkv7-remote` 的 `model_args` 只包含 `model`、`base_url`、
  `max_length`，以及由 `max_num_seqs` 派生的 `num_concurrent`。
- `config.truncation` 会输出真实的完成数/截断数/截断率（原理见第 9 节），
  不再是恒为 `null` 的占位值。

### 10.2 loglikelihood/多选任务（如 RACE）：`tokenizer_name`

`race` 等 `multiple_choice`/loglikelihood 任务需要在本地算出 prompt/continuation
的 token 边界，因此必须设置 `tokenizer_name` 指向一个 `transformers.AutoTokenizer`
可加载的分词器（本地目录路径或 HF Hub repo id 均可）：

```toml
schema_version = 1
backend = "rwkv7-remote"

model_name = "rwkv7-g1j-1.5b"
base_url = "https://api.rwkv.rs/v1/completions"
tokenizer_name = "<与远端权重词表一致的官方 tokenizer 路径或 repo id>"
# 若该 tokenizer 仓库依赖自定义分词器代码（如社区转换的 RWKV World
# tokenizer），还需要显式打开：
# trust_remote_code = true
benchmarks = ["race"]
output_dir = "results/rwkv7-g1j-1.5b-remote-race"

max_length = 16384
apply_chat_template = false
fewshot_as_multiturn = false

[rwkv_profile]
prompt_template = "assistant"
generation_prompt = "fake_think"
wkv_mode = "fp32io16"
```

要点：

- **正确性前提，代码不做校验，必须人工核对**：`tokenizer_name` 必须和
  `api.rwkv.rs` 上该 `model_name` 实际使用的词表完全一致。设置后请求会以
  `tokenized_requests=True` 方式发送——本地用该 tokenizer 把 context/continuation
  编码成 token-id 数组，直接作为 `prompt` 发给服务端，服务端不会重新分词。
  如果词表不一致，不会报错，只会**静默产生错误的 loglikelihood 分数**。
  正式测评前应确认这份 tokenizer 与该权重在 `rwkv7-http`/`vllm-rwkv` 侧已经
  验证过的官方 tokenizer 资产是同一个。
- `trust_remote_code` 仅在设置了 `tokenizer_name` 时才允许出现，否则报错
  `trust_remote_code requires tokenizer_name to be set`；在 `rwkv7-http` 上
  设置 `tokenizer_name`（无论是否带 `trust_remote_code`）会报错
  `tokenizer_name is only supported for backend = rwkv7-remote`。
- 设置 `tokenizer_name` 后 `model_args` 会多出 `tokenizer`（以及可选的
  `trust_remote_code`）两项。
- 若运行环境访问 HF Hub 需要代理，`AutoTokenizer.from_pretrained` 会按
  `transformers`/`huggingface_hub` 的标准方式读取 `HTTP_PROXY`/`HTTPS_PROXY`
  环境变量，不需要仓库代码做任何额外处理；`tokenizer_name` 指向本地目录时
  同样适用，不依赖网络。

### 10.3 鉴权与未使用的端点

- 鉴权：运行前导出 `RWKV_API_KEY` 环境变量，例如：

  ```bash
  export RWKV_API_KEY="<your-api-key>"
  ```

  绝不要把密钥写进 TOML——`RWKV_API_KEY` 只在进程环境中读取，不会进入
  `model_args` 或结果文件的 metadata。未设置该变量时，`rwkv7-remote` 会在
  构造模型时直接报错，而不是发送空 Bearer token。
- 该服务同时暴露的 `/v1/chat/completions`（含 `chat_template_kwargs`）和
  `/v1/responses` 目前未被本仓库任何 backend 使用；`rwkv7-remote` 只走
  `/v1/completions`。

### 10.4 并发与断点续跑（crash-safe resume）

并发策略与 lighteval-rwkv 保持一致，评估端只有两个概念：

- **请求并发预算**：每个 endpoint 只声明自己的推理容量 `max_num_seqs`，
  评估端的请求并发数直接取该值（单一扁平预算，见
  `lm_eval/models/api_models.py` 的 `get_batched_requests`；调度层本身是
  完成即调度、没有 batch barrier）。不再需要评估端手工填写 `num_concurrent`，
  也没有多 replica 的动态容量调度。
- **benchmark 顺序**：严格一次只跑一个 benchmark，顺序由
  `lm_eval/api/benchmark_scheduler.py` 的 `order_benchmarks` 决定（移植
  lighteval-rwkv 的 `_order_benchmarks`）：无需生成的 benchmark 在前，其余按
  剩余 rollout 数从少到多，跑完一个才进入下一个。这样短 benchmark 会尽早
  计分并上传看板，同时请求预算不会被并行 benchmark 成倍放大。

针对长时间、多 benchmark 的 `rwkv7-remote` 跑批，仓库额外提供了 opt-in 的
断点续跑机制，用于从网络抖动、远端 5xx 重试耗尽或进程被杀（`kill -9`）等
崩溃中恢复，而不必从头重跑：

```bash
uv run --no-sync python -m lm_eval run \
  --config configs/eval/rwkv7_g1j_1.5b_remote_race_scoreboard_example.toml \
  --resume
```

或直接在 TOML 里持久开启：

```toml
[checkpoint]
enabled = true
# dir = "results/<name>/checkpoints"   # 默认 output_dir/checkpoints
# fresh = true                          # 显式丢弃旧 checkpoint、重新开始
```

行为说明：

- 每个请求项（含 `repeats`/自洽采样的每一次重复）成功返回后立即落盘到
  `checkpoint.dir` 下的 `run.sqlite3`；重新以相同命令启动时，已完成的
  benchmark 不会发起任何新请求，未跑完的 benchmark 只会重新调度尚未完成的
  那部分请求项。
- `run_id` 由本次运行的完整有效配置（模型、`model_args`、任务、
  `num_fewshot`、`gen_kwargs` 等）算出。若指向同一个 `checkpoint.dir` 但配置
  变了，续跑会直接报错拒绝执行，而不是静默复用旧结果——需要有意丢弃旧
  checkpoint 时显式设置 `checkpoint.fresh = true`。
- 调度类参数（`max_num_seqs` 派生出的 `num_concurrent`、`timeout`、
  `max_retries`）不参与任何一层身份判定：既不算进 `run_id`，也不算进每条请求
  项的 `request_fingerprint`（`lm_eval/api/checkpoint.py` 的
  `OPERATIONAL_MODEL_ARGS` 是唯一出处，`lm_eval/_cli/run.py` 与
  `compute_task_fingerprint` 共用它）。因此"调大并发/超时后 `--resume`"只会
  补齐未完成的项，不会让已完成项失效；改动 prompt、过滤器、metric、
  `gen_kwargs`、模型名或采样配置仍会照常触发重新调度。
- 迁移说明：指纹定义修复前写下的 `run.sqlite3`（例如 2026-10-10 之前的跑批）
  与现在的身份规则不兼容，直接 `--resume` 会把已完成项全部重新派发。确实要
  复用这些旧结果时，用显式信任开关
  `LMEVAL_TRUST_COMPLETED_CHECKPOINT=1`（会打印逐项恢复日志）。
- 若某个 benchmark 已经跑完但发布（scoreboard publication）当时失败或未及
  完成，续跑只会重试发布这一步，不会重新推理、不会重新计分。
- 发布粒度是整个 benchmark，不是其内部叶子 task：`wmdp` 这类多叶子
  benchmark（`wmdp_bio`/`wmdp_chem`/`wmdp_cyber`）只会在它自己的全部叶子都
  跑完后合并聚合成一条看板记录再发布一次，叶子本身永远不会单独作为子条目
  上传看板。没有正式 `aggregate_metric_list` 的 benchmark（如
  `model_written_evals`、`xquad` 这类 tag/group）按叶子间共同拥有的 metric
  名自动加权平均；叶子间 metric 名完全不重叠时（如 `infinitebench`）不产出
  合成主指标，只保留各叶子自己的分项。相应地，`configs/eval/*.toml` 里的
  `[publication.tasks.<name>]` 现在按 benchmark 名配置（例如
  `[publication.tasks.wmdp]`，不再是 `[publication.tasks.wmdp_bio]` 之类的
  叶子名）；单叶子 benchmark（如 `race`）benchmark 名与叶子名相同，配置写法
  不变。
- 崩溃语义是 at-least-once：若远端请求已经成功但进程在结果落盘前崩溃，该
  请求项会被判定为未完成并在续跑时重新发送，可能造成极少量重复调用，但绝不
  会静默丢失或损坏已完成的结果。
- `checkpoint`/`--resume` 与 `--use_cache`（`CachingLM`）不能同时使用——两者
  的缓存/去重语义不兼容，同时设置会在启动早期直接报错，需要二选一。
- 自动重启进程本身不在 `lm_eval run` 的职责范围内：`lm_eval run --resume`
  只负责"重新拉起后正确恢复评估状态"，进程崩溃后重新拉起进程是外部
  supervisor（例如 systemd、supervisord 或自写的重试脚本）的职责。

## 11. 常见错误

- `rwkv7-http requires tokenizer_backend=remote`：不要给评估端配置本地 tokenizer。
- `/tokenizer_info` 缺失：服务启动时补上
  `--enable-tokenizer-info-endpoint`。
- 模型 ID 不匹配：让 TOML 的 `model_name` 与 `--served-model-name` 完全一致。
- 请求超过上下文：让服务端 `--max-model-len` 与评估端 `max_length` 匹配，
  并检查 benchmark 的输入与输出 token 上限。
- 采样参数不生效：采样参数固定来自 `rwkv_profile`，检查
  `generation_prompt` 是否为期望的 `open_think`/`fake_think`。
- `API key not found. Please set the RWKV_API_KEY environment variable.`：
  使用 `rwkv7-remote` 前需导出 `RWKV_API_KEY`，见第 10 节。
- `Tokenizer is required for loglikelihood tasks to compute context lengths.`：
  `rwkv7-remote` 跑 `race` 等 loglikelihood/多选任务时必须设置
  `tokenizer_name`，见第 10.2 节。
- `tokenizer_name is only supported for backend = rwkv7-remote` /
  `trust_remote_code requires tokenizer_name to be set`：
  `tokenizer_name`/`trust_remote_code` 的字段依赖关系配置错误，见第 10.2 节。
- loglikelihood 分数明显不合理但没有报错：优先怀疑 `tokenizer_name` 与远端
  权重词表不一致（代码不会校验这一点），见第 10.2 节。
- `checkpoint (or --resume) and --use_cache cannot be combined`：
  `[checkpoint]`/`--resume` 与 `--use_cache` 二选一，见第 10.4 节。
- `checkpoint store at ... belongs to a different effective config`：
  `checkpoint.dir` 指向的 `run.sqlite3` 记录的是另一次运行的配置指纹；确认
  `checkpoint.dir` 没有被复用于不同 TOML，或有意丢弃时设置
  `checkpoint.fresh = true`，见第 10.4 节。
- 续跑日志里出现大量 `Checkpoint item ... changed content since the last run;
  redispatching`，进度条从头开始：本次续跑的有效配置与写入 checkpoint 的那次
  不一致，harness 正在重跑而不是复用已完成项。先确认是否真的改了 prompt/
  过滤器/模型/`gen_kwargs`；若只改了并发、超时或重试次数，说明用的是修复前的
  旧 checkpoint（身份规则已变），见 10.4 节的迁移说明。
- GPU 空载但评估未结束：检查 SQLite/Hugging Face 缓存是否位于 Windows 挂载盘，
  以及数据集是否在离线模式下反复查询网络元数据。

本仓库只新增 LightEval 和 EvalScope 尚未收录的 benchmark。新增或调整任务时，
应先核对这两个框架的当前可运行注册表，再按 lm-eval 的 task YAML、filter 和 metric
边界实现，不应把模型专用 prompt 或判分逻辑写入 HTTP 后端。
