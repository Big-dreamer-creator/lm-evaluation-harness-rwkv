# 上传成绩到 scoreboard-rwkv

## 统一 TOML 流程（推荐）

普通的版本化 TOML 可以直接开启看板发布；RACE、DROP 以及其他已经注册的
lm-eval task 都走同一条路径，不需要 `run_rwkv_five_benchmarks.py`：

```toml
[publication]
enabled = true
```

部署地址、token、超时、是否 finalize 和模型版本信息默认从环境变量读取：

```bash
export SCOREBOARD_BASE_URL='http://127.0.0.1:7860'
export SCOREBOARD_PUBLICATION_TOKEN='从看板部署方获取的 token'
export SCOREBOARD_UPLOAD_TIMEOUT='3600'
export SCOREBOARD_UPLOAD_FINALIZE='true'
export SCOREBOARD_MODEL_SHA256='0123456789abcdef0123456789abcdef0123456789abcdef0123456789abcdef'
export SCOREBOARD_MODEL_REVISION='可选的权重 revision'

uv run --no-sync python -m lm_eval run --config /path/to/eval.toml
```

若部署使用自定义 token 变量名，设置
`SCOREBOARD_PUBLICATION_TOKEN_ENV` 为该变量名即可；脚本不会把 token 本身写入
配置或结果。TOML 中显式填写的 `base_url`、`token_env`、`timeout`、`finalize`、
`model_sha256`、`model_revision` 优先于环境变量。手工执行上传脚本时，对应 CLI
参数再优先于环境变量。启用发布时必须提供真实的 64 位小写权重 SHA-256；系统
不会猜测或伪造模型身份。

评测器会先保存标准 `results_*.json`、逐任务 `samples_*.jsonl`，再在同一个
输出目录的 `<model>/publication/` 写入 `raw_results.json`、campaign/task
载荷和 `status.json`，最后调用看板的 preflight → campaign → task → finalize
流程。`status.json` 明确区分 `evaluation=complete` 与
`publication=complete|failed|disabled`；网络或契约错误不会删除本地证据，也
不会把“评测完成、发布未完成”显示成已上传。发布载荷使用 task 名称和 task
配置动态生成，因此不依赖某个固定 benchmark 清单。若后端没有返回原始响应
及 token evidence，系统同样只保留本地结果并将发布标记为失败。
部署端需要在 publication preflight 中声明并实现 `lm-eval-campaign-v1`；仍只
声明旧 LightEval contract 的服务会被明确记录为发布失败，而不会把结果冒充成
已上传。

## 旧版五 benchmark producer（兼容保留）

producer 不依赖看板服务器。它会检查本地 `vllm-rwkv`、按模式启动服务，默认
依次运行 `fp16` 和 `fp32io16`，并在已有 task 结果上自动断点恢复：

```bash
uv sync --locked
uv run python scripts/run_rwkv_five_benchmarks.py --wkv-mode both --verify --digest
```

只检查命令、服务探测和两组任务展开，不启动推理：

```bash
uv run python scripts/run_rwkv_five_benchmarks.py --dry-run --wkv-mode both
```

如果这次结果后续要发布到当前 scoreboard-rwkv，评测时必须显式使用看板兼容协议：

```bash
uv run python scripts/run_rwkv_five_benchmarks.py \
  --wkv-mode both --scoreboard-compatible --verify --digest
```

这个选项会把 `open_think`、看板规定的采样参数、`max_gen_toks=8192`、
逐样本原始响应和 token 证据记录到 producer provenance 中。默认的
`fake_think` 结果仍然可以作为本地评测结果保存，但转换器会拒绝把它伪装成
当前 scoreboard DTO。

看板当前 DTO 还要求真实的 `lighteval==0.13.0` 运行依赖。producer 会在启动
前检查这一点；本仓库环境没有该依赖时会明确停止，而不是把 `lm_eval` 版本
冒充成 LightEval 版本。要让这个模式真正上传，还需要在项目 `.venv` 中按看板
契约安装并锁定该版本，或先让 scoreboard-rwkv 提供 lm-eval 原生 DTO。

只做严格的 lockfile/backend/GPU/service 前置检查：

```bash
uv run python scripts/run_rwkv_five_benchmarks.py --preflight-only --wkv-mode both
```

完成后，稳定文件位于 `results/formal-rwkv-five-benchmarks-20260821/`：
`campaign_manifest.json`、`tasks/<mode>__<task>/{results,samples}.json*`、
`publication/campaign.json`、`publication/tasks/*.json` 和
`publication/upload_spool.jsonl`。本地校验可单独执行：

每条 `samples.jsonl` 记录都包含 prompt、input/output token IDs、原始 HTTP
response、reasoning、后处理答案、finish reason、截断标记、指标和 SHA-256
字段；loglikelihood 请求同样保留 tuple-compatible 的响应证据。

启动前 producer 会检查 FlashRWKV2 扩展的编译架构与当前 GPU。当前本机
`FlashRWKV2==0.1.0a6` 扩展含 SM90/SM120 kernels，而 RTX 4060 是 SM89，
所以会在服务启动前明确失败并写入
`results/formal-rwkv-five-benchmarks-20260821/preflight.json`，不会生成伪造分数。

```bash
uv run python scripts/run_rwkv_five_benchmarks.py --verify-only
```

这些 producer 文件使用 `rwkv-producer-campaign-v1` /
`rwkv-producer-task-v1`，并标记 `scoreboard_upload_ready: false`。上传入口现在
包含一个严格的 producer → `lighteval-*` 转换层：它只映射 producer 已经保存的
模型、运行、指标、逐样本响应和 token 证据；缺字段、fake-think、采样参数不匹配
或缺少双 WKV mode 时直接失败，不会补造数据。

本仓库的上传脚本对接的是 [scoreboard-rwkv](https://github.com/rwkv-rs/scoreboard-rwkv)
版本化发布 API。统一入口生成 `lm-eval-campaign-v1` /
`lm-eval-task-v1`；旧 producer 仍使用 `lighteval-campaign-v3` /
`lighteval-task-v2`。两种载荷都必须包含完整的逐样本详情，原始
`results_*.json` 汇总文件仍不能直接上传。

## 旧 producer 载荷的兼容说明
当前已部署的旧服务若 preflight 尚未声明 `lm-eval-campaign-v1`，统一 TOML
流程会保留本地证据并将 `status.json` 标为发布失败；不要为了绕过该检查而
伪造 LightEval 版本、模型执行信息、逐样本输出或 token 证据。

## 准备 payload

准备一个 campaign JSON 和每个 `expected_tasks` 对应的 task JSON。task 文件中的
`campaign_id` 会由脚本在创建 campaign 后替换成服务端返回的 UUID；其余字段必须与
campaign 的 `expected_tasks` 完全一致，并满足看板仓库当前 DTO 校验。

campaign 必须使用 `schema_version: "lighteval-campaign-v3"`、
`lighteval_version: "0.13.0"`，每个 task 必须使用
`schema_version: "lighteval-task-v2"`。当前服务还要求同一权重同时提交 `fp16` 和
`fp32io16` 两个 WKV mode。

对新生成的 producer artifacts，可以直接转换并上传，不必手工改 JSON：

```bash
export SCOREBOARD_BASE_URL=https://eval.rwkv.rs
export SCOREBOARD_PUBLICATION_TOKEN='从看板部署方获取的 token'

uv run python scripts/upload_scoreboard.py \
  --producer-campaign results/formal-rwkv-five-benchmarks-20260821/publication/campaign.json \
  --producer-task results/formal-rwkv-five-benchmarks-20260821/publication/tasks/fp16__moral_stories.json \
  --producer-task results/formal-rwkv-five-benchmarks-20260821/publication/tasks/fp16__haerae.json \
  --producer-task results/formal-rwkv-five-benchmarks-20260821/publication/tasks/fp16__jsonschema_bench.json \
  --producer-task results/formal-rwkv-five-benchmarks-20260821/publication/tasks/fp16__gsm8k_platinum.json \
  --producer-task results/formal-rwkv-five-benchmarks-20260821/publication/tasks/fp16__aexams.json \
  --producer-task results/formal-rwkv-five-benchmarks-20260821/publication/tasks/fp32io16__moral_stories.json \
  --producer-task results/formal-rwkv-five-benchmarks-20260821/publication/tasks/fp32io16__haerae.json \
  --producer-task results/formal-rwkv-five-benchmarks-20260821/publication/tasks/fp32io16__jsonschema_bench.json \
  --producer-task results/formal-rwkv-five-benchmarks-20260821/publication/tasks/fp32io16__gsm8k_platinum.json \
  --producer-task results/formal-rwkv-five-benchmarks-20260821/publication/tasks/fp32io16__aexams.json
```

先只生成转换后的 DTO，不联网：

```bash
uv run python scripts/upload_scoreboard.py \
  --producer-campaign <producer-campaign.json> \
  --producer-task <producer-task-1.json> \
  --producer-task <producer-task-2.json> \
  --converted-output-dir /tmp/scoreboard-publication
```

## 生产环境上传

publication token 不写入仓库、JSON 或 shell 历史；通过环境变量传入：

```bash
export SCOREBOARD_BASE_URL=https://eval.rwkv.rs
export SCOREBOARD_PUBLICATION_TOKEN='从看板部署方获取的 token'
export SCOREBOARD_UPLOAD_TIMEOUT=3600
export SCOREBOARD_UPLOAD_FINALIZE=true

uv run python scripts/upload_scoreboard.py \
  --campaign /path/to/campaign.json \
  --task /path/to/task-fp16.json \
  --task /path/to/task-fp32io16.json
```

默认流程是 preflight → 创建/恢复 campaign → 按 campaign 顺序上传 task → finalize。
请求带有 scoreboard-rwkv 规定的 gzip 和幂等键；中断后用相同文件重新执行即可跳过
服务端已经确认的 task。将 `SCOREBOARD_UPLOAD_FINALIZE=false` 或传入
`--no-finalize` 可只上传并保留 incomplete 状态；`--finalize` 可覆盖环境变量并明确
完成 campaign。

部署到 `/test` 前缀时，将 `SCOREBOARD_BASE_URL` 改为
`https://eval.rwkv.rs/test`。先只检查 token 和远端契约：

```bash
uv run python scripts/upload_scoreboard.py --preflight-only
```

上传前只做本地校验、不联网：

```bash
uv run python scripts/upload_scoreboard.py \
  --campaign /path/to/campaign.json \
  --task /path/to/task-fp16.json \
  --task /path/to/task-fp32io16.json \
  --dry-run
```
