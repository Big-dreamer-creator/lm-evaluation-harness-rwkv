# 上传成绩到 scoreboard-rwkv

本仓库的上传脚本对接的是 [scoreboard-rwkv](https://github.com/rwkv-rs/scoreboard-rwkv)
当前的版本化发布 API。API 只接受完整的 `lighteval-campaign-v3` 和
`lighteval-task-v2` payload；lm-eval 的 `results_*.json` 汇总文件不是该协议，不能直接上传。
不要为了上传而伪造缺失的模型执行信息、逐样本输出或 token 证据。

## 准备 payload

准备一个 campaign JSON 和每个 `expected_tasks` 对应的 task JSON。task 文件中的
`campaign_id` 会由脚本在创建 campaign 后替换成服务端返回的 UUID；其余字段必须与
campaign 的 `expected_tasks` 完全一致，并满足看板仓库当前 DTO 校验。

campaign 必须使用 `schema_version: "lighteval-campaign-v3"`、
`lighteval_version: "0.13.0"`，每个 task 必须使用
`schema_version: "lighteval-task-v2"`。当前服务还要求同一权重同时提交 `fp16` 和
`fp32io16` 两个 WKV mode。

## 生产环境上传

publication token 不写入仓库、JSON 或 shell 历史；通过环境变量传入：

```bash
export SCOREBOARD_BASE_URL=https://eval.rwkv.rs
export SCOREBOARD_PUBLICATION_TOKEN='从看板部署方获取的 token'

uv run python scripts/upload_scoreboard.py \
  --campaign /path/to/campaign.json \
  --task /path/to/task-fp16.json \
  --task /path/to/task-fp32io16.json
```

默认流程是 preflight → 创建/恢复 campaign → 按 campaign 顺序上传 task → finalize。
请求带有 scoreboard-rwkv 规定的 gzip 和幂等键；中断后用相同文件重新执行即可跳过
服务端已经确认的 task。`--no-finalize` 可只上传并保留 incomplete 状态。

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
