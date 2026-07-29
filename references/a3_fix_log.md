# A3 LRU Expert Cache 修复记录

## 发现的问题（3个）

### 1. `--expert-cache` CLI 参数未传后端
- `common/arg.cpp:2260` 把值存到 `params.expert_cache_fraction`
- **从未转成环境变量**，后端 `expert_cache_maybe_init` 读的是 `GGML_CUDA_EXPERT_CACHE`
- 必须显式 `GGML_CUDA_EXPERT_CACHE=1`

### 2. 缓存 key 错误包含 expert_id
- slot 结构用 `(cpu_src, expert_id)` 作 key
- `mul_mat` 中调用缓存时 `expert_id` 硬编码为 `0`
- 128 个 expert 互相覆盖，只缓存了一个 slot
- **修复**：key 只留 `cpu_src`

### 3. DS-V2-Lite 走 mmvq 路径，缓存代码从未执行
- `ggml_cuda_mul_mat()` 中单 token decode 走 **mmvq** 核（量化×向量）
- 缓存代码放在 `cublas_impl`，**一次没跑到**
- **修复**：在 mmvq 检查前插入强制 cublas 路径（`GGML_CUDA_EXPERT_CACHE` + 量化类型时跳过 mmvq/mmq）

## 修复后实测

| 指标 | 无缓存 | 有缓存 | 变化 |
|------|--------|--------|------|
| VRAM | 1275 MiB | **4895 MiB** | ✅ +3.6 GiB |
| Gen | 8.6 t/s | 7.2 t/s | ↓ 略降 |

**结论**：缓存代码正确工作了（128 expert × 权重已分配）。但 mmvq 本身对单 token 已最优，强制 cublas 反而慢了一点。缓存预期在批量推理时更有价值。

## 补丁文件

- `ggml-cuda.cu` — `ggml_cuda_mul_mat()` 1855-1863 行，强制走 cublas
- `expert-cache.cu` — key 简化为仅 `cpu_src`
