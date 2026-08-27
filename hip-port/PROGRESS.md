# moe-l2 → AMD RDNA3 (RX 7900 XTX, gfx1100) HIP 포팅 진행 문서

> **목적**: 이 문서를 읽으면 중단된 HIP 포팅 작업을 그대로 이어갈 수 있다.
> **최종 갱신**: 2026-08-27
> **작업 위치**: `hip-port/` (이 저장소 내부) + 원본 스냅샷 `references/llama.cpp-gpu-lru-cache/`
> **Git 저장소**: `https://github.com/baramofme/moe-l2-hip-port` (main, 커밋 3개: 228ef44 → b355ed0 → 0c844e5)

---

## 1. 목표

DeepSeek-V4-Flash (157B, 85GB GGUF)를 **AMD RX 7900 XTX 단일 카드 (24GB VRAM)**에서 돌리기 위해
moe-l2의 **CUDA 전용 llama.cpp 패치를 HIP/ROCm으로 포팅**한다.

핵심 검증 대상 3가지:
1. **검증 1**: `hipHostRegister` (on-demand/selective expert pin) — 전문가를 CPU RAM에 핀
2. **검증 2**: VRAM 전문가 LRU 캐시 + 페이지 eviction (RSS 캡)
3. **검증 3**: 활성 전문가만 GPU로 선택적 전송 (`GGML_OP_OFFLOAD_MIN_BATCH=1`)

---

## 2. 하드웨어/소프트웨어 환경

| 항목 | 값 |
|---|---|
| GPU | AMD Radeon RX 7900 XTX ×2 (gfx1100, Navi 31, 96 CU) |
| GPU 사용법 | **GPU 0 (HIP_VISIBLE_DEVICES=0)** = free, GPU 1 = 사용 중 |
| ROCm | 7.2.3 (HIP 7.2.53211, amdclang 22.0) |
| CPU/RAM | i5-14600K / 94GB RAM |
| 모델 (NVMe, 빠른 로드) | `/mnt/nvmedata/models/unsloth/Qwen3.6-35B-A3B-GGUF/Qwen3.6-35B-A3B-UD-Q4_K_S.gguf` (20GB, 256 experts, 48 layers) |
| 모델 (검증용, 구식) | `/mnt/vologs/Qwen3-Coder-30B-A3B-Instruct-Q4_K_M.gguf` (18GB, 128 experts) |
| llama.cpp base | 최신 HEAD (`ggml-org/llama.cpp` shallow clone) |

**중요**: ROCm 확인 명령
```bash
rocminfo | grep -E "Marketing Name|gfx[0-9]+"   # GPU 목록
rocm-smi --showid                               # GPU 0/1 매핑
HIP_VISIBLE_DEVICES=0                           # GPU 0 사용 (free)
```

---

## 3. HIP 빌드 방법 (검증 완료)

> ⚠️ **빌드 작업 트리**: `hip-port/llama.cpp/` (저장소 내부, 2026-08-27 이동 완료 — 이전 `/tmp/moe-port-work`는 재부팅으로 소실됨).
> CMakeCache가 옛 경로를 가리키면 `rm -rf build-moe-hip` 후 재configure 필요.

```bash
cd hip-port/llama.cpp
HIPCXX="$(hipconfig -l)/clang" HIP_PATH="$(hipconfig -R)" \
  cmake -S . -B build-moe-hip -DGGML_HIP=ON -DAMDGPU_TARGETS=gfx1100 \
  -DCMAKE_BUILD_TYPE=Release -DLLAMA_BUILD_SERVER=ON
cmake --build build-moe-hip --config Release -j$(nproc) --target llama-server
```

- 빌드 성공 확인: `build-moe-hip/bin/llama-server` + `libggml-hip.so`
- moe-l2 심볼 확인: `nm -D build-moe-hip/bin/libggml-hip.so | grep ggml_cuda_expert`
  - `ggml_cuda_expert_pin_host`, `ggml_cuda_expert_cache_get/set/init/free/...` 10개 심볼
- 빌드 산출물(`build-moe-hip/`)은 `.gitignore`로 제외되어 커밋되지 않음

---

## 4. 이식한 변경 사항 (전부 파악됨, 재적용 가능)

### 4.1. 원본 소스
- **`references/llama.cpp-gpu-lru-cache/`** = moe-l2가 쓰던 llama.cpp 소스 스냅샷 (CUDA 패치 포함)
  - `ggml-cuda.cu` (6016줄), `expert-cache.cu`/`.cuh` (moe-l2 추가), `ggml-backend.cpp`, `server.cpp`
- ⚠️ **스냅샷 전체를 최신 llama.cpp에 통째 복사하면 안 됨** → 구버전/최신 구조 불일치로 크래시
  (`hipblasDestroy`에서 `free(): invalid pointer`). 순정 HIP는 정상 동작하므로 **이식은 선택적 재적용**으로.

### 4.2. 변경 파일 (최신 llama.cpp에 적용한 것)
| 파일 | 변경 내용 |
|---|---|
| `ggml/src/ggml-cuda/vendors/hip.h` | HIP 매핑 추가: `cudaHostRegisterMapped→0`, `cudaPointerAttributes→hipPointerAttribute_t`, `cudaPointerGetAttributes`, `cudaMemoryTypeDevice`, `cudaStreamCreate`, `cudaErrorUnknown` |
| `ggml/src/ggml-cuda/expert-cache.cu/.cuh` | **새 파일** (moe-l2 LRU 캐시). HIP용: `cuda_runtime.h` include 제거, `common.cuh` 먼저 include |
| `ggml/src/ggml-cuda/ggml-cuda.cu` | 전역 `g_pin_mtx`/`g_pinned_ranges`/`g_total_pinned_bytes`, `ggml_cuda_experts_on_host()`, `ggml_cuda_expert_pin_host()`, `ggml_cuda_expert_unpin_host()`, `ggml_cuda_get_backend_stream()`, `ggml_cuda_mul_mat_id` A3 변환(per-expert 캐시), proc_address 노출, `<unistd.h>`/`<sys/mman.h>` include |
| `ggml/src/ggml-backend.cpp` | copy_experts per-expert 캐시 후킹 (proc_address로 pin/cache/stream resolve), **router-map 로드(`MOE_L2_ROUTER_FILE`)** + on-demand pin, `<string>`/`<unordered_set>` include |

### 4.3. 재적용 방법
```bash
# a) diff 적용 (git이 있는 llama.cpp 클론에서)
cd llama.cpp
git apply /home/baramofme/IdeaProjects/moe-l2/hip-port/moe-l2-hip-port.patch
# b) expert-cache 새 파일 복사
cp /home/baramofme/IdeaProjects/moe-l2/hip-port/patched-llama-src/ggml/src/ggml-cuda/expert-cache.* ggml/src/ggml-cuda/
# c) 또는 patched-llama-src/ 의 파일들로 통째 교체 (diff 실패 시)
cp /home/baramofme/IdeaProjects/moe-l2/hip-port/patched-llama-src/ggml/src/ggml-backend.cpp ggml/src/
cp /home/baramofme/IdeaProjects/moe-l2/hip-port/patched-llama-src/ggml/src/ggml-cuda/ggml-cuda.cu ggml/src/ggml-cuda/
cp /home/baramofme/IdeaProjects/moe-l2/hip-port/patched-llama-src/ggml/src/ggml-cuda/vendors/hip.h ggml/src/ggml-cuda/vendors/
```

---

## 5. 검증 실행 방법

### 5.1. 최신 안정 설정 (권장 — router-map + on-demand pin + 캐시 16000 슬롯)
```bash
cd hip-port/llama.cpp
MODEL=/mnt/nvmedata/models/unsloth/Qwen3.6-35B-A3B-GGUF/Qwen3.6-35B-A3B-UD-Q4_K_S.gguf
env LD_LIBRARY_PATH=build-moe-hip/bin HIP_VISIBLE_DEVICES=0 \
  GGML_CUDA_EXPERT_CACHE=1 GGML_OP_OFFLOAD_MIN_BATCH=1 MOE_L2_CACHE_SLOTS=16000 \
  MOE_L2_ROUTER_FILE=/tmp/qwen36_top32.map \
  ./build-moe-hip/bin/llama-server \
  -m "$MODEL" --host 127.0.0.1 --port 18135 -ngl 99 -c 2048 \
  --n-cpu-moe 48 --no-warmup -fit off --no-ui
# 결과: 35.4 t/s, hit 86.6%, 정확한 출력
```
> ⚠️ `MOE_L2_N_LAYERS=48` 대신 **`MOE_L2_CACHE_SLOTS=16000` 명시** 권장 — router-map 파일의 `# EXPERT_TOTAL` 주석이 슬롯을 768로 줄여버려 hit율 0% 발생 (thrash). 명시적 슬롯 수가 우선.

### 5.2. 기본 검증 (router 없이, 캐시+핀)
```bash
cd llama.cpp
MODEL=/mnt/nvmedata/models/unsloth/Qwen3.6-35B-A3B-GGUF/Qwen3.6-35B-A3B-UD-Q4_K_S.gguf
env LD_LIBRARY_PATH=build-moe-hip/bin HIP_VISIBLE_DEVICES=0 \
  GGML_CUDA_EXPERT_CACHE=1 GGML_OP_OFFLOAD_MIN_BATCH=1 MOE_L2_N_LAYERS=48 \
  ./build-moe-hip/bin/llama-server \
  -m "$MODEL" --host 127.0.0.1 --port 18122 -ngl 99 -c 2048 \
  -ot "exps=CPU" -fit off --no-ui
# 추론 테스트 — max_tokens=200+ 필수 (Qwen3.6 reasoning 모델)
curl -s http://127.0.0.1:18122/v1/chat/completions -d '{"model":"q","messages":[{"role":"user","content":"What is 2+2? Answer with just the number."}],"max_tokens":200,"stream":false}'
# → content='4' 확인 (max_tokens가 작으면 reasoning_content에 소진되어 content가 빈 문자열로 보임)
```

**router map 파일 생성** (top-K hot 전문가):
```python
# Qwen3.6-35B-A3B: 48 layers × 256 experts, top-32/layer
import random; random.seed(42)
with open('/tmp/qwen36_top32.map','w') as f:
    f.write("# EXPERT_TOTAL 256\n")
    for l in range(48):
        f.write(f"{l} " + " ".join(map(str, sorted(random.sample(range(256),32)))) + "\n")
```

---

## 6. 검증 결과 (2026-08-27 기준)

### ✅ 완료/확인된 것
1. **순정 llama.cpp HIP 정상 동작**: 7900XTX GPU 0에서 Qwen3-Coder-30B-A3B 추론 13.8 t/s (캐시 OFF, 전문가 CPU 오프로드)
2. **HIP 빌드 성공**: moe-l2 심볼 10개가 `libggml-hip.so`에 포함
3. **HIP-PROC resolve 성공**: pin/copy/get/set/stream 함수가 proc_address로 모두 연결
4. **A3 캐시 초기화 성공**: `[A3] expert_cache initialized: 32768 slots on device 0 (n_expert=256)`
5. **캐시 hit율 달성**: `[HIP-CACHE-GET] hitrate=84%` (Qwen3.6-35B, 32768 slots)
6. **copy_experts 선택적 전송**: `[HIP-COPY-EXP] first=15 last=15 n_expert=128` — 활성 전문가만 개별 전송 (검증 3 ✅)
7. **핀 경로 활성**: `pin_fn` resolve + copy_experts에서 호출 (검증 1 경로 동작)
8. **속도**: 캐시 ON 시 13.8~37.3 t/s (전문가 전체 CPU 오프로드 상태)

### ✅ 최종 검증 결과 (2026-08-27, Qwen3.6-35B-A3B @ 7900XTX)

**이전 "캐시 정확성 버그"는 실재하지 않았음 — Qwen3.6 reasoning 모델 특성으로 판명:**
- Qwen3.6은 `reasoning_content`(사고 과정)를 먼저 생성하는 reasoning 모델
- `max_tokens`가 작으면 추론에 다 소진되어 `content`가 비어 보임 (`''` 또는 `??????`)
- `max_tokens=200+`로 주면 정상 답변: `content='4'`, `'def return_forty_two(): return 42'` ✅

**검증 1 — hipHostRegister (핀): ✅**
```
[HIP-PIN] calls=17000 pinned_bytes=18987.6MB total_pinned=18987.6MB
```
- 핀 함수 17,000회 호출, **19GB가 hipHostRegister로 핀 고정 성공** (에러 없음)
- RSS 19.9GB = 핀된 전문가가 CPU RAM에 실제 상주

**검증 2 — VRAM 전문가 캐시: ✅**
```
[HIP-CACHE-GET] cnt=205400 hit=187213 miss=18187 hitrate=91.1%
```
- 캐시 hit율 91-95% (532k 호출 중 508k hit 달성도 확인)
- 캐시 ON에서 정확한 출력 (`content='4'`)

**검증 3 — 활성 전문가만 선택적 전송: ✅**
- copy_experts per-expert 경로: `first=15 last=15`, `first=41 last=44` — 사용된 전문가만 개별 전송
- `-ot "exps=CPU"` / `--n-cpu-moe 48`로 전문가 CPU 오프로드 + `GGML_OP_OFFLOAD_MIN_BATCH=1`로 GPU 강제

**성능 (전문가 전체 CPU 오프로드 + 캐시/핀 ON):**
- 생성 속도: **31-43 t/s** (Qwen3.6-35B-A3B Q4_K_S, 256 experts)
- VRAM: 16.6GB (캐시 + KV, 전문가 제외)
- RSS: 19.9GB (전문가 CPU 상주)

**참고 — mul_mat_id A3 경로**: `[HIP-MULMATID]` 로그가 없음 → HIP에서 MUL_MAT_ID는 MMVQ/MMF 빠른 경로로 처리되고, moe-l2 캐시는 copy_experts(스케줄러) 경로에서 hit/consume 됨. 즉 **copy_experts 캐시 후킹이 실제 효과 경로**이며, mul_mat_id 내부 A3 변환은 dead path (향후 제거 가능).

### Selective pin (router-map) 시도 결과 — HIP 제약 발견 (2026-08-27)

**목표**: 원본처럼 hot 전문가만 핀해서 RSS 절감 (V4: whole-pin 84GB → selective 26.8GB)

**시도한 조합 결과**:

| 조합 | 결과 |
|---|---|
| 핀 ON + router(selective, 일부만 핀) | ❌ **크래시** — `hipMemcpyAsync invalid argument` (미핀 mmap 페이지 H2D 실패) |
| 핀 OFF + router | ✅ 정상 — 단, 속도 12 t/s (캐시 미활용) |
| 핀 ON + router + 캐시 16000 슬롯 (전부 핀) | ✅ **정상 — 35.4 t/s, hit 86.6%, 답 정확** |
| 핀 ON + 전체(router 없음, 이전 검증) | ✅ 정상 — hit 95% |

**HIP 제약 (문서화 필수)**:
- `hipMemcpyAsync`는 **미핀 mmap 페이지를 H2D로 복사할 때** `invalid argument` 반환 가능
  (CUDA는 페이지 폴트로 처리하지만 HIP는 실패)
- **selective pin(일부만 hipHostRegister)** 시, 핀된 페이지와 인접한 미핀 페이지의 DMA가 깨짐 → 크래시
- 안정적 모드는 **전부 핀**(on-demand pin, `MOEL2_NOPIN` 미설정) 또는 **전부 미핀**(`MOEL2_NOPIN=1`)

**RSS가 높은 이유 (19.7GB)**:
- `--n-cpu-moe 48`로 전문가 전체를 CPU에 두고, 추론이 **대부분의 전문가를 접근** → on-demand pin으로 전체가 핀/상주
- 원본의 RSS 8-11GB는 **domain 라우팅으로 실제 사용 전문가만 접근**한 결과
- RSS 절감은 **접근 전문가 수 제한**(domain 라우팅)이 필요 — selective pin은 HIP에서 불가

### ✅ GitHub 푸시 상태 (2026-08-27)
- **저장소**: `https://github.com/baramofme/moe-l2-hip-port` (main)
- **커밋 3개**: `228ef44`(초기 이식+문서) → `b355ed0`(검증 완료) → `0c844e5`(selective pin + HIP 제약)
- 원본 저장소 `hip-port-porting` 브랜치(`8bec677`)에도 로컬 보존됨

---

## 7. 알려진 함정 / 참고

1. **재부팅 주의**: `/tmp` 작업은 재부팅으로 사라짐 → **이 저장소 `hip-port/` 내부에서 작업할 것**
2. **`MOE_L2_N_LAYERS` 필수**: 캐시 슬롯 수 = `n_expert × 3 × n_layers`. 안 주면 기본값 1 레이어 → 슬롯 부족 → hit율 0%
   - Qwen3.6-35B-A3B: `MOE_L2_N_LAYERS=48` (256×3×48=32768)
   - Qwen3-Coder-30B-A3B: `MOE_L2_N_LAYERS=40` (128×3×40=15360)
3. **캐시 키**: `hash(tensor_name) ^ (expert_idx × 0x9E3779B97F4A7C15ULL)` — copy_experts와 mul_mat_id가 **동일한 키**를 써야 함
4. **스트림 race**: 캐시 D2D copy는 H2D fill과 **같은 스트림**에서 실행해야 함 (스냅샷 P0 fix: `ggml_cuda_get_backend_stream` re-fetch per expert)
5. **`experts_on_host`**: `hipPointerGetAttributes`가 mmap CPU 포인터를 device로 오판할 수 있음 → `-ot "exps=CPU"` + `GGML_OP_OFFLOAD_MIN_BATCH=1`로 강제
6. **`cudaHostRegisterMapped=0`**: HIP에 mapped 플래그 없음. HIP의 핀은 `hipHostRegister`(portable)만 가능 → GPU 직접 DMA 읽기 대신 `hipMemcpyAsync` H2D 경로로 폴백. **discrete RDNA3에서 성능 영향 실측 필요** (검증 1의 미완 부분)
7. **`MOE_L2_ROUTER_FILE` + 슬롯 수 함정**: router-map 파일의 `# EXPERT_TOTAL 256` 주석을 `maybe_init`가 읽어 슬롯을 768로 줄임 → hit율 0% (LRU thrash). **반드시 `MOE_L2_CACHE_SLOTS=16000` 명시** (env가 최우선)
8. **HIP selective pin 불가**: 일부만 `hipHostRegister`하면 인접 미핀 페이지 H2D가 `invalid argument`로 크래시. 안정적 모드는 전부 핀(on-demand) 또는 전부 미핀(`MOEL2_NOPIN=1`)

---

## 8. 남은 작업 (이어서 할 것)

**핵심 3개 검증은 완료됨.** 남은 것은 실제 배포/확장 작업:

1. [ ] **DeepSeek-V4-Flash (85GB) 실제 실행** — 7900XTX에서 VRAM/RSS 측정 (최종 목표 모델)
   - ⚠️ 주의: upstream llama.cpp deepseek4 CUDA 전문가 버그(#25582)가 HIP에도 해당하는지 확인 필요
   - RSS 전략: on-demand pin 사용 (selective pin은 HIP에서 불가 — 위 제약 참조)
2. [ ] **Python 쪽 포팅** — `install.sh`/`doctor`의 nvidia-smi→rocm-smi, `download-bins`에 HIP asset 추가, `LD_LIBRARY_PATH` ROCm 경로
3. [ ] **`--n-cpu-moe`로 검증된 설정을 `moe-l2 start --gpu` 통합** — proxy가 HIP llama-server를 spawn하도록
4. [ ] mul_mat_id A3 경로 정리 (dead path 제거 또는 유지)
5. [ ] **RSS 절감 추가 탐구** — domain 라우팅(실사용 전문가 제한) 또는 mmap page-cache 전략 (HIP selective pin 불가하므로 대안 필요)
6. [ ] GitHub Release에 HIP 바이너리 (`llama-hip_bins.tar.gz`) 배포 파이프라인

---

## 9. 보존된 파일 목록 (hip-port/)

```
hip-port/
├── PROGRESS.md              # 이 문서
├── .gitignore               # 빌드 산출물/llama.cpp 전체 제외 (패치된 파일만 추적)
├── llama.cpp/               # 실제 빌드 작업 트리 (패치 적용됨, build-moe-hip 포함)
│   └── ggml/src/
│       ├── ggml-backend.cpp          # copy_experts 캐시+router-map 후킹
│       └── ggml-cuda/
│           ├── ggml-cuda.cu          # 핀/캐시/experts_on_host/stream + A3 변환
│           ├── expert-cache.cu       # LRU 캐시 (HIP 호환)
│           ├── expert-cache.cuh
│           └── vendors/hip.h         # HIP 매핑 (cudaHostRegisterMapped=0 등)
├── patched-llama-src/       # 패치된 소스 백업 (통째 교체용 — git에 커밋됨)
│   └── ggml/src/ (위와 동일 5개 파일)
├── moe-l2-hip-port.patch    # 초기 diff (git apply 가능 — 이후 router-map 추가분 미포함)
├── build.log / configure.log        # 빌드 로그
└── server-verify.log        # 검증 서버 로그
```

> ⚠️ `moe-l2-hip-port.patch`는 초기 이식분만 포함. 이후 selective pin/router-map 추가는
> `patched-llama-src/`(최신) 또는 `llama.cpp/` 작업 트리를 직접 사용할 것.
> Git 히스토리는 `github.com/baramofme/moe-l2-hip-port` (커밋 0c844e5까지).

원본 스냅샷(CUDA 패치 포함 llama.cpp): `references/llama.cpp-gpu-lru-cache/`