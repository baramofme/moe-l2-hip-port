# moe-l2 → AMD RDNA3 (RX 7900 XTX, gfx1100) HIP 포팅 진행 문서

> **목적**: 이 문서를 읽으면 중단된 HIP 포팅 작업을 그대로 이어갈 수 있다.
> **최종 갱신**: 2026-08-27
> **작업 위치**: `hip-port/` (이 저장소 내부) + 원본 스냅샷 `references/llama.cpp-gpu-lru-cache/`

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

```bash
cd llama.cpp
HIPCXX="$(hipconfig -l)/clang" HIP_PATH="$(hipconfig -R)" \
  cmake -S . -B build-moe-hip -DGGML_HIP=ON -DAMDGPU_TARGETS=gfx1100 \
  -DCMAKE_BUILD_TYPE=Release -DLLAMA_BUILD_SERVER=ON
cmake --build build-moe-hip --config Release -j$(nproc) --target llama-server
```

- 빌드 성공 확인: `build-moe-hip/bin/llama-server` + `libggml-hip.so`
- moe-l2 심볼 확인: `nm -D build-moe-hip/bin/libggml-hip.so | grep ggml_cuda_expert`
  - `ggml_cuda_expert_pin_host`, `ggml_cuda_expert_cache_get/set/init/free/...` 10개 심볼

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
| `ggml/src/ggml-backend.cpp` | copy_experts per-expert 캐시 후킹 (proc_address로 pin/cache/stream resolve), `<string>` include |

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

```bash
cd llama.cpp
MODEL=/mnt/nvmedata/models/unsloth/Qwen3.6-35B-A3B-GGUF/Qwen3.6-35B-A3B-UD-Q4_K_S.gguf
# 캐시 ON + 전문가 CPU 오프로드 + 선택적 전송
env LD_LIBRARY_PATH=build-moe-hip/bin HIP_VISIBLE_DEVICES=0 \
  GGML_CUDA_EXPERT_CACHE=1 GGML_OP_OFFLOAD_MIN_BATCH=1 MOE_L2_N_LAYERS=48 \
  ./build-moe-hip/bin/llama-server \
  -m "$MODEL" --host 127.0.0.1 --port 18122 -ngl 99 -c 2048 \
  -ot "exps=CPU" -fit off --no-ui
# 추론 테스트
curl -s http://127.0.0.1:18122/v1/chat/completions -d '{"model":"q","messages":[{"role":"user","content":"What is 2+2? Answer with just the number."}],"max_tokens":10,"stream":false}'
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

### ❌ 미해결 이슈 (핵심!): 캐시 정확성
- **증상**: 캐시 ON(`GGML_CUDA_EXPERT_CACHE=1`) 시 출력이 깨짐
  - 초기: `??????????` (hit율 97.8%, 15360 slots, Qwen3-Coder)
  - per-expert + 같은 스트림 수정 후: **빈 문자열** (hit율 84%, 32768 slots, Qwen3.6)
  - 캐시 OFF(대조)에서는 정상 출력 (`4`)
- **원인 추정**: 캐시 키 불일치
  - copy_experts의 `input->name`이 **모든 레이어에서 `blk.0.ffn_gate_exps.weight`로 동일**하게 나옴 (레이어 구분 실패)
  - mul_mat_id의 `src0->name`은 `blk.0`, `blk.1`... 레이어별로 다름
  - → copy_experts가 저장한 키와 mul_mat_id가 조회하는 키가 레이어에서 불일치 → **잘못된 전문가 데이터로 hit**
- **다음 단계 (미완료)**:
  1. copy_experts에서 `input->name`이 레이어별로 실제로 같은지 **로그로 확정** (`[HIP-KEY] input name=...` 로그 추가됨, 재빌드/실행 필요)
  2. `input`(원본)이 아니라 `input->view_src` 또는 텐서 메타데이터에서 **레이어 번호를 얻어** 캐시 키에 포함
  3. 또는 스냅샷의 `router_layer_of(input)` 방식으로 레이어 파싱 (`blk.N` → N)
  4. 수정 후 캐시 ON에서 **정상 출력 확인**이 최종 판정

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

---

## 8. 남은 작업 (이어서 할 것)

1. [ ] **캐시 정확성 수정** (위 6번 미해결) — 키에 레이어 번호 포함
2. [ ] 캐시 ON에서 정상 출력 + hit율 확인 (검증 2 완료)
3. [ ] 검증 1: `hipHostRegister`(핀) 실제 성공 여부 + RSS 감소 실측 (`g_total_pinned_bytes` 로그)
4. [ ] 검증 2: 페이지 eviction (`MOE_L2_LRU_MAX_EXPERTS`) + RSS 캡 실측
5. [ ] 검증 3: 선택적 전송 확인 (이미 로그로 확인됨 — 최종 정리)
6. [ ] DeepSeek-V4-Flash (85GB) 실제 실행 — VRAM/RSS 측정
7. [ ] Python 쪽 포팅 (install.sh, doctor, nvidia-smi→rocm-smi, download-bins HIP asset)
8. [ ] 최종 보고서 작성

---

## 9. 보존된 파일 목록 (hip-port/)

```
hip-port/
├── PROGRESS.md              # 이 문서
├── moe-l2-hip-port.patch    # 최신 llama.cpp에 적용한 diff (git apply 가능)
├── patched-llama-src/       # 패치 적용된 전체 소스 (통째 교체용)
│   └── ggml/src/
│       ├── ggml-backend.cpp
│       └── ggml-cuda/
│           ├── ggml-cuda.cu
│           ├── expert-cache.cu
│           ├── expert-cache.cuh
│           └── vendors/hip.h
├── build.log                # 마지막 HIP 빌드 로그 (성공)
└── server-verify.log        # 검증 서버 실행 로그 (캐시 hit율 84% 포함)
```

원본 스냅샷(CUDA 패치 포함 llama.cpp): `references/llama.cpp-gpu-lru-cache/`