# HIP 포트 제4차 시도 — Pinned Partition (정적 파티션)

> 목표: llama.cpp PR #25294(`moe-stream-partition`)에서 도출한 설계 아이디어 2가지를
> 현재 HIP 포트(직접 DMA + NO_EVICT 정착 상태)에 적용해 실측한다.
> 3차 시도(FreeToken)와 병행하며, 각 실험 결과를 게이트로 설계에 반영한다.
>
> 모든 작업은 `hip-port/` 아래에서 수행한다.

---

## 배경 — 왜 4차인가

| 차수 | 시도 | 결과 |
|---|---|---|
| 1차 | hipHostRegister selective pin | 부분 등록 시 인접 미핀 H2D `invalid argument` 크래시 |
| 2차 | Bounce staging (memcpy + pinned 더블버퍼 + async H2D) | crash 0 달성, 그러나 staging 오버헤드(209µs)로 11 t/s 한계 |
| 3차 | 직접 DMA (NO_STAGING) + NO_EVICT + 도메인 테이블 | **19.5-21.3 t/s**, hit 90%, majflt 0. RDNA3 계산 천장 ~38 t/s 확인 |
| 3차 보조 | FreeToken (prefill pruning/merging) | 계획 수립, 실험 대기 |

2차·3차 실측으로 확정된 사실:
1. **eviction(MADV_PAGEOUT)이 속도를 죽인다** — hot expert 페이지까지 RAM에서 방출,
   VRAM LRU 미스(11%)마다 NVMe 재읽기(1.7ms) 폭주. NO_EVICT=1 → majflt 0, 37.7 t/s.
2. **staging이 직접 DMA보다 느리다** — memcpy+H2D+뮤텍스/이벤트(209µs) vs
   on-demand pin + hipMemcpyAsync 직접 DMA(127µs).
3. **도메인 테이블이 hit율을 지배** — 테이블 없이 55% → math 테이블 90.1%.

여기에 llama.cpp PR #25294(`moe-stream-partition`)의 설계 아이디어를 결합한다:

**아이디어 1 — 정적 파티션 (static partition):**
> "every expert lives in exactly one place, pinned in VRAM or mirrored in pinned RAM,
> so the SSD is only read once at load and eviction never writes anything back."

즉 LRU 동적 캐시 대신 **hot expert는 VRAM에 영구 고정, cold expert는 RAM 상주**.
현재 NO_EVICT(전체 RAM 상주 + LRU 캐시)에서 한 단계 더 나아가,
도메인 테이블의 top-K hot expert를 VRAM에 **고정(pre-pin)** 하고 나머지는 RAM에 둔다.
→ LRU miss가 원천적으로 사라지고, 캐시 슬롯 경쟁도 없다.

**아이디어 2 — per-layer miss 복사 오버랩:**
> "each layer is an independent draw of 6 experts out of 256 ... the miss copies slip
> in behind the shared expert and only the leftover wait costs time."

레이어별로 콜드 expert의 PCIe 복사를 **상주 expert 계산 뒤에 숨긴다** (timeline overlap).
현재는 copy_experts가 그룹 내 expert 연속 처리 + WILLNEED prefetch만 있고,
복사와 matmul 커널 실행이 직렬화되어 있다. 레이어 간 스트림 오버랩으로
miss 대기시간을 계산에 숨긴다.

---

## 실험 설계

### 실험 A — 정적 파티션: hot expert VRAM 고정 vs LRU (아이디어 1)

**가설**: 도메인 테이블의 hot expert를 LRU 대신 VRAM에 영구 고정하면
miss가 0에 수렴하고, 캐시 슬롯 경쟁이 사라져 현재 19.5-21.3 t/s 를 넘는다.

**설계**:
- 현재: `MOE_L2_CACHE_SLOTS=8000` + LRU (hit 90.1%)
- 제안: 도메인 테이블 top-K expert (레이어별)를 `cudaHostRegister`/hip 고정 영역으로
  VRAM에 pre-pin. cold expert는 NO_EVICT RAM에서 직접 DMA.
- 게이트: `hit율 ≥ 95%` AND `t/s > 21.3` (현재 최고) → 정적 파티션 확정

**변형**: 정적 파티션 VRAM 크기 파라미터 (top-50/top-100 등) 스윕.

### 실험 B — per-layer miss 오버랩 (아이디어 2)

**가설**: 콜드 expert 복사를 별도 스트림으로 보내 상주 expert의 matmul 커널 뒤에
숨기면 miss 대기(127µs)가 토큰 시간에서 사라진다.

**설계**:
- copy_experts miss 경로를 전용 스트림으로 분리 (현재 default stream 직렬화)
- 레이어 N의 miss 복사 + 레이어 N-1의 matmul 오버랩
- 게이트: miss-dominated 설정(캐시 4500)에서 t/s 증가 확인

### 실험 C — FreeToken (3차 계획 이어서)

배경 문서 `hip_port_third_try_freetoken_260828.md` 참조. 프롬프트 토큰
pruning/merging으로 prefill FLOPs를 줄이는 접근. moe-l2의 DMA I/O 병목과
직교하므로 결합 시 시너지 가능.

---

## 환경

| 항목 | 값 |
|---|---|
| GPU | AMD Radeon RX 7900 XTX (gfx1100), GPU 0 사용 |
| ROCm | 7.2.3 (HIP 7.2.53211) |
| CPU/RAM | i5-14600K / 94GB RAM |
| 모델 (실측) | Qwen3.6-35B-A3B Q4_K_XL (22GB, ExpertClone) |
| 모델 (최종) | DeepSeek-V4-Flash (85GB, 3-shard, 00003 미완) |
| 빌드 | `hip-port/llama.cpp` (bounce staging → 직접 DMA + NO_EVICT 정착) |

---

## 실험 결과

(실험 진행하며 기록)

### ⚠️ 실험 A/B 결론 수정 (2026-08-28 후반) — math 테이블이 오히려 30-35% 손해

실험 A 의 "캐시 슬롯 무의미" 와 2차 문서의 "테이블 hit 90% = 19.5-21.3" 은 **math 테이블
(listed 게이트) 의 왜곡**이었음이 밝혀짐. 그래프 ON 복원 후 무테이블/테이블 A/B:

| 설정 (그래프 ON) | hit | gen t/s |
|---|---|---|
| 전량 GPU | - | **98.4** |
| **오프로드, 무테이블, 캐시 16000** | 92.9% | **33.5-34.6** |
| 오프로드, 무테이블, 캐시 8000 | 77.8% | 26.0-27.2 |
| 오프로드, +math 테이블, 캐시 16000 | 94.4% | 22.1-22.5 |
| 오프로드, +math 테이블, 캐시 8000 | 82.9% | 22.4 |

**원인**: math 테이블의 `listed` 게이트가 **테이블 밖 expert 를 캐시 참여에서 제외** →
라우팅이 테이블 밖 expert 를 고르면 매 토큰 H2D(127µs) 반복 + 캐시 저장 안 됨.
이 모델(일반 Q4_K_S)의 실제 라우팅은 math 도메인 top-100 과 불일치 → unlisted H2D 가 지배.
**hit율이 높아도 (94.4%) 속도가 33% 낮음.** 2차 문서의 테이블 실측(19.5-21.3)도 이 손해 포함.

**수정된 결론**:
1. **캐시 슬롯은 무테이블에서 확실한 효과** (+28%, 26→34 t/s). 실험 A 의 "miss 제거 무의미" 는 테이블 왜곡.
2. **그래프 ON 이 오프로드를 크게 개선** (+37%, 22.7→31-34). OFF 는 -11%(전량)/-37%(오프로드) 손해.
3. **무테이블 + 캐시 16000 + 그래프 ON = 최선의 오프로드 33.5-34.6 t/s** (7900 XTX).
4. 전량 GPU(98.4) 는 여전히 3x — Qwen 은 전량이 정답, 오프로드는 V4 급에서만.

### 실행 환경 확립 (2026-08-28)

**중요 발견 — moe-l2 정상 실행 방법**: `GGML_OP_OFFLOAD_MIN_BATCH=1` + `-ot exps=CPU` 조합이 필수.
`-ot exps=CPU`만 쓰면 expert가 CPU에서 계산되어 copy_experts(활성 expert GPU 복사)가 꺼지고 11-12 t/s.
`GGML_OP_OFFLOAD_MIN_BATCH=1`(decode batch=1에서 expert matmul을 GPU로 강제)을 함께 쓰면
스케줄러가 활성 expert만 GPU로 복사 + VRAM 캐시 동작 → 22-24 t/s.

| 설정 | copy_experts | gen t/s | VRAM |
|---|---|---|---|
| `-ot exps=CPU`만 | 꺼짐 (CPU 계산) | 11-12 | 5.0 GB |
| `-ot exps=CPU` + `MIN_BATCH=1` | 켜짐 | 22-24 | 7.3 GB |
| `-ngl 99`만 (전량 GPU) | 불필요 | **98.4** | 23.1 GB |

전량 GPU 98.4 t/s는 2차 문서의 "RDNA3 계산 천장 ~38 t/s" 주장이 **오판**이었음을 보여준다
(2차는 오프로드 상태만 측정). expert 오프로드는 이 모델에서 4.4x 손해.

### 실험 A — 캐시 슬롯 스윕 (정적 파티션 근사)

일반 Qwen3.6-35B-A3B-UD-Q4_K_S, math 테이블(4000 experts), NO_STAGING + NO_EVICT:

| 캐시 슬롯 | hit | gen t/s | VRAM |
|---|---|---|---|
| 8000 | 82.9% (188k/38k) | 22.4 | 7.3 GB |
| 16000 | **95.4%** (235k/11k) | 22.7-24.3 | 13.0 GB |

**결론: hit 82.9 → 95.4% (+12.6pp) 여도 t/s 는 거의 불변 (22.4 → 22.7-24.3).**
miss(H2D 127µs) 를 제거해도 D2D hit(30µs x ~700 expert/토큰 ≈ 20ms) 가 그대로 병목.
→ **정적 파티션(영구 고정, hit 100% 목표) 은 LRU 캐시가 충분히 클 때와 성능이 거의 동일.
   miss 4.6% 제거 효과는 t/s +1~2 수준.** LRU → 영구 고정 전환은 무의미.

**진짜 병목은 expert 전송 자체**: 전량 GPU 98.4 vs 오프로드 22.4 t/s.
hit 든 miss 든 expert weight 를 매 토큰 GPU 로 옮기는 것이 지배 비용.
→ 개선 방향은 "캐시 hit 증가" 가 아니라 "전송 자체 제거/오버랩" (실험 B).

### 실험 B — miss 복사 오버랩 (PR 아이디어 2)

**결론: 이득 없음 — 이미 2차 시도에서 검증됨 + miss 비율이 너무 낮음.**

1. **오버랩(staging)은 직접 DMA보다 이미 느림** (2차 문서 실측): staging(memcpy+H2D 더블버퍼,
   209µs) vs 직접 DMA(127µs). 오버랩 구조가 뮤텍스/이벤트 오버헤드로 오히려 손해.
2. **miss가 4.6%뿐** (캐시 16000, hit 95.4%): miss 오버랩이 숨길 대상이 거의 없음.
3. **진짜 병목은 hit D2D 전송 자체**: hit 95.4%여도 토큰당 ~700회 D2D(30µs) ≈ 20ms.
   오버랩 대상(miss)이 아니라 전송 자체(전체)가 병목.

**PR #25294 두 아이디어의 실측 결론 (Qwen3.6-35B-A3B-UD-Q4_K_S @ 7900 XTX):**

| 아이디어 | 실측 | 판정 |
|---|---|---|
| 1. 정적 파티션 (영구 고정) | hit 82.9→95.4%여도 t/s 22.4→22.7-24.3 (미미) | **도움 안 됨** — 병목은 miss가 아니라 hit D2D 자체 |
| 2. miss 복사 오버랩 | staging 오버랩이 직접 DMA보다 느림 (2차) + miss 4.6% | **도움 안 됨** |

**핵심 통찰 — 오프로드 자체가 4.4x 손해:**
- 전량 GPU (expert 전부 VRAM): **98.4 t/s**
- 오프로드 (expert CPU + 캐시 hit 95%): 22.4-24.3 t/s
- 2차 문서의 "RDNA3 계산 천장 ~38 t/s"는 **오판** (전량 GPU 미측정).

**의미**: Qwen 21GB 모델은 VRAM 24GB에 전량 로드 가능 → 오프로드 불필요 (전량 GPU 98.4).
오프로드가 정당화되는 건 V4(85GB)처럼 VRAM에 안 들어가는 모델뿐. 그 경우에도
"캐시 hit 증가/오버랩"이 아니라 **전송 자체 제거**(mul_mat_id가 캐시에서 직접 계산하는
2107 경로, experts_on_host 필요) 가 유일한 개선 경로.

### 실험 B2 — mul_mat_id 직접 캐시 (2107, experts_on_host) — 유일 개선 후보 검증 ❌

"전송 제거" 후보 (copy_experts D2D 대신 mul_mat_id가 캐시에서 직접 계산)를 실측.
HIP 빌드에서 2107 경로는 experts_on_host=false (copy_experts가 src0 를 GPU 로) 로 항상 꺼져 있었음.

**활성화 작업:**
- `GGML_CUDA_FORCE_CPU_EXPERTS=1` env 추가 (CUDA ref 의 d2h 실험 게이트 이식, 1960 skip 판별 전으로 이동)
- `GGML_ASSERT(mul_mat_id_needs_sync)` 제거 (CUDA ref 와 동일 — decode 배치에서 MMVQ 가능해도
  experts_on_host 경로 허용. **단 GGML_HIP_GRAPHS=OFF 필요**, stream sync 경로라 그래프 캡처와 충돌)
- `temp_gpu.alloc` 을 루프 밖 1회로 (CUDA ref 구조 — 루프 안 miss 마다 alloc 시 pool assert)

**실측 (Qwen3.6-35B-A3B-UD-Q4_K_S, 캐시 2000, 전량 GPU 로드 + FORCE_CPU d2h):**

| 경로 | hit | gen t/s | 비고 |
|---|---|---|---|
| copy_experts (D2D) | 42.7% | 16.1-17.9 | `-ot exps=CPU` 오프로드 |
| **2107 직접 캐시 (전송 0)** | **65.9%** | 15.4-15.5 | FORCE_CPU |

**결론: 2107 경로가 hit 율 23pp 높음에도 속도가 비슷/느림.**
hit 시 expert 전송이 0 이어도, **src0_slice 개별 커널 실행(전송 제거 경로)이 D2D + 기존 커널
(copy_experts) 보다 느려서** 전체 성능이 오르지 않는다. → **"전송 제거" 도 개선 경로가 아님.**

**한계**: FORCE_CPU 는 전량 GPU 상태에서 d2h 사본을 만들므로 expert(21GB) + 캐시로 OOM →
캐시 2000 제한. 진짜 llama-model-loader 패치(expert host 배치 + GPU backend)면 캐시를 키울 수
있지만, 계산 방식(src0_slice 커널)이 동일해 큰 개선은 기대하기 어렵다.

**최종 판정 (모든 오프로드 방식):**

| 방식 | gen t/s | 비고 |
|---|---|---|
| 전량 GPU (expert 전부 VRAM) | **98.4** | 이 모델의 진짜 성능 |
| copy_experts + D2D 캐시 (hit 95%) | 22.7-24.3 | 캐시 16000 |
| copy_experts + D2D 캐시 (hit 43%) | 16.1-17.9 | 캐시 2000 |
| 2107 직접 캐시 (hit 66%) | 15.4-15.5 | 캐시 2000, 전송 0 |

**오프로드는 본질적으로 4-6x 손해** — expert 전송(어떤 방식이든) + 개별 expert 커널 실행이
전량 GPU 의 전체-텐서 커널보다 느리다. Qwen 21GB 는 VRAM 24GB 에 전량 로드가 정답.
오프로드는 V4(85GB) 같은 VRAM 초과 모델에서만 불가피하고, 그때도 15-24 t/s 수준이 한계.

**V4 (85GB) 실측이 유일하게 남은 검증** — 단 shard 00003 = 0B, 00001 = 5MB 로 불완전.

### 실험 D — Wave64 (GGML_HIP_WAVE64, RDNA3) — ❌ 이 워크로드에서 기각

llama.cpp #20934 논의의 Wave64 제안 이식 (CMake-only, off by default):
`-mwavefrontsize64` + WMMA/MMQ/MMF/fattn 파일은 wave32 예외 (`-mno-wavefrontsize64`).
커뮤니티 실측 (Qwen3.6-35B-A3B Q6_K): +4.5% (d0) ~ +12% (d32768).

**실측 (Q4_K_S, 짧은 컨텍스트):**

| 워크로드 | wave32 | wave64 |
|---|---|---|
| 전량 GPU | 98.4 | 96-97 (-1%) |
| 오프로드 무테이블 캐시 16000 | 33.5-34.6 | 30.6-33.3 (-5%) |

**결론: 기각.** 이유:
1. mmq.cu (Q4_K_S 의 matmul 경로) 가 wave32 예외 → 양자화 matmul 은 wave64 미적용.
2. wave64 이득은 dense/FP 경로 + 긴 컨텍스트(d32768) 에서만 — 커뮤니티도 Q6_K + 긴 컨텍스트에서 최대.
3. 짧은 컨텍스트 Q4_K_S 에선 오히려 소폭 손해.

### 실험 E — Vulkan 백엔드 — ⏸ SDK 없음

#20934 실측: 7900 XTX 에서 Vulkan 이 ROCm 보다 tg 20-22% 빠름 (wave64 전용, RADV).
시스템에 dev 헤더/glslc 없음 (sudo 필요) → 빌드 불가. 이후:
`sudo apt install libvulkan-dev glslang-tools shaderc` 로 설치 후 재시도 가치 있음.
(단 moe-l2 커스텀 expert 캐시는 HIP 전용 — 전량 GPU 비교만 가능)

### 실험 F — ZLUDA / SageAttention / rocWMMA FA (웹 검증, 2026-08-28)

**ZLUDA — ❌ 쓸 이유 없음 (웹 확인):**
- ZLUDA 5/6 (2026-01, 2026-06): llama.cpp CUDA 백엔드 완전 지원, ROCm 7 지원.
- **성능: "nearly identical to native ROCm" (<5% 차이)** — 병목(expert 전송/개별 커널)을 해결하지
  않고 성능도 HIP 동급. HIP 포팅 무의미화 + 레이어 오버헤드만 추가. 기각.

**SageAttention — ❌ llama.cpp 와 무관:**
- SageAttention 은 PyTorch/Triton 라이브러리 (v1 은 Triton/ROCm 가능, v2 는 CUDA 전용).
- llama.cpp(C++) 에 적용 불가. 설령 적용해도 decode 에서 attention <1% (beellama 실측:
  "Attention is <1% of decode time on Qwen3.5-27B, Weight GEMM dominates >99%").
- ComfyUI(이미지 생성) 의 7900 XTX 개선(19%) 은 diT 작업이라 무관. 기각.

**GGML_HIP_ROCWMMA_FATTN — ⏸ 이식 필요:**
- llama.cpp 최신 HIP 옵션 (#15021/#10879): rocWMMA 로 flash attention 가속,
  7900 XTX prefill "huge performance jumps".
- 이 hip-port(3b80fa9) 에 코드 없음 + /opt/rocm 에 librocwmma 없음 (헤더만).
- 최신 llama.cpp 에서 fattn-rocwmma 코드 이식 + rocwmma 패키지 설치 후 실험 가능. 보류.

**커뮤니티 실측과의 일치 (웹 검증 결과):**
- 전량 vs 오프로드 격차: willitrunai "cards that fit whole model in VRAM run far faster
  than ones that offload" — 7900 XTX Q4_K_M(오프로드) 30.8 t/s vs 내 33.5-34.6. 일치.
- GGML_OP_OFFLOAD_MIN_BATCH: PR #18535 env, 기본 32. decode 강제 시 H2D 증가 (#20757) — 조건 의존적.
- ROCm < Vulkan 20-22% (#20934), wave32 고정, HIP 커널이 NVIDIA 설계 공유 + ROCm BLAS 미최적화.
- persistent expert cache RFC (#24528): CPU MUL_MAT_ID + GPU hit 캐시 구조 (4x, 미머지) — 내 2107
  실험(전송 제거)과 다른 구조. V4 실측 시 이 구조도 후보.

### 실험 G — batch/ubatch 크기 (prefill) — ✅ -ub 증가가 prefill +47%

DocShotgun 가이드 (CPU+GPU MoE 는 `-b 4096 -ub 4096` 권장). 긴 프롬프트(774 토큰) A/B:

| 설정 | prefill | gen (긴 컨텍스트) |
|---|---|---|
| 기본 (-b 2048 -ub 512) | 192.2 t/s | 37.6-41.8 t/s |
| **-b 4096 -ub 4096** | **283.4 t/s (+47%)** | 38.4-40.9 t/s |

**부수 발견**: 긴 프롬프트 후 gen 이 37.6-41.8 (짧은 프롬프트 33.5-34.6 보다 빠름) — 프롬프트
처리가 expert 캐시를 워밍업 (수학 도메인 라우팅 집중) 하여 생성 hit 이 높아짐.

**최선 설정 확정 (7900 XTX, Qwen3.6-35B-A3B-UD-Q4_K_S):**
```
GGML_OP_OFFLOAD_MIN_BATCH=1 GGML_CUDA_EXPERT_CACHE=1 MOE_L2_CACHE_SLOTS=16000
MOE_L2_NO_STAGING=1 MOE_L2_NO_EVICT=1 MOE_L2_N_LAYERS=48
llama-server -ngl 99 -ot exps=CPU -c <ctx> -b 4096 -ub 4096
```
→ prefill 283 t/s, gen 38-41 t/s (긴 컨텍스트), VRAM ~13GB. 전량 GPU 는 98.4 (Qwen).

### 실험 H — V4 실측 (DeepSeek-V4-Flash-UD-IQ2_M, 85GB) — 구조적 한계 확인 ✅

**V4 3-shard 완성** (00001 5.2MB 메타데이터 전용 + 00002 50GB + 00003 41GB = ~91GB,
`unsloth/DeepSeek-V4-Flash-0731-GGUF`). 2차 문서의 남은 작업 실측 완료.

**실측 (24GB 7900 XTX, expert 오프로드 `-ot exps=CPU` + MIN_BATCH=1 + NO_STAGING):**

| 캐시 슬롯 | hit | gen | VRAM | 비고 |
|---|---|---|---|---|
| 1000 | 36% | **5.0 t/s** | OK | 안정, RSS 56.8GB |
| 3000 | 57% | 매우 느림 | 25.2GB 초과 | swap 스래싱, 실사용 불가 |

**결과: crash 0 ✓ (85GB 모델이 24GB 카드에서 expert 오프로드로 동작), gen 5 t/s.**

**구조적 한계 3중:**
1. **expert 크기**: V4 expert ~10MB (gate/up/down) — Qwen 576KB 의 17배. 캐시 슬롯당 VRAM 소모 큼 →
   24GB 카드에서 캐시 ~1000 슬롯(10GB) 이 현실적 한계 (3000 슬롯은 30GB → VRAM 초과).
2. **spread routing**: DeepSeek 계열 라우팅이 분산 → 같은 expert 재사용 낮음 → 캐시 hit 본질적으로
   낮음 (36%, 캐시 3000 이어도 57% — LRU thrash).
3. **RAM 부족**: 모델 91GB > available 56GB → NO_EVICT (전량 상주) 불가 → evict 시 NVMe 재읽기.
   swap 스래싱 발생.

**2차 문서 목표 (crash 0 / RSS ≤ 28GB / ≥30 t/s)**: crash 0 달성, RSS 56.8GB / 5 t/s 미달.
**≥30 t/s 는 이 하드웨어(24GB VRAM + 56GB RAM) 에서 expert 오프로드 구조로는 달성 불가** 실측 확인.

**시사점**: V4 급 (85GB) 은 24GB 카드에서 오프로드 본질적 한계 (~5 t/s). 
- 더 큰 VRAM (48GB+) 필요하거나, expert 압축/스파시티 활용 (라우팅 집중 유도) 필요.
- persistent expert cache RFC (#24528, CPU MUL_MAT_ID + GPU hit 캐시) 는 miss 를 CPU 가 계산하므로
  전송 병렬화 측면에서 유일한 구조적 대안 — V4 에서 재평가 가치 있음.

**V4 미세 튜닝 (추가 실측):** 캐시 1200 + `MOE_L2_LRU_MAX_EXPERTS=1500` 도 hit 38% / gen 4.9-6.9 t/s
(캐시 1000 과 동일). **NO_STAGING(직접 DMA) 경로에선 staging_evict_fn 이 nil 이라 RSS 캡이 무효**
(RSS 77GB, 스왑 근접). 캐시 1000-1200 / hit 36-38% / gen 5-7 t/s 가 V4 의 현실적 한계 확정.
RSS 제어하려면 staging(evict) 경로 필요하나, evict 시 NVMe 재읽기(1.7ms)로 속도 손실 (2차에서 확인).

### ⚡ V4 경로 재발견 (2026-08-29) — STAGING(bounce) 이 V4 에서 4x, hit 99%

2차 문서의 "직접 DMA > staging" 은 **모델 의존적**이었음. V4(큰 expert ~2.17MB) 실측:

| 경로 | 캐시 | hit | gen t/s | 비고 |
|---|---|---|---|---|
| NO_STAGING (직접 DMA) | 1000-1200 | 36-38% | 5-7 | 캐시 miss 지속 |
| **STAGING (bounce) + LRU 캡 1200** | 1200 | **99.1%** | **20.0-20.3** | RSS 76.4GB, swap 경계 |

**STAGING 이 4x 빠름 + hit 99%.** 원인 추정: NO_STAGING 의 miss 경로
(`ggml_backend_tensor_set_async`, unpinned mmap → H2D) + `cache_set`(D2D) 이 큰 expert 에서
스트림/저장 문제로 캐시 hit 이 지속되지 않음 (36%). STAGING(pinned bounce) 은 캐시 저장이
정상 → 두 번째 토큰부터 expert 재사용 (99%). Qwen(작은 expert) 에선 반대(직접 DMA 우위).

**V4 최선 설정 확정:**
```
GGML_OP_OFFLOAD_MIN_BATCH=1 GGML_CUDA_EXPERT_CACHE=1 MOE_L2_CACHE_SLOTS=1200
MOE_L2_N_LAYERS=43 MOE_L2_LRU_MAX_EXPERTS=600~1200
llama-server -ngl 99 -ot exps=CPU -c <ctx>
```
(NO_STAGING **제거** — staging bounce 필수) → gen 20 t/s (85GB V4 @ 24GB 7900 XTX).

**LRU 캡 600 vs 1200: gen 동일 (19.8-20.5 t/s), RSS 비슷 (74.7 vs 76.4GB) — RSS 는 캡으로 안 줄어듦**
(모델 85GB > RAM 94GB 의 본질적 문제. expert hit 99% 라 스왑이 gen 에 영향 없음 — 3회 요청 안정).

### 실험 C — FreeToken (계획 갱신)

3차 문서 (`hip_port_third_try_freetoken_260828.md`) 계획 이어서. **실측 전제가 변경됨**:
- FreeToken은 prefill FLOPs 감축 (프롬프트 토큰 pruning/merging) — decode expert 전송 병목과 직교.
- 현재 주요 병목은 decode의 expert 전송 (전량 GPU 98.4 vs 오프로드 22.4)이므로,
  FreeToken으로 개선되는 prefill은 이 병목과 다른 축. V4 (85GB) 실측이 가능해지면
  prefill 대비 의미를 판단.
- V4 3-shard는 **여전히 불완전** (00003 = 0B, 00001 = 5MB) — 실측 대기.

---

## 설계 확정 게이트

| 결정점 | 입력 | 확정 내용 |
|---|---|---|
| hot expert 고정 방식 | 실험 A | **LRU 유지** (정적 파티션 전환 무의미 — hit 95.4%에서 miss 제거 효과 미미) |
| 오버랩 스트림 | 실험 B | **기각** (staging 오버랩 < 직접 DMA, miss 4.6%) |
| 전송 제거 경로 | 실험 A/B | **mul_mat_id 직접 캐시 (2107, experts_on_host) — 유일한 개선 경로, V4에서 검증 필요** |
| FreeToken 채택 | 실험 C | prefill 축 개선, V4 실측 후 판단 |