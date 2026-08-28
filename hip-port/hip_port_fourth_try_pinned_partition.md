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