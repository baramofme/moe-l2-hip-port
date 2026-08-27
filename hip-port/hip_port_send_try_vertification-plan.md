# HIP 포트 — 검증 계획 (Bounce Buffer 전환 선행 실험)

> 목표: 설계 전환(전면 bounce + 페이지캐시 제어)의 **가정 3가지를 실측으로 확정**하고,
> 각 결과가 구현 설계를 어떻게 결정하는지를 **게이트**로 연결한다.
> 모든 작업은 `hip-port/` 아래에서 수행한다.

배경: HIP selective `hipHostRegister`는 일부만 등록하면 인접 미핀 페이지의
H2D DMA가 `invalid argument`로 크래시한다 (PROGRESS 2026-08-27 참조).
mmap 영역에 대한 등록을 아예 제거하고, CPU `memcpy` → pinned mini-buffer →
async H2D(Bounce)로 전환하는 것이 원천 해법이다. 본 계획은 그 전환 전에
HIP 런타임/OS 커널 가정을 검증한다.

---

## Phase 0 — 환경 고정 (~5분)

```bash
hipcc --version && rocm-smi --showproductname && uname -r
#  - ROCm 버전, GPU(7900XTX), 커널 버전 (MADV_PAGEOUT 은 Kernel 5.4+ 필수)
free -g && nproc
```

테스트 데이터 — 실데이터 필수 (sparse `truncate` 는 hole read 가 zero-fill 이라
refault 수치가 가짜가 됨):

```bash
dd if=/dev/urandom of=/tmp/test_gguf.bin bs=1M count=1100   # 1.1GB 실데이터 (실험2 1GB 범위 + 여유)
dd if=/dev/zero   of=/tmp/dummy_gguf.bin bs=1M count=64     # 실험1용
```

선택적 대조군 (E0, ~10분): 현재 selective-pin 빌드로 짧은 생성 1회 →
문서화된 `invalid argument` 크래시 재현 확인. 환경이 제약을 실제로 재현하는지
확정하고, 이후 fix 의 효과를 대조한다. 시간 압박 시 생략 가능.

---

## Phase 1 — 실험 1: Device-Global Poisoning (수정판)

수정 사항 적용: `dummy_gguf.bin` 생성, 파일 크기 fstat 가드, 64MB 전체 복사.

```bash
hipcc -O2 -o /tmp/test_poison hip-port/experiments/test_device_global_poison.cpp
/tmp/test_poison
```

| 결과 | 판정 | 결정 |
|---|---|---|
| `hipSuccess` | poison 없음 | 로딩/비-expert H2D direct 유지 |
| `invalid argument` | poison 존재 | **모든 H2D staging 경유** (H2D 감사 "실패 시" 컬럼 확정) |

보조 probe: sync `hipMemcpy` 도 같은 영역에서 한 번 측정
(unpinned sync 는 스테이징으로 동작하는 것이 문서화된 동작 — 대조군).

---

## Phase 2 — 실험 2: Pagecache 회수 (수정판)

수정 사항 적용: `PROT_READ` only 로 mmap(실제 GGUF 시맨틱),
**메서드별 독립 서브레인지**(순서 오염 제거), fstat clamp, 페이지 정렬.

```
range A (0~256MB)   : touch → posix_fadvise(DONTNEED)      → RssFile 변화
range B (256~512MB) : touch → madvise(MADV_PAGEOUT)        → RssFile 변화
range C (512~768MB) : touch → 아무것도 안 함(대조군)        → 커널 자발적 회수 여부
```

- 각 단계 사이 `echo 3 > /proc/sys/vm/drop_caches` 는 **하지 말 것**
  (회수 메커니즘 자체를 죽임 — cold 재현이 필요한 실험 3 에서만 root 로 사용).
- 측정: `/proc/self/status` 의 **RssFile** (파일백 페이지만 관측).
- Refault: 각 방법으로 방출 직후 1페이지 재접근 지연(ns) + 별도로 1.55MB 연속 재접근 지연.

| 결과 | 판정 |
|---|---|
| PAGEOUT 이 PROT_READ-only 에서 RssFile 감소 | → eviction = **madvise 기반** (fd 불필요, copy_experts 에서 주소만으로 호출 가능) |
| PAGEOUT EINVAL/무효 | → **fadvise + fd/offset 레지스트리** 필요 (모델 mmap base/fd 추적 — DSO 경계 설계 추가) |
| 대조군 C 에서도 회수됨 | → eviction 강도/빈도 낮출 수 있음 (커널 자체 회수) |

---

## Phase 3 — 실험 3: Cold memcpy 지연 (코드 채워서 실행)

수정 사항 적용: 빈 블록에 코드 작성. 1.55MB(DS)와 1MB(Qwen) 두 크기, 3상태 비교:

- **Cold**: fadvise 로 방출 후 → memcpy 지연 (major fault + NVMe I/O 지배)
- **Warm**: touch 만 하고 방출 안 함 → memcpy 지연 (RAM hit)
- **Hit**: 직전에 읽음 → memcpy 지연 (resident)

각 100회 반복, 평균/95분위. `/tmp/test_gguf.bin` 에 대해
`/tmp/test_cold <file> <offset> <size>`.

| 결과 | 판정 | 결정 |
|---|---|---|
| Cold 1.55MB < 2ms | ~1% 주장 유효 | prefetch 생략 가능, eviction 자유 |
| Cold 1.55MB 2~10ms | 콜드 토큰에 눈에 띔 | **WILLNEED prefetch 필수** + "VRAM set 직후 방출" 정책 유지 |
| Cold > 10ms | 토큰 시간 초과 | 방출 정책 완화(방출 지연), prefetch 필수 |

---

## Phase 4 — 설계 확정 게이트 (실험 종합)

| 결정점 | 입력 | 확정 내용 |
|---|---|---|
| H2D 감사 테이블 | 실험 1 | "로딩 direct 유지" vs "전부 staging" 컬럼 확정 |
| eviction API | 실험 2 | `madvise(MADV_PAGEOUT)` vs `posix_fadvise`+레지스트리 |
| prefetch/방출 정책 | 실험 3 | WILLNEED on/off, 방출 시점(즉시 vs 지연) |
| staging 풀 크기 | 모델별 | 캡 32MB + chunked 기본 (Mixtral 2GB 함정 회피) |

---

## Phase 5 — 구현 후 최종 검증 (V4 실측)

```bash
# 1. crash 0건 / VmRSS ≤ 28GB / ≥30 t/s
env HIP_VISIBLE_DEVICES=0 GGML_CUDA_EXPERT_CACHE=1 MOE_L2_CACHE_SLOTS=16000 \
    MOE_L2_ROUTER_FILE=/path/to/ds_v4.map \
    ./build-moe-hip/bin/llama-server -m /mnt/nvmedata/models/DeepSeek-V4-Flash.gguf \
    -ngl 99 --n-cpu-moe 60 -c 4096

# 2. 대조: MOEL2_NOPIN=1 (기존 12 t/s 기준선) — staging 엔진의 실제 이득 측정
# 3. 동시성: 2~4 병렬 세션 (4-slot 파이프라인, 스트림별 오버랩 검증)
```

합격 기준:
1. crash 0, VmRSS ≤ 28GB, ≥30 t/s
2. staging 엔진 > NOPIN 기준선
3. 병렬 세션에서도 crash 0 + throughput 저하 없음

---

## 시간 예산

| Phase | 시간 | 게이트 |
|---|---|---|
| 0 환경+데이터 (+E0 대조) | 5-15분 | 데이터 존재, ROCm/커널 확인 |
| 1 Poison | 5분 | H2D 감사 결정 |
| 2 Pagecache | 10분 | eviction API 결정 |
| 3 Cold latency | 10분 | prefetch/방출 정책 결정 |
| 4 설계 확정 | 즉시 | 위 결과 합성 |
| 5 V4 실측 | 20-30분 | 최종 합격/불합격 |

---

## 실험/구현 시 필수 수정 사항 (실행 전 반드시 반영)

1. **실험 1·2 SIGBUS 방지**: 파일 사전 생성(`ftruncate`/`dd`), touch 범위를
   `fstat` 파일 크기로 clamp.
2. **`mul_mat_id` guard**: 함수 전체 `return` 금지 — 차단 대상은 moe-l2 가
   넣은 direct-H2D/hook 서브패스뿐. 일반 expert 읽기(matmul 커널)는 유지.
3. **Slots 전용 `hipStreamCreate`**: default stream(nullptr) 사용 시
   double-buffer 오버랩 무효화 + `hipEventSynchronize` 가 전 작업까지 대기.
4. **eviction `madvise` 페이지 정렬**: expert 주소는 페이지 미정렬 → EINVAL.
   `getpagesize()` 정렬 (기존 `pin_host` 와 동일 로직).
5. **실험 2**: `PROT_READ` only + 메서드별 독립 서브레인지 (순서 오염 제거).
6. **`posix_fadvise` fd/offset 획득 경로 미정**: copy_experts 는 CPU 포인터만
   앎 → `madvise`(주소만 필요) 가 실무상 우월. fadvise 는 fd/offset 레지스트리가
   이미 있을 때만 사용.
7. **staging 풀 상한**: capacity = max expert 로 정렬하면 Mixtral(252MB)에서
   ~2GB 핀 RAM — 상한(32MB) + chunked fallback 을 기본 경로로.
8. **플랫폼 가드**: `__HIP_PLATFORM_AMD__`/`GGML_USE_HIP` 하에서만 Bounce 활성화.
   `hipHostRegister` no-op wrapper 의 `#define` 은 헤더 include 이후, .cu 파일
   스코프로 한정 (hip_runtime.h 내부 오염 방지).
9. **CUDA 경로 보존**: 기존 selective-pin 경로는 그대로 두고 HIP 에서만
   Bounce 엔진 사용.

---

## 실험 실행 결과 (2026-08-27, ROCm 7.2.3 / RX 7900 XTX / kernel 7.0.0-30)

### Phase 0 — 환경
- HIP 7.2.3, ROCm 7.2.3 (AMD clang 22), GPU: RX 7900 XTX, RAM 94GB, 20 cores
- 커널 7.0.0-30 (MADV_PAGEOUT 지원, Kernel 5.4+)
- 테스트 데이터: `/tmp/test_gguf.bin` (1.1GB urandom 실데이터), `/tmp/dummy_gguf.bin` (64MB)
- V4 모델: `/mnt/nvmedata/models/deepseek-v4-flash-UD-IQ2_M/UD-IQ2_M/` — 3-shard, **00002 30GB 완료 / 00003 0B 미완** → Phase 5 실측 보류

### 실험 1: Device-Global Poisoning — ✅ poison 없음
```
[1] hipHostMalloc(16MB pool): no error
[5] async unpinned mmap H2D: no error
[6] stream sync: no error
[7] sync unpinned mmap H2D: no error
```
**결정: `hipSuccess` — poison이 device-global 아님.** 미니 핀 풀 등록이 unpinned mmap H2D를
깨뜨리지 않음. 로딩/비-expert H2D는 direct 유지 가능.

### 실험 2: Pagecache 회수 — ✅ PAGEOUT 결정적 승리
| 영역 | 방법 | RssFile 변화 | 판정 |
|---|---|---|---|
| A | posix_fadvise(DONTNEED) | 265MB → 265MB | **no-op (무효)** |
| B | madvise(MADV_PAGEOUT) | 527MB → 265MB (−262MB = B 영역 정확) | **동작** ✓ |
| C | 대조군 | 527MB 유지 | 커널 자발적 회수 없음 |

- refault 1페이지 = 456µs (진짜 디스크 I/O, NVMe)
- **추가 발견**: 미정렬 len의 `madvise(MADV_PAGEOUT)`는 ret=0 반환하지만 **조용히 no-op**
  (1.55MB 미정렬은 RssFile 변화 없음, 정렬 1MB는 정확히 방출) → **PAGEOUT은 페이지 정렬 필수**
- **결정: eviction = `madvise(MADV_PAGEOUT)` 확정.** 주소만 필요 (fd/offset 레지스트리 불필요).
  fadvise는 O_RDONLY/PROT_READ/MAP_SHARED에서 쓸모없음.

### 실험 3: Cold memcpy 지연 — ✅ 콜드 패널티 정량화
| 크기 | HIT | WARM | COLD (PAGEOUT 후) |
|---|---|---|---|
| DS 1.55MB | 84.8µs | 65.5µs | **1,744µs** (~1.7ms) |
| Qwen 1MB | 42.7µs | 36.5µs | **1,132µs** (~1.1ms) |

- COLD = HIT × 20~27배. 콜드 페이지 첫 memcpy는 디스크 I/O 지배.
- **결정: `MADV_WILLNEED` prefetch 필수** + "VRAM set 직후 방출" 정책 유지
  (방출은 VRAM 캐시에 안전하게 들어간 expert만 한정 — 같은 expert 재사용 시 1.7ms 패널티 방지).

### 설계 확정 요약 (Phase 4 게이트 결과)
| 결정점 | 결과 |
|---|---|
| H2D 감사 | 로딩/비-expert direct 유지 (poison 없음), copy_experts만 staging |
| eviction API | `madvise(MADV_PAGEOUT)` (페이지 정렬 필수) |
| prefetch | `MADV_WILLNEED` 필수 |
| staging 풀 | 슬롯당 2×정렬된 expert 크기, chunked fallback |

### 남은 작업
- [ ] V4 00003 shard 완성 후 Phase 5 실측 (crash 0 / VmRSS ≤ 28GB / ≥30 t/s)
- [ ] ExpertStagingEngine 구현 (experiments/ 의 검증 코드를 기반으로)

### 구현 완료 (2026-08-27) — ExpertStagingEngine

실험 결과를 반영한 스테이징 엔진 구현 완료. HIP 빌드(llama-server) 컴파일 통과.

**추가된 파일:**
- `llama.cpp/ggml/src/ggml-cuda/expert-staging.h` — API 선언
- `llama.cpp/ggml/src/ggml-cuda/expert-staging.cu` — HIP 전용 구현 (`#if defined(GGML_USE_HIP)`)

**수정된 파일:**
- `llama.cpp/ggml/src/ggml-cuda/ggml-cuda.cu` — include 추가 + proc_address 3종 등록 (HIP 가드)
- `llama.cpp/ggml/src/ggml-backend.cpp` — staging resolve + copy_experts 통합

**구현 내용:**
1. **staging_copy**: mmap → (CPU memcpy) → pinned 더블버퍼 → async H2D. 페어당 2버퍼 교대 +
   이벤트 동기화로 memcpy(k+1)과 DMA(k) 오버랩. 기본 4페어 × 2 × 2MB~32MB (라운드로빈 +
   페어별 mutex로 스레드 안전). `MOE_L2_STAGING_PAIRS` env로 페어 수 조절.
2. **staging_evict**: `madvise(MADV_PAGEOUT)` 페이지 정렬 (실험 2 검증: PROT_READ-only
   MAP_SHARED 동작, 미정렬은 조용히 no-op). VRAM cache set 성공 후에만 호출.
3. **pin_fn skip**: staging 활성(HIP) 시 mmap 등록 완전 생략 — CUDA는 원래 selective pin 유지.
4. **miss 경로 교체**: `ggml_backend_tensor_set_async`(unpinned mmap 직접 H2D) → `staging_copy`
   (HIP). CUDA는 원래 경로 유지.

**빌드 검증:**
- `cmake --build . --target ggml-hip` → 성공
- `cmake --build . --target llama-server` → 성공
- 백업 트리 `patched-llama-src/`에 미러링 완료

**남은 작업:**
- [ ] V4 00003 shard 완성 후 실측 (Phase 5): crash 0 / VmRSS ≤ 28GB / ≥30 t/s
- [ ] `MADV_WILLNEED` prefetch (실험 3: 콜드 1.7ms 패널티 → 라우터 예측 기반 선반입)
- [ ] 2~4 병렬 세션 동시성 검증

### 추가 실험 — Zero-Copy Fallback 후보 (2026-08-27)

이식 문서의 `cudaHostRegisterMapped → 0`(mapped 미지원)을 재검증. bounce 설계가
부진할 때의 대안 후보로서 hipHostMallocMapped / hipHostRegisterMapped 를 실측.

**환경**: ROCm 7.2.3 / RX 7900 XTX / kernel 7.0.0-30. 코드: `experiments/test_zero_copy.cpp`

| 항목 | 결과 | 판정 |
|---|---|---|
| A. hipHostMallocMapped → getDevicePointer → 커널 읽기 | sum_match=YES, 12.8 GB/s | ✅ 동작 |
| B. mmap + hipHostRegister(mapped) → 커널 읽기 | sum_match=YES, 12.5 GB/s | ✅ 동작 |
| C. RSS (64MB 등록 직후) | delta=+0KB | ✅ **Lazy pin** (등록은 fault-in 안 함) |
| D. 등록 후 인접 미핀 페이지 H2D | no error | ✅ **Poison 없음** (mapped 등록은 인접 미핀을 안 깨뜨림) |
| E. pinned H2D 복사 기준선 | 13.0 GB/s | 기준 |
| A/B zero-copy 읽기 vs E | 12.5-12.8 vs 13.0 GB/s | 동급 (~97%) |

**결론 — 이식 문서의 "mapped 미지원"은 런타임 한계가 아니라 미시도/보수적 선택이었음.**
ROCm 7.2.3에서 mapped가 실제 동작: 디바이스 포인터 정상, 커널 직접 읽기 정확,
부분 등록 후 인접 미핀 H2D 안전(원래 selective-pin 크래시와 다른 동작), lazy pin.

**Fallback 후보 (bounce 부진 시):**
```
mmap expert -> hipHostRegister(mapped, 페이지 정렬) -> GPU 커널 직접 읽기
(H2D 복사 불필요 — CUDA selective pin과 동일 모델)
```
- 대역폭 동급, RSS lazy pin으로 selective pin의 RSS 절감이 HIP에서 원천 가능
- poison 없음으로 hot expert만 부분 등록 가능

**주의 (실험 중 발견)**: `sum_kernel` 초기 버전은 블록 리덕션 누락으로
`partial[blockIdx.x]`에 256개 스레드가 race → 1/256 크기 오합. 대조군(실제
디바이스 버퍼)이 같은 버그를 재현해 국소화. 공유 메모리 리덕션 추가로 해결.

**유보 사항 (V4 실측으로 확인 필요):**
1. 다중 소형 슬라이스 등록 + 사이 미핀 접근에서 poison 없는지
2. GPU가 터치한 mapped 페이지의 상주/방출 동작 (MADV_PAGEOUT 결합 가능성)
3. 1.55MB 소형 슬라이스에서도 zero-copy 대역폭 유지되는지

### 유보 사항 실측 (2026-08-27) — test_reservations.cpp

**R1. 다중 소형 슬라이스 + 사이 미핀 접근 — poison 없음 ✅**
16개 슬라이스(1.55MB, 48MB 간격, 페이지 정렬)를 `hipHostRegisterMapped`로 등록 후:
- 등록 슬라이스 H2D: no error
- **등록 사이 미핀 슬라이스 H2D: no error** (실제 모델 interleaved 패턴 안전)
- 원거리 미핀 H2D: no error
- 등록 슬라이스 커널 직접 읽기: no error
→ 원래 selective-pin 크래시(비-mapped)와 달리 mapped 등록은 다중 슬라이스에서 안전.

**R2. 터치된 mapped 페이지 RSS — 상주 안 함 + PAGEOUT 동작 ✅**
256MB mapped 등록 후 32MB 커널 읽기: RSS delta = **+0KB** (HMM/온디맨드 페이징 —
GPU가 읽어도 물리 RAM에 상주 안 함). `madvise(MADV_PAGEOUT)`(32MB): RSS -32MB 정확히 감소.
→ mapped 경로는 RSS 관리가 원천적으로 쉬움 (bounce의 eviction 훅 불필요).

**R3. 1.55MB 소형 슬라이스 — zero-copy 1.63x 느림 ⚠️**
- zero-copy 커널 읽기: min 213us / pinned H2D: min 131us / ratio 1.63x
- 64MB에서는 동급(~97%)이었으나 소형 전송에서는 커널 launch + PCIe 지연이 지배.
→ fallback으로 쓰려면 성능 손실 1.63x 감수 필요.

### 최종 평가 — fallback 후보 (bounce vs zero-copy mapped)

| 차원 | Bounce (현재 구현) | Zero-copy mapped (fallback) |
|---|---|---|
| 성능 (1.55MB) | H2D 131us + memcpy ~78us | 커널 읽기 213us |
| RSS 제어 | PAGEOUT eviction 훅 필요 | 터치해도 안 상주 + PAGEOUT 동작 (우월) |
| poison | 회피 (등록 안 함) | 없음 (확정) |
| 구현 상태 | 구현+빌드 완료 | 미구현 |

**하이브리드 후보 (RSS 병목 시나리오, 예: 85GB V4):**
- hot expert → mapped 등록 (터치해도 RSS 안 늘음, 커널 직접 읽기)
- cold expert → bounce (H2D 131us, 성능 우위)
- RSS가 원천적으로 바운드되므로 bounce의 eviction 훅이 불필요해짐
- 성능은 hot(213us)이 cold(131us+memcpy)보다 느리므로 hot hit율이 낮은 모델에 유리

**남은 작업:**
- [ ] V4 00003 shard 완성 후 실측 (Phase 5): crash 0 / VmRSS <= 28GB / >=30 t/s
- [ ] `MADV_WILLNEED` prefetch (실험 3: 콜드 1.7ms 패널티)
- [ ] 2~4 병렬 세션 동시성 검증
- [ ] (선택) 하이브리드 hot/cold 분기 실험

### 하이브리드 실험 — 기각 (2026-08-27) — test_hybrid.cpp

실측 상수 시뮬레이션 (D2D hit 30us / bounce miss 209us / zero-copy 213us):
hotset 30-100, VRAM 캐시 0-100슬롯, 2000토큰 x 6 experts, hot 85%.

| 시나리오 | bounce-only | zc-only | hybrid |
|---|---|---|---|
| 캐시 100 | 588-1066ms | 2556ms | 2360-2409ms |
| 캐시 10 | 1998-2350ms | 2556ms | 2532-2534ms |
| 캐시 0 | 2508ms | 2556ms | 2548ms |

**결론: hybrid가 이길 수 있는 설정 없음 — 기각.**
1. hot expert를 zero-copy(213us)로 돌리면 VRAM 캐시 히트(30us)라는 초고속 경로를 버림 (7x 손실).
   "hot -> zero-copy" 직관이 반대였음.
2. RSS 장점도 무효: bounce의 MADV_PAGEOUT eviction이 이미 RSS를 바운드.
3. 유일한 동률(캐시 0)에서도 bounce 근소 우위.

**최종: bounce-only (VRAM 캐시 + MADV_PAGEOUT) 확정.** zero-copy mapped 는 fallback 후보에서
제외. VRAM 캐시 히트(30us)가 zero-copy(213us)를 압도하고, expert 슬라이스 크기가 KV 캐시 대비
미미해 zero-copy의 VRAM 절약 이점도 실익 없음.

### 실측 검증 — Qwen3.6-35B-A3B (2026-08-28) ✅

llama-server (bounce staging engine 빌드), Qwen3.6-35B-A3B-expert_clone_logic-UD-Q4_K_XL (22GB),
`-ngl 99 --n-cpu-moe 48 -ot "exps=CPU" -c 4096`, `MOE_L2_CACHE_SLOTS=16000 MOE_L2_N_LAYERS=48`.

| 지표 | 값 | 판정 |
|---|---|---|
| crash (invalid argument) | 0건 | ✅ HIP 크래시 해결 |
| 캐시 hit율 | 90.5% (217k hit / 22.9k miss) | ✅ VRAM LRU 정상 |
| RSS | 로드 19.8GB -> 생성 중 7.98GB (-60%) | ✅ PAGEOUT eviction 실증 |
| 생성 속도 | 11.0 t/s (200 tokens) | 무겁지만 정상 |
| 출력 | reasoning + content 정상 | ✅ |

핵심: staging 경로로 로드부터 생성까지 crash 0. RSS -60% 는 MADV_PAGEOUT eviction 의
실모델 증거. hit율 90.5% 로 VRAM LRU 정상. Qwen 은 전문가가 작아(~1MB) 캐시 이득이
크지 않은 모델이라 11 t/s 는 예상 범위 (DS/V4 처럼 전문가가 큰 모델에서 이득 큼).

### MADV_WILLNEED prefetch (2026-08-28)

copy_experts 가 같은 그룹의 expert 를 연속 처리하는 점을 활용 — eid 처리 후 다음
expert(eid+1) 의 mmap 페이지에 WILLNEED 를 미리 줘 콜드 fault(1.7ms) 회피.
- expert-staging.h/.cu: `ggml_cuda_expert_staging_prefetch()` (MADV_WILLNEED, 페이지 정렬)
- ggml-cuda.cu: proc_address 등록
- ggml-backend.cpp: resolve + miss 경로에서 `eid < last_id` 시 다음 expert prefetch
- HIP 빌드 llama-server 통과, 백업 트리 미러링 완료
