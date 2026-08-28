# HIP 포트 제3차 시도 — FreeToken (Prefill 가속) (2026-08-28)

> 목표: FlashML-org/FreeToken (https://github.com/FlashML-org/FreeToken) 을 AMD ROCm/HIP
> (RX 7900 XTX, ROCm 7.2.3) 환경으로 포팅해 Prefill 속도 향상을 실측한다.
>
> V4 실측 대신 진행하는 새 실험. moe-l2 는 메모리 I/O 오프로딩(전문가 CPU 상주)으로
> Prefill 시 DMA I/O 병목이 발생할 수 있고, FreeToken 은 프롬프트 토큰을 실시간
> pruning/merging 하여 연산량(N) 자체를 줄여 Prefill/TTFT 를 개선한다.

---

## 배경 — FreeToken vs moe-l2

| 기술 | 원리 | Prefill 영향 |
|---|---|---|
| FreeToken | 중복/비활성 토큰 실시간 솎아내기 (Pruning/Merging) | FLOPs 감축 -> Latency/TTFT 대폭 개선 |
| moe-l2 | 전문가 가중치 CPU RAM 오프로딩 | 실행 가능하게 해주지만 DMA I/O 병목 가능 |

---

## 포팅 핵심 포인트 5가지 (문서 기준)

1. **빌드 시스템 전환 + hipify 자동 변환**
   - `setup.py`/CMake: CUDAExtension -> HIP 익스텐션, `-arch=sm_xx` -> `--offload-arch=gfx1100`
   - hipify-python 으로 `cuda*` API / `<cuda_runtime.h>` -> `hip*` / `<hip/hip_runtime.h>` 자동 변환
2. **Wavefront 크기/비트마스크 하드코딩 제거**
   - Warp=32 고정 vs Wave32/64 가변 (RDNA3 = Wave32/64 가변)
   - `0xffffffff` 마스크 -> `wavefront_size` 동적 참조 / `uint64_t`
3. **CUDA 워프 프리미티브/인트린식 교체**
   - `__shfl_sync`/`__ballot_sync` -> HIP `__shfl`/`__ballot`
   - FP16/BF16, WMMA -> MFMA(CDNA)/WMMA(RDNA) 재매핑
4. **CUB 및 외부 라이브러리 교체**
   - `<cub/cub.cuh>` -> `<hipcub/hipcub.hpp>` / rocPRIM
   - FlashAttention/Triton 연동부 분기 (AMD FA / CK / ROCm Triton)
5. **LDS 뱅크 충돌 / Occupancy 튜닝**
   - LDS 뱅크 구조 및 패딩 조정, `__launch_bounds__` 재조정

---

## 환경 조사 (2026-08-28)

**시스템**: ROCm 7.2.3 / RX 7900 XTX (gfx1100) / kernel 7.0.0-30 / 94GB RAM

**Python/torch**: 시스템에 torch 없음. hipify-clang/perl 은 `/opt/rocm-7.2.3/bin/` 에 있음.

**Docker 후보** (ROCm PyTorch):
- `baramofme/comfyui-rocm:rocm7.14-py3.14-torch2.12.0-triton3.7.1-comfy0.33.1` (33GB) — torch 2.12 + triton 3.7.1
- `baramofme/unsloth-rocm-gfx1100:260805-marimo` (31.8GB)
- `baramofme/llama-cpp-rocm:gfx1100-rocm7.2-tbqplus-a8ec5c276-adb55e514` (22.4GB)

Docker GPU 접근 확인됨 (`--device=/dev/kfd --device=/dev/dri` 로 rocm-smi 동작).

---

## 실험 계획 (초안)

1. FreeToken 저장소 구조/의존성 파악 (librarian 조사 완료 대기)
2. 실행 환경 결정: Docker (comfyui-rocm) vs 로컬 venv + torch ROCm
3. CUDA 커널 실측: hipify 변환 후 컴파일 가능 여부, gfx1100 타깃
4. 벤치마크: FreeToken Prefill speedup vs baseline (프롬프트 길이별 TTFT)
5. (선택) moe-l2 와 결합 시 Prefill I/O 병목 개선 여부

---

## 진행 상태

- [ ] FreeToken 저장소 구조 조사 (librarian bg task)
- [ ] 환경 확정 (Docker vs venv)
- [ ] hipify 변환 시도
- [ ] 컴파일/실행 검증
- [ ] Prefill 벤치마크
---

## 실측 결과 — FreeToken 로직 hip-port 적용 판정 (2026-08-28)

### FreeToken의 실제 처리 로직 (문서 주장과 불일치)

FreeToken 코드에 "토큰 pruning/merging"은 **없음**. 실제 핵심:
1. `moe_prefill_overlap` — 레이어 L+1 H2D 프리페치를 레이어 L GEMM과 오버랩 (double-buffer)
2. sparse attention — DSv4/GLM/minimax **모델 전용** 커널 (Qwen에 없음)
3. q* policy — CPU/GPU 레이어 분배

### 병목 위치 실측 (Qwen3.6-35B-A3B, 970-token prefill)

| 설정 | prefill | tps | 비고 |
|---|---|---|---|
| 전문가 CPU (moe-l2 현재) | 60,348ms | 16.1 | HIP에서 H2D 경유 |
| 전문가 GPU (전체 상주) | 745ms | **1,302** | GEMM 주도 |

**전문가 GPU 상주 시 prefill 81배 빠름** -> 병목은 PCIe 대역폭이 아니라
**HIP의 전문가 H2D 경로 자체** (bounce: memcpy + H2D 직렬).

### 결론: FreeToken 레이어 오버랩 이식 -> 성능 향상 없음

- H2D 60초가 GEMM 0.75초를 압도 -> 오버랩해도 H2D가 지배 (개선 수% 미만)
- FreeToken은 CUDA에서 expert가 pinned host + zero-copy GPU 직접 읽기 전제인데,
  HIP엔 그 경로가 없어 H2D 복사가 병목
- 진짜 해법은 오버랩이 아니라 **H2D 경로 자체 개선**:
  a) CPU memcpy 제거 (직접 pinned H2D) — staging 엔진 개선
  b) 전체 GPU 상주가 가능한 모델은 prefill 81배 (VRAM 여유 시)

### 실험 환경 (ROCm 7.14 Docker)

- FreeToken HIP 빌드 성공: hipify 변환(pinned_tensor/cpu_moe/gguf 커널) + setup.py
  HIP화 (ROCM_PATH 탐색, amdhip64 링크) + CUDART_CB/hipLaunchHostFunc 수정
- `freetoken-dev` Docker 컨테이너 (torch 2.12.0+rocm7.14.0, triton 3.7.1)
- torch>=2.11 제약 완화, flashlib 설치, 전체 import OK, ft CLI 동작

### 결론 정리

FreeToken 처리를 hip-port에 적용하는 것은 **prefill 성능 향상에 효과 없음** (실측).
대신 **staging 엔진 H2D 경로 개선**(memcpy 제거, DMA 파이프라인)이 실제 병목 해결책.
