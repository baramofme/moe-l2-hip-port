# HIP 포트 제5차 시도 — Qwen3.8-27B (MTP 스펙 디코딩 + Context 스윕)

> 목표: Qwen3.8-27B (dense 27B) 를 RX 7900 XTX 에서 실측 — MTP 내장 확인,
> 스펙 디코딩 효과, 최대 context 한계.
>
> 모든 작업은 `hip-port/` 아래에서 수행한다.

---

## 배경

Qwen3.8-27B 는 **dense 27B** 모델 (MoE 아님, qwen35 arch). unsloth GGUF:
`unsloth/Qwen3.8-27B-GGUF/Qwen3.8-27B-Q3_K_M.gguf` (13.8GB) + mmproj-BF16 (VLM).

Qwen3 시리즈는 **MTP (Multi-Token Prediction) 내장** — llama.cpp `--spec-type draft-mtp`
로 스펙 디코딩 가능 (추측 토큰을 한 번에 검증 → 생성 가속).

## 환경

| 항목 | 값 |
|---|---|
| GPU | RX 7900 XTX (gfx1100), 24GB |
| ROCm | 7.2.3 |
| 빌드 | hip-port/llama.cpp build-moe-hip (그래프 ON, wave32, MTP 지원) |
| 모델 | Qwen3.8-27B-Q3_K_M.gguf (13.8GB, dense 27B) |
| 설정 | 전량 GPU (-ngl 99) + MTP 스펙 + KV 양자화 스윕 |

---

## 실측 결과

### 1. 모델 구조 확인 — dense 27B + MTP 내장

- expert 텐서 0개 → **dense** (MoE 아님)
- **MTP 텐서 4개**: `blk.64.nextn.eh_proj.weight`, `.enorm`, `.hnorm`, `.shared_head_norm`
  (65 layers, 마지막 레이어에 MTP 헤드) → `--spec-type draft-mtp` 자동 감지됨

### 2. MTP 스펙 디코딩 효과 — gen +77%

| 설정 | gen t/s | draft acceptance |
|---|---|---|
| 기본 (스펙 없음) | 34.6 | - |
| **+ `--spec-type draft-mtp --spec-draft-n-max 4`** | **59.9-61.3** | 76.9% (150/195, mean len 4.06) |

- MTP 스펙 디코딩이 dense 모델에서도 큰 효과 (+77%)
- acceptance 76.9% = 정상 검증 (쓰레기 없음, Qwen3.8-27B 는 reasoning 모델이 아님)

### 3. Context 스윕 (KV 양자화) — 128K 까지 가능

| Context | KV type | gen t/s | prefill t/s | VRAM |
|---|---|---|---|---|
| 16K | q8_0 | 61.4 | 25.6 | 19.2GB |
| 32K | q8_0 | 59.8 | 90.8 | 20.3GB |
| 64K | q8_0 | 59.1 | 90.1 | 21.8GB |
| **128K** | **q4_0** | **60.1** | 96.9 | 22.7GB |

- gen 은 context 와 무관하게 ~60 t/s (MTP 유지)
- 64K 까지 KV q8_0, 128K 는 q4_0 필요 (KV 용량 절반)
- 256K (모델 원래 한계) 는 KV 가 2배 더 필요 → 24GB 카드에서 불가

### 4. 참고 — 이 세션 실측 모델 비교

| 모델 | 종류 | 크기 | gen t/s |
|---|---|---|---|
| Qwen3.8-27B (Q3_K_M) | dense | 13.8GB | 34.6 / **60 (MTP)** |
| Qwen3.6-35B-A3B (Q4_K_S) | MoE | 21GB | 98.4 |
| Qwen3.6-35B-A3B (IQ4_NL_MTP) + MTP | MoE | 18.5GB | 117-122 |
| DeepSeek-V4-Flash (IQ2_M) 오프로드 | MoE | 85GB | 20 |

- dense 27B(13.8GB) 가 MoE 35B(21GB) 보다 3x 느림 — dense 는 매 토큰 전체 파라미터를
  읽지만, MoE 는 활성 expert 만 읽어 대역폭 부담이 훨씬 작음

---

## 실사용 추천 설정

```bash
# 64K (품질 우선)
llama-server -m Qwen3.8-27B-Q3_K_M.gguf -ngl 99 \
  --spec-type draft-mtp --spec-draft-n-max 4 \
  -c 65536 --cache-type-k q8_0 --cache-type-v q8_0

# 128K (최대 context)
llama-server -m Qwen3.8-27B-Q3_K_M.gguf -ngl 99 \
  --spec-type draft-mtp --spec-draft-n-max 4 \
  -c 131072 --cache-type-k q4_0 --cache-type-v q4_0
```

→ gen 60 t/s, prefill 90-97 t/s.

---

## 결론

1. **Qwen3.8-27B 는 MTP 내장** (사용자 확인) — `--spec-type draft-mtp` 로 gen 34.6 → 60 t/s (+77%).
2. **128K context** 까지 24GB 카드에서 가능 (KV q4_0), gen 유지.
3. dense 모델이지만 MTP 덕에 일상 대화용으로 충분히 빠름.
4. MoE (Qwen3.6-35B-A3B) 를 쓸 수 있다면 gen 98-122 t/s 로 더 빠름 — 모델 선택 기준 제공.