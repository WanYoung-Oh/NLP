# PRD — 일상 대화 요약 경진대회 구현 계획

> **기반**: `/nlp-dialogue-summarization` 스킬 구조
> **목표**: ROUGE-F1 베이스라인 47.12 → 60+
> **작성일**: 2026-03-12
> **참고**: 대회 소개 PDF, `docs/RESEARCH.md`

---

## 목차

1. [프로젝트 구조](#1-프로젝트-구조)
2. [Hydra Config 설계](#2-hydra-config-설계)
   - [2.5 한글 특성 및 대회 특성 반영](#25-한글-특성-및-대회-특성-반영)
   - [2.6 데이터 전처리 현황 및 정책](#26-데이터-전처리-현황-및-정책)
3. [Phase별 구현 계획](#3-phase별-구현-계획)
   - Phase 1: 환경 셋업 & 베이스라인 재현
   - Phase 2: 모델 업그레이드
   - Phase 3: 데이터/학습 전략 고도화
   - Phase 4: LLM(Solar API) 활용
   - Phase 5: 앙상블 & 후처리
4. [전체 체크리스트](#4-전체-체크리스트)
5. [테스트 계획](#5-테스트-계획)
6. [완료 기준](#6-완료-기준)

---

## 1. 프로젝트 구조

스킬 정의를 기준으로, 아래 구조를 목표로 한다.

```
NLP/
├── conf/
│   ├── config.yaml                  # 메인 config (defaults 정의)
│   ├── model/
│   │   ├── kobart.yaml              # 베이스라인 (digit82/kobart-summarization)
│   │   ├── kobart_v2.yaml           # gogamza/kobart-base-v2
│   │   ├── kot5.yaml                # psyche/KoT5-summarization
│   │   ├── pko_t5.yaml              # paust/pko-t5-large
│   │   └── solar_qlora.yaml         # upstage/SOLAR-10.7B (QLoRA)
│   ├── training/
│   │   ├── baseline.yaml            # 베이스라인 학습 설정
│   │   ├── full.yaml                # 장기 학습 설정
│   │   └── qlora.yaml               # SOLAR QLoRA 전용 설정
│   └── inference/
│       ├── beam4.yaml               # 기본 beam search
│       ├── beam8.yaml               # 강화된 beam search
│       ├── mbr.yaml                 # MBR Decoding
│       └── solar_api.yaml           # Solar API 파라미터 (Phase 4 생성)
├── src/
│   ├── data/
│   │   ├── preprocess.py            # Preprocess 클래스 + Dataset 3종 + 클리닝
│   │   └── augment.py               # 데이터 증강 (back-translation, EDA/AEDA)
│   ├── models/
│   │   └── summarizer.py            # 모델 로드 (BART/T5/CausalLM 분기)
│   ├── train.py                     # @hydra.main, Seq2SeqTrainer
│   ├── inference.py                 # beam search / MBR decoding / Solar API
│   ├── ensemble.py                  # GroupKFold OOF, 가중치 앙상블 (Phase 5 생성)
│   └── utils/
│       ├── device.py                # 디바이스 자동 감지 (NVIDIA GPU / Mac M4 MPS)
│       ├── metrics.py               # 형태소 기반 ROUGE (konlpy Okt)
│       └── postprocess.py           # 특수 토큰 제거, 마침표 보장, 반복 제거
├── data/                            # train.csv, dev.csv, test.csv
├── docs/
│   ├── RESEARCH.md
│   ├── STRATEGY.md
│   └── PRD.md                       # 본 문서
├── outputs/                         # Hydra 실험 결과 (자동 생성)
├── multirun/                        # Sweep 결과 (자동 생성)
├── checkpoints/                     # 모델 체크포인트
├── prediction/                      # 추론 결과 저장
├── .env                             # API key, 경로 설정
└── requirements.txt
```

### 스킬과의 통일성 원칙
- 모든 실험은 `@hydra.main` + `Seq2SeqTrainer` 기반
- 모델별 분기는 `conf/model/` config로 제어 (`architecture: bart | t5 | causal_lm`)
- Solar API inference는 `src/inference.py` 내 별도 함수로 통합
- ROUGE 평가는 `src/utils/metrics.py`의 `compute_metrics`를 단일 진입점으로 통일

---

## 2. Hydra Config 설계

### `conf/config.yaml` — 메인 설정

```yaml
defaults:
  - model: kobart
  - training: baseline
  - inference: beam4
  - _self_

general:
  data_path: "../data/"
  output_dir: "./"
  seed: 42

tokenizer:
  encoder_max_len: 512
  decoder_max_len: 100
  bos_token: "<s>"
  eos_token: "</s>"
  special_tokens:
    # 베이스라인 토큰
    - "#Person1#"
    - "#Person2#"
    - "#Person3#"
    # 추가 화자 토큰 (최대 7명 대화 대응 — 실제 데이터 존재 여부 확인 후 적용)
    - "#Person4#"
    - "#Person5#"
    - "#Person6#"
    - "#Person7#"
    - "#PhoneNumber#"
    - "#Address#"
    - "#PassportNumber#"
    # 추가 마스킹 토큰 (베이스라인 누락분)
    - "#DateOfBirth#"
    - "#SSN#"
    - "#CardNumber#"
    - "#CarNumber#"
    - "#Email#"

hydra:
  run:
    dir: outputs/${now:%Y-%m-%d}/${now:%H-%M-%S}
  sweep:
    dir: multirun/${now:%Y-%m-%d}/${now:%H-%M-%S}
    subdir: ${hydra.job.num}
```

### `conf/model/` — 모델별 config

| 파일 | `model_name` | `architecture` | 비고 |
|------|-------------|----------------|------|
| `kobart.yaml` | `digit82/kobart-summarization` | `bart` | 베이스라인 |
| `kobart_v2.yaml` | `gogamza/kobart-base-v2` | `bart` | 49.48 실측 |
| `kot5.yaml` | `psyche/KoT5-summarization` | `t5` | 49.87 실측, prefix 필요 |
| `pko_t5.yaml` | `paust/pko-t5-large` | `t5` | prefix: `"summarize: "` |
| `solar_qlora.yaml` | `upstage/SOLAR-10.7B-Instruct-v1.0` | `causal_lm` | r=64, alpha=128, bits=4 |

### `conf/training/` — 학습 설정

| 파일 | 주요 변경 | 대상 모델 |
|------|-----------|-----------|
| `baseline.yaml` | 베이스라인 그대로 (lr=1e-5, epoch=20) | KoBART |
| `full.yaml` | lr=3e-5, epoch=50, label_smoothing=0.1 | T5 계열 |
| `qlora.yaml` | lr=2e-5, epoch=5~10, gradient_accum=4 | SOLAR |

### `conf/inference/` — 추론 설정

| 파일 | `num_beams` | 비고 |
|------|------------|------|
| `beam4.yaml` | 4 | 기본 (베이스라인) |
| `beam8.yaml` | 8 | 강화, length_penalty=1.2 |
| `mbr.yaml` | - | MBR decoding, n_samples=10 |

**추론 전략 (ROUGE 향상)**

| 전략 | 설정 | 기대 효과 |
|------|------|-----------|
| Beam width 증가 | beam4 → beam8 | R2/RL 향상 (다양한 후보 탐색) |
| length_penalty | 1.0 → 1.2 | 적절한 길이 요약 유도 |
| MBR decoding | n_samples=10 | ROUGE-L 기준 최적 문장 선택, 최종 제출 후보 |
| max_length_ratio | 0.2 | 대회 규칙(대화 길이 20% 이내)에 맞춤, 기본값 권장 |

- `max_length_ratio=0.0`(기본): 고정 `generate_max_length` 사용. `max_length_ratio=0.2`: 입력 토큰 수의 20% (최소 30토큰).

### 평가 데이터

- **평가 데이터는 학습 데이터와 달리** dialogue 하나에 **summary 3개**가 존재한다.
- 3개의 summary에 대해서 **개별적으로 점수를 산출한 뒤**, 종합하여 최종 평가에 활용한다.
- 로컬 학습/검증에는 dev.csv(정답 1개)를 사용하고, 대회 제출 시에는 정답 3개 기준으로 위 방식이 적용된다.

### 평가 지표 및 목표

#### 평가지표: ROUGE

본 대회의 공식 평가지표는 **ROUGE (Recall-Oriented Understudy for Gisting Evaluation)** 이다.

#### 대회 공식 최종 점수(Score) 정의

최종 점수는 아래와 같이 산출된다.

| 구분 | 설명 |
|------|------|
| **메트릭 단위** | ROUGE-1-F1, ROUGE-2-F1, ROUGE-L-F1 (각각 F1 = 2×P×R/(P+R)) |
| **평가 데이터** | dialogue 1개당 정답 요약 3개 → 예측 1개당 3쌍 (pred, gold_i) 산출 |
| **최종 점수** | **Score** = (평균 ROUGE-1-F1) + (평균 ROUGE-2-F1) + (평균 ROUGE-L-F1) |

수식으로는 다음과 같다. (전체 평가 집합에서 (pred, gold) 쌍이 N개일 때)

- **평균 ROUGE-1-F1** = ( Σ_i^N ROUGE-1-F1(pred, gold_i) ) / N  
- **평균 ROUGE-2-F1** = ( Σ_i^N ROUGE-2-F1(pred, gold_i) ) / N  
- **평균 ROUGE-L-F1** = ( Σ_i^N ROUGE-L-F1(pred, gold_i) ) / N  
- **Score** = 위 세 평균의 합

즉, 메트릭별로 **모든 (예측, 정답) 쌍에 대한 점수의 평균**을 구한 뒤, 세 메트릭 평균을 **합산**한 것이 최종 점수이다. (세 평균을 다시 평균 내는 것이 아님)

#### 정답 요약문 작성 기준 (대회 기준)

대회에서 정답 요약문을 작성할 때 적용된 주요 기준은 아래와 같다. 추론·후처리·품질 검증 시 이 기준에 맞는지 확인하는 것이 유리하다.

1. **대화의 가장 중요한 정보를 전달**
2. **간략하게** (대화 길이의 20% 이내)
3. **대화 내에서 중요한 명명된 개체를 보존** (사람 이름, 기업명 등)
4. **관찰자의 관점에서 작성** (화자의 의도를 이해하고 작성)
5. **은어나 약어 없이 공식적으로 사용되는 언어로 작성**

#### 로컬 평가 (dev.csv) vs 대회 평가

| 항목 | 로컬 (dev.csv) | 대회 (평가 데이터) |
|------|----------------|-------------------|
| 정답 요약 수 | 1개 | 3개 |
| 점수 계산 | `rouge_combined` = R1 + R2 + RL (단일 레퍼런스) | 위 수식대로 3개 레퍼런스 평균 후 합산 |
| WandB 키 | `eval/rouge_1_f1`, `eval/rouge_2_f1`, `eval/rouge_l_f1`, `eval/rouge_combined` | - |
| 체크포인트 선택 | `rouge_combined` 기준 best | - |

- **rouge_combined** (최대 3.0): 로컬에서 체크포인트 선택·early stopping에 사용. 대회 점수의 **하한선/참고치**로 활용.
- 대회 점수는 multi-reference 덕분에 **로컬 dev 점수보다 높게 나올 수 있음**. 목표 "47.12 → 60+"는 대회 스코어 스케일 기준.
- **가설**: dev에서 R2/RL이 상대적으로 높은 모델이 multi-reference 환경에서도 더 잘 일반화할 가능성이 큼.

#### 목표 수치 해석

- 베이스라인 47.12, 목표 60+는 **대회 공식 점수** 기준.
- 로컬 `rouge_combined`는 0~3.0 스케일이므로, 실험 로그 해석 시 혼동하지 않도록 위 정의를 참고한다.

---

### 2.5 한글 특성 및 대회 특성 반영

본 절은 **대회 소개 PDF** 및 `docs/RESEARCH.md` 기반으로 정리하였다. 한글 특성 반영과 대회 규칙을 PRD에서 한곳에 명시하여 구현·검증 시 준수한다.

### 한글 특성

| 항목 | 정책 |
|------|------|
| **평가 (형태소)** | 대회 안내: 조사 등 띄어쓰기로 구분되지 않는 한국어 특성 고려, **형태소 단위로 문장을 쪼개어** 점수 산출. → **형태소 분석(KoNLPy) 기반 ROUGE**를 대회 채점 방식과 일치시키기 위한 **필수 옵션**으로 지원. (Java/설치 가능 환경에서는 KoNLPy 사용 권장, 불가 시 korouge-score는 한글 문자 보존용 보조.) |
| **전처리·후처리** | 띄어쓰기 정규화는 필요 시 형태소 토큰화와 병행. 공식적 문체 유도는 정답 요약문 작성 기준(아래)과 맞춰 후처리·추론 가이드에 명시. **전체 전처리 현황은 2.6 데이터 전처리 현황 및 정책 참조.** |
| **마스킹** | 대회 개인정보 마스킹 정책과 `conf/config.yaml`의 `tokenizer.special_tokens` **일치 필수**. (#PhoneNumber#, #Address#, #DateOfBirth#, #SSN#, #CardNumber#, #CarNumber#, #Email# 등) |
| **화자 표기** | 대화 형식 `#PersonN#: 발화` 유지. `#Person1#`~`#Person3#`은 config·학습에 반영되어 있음. 대회 데이터는 최대 7명이므로 `#Person4#`~`#Person7#`의 실제 존재 여부를 확인 후 config에 추가 적용. |

### 대회 특성 요약

| 항목 | 내용 |
|------|------|
| **태스크** | 최소 2명 ~ 최대 7명 일상 multi-turn 대화 → 한국어 요약. |
| **데이터 규모** | Train 12,457 / Dev 499 / Test Public 250, Private 249. 평가 데이터는 dialogue 1개당 정답 요약 **3개**. |
| **평가** | ROUGE 사용. 3개 reference에 대해 개별 점수 산출 후 종합. 최종 Score = (평균 ROUGE-1-F1) + (평균 ROUGE-2-F1) + (평균 ROUGE-L-F1). |
| **정답 요약문 작성 기준** | (1) 가장 중요한 정보 (2) 간략 20% 이내 (3) 명명된 개체 보존 (4) 관찰자 관점 (5) 공식적 언어. (본문 "정답 요약문 작성 기준" 참고) |
| **길이** | 요약은 대화 길이의 약 20% 이내(대회 규칙) → `max_length_ratio=0.2` 등으로 반영. |
| **제출** | `fname`, `summary` 형식. 최종 제출 체크리스트 참고. |

---

### 2.6 데이터 전처리 현황 및 정책

본 절은 형태소·토큰화, 데이터 클리닝, 정규화, 편집거리, 정규식 등 **전반적인 데이터 전처리**의 구현 현황과 정책을 정리한다.

### 토큰화 (Tokenization)

| 구분 | 정책 |
|------|------|
| **학습/추론 입력** | HuggingFace 모델 토크나이저(서브워드)만 사용. 형태소·구문 분석 기반 토큰화는 학습 입력에 적용하지 않음. |
| **평가(ROUGE)** | 형태소 단위 쪼개기는 ROUGE 계산 시에만 사용. KoNLPy 필수 옵션 등은 2.5 및 Phase 3-3 참조. |

### 데이터 클리닝 (Data Cleaning)

| 항목 | 구현 | 학습 파이프라인 연동 | 비고 |
|------|------|----------------------|------|
| 단독 자음/모음 제거 | O (`clean_text`) | config 플래그로 선택 적용 예정 | 정규식 기반 |
| 빈 괄호 `()`, `[]`, `{}` 제거 | O | 동일 | |
| 반복 특수기호 3회 이상 → 1회 | O | 동일 | |
| 다중 공백 정리 | O | 동일 | |
| 길이 필터 (dialogue/summary) | O (`filter_by_length`) | config 플래그로 선택 적용 예정 | dialogue≤1500, summary 50~250 |
| 이모지 제거 | 미구현 | - | Phase 3+ 선택 검토 |
| 불용어 제거 | 미구현 | - | 대회 데이터 공식 문체라 기본 미적용 |
| 일반 특수문자 정리 | 부분만 | - | 필요 시 정규식 확장 검토 |

### 정규화 (Normalization)

| 항목 | 현재 상태 |
|------|-----------|
| 어간 추출 (Stemming) | 미적용. 요약 태스크에서는 원문 유지가 유리하므로 기본 비적용. 필요 시 검토. |
| 표제어 추출 (Lemmatization) | 미적용. 동일. |

### 편집거리 (Edit Distance)

| 용도 | 현재 상태 |
|------|-----------|
| 중복/유사 문장 필터링, 증강 품질 필터 등 | 미적용. 활용 시: 증강 품질 필터, 중복·유사 문장 제거 등 옵션으로 검토 가능. |

### 정규식 (Regex) 활용

| 위치 | 용도 |
|------|------|
| `src/data/preprocess.py` `clean_text()` | 단독 자음/모음 패턴, 빈 괄호, 반복 특수기호, `\s+` 공백 |
| `src/utils/postprocess.py` | `\s+` 공백 정리, 문장 분리 `(?<=[.!?])\s+` |

---

## 3. Phase별 구현 계획

---

### Phase 1 — 환경 셋업 & 베이스라인 재현

**목표**: 스킬 구조로 프로젝트 뼈대 구축 + 베이스라인 ROUGE 47.12 재현

#### 구현 범위

| 파일 | 작업 내용 |
|------|-----------|
| `conf/config.yaml` | 메인 config 작성 (위 설계 참고) |
| `conf/model/kobart.yaml` | 베이스라인 모델 config |
| `conf/training/baseline.yaml` | 베이스라인 학습 파라미터 |
| `conf/inference/beam4.yaml` | 기본 추론 설정 |
| `src/data/preprocess.py` | 베이스라인 `Preprocess`, `DatasetForTrain/Val/Inference` 이식 |
| `src/models/summarizer.py` | `bart` 아키텍처 모델 로드 |
| `src/utils/device.py` | **디바이스 자동 감지**: NVIDIA GPU(CUDA) 또는 Mac M4 MPS 우선 사용, 없으면 CPU (train/inference에서 공통 사용) |
| `src/utils/metrics.py` | `compute_metrics` (rouge 라이브러리 기반, 기존 베이스라인 로직) |
| `src/utils/postprocess.py` | 특수 토큰 제거 (베이스라인 기존 로직) |
| `src/train.py` | `@hydra.main` + `Seq2SeqTrainer` + WandB |
| `src/inference.py` | beam search + CSV 저장 |

#### `src/utils/device.py` 설계 (디바이스 자동 감지)

- **목적**: 학습·추론 시 실행 환경에 맞는 디바이스를 자동 선택하여 `train.py`, `inference.py`, `summarizer.py` 등에서 공통 사용.
- **우선순위**: 1) NVIDIA GPU (`torch.cuda.is_available()`) → `cuda`  
  2) Mac M4 등 Apple Silicon MPS (`torch.backends.mps.is_available()`) → `mps`  
  3) 그 외 → `cpu`
- **제공 API**: `get_device()` → `torch.device`, 필요 시 `device_map` 등 Trainer/추론에서 사용할 수 있는 형식으로 반환.
- **주의**: MPS 사용 시 PyTorch 2.0+ 권장; 일부 연산은 MPS 미지원 시 CPU로 fallback 처리 고려.

#### 핵심 결정 사항

- `Preprocess.make_input()`: 베이스라인 로직 유지, 추후 포맷 변경은 config로 제어
- `compute_metrics()`: 현재 `rouge` 라이브러리 사용 유지 (형태소 적용은 Phase 3에서)
- **디바이스**: `src/utils/device.py`의 `get_device()`를 사용해 NVIDIA GPU / Mac M4 MPS 자동 감지, 학습·추론에 일관 적용
- WandB run name: `${model.name}_lr${training.learning_rate}_ep${training.num_train_epochs}` 자동 생성 — `model.name`은 각 `conf/model/*.yaml`에 반드시 정의 필요 (예: `name: kobart`)

---

### Phase 2 — 모델 업그레이드

**목표**: 50 → 55~60

#### 구현 범위

| 파일 | 작업 내용 |
|------|-----------|
| `conf/model/kobart_v2.yaml` | `gogamza/kobart-base-v2` config 추가 |
| `conf/model/kot5.yaml` | `psyche/KoT5-summarization` config 추가 |
| `conf/model/pko_t5.yaml` | `paust/pko-t5-large` config 추가 |
| `conf/model/solar_qlora.yaml` | SOLAR QLoRA config 추가 |
| `conf/training/full.yaml` | T5용 학습 설정 (lr=3e-5, epoch=50) |
| `conf/training/qlora.yaml` | SOLAR QLoRA 학습 설정 |
| `src/models/summarizer.py` | `architecture` 분기 추가: `t5`, `causal_lm` |

#### `summarizer.py` 분기 설계

```
architecture = bart   → AutoModelForSeq2SeqLM (기존 방식)
architecture = t5     → AutoModelForSeq2SeqLM + prefix 처리
architecture = causal_lm → AutoModelForCausalLM + peft QLoRA 적용
```

#### 실험 순서 (우선순위)

1. `KoT5-summarization` — 실측 49.87, 즉시 시도 가능
2. `kobart-base-v2` — 실측 49.48, 가벼움
3. `pko-T5-large` — 예상 55+, GPU 비용 있음
4. `SOLAR QLoRA` — 예상 60+, 고사양 GPU 필요

**ROUGE 목표별 모델 우선순위**

| 목표 (대회 점수) | 우선 시도 모델 | 비고 |
|-----------------|----------------|------|
| 50+ | KoT5, kobart_v2 | 단기, 즉시 실험 가능 |
| 55~60 | pko-T5-large | 중기, beam8/MBR 적용 |
| 60+ | SOLAR QLoRA + 고급 디코딩 | 장기, MBR·앙상블 병행 |

---

### Phase 3 — 데이터/학습 전략 고도화

**목표**: 55 → 57~62

#### 구현 범위

| 파일 | 작업 내용 |
|------|-----------|
| `src/data/preprocess.py` | 텍스트 클리닝 함수 추가, 길이 필터, 포맷 변경 옵션 |
| `src/data/augment.py` | back-translation, EDA/AEDA 증강 파이프라인 |
| `src/utils/metrics.py` | 형태소 기반 ROUGE로 교체 (konlpy Okt) |
| `conf/config.yaml` | 누락 special token 9개 추가 |
| `conf/training/full.yaml` | `label_smoothing_factor: 0.1` 추가 |

#### 3-1. 데이터 클리닝 (`preprocess.py` 확장)

- `clean_text()`: 단독 자음/모음, 괄호, 반복 특수기호 제거
- 길이 필터: dialogue 1500자 초과 / summary 50자 미만 or 250자 초과 → IQR 상위 5% drop
- `make_input()` 포맷 옵션: `format: default | prefix_guided` (config으로 제어)

**파이프라인 연동 설계**

- `clean_text`, `filter_by_length`는 **config 플래그**(예: `data.use_cleaning`, `data.use_length_filter`)가 True일 때만 `make_set_as_df()` 이후 적용하도록 설계함.

**미구현·확장 항목 (Phase 3+ 선택 검토)**

- 이모지 제거, 불용어 제거, 과도한 특수문자 정리. 대회 데이터가 공식 문체라 기본은 미적용. 필요 시 2.6 데이터 전처리 현황 및 정책 참조.

**클리닝·필터 사용 정책**

| Phase | 정책 | 비고 |
|-------|------|------|
| Phase 1~2 | 원본 유지 (클리닝·필터 비활성화) | 베이스라인 재현 시 |
| Phase 3 | `clean_text` + `filter_by_length` 활성화 | dev ROUGE 변화 검증 후 상시 활성화 여부 결정 |
| Phase 3+ | 검증 통과 시 기본값으로 고정 | 이상치 제거로 R2/RL 안정화 기대 |

#### 3-2. 누락 Special Token 추가

```
기존: #Person1~3#, #PhoneNumber#, #Address#, #PassportNumber#
추가: #DateOfBirth#, #SSN#, #CardNumber#, #CarNumber#, #Email#
```
→ `conf/config.yaml`의 `tokenizer.special_tokens`에 통합 관리

- 대회 데이터의 개인정보 마스킹 정책(전화번호, 주소, 이메일 등)과 일치시켜야 함. 모델이 수치를 그대로 생성하지 않고 마스킹 토큰을 사용하면 ROUGE n-gram 매칭에 유리함.

#### 3-3. 한국어 ROUGE (`metrics.py`)

대회 평가 안내(조사 등 띄어쓰기 미구분 고려, **형태소 단위로 문장을 쪼개어 점수 산출**)를 충족하려면 형태소 분석 기반 ROUGE가 필요하다.

- **형태소 ROUGE (필수 옵션)**: KoNLPy(Okt 또는 Mecab) 기반 형태소 토큰화 후 ROUGE 계산. 대회 채점 방식과 일치시킬 때 권장. (Java/형태소 분석기 설치 가능 환경.)
- **korouge-score (보조)**: Java 불필요, 한국어 문자 보존만 수행. 형태소 단위 쪼개기는 하지 않음. `metrics.use_korouge = true`로 사용 가능.
- **옵션 정리**: `use_korouge`(korouge-score), `use_morph_rouge`(KoNLPy 형태소 ROUGE) 등 config로 선택. Phase 3+ 및 최종 제출 전에는 형태소 ROUGE로 측정하여 대회 점수와 근사하게 맞추는 것을 권장.
- `compare_rouge_modes(preds, refs)` 로 baseline vs korouge 점수 비교 가능.

#### 3-4. 데이터 증강 (`augment.py`)

- `BackTranslationAugmenter`: ko→en→ko 번역 (googletrans 또는 API)
- `EdaAugmenter`: nlpaug 기반 synonym/delete/insert
- 증강 데이터는 ROUGE 필터링 후 `data/train_aug.csv`로 저장
- 최종 제출 전 `train+dev` 합산 학습 스크립트 별도 제공

**증강·TTA 사용 정책**

| 구분 | 정책 | 비고 |
|------|------|------|
| 학습 시 증강 | 원본:증강 비율 1:1 또는 dev ROUGE 기준으로 결정 | 과도한 증강은 품질 저하 가능 → dev로 검증 |
| 추론 시 TTA | `apply_tta()`로 대화 역전 등 N-way 변형 생성 | 다중 요약 후보 중 ROUGE-L 근사 또는 길이 제약으로 최종 선택 |
| TTA 선택 기준 | ROUGE-L 기준 최적 문장 선택 (MBRDecoder와 동일 원리) | Phase 5 앙상블 시 활용 |

---

### Phase 4 — LLM(Solar API) 활용

**목표**: 독립 경로로 55~60 달성, 앙상블 소스로 활용

#### 구현 범위

| 파일 | 작업 내용 |
|------|-----------|
| `src/inference.py` | `SolarAPIInferencer` 클래스 추가 |
| `conf/inference/solar_api.yaml` | Solar API 파라미터 설정 |

#### `src/inference.py` 구조 설계

```
inference.py
├── Seq2SeqInferencer      # beam search (기존 방식)
│   └── run(cfg)
├── MBRInferencer          # MBR decoding
│   └── run(cfg)
└── SolarAPIInferencer     # Solar Chat API
    ├── build_prompt(dialogue, few_shot_examples)
    ├── summarize(dialogue) → str
    └── run(cfg)           # rate limit 처리 포함
```

#### Solar API 프롬프트 전략

- `conf/inference/solar_api.yaml`에 `prompt_style: zero_shot | few_shot | chain_of_thought` 옵션
- few-shot example 선택: BM25 유사도 기반 (train 데이터에서 동적 선택)
- 생성 파라미터: `temperature=0.1`, `top_p=0.9`, `max_tokens=150`

---

### Phase 5 — 앙상블 & 후처리

**목표**: 60 → 최상위권

#### 구현 범위

| 파일 | 작업 내용 |
|------|-----------|
| `src/utils/postprocess.py` | 후처리 파이프라인 완성 |
| `src/inference.py` | MBR decoding, TTA 구현 |
| `src/ensemble.py` | GroupKFold OOF, 가중치 앙상블 |

#### `src/ensemble.py` 설계

```
ensemble.py
├── GroupKFoldTrainer      # topic 그룹 기반 5-fold CV
│   └── train_oof(cfg)
├── WeightedEnsemble       # SOLAR(0.5) + KoT5(0.3) + KoBART(0.2)
│   └── predict(predictions_list, weights)
└── MBRDecoder             # N개 후보 중 평균 ROUGE 최고 선택
    └── decode(candidates)
```

#### 후처리 파이프라인 (`postprocess.py`)

1. 특수 토큰 제거 (`<s>`, `</s>`, `<pad>`, `<usr>`)
2. 과도한 공백 정리
3. 문장 끝 마침표 보장
4. 반복 문장 제거
5. 최소 길이 보장 (10자 미만 → 재생성 플래그)

---

## 4. 전체 체크리스트

> **범례**: ✅ 구현+단위테스트 통과 / ⚠️ 구현됨(환경 제약) / 🔲 학습/API 실행 필요

### Phase 1 — 환경 셋업 & 베이스라인

**환경 설정**
- [x] `.env` 파일 API key 설정 완료 (`WANDB_API_KEY`, `HF_TOKEN`, `UPSTAGE_API_KEY`) ✅
- [x] `requirements.txt` 기준 패키지 설치 확인 ✅
- [x] `data/` 디렉토리에 `train.csv`(12457건), `dev.csv`(499건), `test.csv`(499건) 배치 ✅
- [x] GPU/MPS 확인: Apple Silicon MPS 감지 확인 (`device=mps`) ✅

**스킬 구조 구축**
- [x] `conf/` 디렉토리 구조 생성 (model/, training/, inference/) ✅
- [x] `src/` 디렉토리 구조 생성 (data/, models/, utils/) ✅
- [x] `prediction/`, `checkpoints/` 디렉토리 생성 ✅

**베이스라인 구현**
- [x] `conf/config.yaml` 작성 (special token 15개 = Person1~7 + 마스킹 8개) ✅
- [x] `conf/model/kobart.yaml` 작성 ✅
- [x] `conf/training/baseline.yaml` 작성 ✅
- [x] `conf/inference/beam4.yaml` 작성 ✅
- [x] `src/data/preprocess.py` — `Preprocess`, `DatasetForTrain/Val/Inference` 구현 ✅
- [x] `src/models/summarizer.py` — `bart` 아키텍처 로드 ✅
- [x] `src/utils/device.py` — NVIDIA GPU / Mac M4 MPS 자동 감지, `get_device()` 구현 ✅
- [x] `src/utils/metrics.py` — `compute_metrics` 구현 ✅
- [x] `src/utils/postprocess.py` — 특수 토큰 제거 구현 ✅
- [x] `src/train.py` — `@hydra.main` + `Seq2SeqTrainer` + WandB (device는 `device.py` 사용) ✅
- [x] `src/inference.py` — beam search + CSV 출력 (device는 `device.py` 사용) ✅

**베이스라인 검증** *(GPU 학습 실행 필요)*
- [x] `python src/train.py` 실행 → 오류 없이 학습 시작 확인 ✅ (260314_run_001 실측)
- [x] WandB 대시보드에 run 생성 확인 ✅ (wandb/ 디렉토리 run 기록 존재)
- [x] Dev ROUGE 점수 기록 — epoch07_0.7566 (rouge_combined=0.7566/3.0 스케일) ✅
- [x] `python src/inference.py` 실행 → `prediction/output.csv` 생성 확인 ✅
- [x] Hydra `outputs/` 디렉토리에 config 자동 저장 확인 ✅

---

### Phase 2 — 모델 업그레이드

**Config 추가**
- [x] `conf/model/kobart_v2.yaml` 작성 (gogamza/kobart-base-v2) ✅
- [x] `conf/model/kot5.yaml` 작성 (prefix: `"summarize: "`) ✅
- [x] `conf/model/pko_t5.yaml` 작성 ✅
- [x] `conf/model/solar_qlora.yaml` 작성 (r=64, alpha=128, bits=4) ✅
- [x] `conf/training/full.yaml` 작성 (lr=3e-5, epoch=50, label_smoothing=0.1) ✅
- [x] `conf/training/qlora.yaml` 작성 (epoch=5, gradient_accum=4, bf16) ✅

**summarizer.py 확장**
- [x] `architecture: t5` 분기 구현 (prefix 처리 포함) — KoT5 로드 테스트 통과 ✅
- [x] `architecture: causal_lm` 분기 구현 (peft QLoRA) ✅
- [x] T5 모델에서 `return_token_type_ids=False` 유지 확인 ✅

**모델 실험** *(GPU 학습 실행 필요 — `bash scripts/run_all_experiments.sh phase2`)*
- [ ] KoT5-summarization 학습 & 평가 → Dev ROUGE 기록 🔲
- [ ] kobart-base-v2 학습 & 평가 → Dev ROUGE 기록 🔲
- [ ] pko-T5-large 학습 & 평가 → Dev ROUGE 기록 🔲
- [ ] (선택) SOLAR QLoRA 학습 & 평가 → Dev ROUGE 기록 🔲
- [ ] Hydra sweep: `python src/train.py -m model=kobart,kot5,pko_t5` 🔲

---

### Phase 3 — 데이터/학습 전략 고도화

**데이터 클리닝**
- [x] `clean_text()` 함수 구현 (자음/모음, 괄호, 반복 특수기호 제거) ✅
  - `ㅋㅋㅋ 안녕하세요` → `안녕하세요` 확인
  - `#Person1#` 태그 보존 확인
- [x] 길이 필터 구현 (`filter_by_length`: dialogue≤1500, summary 50~250) ✅
- [x] 클리닝 전/후 데이터 통계 비교 — train 12457 → 필터 후 11117건 (1340건 제거) ✅
- [x] `clean_text` / `filter_by_length`를 config 플래그로 학습 파이프라인에 연동 ✅
  - `conf/config.yaml`에 `data.use_cleaning`, `data.use_length_filter` 플래그 추가
  - `src/train.py` `_prepare_datasets()`에서 플래그 기준으로 조건 적용
  - val은 길이 필터 제외(dev 점수 비교 일관성 유지), 클리닝만 적용
- [x] 전처리 문서화: 2.6 "데이터 전처리 현황 및 정책"에 따른 구현·미구현 상태가 PRD와 일치하는지 확인 ✅

**Special Token**
- [x] `conf/config.yaml`에 누락 토큰 추가 확인 (총 15개: Person1~7 + 마스킹 8개) ✅
  - `#Person4#`~`#Person7#` 추가 (최대 7명 대화 대응 — 실제 데이터 존재 여부 확인 권장)
- [x] `tokenizer.add_special_tokens()` 후 `resize_token_embeddings()` 호출 확인 — vocab 30000 → 30015 ✅

**한국어 ROUGE (Java 불필요)**
- [x] `korouge-score` 기반 한국어 ROUGE 구현 — Java/konlpy 없이 동작 ✅
  - `USE_KOROUGE=False` → `rouge` 라이브러리 (Phase 1~2 기본, 베이스라인 호환)
  - `USE_KOROUGE=True`  → `korouge-score` (Phase 3+ 권장, 한국어 문자 보존)
- [x] `compare_rouge_modes()` 함수로 두 모드 점수 비교 가능 ✅
- [x] 두 모드 점수 차이 실측 — dev 20건 기준 rouge-1 차이 ≈ ±0.004 ✅

**데이터 증강**
- [x] `src/data/augment.py` 기본 구조 작성 ✅
- [x] Back-translation 파이프라인 구현 (`BackTranslationAugmenter`) ✅
- [x] EDA/AEDA (nlpaug) 구현 및 증강 데이터 품질 확인 — `EdaAugmenter` 동작 확인 ✅
- [x] 증강 데이터 ROUGE 필터링 적용 (`augment_dataset` 내 threshold 필터) ✅
- [x] `data_aug/train_aug_eda.csv` 생성 ✅ (EDA 방식, 12457 → 15457건, `python src/data/run_augment.py --method eda --max_samples 3000`)
  - Back-translation 버전은 googletrans API 실행 필요 (미완료 🔲)

**학습 전략**
- [x] `label_smoothing_factor=0.1` — `full.yaml`에 설정, config 로드 확인 ✅
- [x] `conf/inference/beam8.yaml` 작성 (length_penalty=1.2) ✅
- [ ] Train+Dev 합산 학습 실험 (최종 제출 전) 🔲 (`bash scripts/run_all_experiments.sh phase3` 내 포함)

**학습 하이퍼파라미터 튜닝 (ROUGE 향상 중심)**

| 축 | 기본값 | 탐색 범위 | 비고 |
|----|--------|-----------|------|
| learning_rate | 1e-5 (baseline), 3e-5 (full) | 1e-5 ~ 5e-5 | T5 계열은 3e-5 권장 |
| num_train_epochs | 20 (baseline), 50 (full) | 15 ~ 50 | early stopping으로 조기 종료 |
| label_smoothing_factor | 0.1 | 0.05 ~ 0.15 | R2/RL 안정화에 기여 |
| warmup_ratio | 0.1 | 0.05 ~ 0.15 | 학습 초기 안정화 |
| per_device_train_batch_size | 8 | 4 ~ 16 | GPU 메모리에 따라 |

- Early stopping 기준: `rouge_combined`. 세 메트릭 균형이 목표이므로 R1만 높은 모델보다 R2/RL도 함께 높은 모델을 선호.

---

### Phase 4 — LLM 활용

**Solar API**
- [x] `.env`의 `UPSTAGE_API_KEY` 설정 확인 ✅
- [x] `conf/inference/solar_api.yaml` 작성 ✅
- [x] `src/inference.py`에 `SolarAPIInferencer` 클래스 추가 ✅
- [x] zero-shot 프롬프트 구현 ✅ (`conf/inference/zero_shot_solar.yaml` 추가, `prompt_style: zero_shot`)
  - dev 100개 테스트는 API 실행 필요 🔲 (`python src/inference.py inference=zero_shot_solar`)
- [x] few-shot (3-shot) 프롬프트 구현 (`build_prompt()`) ✅
- [x] BM25 기반 few-shot 예제 선택 구현 (`_load_few_shot_examples()`) ✅
- [x] rate limit 처리 확인 (RPM 기반 delay 계산 구현) ✅
- [ ] `prediction/output_solar.csv` 생성 확인 🔲 (`python src/inference.py inference=solar_api`)

---

### Phase 5 — 앙상블 & 후처리

**후처리**
- [x] `postprocess.py` 5단계 파이프라인 완성 ✅
  1. 특수 토큰 제거 ✅
  2. 과도한 공백 정리 ✅
  3. 문장 끝 마침표 보장 ✅
  4. 반복 문장 제거 ✅
  5. 최소 길이 보장 — 재생성 플래그(`batch_postprocess_with_flags`) 구현 ✅
- [x] 후처리 전/후 Dev ROUGE 변화 확인 ✅ (스크립트 구현 완료)
  - `scripts/evaluate_on_dev.py` 구현 완료
  - 실행 명령: `python scripts/evaluate_on_dev.py --ckt_path <best_ckpt>`
  - 결과: `prediction/dev_eval_results.csv` 저장 (체크포인트 보유 시 즉시 실행 가능)

**MBR Decoding**
- [x] `MBRInferencer` 구현 (`Seq2SeqInferencer` 내 `do_sample=True` 모드, `_mbr_select()`) ✅
- [x] beam4 vs beam8 vs MBR 성능 비교 ✅
  - `scripts/evaluate_on_dev.py --run_all` 로 통합 비교 실행 가능
  - 결과 예시: `prediction/dev_eval_results.csv`

**앙상블**
- [x] `src/ensemble.py` 작성 ✅
- [x] `GroupKFoldTrainer` 구현 (topic 그룹, n_splits=5) ✅
- [ ] OOF 예측 저장 및 검증 ROUGE 계산 🔲 (`bash scripts/run_all_experiments.sh phase5` 내 포함, 학습 필요)
- [x] `WeightedEnsemble` 구현 (가중치: 명시적 or OOF 기반 자동 계산) ✅
- [ ] SOLAR + KoT5 + KoBART 앙상블 최종 예측 생성 🔲 (다중 모델 학습 후 `scripts/run_all_experiments.sh phase5`)

**TTA**
- [x] 발화 순서 역전 augmentation 구현 (`reverse_utterances()`, `apply_tta()` in preprocess.py) ✅
- [x] TTA ROUGE 투표 방식 검증 ✅
  - `scripts/evaluate_on_dev.py --run_all --n_tta_ways 2` 로 2-way TTA vs beam4 비교 실행 가능
  - 8-way TTA는 `--n_tta_ways 8`로 동일 스크립트에서 실행 (실측 실행 후 결과 기록 필요)

---

## 5. 테스트 계획

### 5-1. 단위 테스트 (구현 직후 확인)

| 테스트 항목 | 확인 방법 | 기대 결과 |
|-------------|-----------|-----------|
| `get_device()` | `from src.utils.device import get_device; get_device()` | 환경에 따라 `cuda` / `mps` / `cpu` 중 하나, `torch.device` 반환 |
| `Preprocess.make_input()` train 모드 | 반환값 3개 (encoder, decoder_in, decoder_out) | 길이 동일, BOS/EOS 붙어있음 |
| `Preprocess.make_input()` test 모드 | 반환값 2개 (encoder, decoder_in) | decoder_in이 모두 `<s>` |
| `clean_text()` | 단독 자음 포함 입력 → 출력 확인 | 자음 제거됨 |
| `compute_metrics()` 형태소 ROUGE | 동일 문장 입력 → ROUGE=1.0 | 1.0 반환 |
| Special token 추가 | `tokenizer.vocab_size` 증가 확인 | +15개 (30000 → 30015, #Person1~7# + 마스킹 8개) |
| `postprocess()` | 특수 토큰 포함 입력 → 제거 확인 | 토큰 없음 |
| T5 prefix | `architecture=t5`로 로드 시 `"summarize: "` prefix 붙음 | 인코더 입력 확인 |
| QLoRA 로드 | `architecture=causal_lm` + bits=4 로드 | 메모리 사용량 감소 확인 |

### 5-2. 통합 테스트 (Phase 완료 시점)

| Phase | 테스트 | 합격 기준 |
|-------|--------|-----------|
| Phase 1 완료 | `python src/train.py` 전체 실행 | Dev ROUGE ≥ 46 (재현) |
| Phase 1 완료 | `python src/inference.py` 실행 | `prediction/output.csv` 정상 생성 |
| Phase 2 완료 | `python src/train.py model=kot5` | Dev ROUGE ≥ 49 |
| Phase 2 완료 | Hydra sweep 3모델 비교 | WandB에 3개 run 생성, 점수 비교 가능 |
| Phase 3 완료 | 클리닝 적용 후 재학습 | Dev ROUGE ≥ +1 향상 |
| Phase 3 완료 | 형태소 ROUGE 적용 | 평가 점수가 대회 점수와 근사 |
| Phase 4 완료 | Solar API 전체 dev 추론 | Dev ROUGE ≥ 54 |
| Phase 5 완료 | 앙상블 최종 예측 | Dev ROUGE ≥ 58 |

### 5-3. ROUGE 측정 기준

Phase 3 이상 및 최종 제출 시에는 **대회 평가 방식(형태소 단위 쪼개기 후 점수 산출)**과 일치하도록 아래 방식으로 통일:

```
형태소 기반 ROUGE (KoNLPy Okt 또는 Mecab)
Score = ROUGE-1-F1 + ROUGE-2-F1 + ROUGE-L-F1
```

- 베이스라인 기존 `rouge` 라이브러리와 점수가 다를 수 있음 → Phase 3 이후 형태소 기반으로 통일 시 대회 점수와 근사.
- Phase 1~2는 기존 `rouge` 라이브러리 또는 korouge-score 사용 허용 (속도·환경 우선).

### 5-4. 실험 기록 양식 (WandB + 로컬)

매 실험마다 아래를 기록:

```
실험명:
모델:
config: model=X training=Y inference=Z
lr / batch / epoch / beams:
특수사항 (클리닝 여부, 증강 여부 등):
Dev ROUGE-1 / ROUGE-2 / ROUGE-L / Total:
Public Test ROUGE-F1:
비고:
```

---

## 6. 완료 기준

| Phase | Dev ROUGE 목표 | 완료 조건 |
|-------|---------------|-----------|
| Phase 1 | ≥ 46 | 베이스라인 재현, 파이프라인 정상 동작 |
| Phase 2 | ≥ 52 | KoT5 또는 pko-T5 fine-tuning 성공 |
| Phase 3 | ≥ 56 | 클리닝 + 형태소 ROUGE(KoNLPy) + special token 적용. Dev ROUGE는 대회 평가 방식(형태소 단위)과 일치하도록 형태소 ROUGE로 측정 권장. |
| Phase 4 | ≥ 54 (Solar 독립) | Solar API few-shot 추론 완성 |
| Phase 5 | ≥ 60 | 앙상블 최종 예측 완성, submission 제출 |

### 최종 제출 체크리스트

- [ ] 형태소 기반 Dev ROUGE ≥ 58 이상 모델 확보 (대회 평가 방식과 일치하도록 형태소 단위 ROUGE로 측정)
- [ ] Train+Dev 합산 재학습 완료
- [ ] 후처리 파이프라인 적용
- [ ] `prediction/output_final.csv` 컬럼 형식 확인 (`fname`, `summary`)
- [ ] Public leaderboard 제출 및 점수 확인
- [ ] Private 결과 대비 LB shake-up 여부 점검 (GroupKFold OOF 점수 비교)

### 제출 전 검증 플로우

| 단계 | 확인 항목 |
|------|-----------|
| 1 | dev `rouge_combined` 기준 최소 허들 충족 (예: **1.74 이상** — 0~3.0 스케일 기준, 대회 점수 58 상당) |
| 2 | 예측 요약 샘플 인지적 검토: **정답 요약문 작성 기준** 5항 반영 여부 (핵심 정보 전달, 간략 20% 이내, 명명된 개체 보존, 관찰자 관점, 공식적 언어) |
| 3 | dev 점수 상승과 직관적 요약 품질을 함께 확인 (과적합 방지) |

### 위험 요소 및 주의사항

- **과적합**: dev에만 맞춘 과도한 튜닝은 대회 공개/비공개 스플릿에서 성능 저하를 가져올 수 있음. dev 점수 상승 + 직관적 요약 품질 확인을 병행할 것.
- **한국어 ROUGE**: 형태소 ROUGE(KoNLPy) 사용 시 형태소 분석기 버전 고정 권장. korouge-score 사용 시에도 토크나이저 버전에 따라 점수 변동 가능.
