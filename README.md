# Synthetic Data Service

산불(wildfire)과 발 병변(foot lesion) 합성 이미지를 동일한 웹 UI에서 생성·저장·라벨링하는 서비스입니다.
FastAPI 백엔드 + React/Vite 프론트엔드 + MinIO 객체 저장소로 구성되며, Docker Compose로 일괄 배포합니다.

---

## 폴더 구조

```
multi_synth_streamlit/
├── docker-compose.yml          # 모든 서비스(MinIO/API/Web/Streamlit) 정의
├── Dockerfile                  # Streamlit 데모(레거시)
├── .env.example                # 환경 변수 템플릿
├── app.py                      # Streamlit 데모 앱 (레거시)
├── wildfire_mask_test.py       # 산불 마스크 단독 실험 스크립트
├── README.md
│
├── web/                        # 메인 서비스
│   ├── backend/
│   │   ├── Dockerfile
│   │   ├── main.py             # FastAPI 엔드포인트
│   │   ├── synth_service.py    # 합성 파이프라인 (FLUX/SD1.5/CLIP-Seg)
│   │   ├── storage_service.py  # MinIO 업로드/리스트
│   │   └── requirements.txt
│   └── frontend/
│       ├── Dockerfile
│       ├── package.json
│       ├── vite.config.ts
│       └── src/
│           ├── main.tsx        # UI 전체
│           ├── components/MaskCanvas.tsx
│           └── lib/api.ts      # 백엔드 호출
│
└── outputs/                    # 로컬 결과 (gitignore)
    ├── wildfire/
    └── foot/
```

외부 마운트(컨테이너에서만):
```
../foot_make_dataset/   # 발 병변 파인튜닝 체크포인트
../diffusers/           # (레거시 Gradio 실험 코드, 읽기 전용)
```

---

## 실행 방법

### 사전 준비

1. **HuggingFace 토큰** — FLUX 모델은 gated이므로 [HF 토큰](https://huggingface.co/settings/tokens) 발급 + 모델 페이지에서 접근 동의
   - [FLUX.1-Fill-dev](https://huggingface.co/black-forest-labs/FLUX.1-Fill-dev)
   - [FLUX.1-Kontext-dev](https://huggingface.co/black-forest-labs/FLUX.1-Kontext-dev)
2. **GPU** — NVIDIA GPU + 최신 driver (코드에는 A100 80GB 기준 설정. `docker-compose.yml`의 `device_ids`로 사용할 GPU 지정)
3. **Docker / Docker Compose**

### 시작

```bash
git clone git@github.com:byeongsun555/girt-synth-streamlit.git
cd girt-synth-streamlit

cp .env.example .env
# .env 안의 HF_TOKEN 값 채우기

docker compose up -d
```

서비스 접속:

| 서비스 | URL | 설명 |
|---|---|---|
| **Web UI** | http://localhost:5173 | 메인 서비스 (React) |
| API | http://localhost:8000/docs | FastAPI Swagger |
| MinIO Console | http://localhost:9021 | 객체 저장소 관리자 (계정: minioadmin / minioadmin) |
| Streamlit (레거시) | http://localhost:8501 | 초기 데모 |

### 첫 실행 시

- HuggingFace 모델(FLUX 약 24GB, CLIP-Seg 150MB)이 `.cache/huggingface/`에 다운로드됩니다. 첫 추론은 5~10분 소요.
- MinIO `synthetic-data` 버킷이 자동 생성됩니다.

### 코드 변경 반영

- 백엔드 코드: 컨테이너 재시작 필요 → `docker compose restart synth-api`
- 프론트 코드: Vite HMR로 자동 반영

---

## 산불 합성 (Wildfire)

원본 이미지에 사용자가 마스크를 그리면 해당 식생 영역에 자연스러운 화염·연기를 합성합니다.

### 핵심 기술

| 항목 | 내용 |
|---|---|
| **모델** | `black-forest-labs/FLUX.1-Fill-dev` (정밀 마스크 inpainting) + `black-forest-labs/FLUX.1-Kontext-dev` (자연어 편집, 옵션) |
| **마스크 정제** | `CIDAS/clipseg-rd64-refined` (CLIP-Seg)로 사용자 마스크 안에서 식생만 추출, 비식생(도로·건물·물) 자동 제외 |
| **합성 모드** | `balanced` / `flame_dominant` / `full_flame` / `smoke_dominant` 4종 |
| **마스크 외곽 perturbation** | 마스크 모양 그대로 화염 띠가 되지 않도록 외곽 흐트러뜨림 |
| **Smoke headroom** | 마스크 위쪽으로 alpha gradient 추가하여 연기가 자연스럽게 위로 페이드 |

### 처리 흐름

```
원본 + 사용자 마스크
    ↓
CLIP-Seg ensemble(식생 - 비식생) → refined mask
    ↓
mask perturb (외곽 흐트러뜨림 + smoke headroom)
    ↓
crop 영역 추출 (마스크 bbox + 패딩)
    ↓
FLUX Fill 추론 (work resolution 768~1024)
    ↓
원본 해상도에 alpha composite
    ↓
MinIO + 로컬 저장 + 메타데이터 JSON
```

### 주요 파라미터

| 파라미터 | 기본값 | 설명 |
|---|---|---|
| `Fill Guidance` (CFG) | **9** | 자연스러움 / 프롬프트 충실도 균형 (낮을수록 자유, 높을수록 strict) |
| `Steps` | 34 | denoising 반복 횟수 |
| `Smoke Area` (headroom %) | 30 | 연기가 위쪽으로 확장될 비율 |
| `Vegetation Threshold` | 0.30 | CLIP-Seg 식생 인식 임계값 |
| `Vegetation Min Coverage` | 0.15 | 정제 fallback 발생 임계 |
| `Mask Dilate / Feather` | 2 / 8 | 마스크 후처리 |

---

## 발 병변 합성 (Foot Lesion)

플랜타 발 사진에 사용자가 마스크를 그리면 티눈(corn) 또는 균열(crack)을 합성합니다.

### 핵심 기술

| 항목 | 내용 |
|---|---|
| **모델** | 파인튜닝된 Stable Diffusion 1.5 Inpainting 체크포인트 (`foot_make_dataset/output/corn_inpaint_sd15_*`, `crack_inpaint_sd15_*`) |
| **학습 데이터** | 임상 사진 + LLaVA-Med 자동 캡션 (`*_captions_llava_med_v1.jsonl`) |
| **프롬프트 풀** | 모델별 다중 의학 표현 (병변 위치/크기/표면 묘사)을 자동 셔플 |
| **자동 다수 생성** | 1회 실행 시 N장(기본 5장) 자동 생성, seed는 base+1,2,... |

### 처리 흐름

```
발 사진 + 사용자 마스크
    ↓
512×512 표준화
    ↓
N개 프롬프트 자동 선택 (seed 기반 셔플)
    ↓
SD1.5 inpainting × N회
    ↓
MinIO 업로드 (입력/마스크/preview/결과 N장)
    ↓
대표 이미지 = result_01.png
```

### 주요 파라미터

| 파라미터 | 기본값 | 설명 |
|---|---|---|
| Model | corn / crack | 병변 유형 선택 |
| 생성 수 | 5 | 한 번에 만들 결과 수 |
| Steps | 50 | SD1.5 denoising 스텝 |
| Guidance Scale | 6.5 | SD1.5 CFG |
| Strength | 0.85 | 마스크 영역의 노이즈 비율 |
| Mask Dilate / Feather | 5 / 4 | 마스크 후처리 |

---

## 데이터 저장

### MinIO 구조

```
synthetic-data/
├── wildfire/
│   └── wildfire_<timestamp>_seed<N>/
│       ├── image.png
│       ├── event_region_mask.png        # 사용자 마스크
│       ├── generation_edit_mask.png     # perturbation + headroom 후
│       └── metadata.json
│
└── foot/
    └── <model>/
        └── <model>_<timestamp>_seed<N>/
            ├── input.png
            ├── mask.png
            ├── preview.png              # 마스크 오버레이
            ├── images/result_{01..N}.png
            ├── metadata/result_{01..N}.json
            └── metadata.json            # run-level
```

### 메타데이터(JSON) 예시

각 생성마다 모델/시드/스텝/가이던스/프롬프트/마스크 정제 정보가 자동 기록되어 데이터셋 라벨링·재현에 활용 가능합니다.

```json
{
  "run_id": "wildfire_20260615_120306_seed109808443",
  "model": "black-forest-labs/FLUX.1-Fill-dev",
  "seed": 109808443,
  "steps": 34,
  "fill_guidance": 9.0,
  "composition_mode": "flame_dominant",
  "selected_prompt_variant": "...",
  "mask_refinement": {
    "applied": true,
    "coverage_ratio": 0.51,
    "nonveg_pixels_removed": 689,
    "positive_queries": ["vegetation, ..."],
    "negative_queries": ["road, ..."]
  },
  "final_prompt": "...",
  "storage_image": "wildfire/.../image.png"
}
```

---

## API 엔드포인트

| Method | Path | 용도 |
|---|---|---|
| GET | `/api/health` | 디바이스 / 모델 / MinIO 상태 |
| GET | `/api/defaults` | UI 기본값 |
| POST | `/api/wildfire/generate` | 산불 합성 (multipart) |
| POST | `/api/foot/preview` | 발 마스크 미리보기 |
| POST | `/api/foot/generate` | 발 병변 다중 생성 |
| GET | `/api/datasets/recent` | 최근 합성 데이터셋 목록 |
| GET | `/api/storage/object?object_name=...` | MinIO 객체 proxy 다운로드 |

자세한 스키마는 http://localhost:8000/docs 참고.

---

## 라이선스 / 모델 사용 정책

- **FLUX.1-Fill-dev**, **FLUX.1-Kontext-dev**: Black Forest Labs Non-Commercial License (연구·테스트 용도). 상업적 사용은 별도 라이선스 필요.
- **CLIP-Seg** (`CIDAS/clipseg-rd64-refined`): 자유 사용.
- **Stable Diffusion 1.5**: CreativeML Open RAIL-M.
- 발 병변 학습 데이터는 외부 임상 데이터 기반이므로 별도 IRB·데이터 사용 동의를 따라야 합니다.

---

## 알려진 한계

- 마스크가 도로/건물 위에 그려지면 모델이 그 영역에 char/scorched 효과를 그릴 수 있어, CLIP-Seg negative 쿼리로 보정하지만 완벽하진 않습니다.
- 작은 마스크에서는 화염이 단일 덩어리로 압축되는 경향. 식생 영역을 따라 조금 더 넓게 그리는 것을 권장.
- FLUX Fill의 `Fill Guidance`는 7~12 범위가 자연스럽고, 20+에서는 stickerlike 인공물이 늘어남.

---

## 컨테이너 관리

```bash
# 전체 시작
docker compose up -d

# 백엔드만 재시작 (코드 변경 후)
docker compose restart synth-api

# 로그 확인
docker compose logs -f synth-api

# 정지
docker compose down
```
