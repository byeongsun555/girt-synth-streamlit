# Synthetic Data Service

React + FastAPI 기반 합성 데이터 시연 서비스입니다. 기존 Streamlit 앱도 백업용으로 함께 유지됩니다.

- 산불: `image + prompt` 방식의 FLUX Kontext 이미지 생성
- 발 병변: `foot_make_dataset` 파인튜닝 모델 기반 인페인팅
- 저장소: MinIO 기반 합성데이터 이미지/메타데이터 저장 및 최근 결과 조회

## 실행

```bash
cd /mnt/volume3/multi_synth_gradio
docker compose up -d --build
```

브라우저:

```text
React 시연 화면: http://localhost:5173
API 문서: http://localhost:8000/docs
MinIO 콘솔: http://localhost:9021
Streamlit 백업 화면: http://localhost:8501
```

MinIO 기본 계정은 별도 설정이 없으면 아래 값을 사용합니다.

```text
ID: minioadmin
PW: minioadmin
Bucket: synthetic-data
```

코드만 바뀐 경우:

```bash
cd /mnt/volume3/multi_synth_gradio
docker compose restart synth-api synth-web
```

산불 탭은 Hugging Face gated model인 `black-forest-labs/FLUX.1-Kontext-dev`를 사용합니다. 처음 실행 전 Hugging Face에서 모델 접근 동의를 완료하고 토큰을 넘겨주세요.

```bash
export HF_TOKEN=hf_xxx
export MINIO_ACCESS_KEY=minioadmin
export MINIO_SECRET_KEY=minioadmin
cd /mnt/volume3/multi_synth_gradio
docker compose up -d --force-recreate
```

브라우저에서 MinIO presigned URL을 열 수 있도록 `MINIO_PUBLIC_ENDPOINT`는 호스트에서 접근 가능한 주소로 둡니다. 기본값은 `localhost:9020`입니다.

## 웹 구조

```text
web/backend/   FastAPI 모델 호출 API
web/frontend/  React 19 + TypeScript + Vite + TailwindCSS + Base UI 화면
minio-data/    MinIO 객체 저장 볼륨
```

## 탭 구성

### 산불 이미지 생성

원본 이미지를 업로드하고 프롬프트를 입력하면 image-to-image 방식으로 산불 장면을 생성합니다.

현재 모델:

```text
black-forest-labs/FLUX.1-Kontext-dev
```

결과 저장 위치:

```text
outputs/wildfire/
MinIO: synthetic-data/wildfire/
```

### 발 병변 인페인팅

발 이미지를 업로드하면 512x512 캔버스 위에 마스크를 그릴 수 있습니다. 먼저 `마스크 미리보기`를 완료한 뒤 `발 병변 합성 실행`을 누르면, 앱 내부 프롬프트 풀에서 요청한 생성 수만큼 프롬프트를 뽑아 여러 장을 생성합니다.

사용 모델:

```text
foot_make_dataset/output/corn_inpaint_sd15_200_4_5e-6_newcap
foot_make_dataset/output/crack_inpaint_sd15_200_4_5e-6
```

결과 저장 위치:

```text
outputs/foot/
MinIO: synthetic-data/foot/
```

## 저장 및 관리

생성 결과는 기존 로컬 `outputs/`에 저장하면서, MinIO가 사용 가능하면 같은 run 단위로 이미지와 JSON 메타데이터를 업로드합니다. React 화면 상단에서 MinIO 연결 상태를 확인할 수 있고, 하단 `저장된 합성데이터` 영역에서 최근 결과 이미지와 메타데이터를 바로 열 수 있습니다.

MinIO가 일시적으로 내려가도 모델 생성은 계속 진행됩니다. 이 경우 로컬 저장은 유지되고, 화면에는 저장소 연결 오류가 표시됩니다.

## 주요 보정

발 병변 인페인팅의 `dilate`와 `feather`는 마스크 미리보기와 실제 생성 모두에 동일하게 적용됩니다. 따라서 미리보기에서 확인한 마스크 영역 기준으로 결과가 생성됩니다.
