from __future__ import annotations

import io

from fastapi import FastAPI, File, Form, HTTPException, Query, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import Response
from PIL import Image

from synth_service import (
    DEVICE,
    FOOT_DEFAULT_DILATE,
    FOOT_DEFAULT_FEATHER,
    FOOT_DEFAULT_GUIDANCE,
    FOOT_DEFAULT_STEPS,
    FOOT_DEFAULT_STRENGTH,
    FOOT_MODEL_CONFIGS,
    WILDFIRE_COMPOSITION_LABELS,
    WILDFIRE_DEFAULT_FILL_GUIDANCE,
    WILDFIRE_DEFAULT_GUIDANCE,
    WILDFIRE_DEFAULT_MASK_DILATE,
    WILDFIRE_DEFAULT_MASK_FEATHER,
    WILDFIRE_DEFAULT_OUTPUT_SIZE,
    WILDFIRE_DEFAULT_SEED,
    WILDFIRE_DEFAULT_SMOKE_HEADROOM,
    WILDFIRE_DEFAULT_STEPS,
    WILDFIRE_DEFAULT_ENGINE,
    WILDFIRE_DEFAULT_VEG_MIN_COVERAGE,
    WILDFIRE_DEFAULT_VEG_REFINE,
    WILDFIRE_DEFAULT_VEG_THRESHOLD,
    WILDFIRE_ENGINES,
    WILDFIRE_MODEL_ID,
    WILDFIRE_NEGATIVE,
    WILDFIRE_PROMPT,
    generate_foot_lesions,
    generate_wildfire_image,
    get_foot_prompt_pool,
    image_to_data_url,
    images_to_zip_base64,
    preview_foot_mask,
    read_image_bytes,
)
from storage_service import get_storage, object_proxy_url


app = FastAPI(title="Synthetic Data Service API", version="1.0.0")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=False,
    allow_methods=["*"],
    allow_headers=["*"],
)


async def upload_to_image(upload: UploadFile) -> tuple[Image.Image, str]:
    data = await upload.read()
    if not data:
        raise HTTPException(status_code=400, detail="이미지 파일이 비어 있습니다.")
    try:
        return read_image_bytes(data)
    except Exception as exc:
        raise HTTPException(status_code=400, detail=f"이미지를 읽을 수 없습니다: {exc}") from exc


@app.get("/api/health")
def health():
    storage = get_storage().status()
    return {
        "ok": True,
        "device": DEVICE,
        "storage": storage,
        "wildfire_model": WILDFIRE_MODEL_ID,
        "foot_models": {
            name: {"label": config["label"], "prompt_count": len(get_foot_prompt_pool(name))}
            for name, config in FOOT_MODEL_CONFIGS.items()
        },
    }


@app.get("/api/defaults")
def defaults():
    return {
        "wildfire": {
            "model": WILDFIRE_MODEL_ID,
            "prompt": WILDFIRE_PROMPT,
            "negative_prompt": WILDFIRE_NEGATIVE,
            "steps": WILDFIRE_DEFAULT_STEPS,
            "guidance": WILDFIRE_DEFAULT_GUIDANCE,
            "fill_guidance": WILDFIRE_DEFAULT_FILL_GUIDANCE,
            "seed": WILDFIRE_DEFAULT_SEED,
            "output_size": WILDFIRE_DEFAULT_OUTPUT_SIZE,
            "smoke_headroom": WILDFIRE_DEFAULT_SMOKE_HEADROOM,
            "mask_dilate": WILDFIRE_DEFAULT_MASK_DILATE,
            "mask_feather": WILDFIRE_DEFAULT_MASK_FEATHER,
            "vegetation_refine": WILDFIRE_DEFAULT_VEG_REFINE,
            "vegetation_threshold": WILDFIRE_DEFAULT_VEG_THRESHOLD,
            "vegetation_min_coverage": WILDFIRE_DEFAULT_VEG_MIN_COVERAGE,
            "wildfire_engine": WILDFIRE_DEFAULT_ENGINE,
            "wildfire_engines": WILDFIRE_ENGINES,
            "composition_mode": "flame_dominant",
            "composition_modes": WILDFIRE_COMPOSITION_LABELS,
        },
        "foot": {
            "steps": FOOT_DEFAULT_STEPS,
            "guidance": FOOT_DEFAULT_GUIDANCE,
            "strength": FOOT_DEFAULT_STRENGTH,
            "seed": 777,
            "count": 5,
            "dilate": FOOT_DEFAULT_DILATE,
            "feather": FOOT_DEFAULT_FEATHER,
            "models": {
                name: {"label": config["label"], "prompt_count": len(get_foot_prompt_pool(name))}
                for name, config in FOOT_MODEL_CONFIGS.items()
            },
        },
    }


@app.get("/api/storage/status")
def storage_status():
    return get_storage().status()


@app.get("/api/storage/object")
def storage_object(object_name: str = Query(..., min_length=1)):
    try:
        data, content_type = get_storage().get_object_bytes(object_name)
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    except Exception as exc:
        raise HTTPException(status_code=404, detail=f"MinIO 객체를 읽을 수 없습니다: {exc}") from exc
    return Response(content=data, media_type=content_type)


def storage_object_url(result: dict, artifact_name: str) -> str | None:
    storage = result.get("storage")
    if not storage:
        return None
    return object_proxy_url(storage.get("objects", {}).get(artifact_name))


@app.get("/api/datasets/recent")
def recent_datasets(limit: int = Query(40, ge=1, le=200)):
    try:
        return {"items": get_storage().list_dataset_items(limit=limit), "storage": get_storage().status()}
    except Exception as exc:
        raise HTTPException(status_code=503, detail=f"MinIO 데이터 목록을 읽을 수 없습니다: {exc}") from exc


@app.post("/api/wildfire/generate")
async def wildfire_generate(
    image: UploadFile = File(...),
    mask: UploadFile | None = File(None),
    prompt: str = Form(""),
    negative_prompt: str = Form(WILDFIRE_NEGATIVE),
    steps: int = Form(WILDFIRE_DEFAULT_STEPS),
    guidance: float = Form(WILDFIRE_DEFAULT_GUIDANCE),
    fill_guidance: float = Form(WILDFIRE_DEFAULT_FILL_GUIDANCE),
    seed: int = Form(WILDFIRE_DEFAULT_SEED),
    output_size: int = Form(WILDFIRE_DEFAULT_OUTPUT_SIZE),
    composition_mode: str = Form("flame_dominant"),
    smoke_headroom: int = Form(WILDFIRE_DEFAULT_SMOKE_HEADROOM),
    mask_dilate: int = Form(WILDFIRE_DEFAULT_MASK_DILATE),
    mask_feather: int = Form(WILDFIRE_DEFAULT_MASK_FEATHER),
    vegetation_refine: bool = Form(WILDFIRE_DEFAULT_VEG_REFINE),
    vegetation_threshold: float = Form(WILDFIRE_DEFAULT_VEG_THRESHOLD),
    vegetation_min_coverage: float = Form(WILDFIRE_DEFAULT_VEG_MIN_COVERAGE),
    wildfire_engine: str = Form(WILDFIRE_DEFAULT_ENGINE),
    save_output: bool = Form(True),
):
    source_image, image_id = await upload_to_image(image)
    mask_image = None
    if mask is not None:
        mask_data = await mask.read()
        if mask_data:
            mask_image = Image.open(io.BytesIO(mask_data)).convert("L")
    try:
        result = generate_wildfire_image(
            source_image=source_image,
            mask_image=mask_image,
            prompt=prompt,
            negative_prompt=negative_prompt,
            steps=steps,
            guidance=guidance,
            seed=seed,
            output_size=output_size,
            save_output=save_output,
            fill_guidance=fill_guidance,
            composition_mode=composition_mode,
            smoke_headroom=smoke_headroom,
            mask_dilate=mask_dilate,
            mask_feather=mask_feather,
            vegetation_refine=vegetation_refine,
            vegetation_threshold=vegetation_threshold,
            vegetation_min_coverage=vegetation_min_coverage,
            wildfire_engine=wildfire_engine,
        )
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    except Exception as exc:
        raise HTTPException(status_code=500, detail=str(exc)) from exc

    return {
        "image_id": image_id,
        "status": result["status"],
        "image": image_to_data_url(result["result"]),
        "image_url": storage_object_url(result, "image.png"),
        "event_region_mask": (
            image_to_data_url(result["event_region_mask"].convert("RGB"))
            if result.get("event_region_mask") is not None
            else None
        ),
        "generation_edit_mask": (
            image_to_data_url(result["generation_edit_mask"].convert("RGB"))
            if result.get("generation_edit_mask") is not None
            else None
        ),
        "saved_path": result.get("saved_path"),
        "metadata_path": result.get("metadata_path"),
        "storage": result.get("storage"),
        "guidance": result.get("guidance"),
        "fill_guidance": result.get("fill_guidance"),
        "crop_padding": result.get("crop_padding"),
        "composition_mode": result.get("composition_mode"),
        "composition_label": result.get("composition_label"),
        "selected_prompt_variant": result.get("selected_prompt_variant"),
        "mask_refinement": result.get("mask_refinement"),
        "wildfire_engine": result.get("wildfire_engine"),
    }


@app.post("/api/foot/preview")
async def foot_preview(
    image: UploadFile = File(...),
    mask: UploadFile = File(...),
    dilate: int = Form(FOOT_DEFAULT_DILATE),
    feather: int = Form(FOOT_DEFAULT_FEATHER),
):
    source_image, image_id = await upload_to_image(image)
    mask_data = await mask.read()
    try:
        mask_image = Image.open(io.BytesIO(mask_data)).convert("L")
        result = preview_foot_mask(source_image, mask_image, dilate=dilate, feather=feather)
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    except Exception as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc

    return {
        "image_id": image_id,
        "mask": image_to_data_url(result["mask"].convert("RGB")),
        "preview": image_to_data_url(result["preview"]),
    }


@app.post("/api/foot/generate")
async def foot_generate(
    image: UploadFile = File(...),
    mask: UploadFile = File(...),
    model_name: str = Form("corn"),
    steps: int = Form(FOOT_DEFAULT_STEPS),
    guidance: float = Form(FOOT_DEFAULT_GUIDANCE),
    strength: float = Form(FOOT_DEFAULT_STRENGTH),
    seed: int = Form(777),
    output_count: int = Form(5),
    dilate: int = Form(FOOT_DEFAULT_DILATE),
    feather: int = Form(FOOT_DEFAULT_FEATHER),
    save_output: bool = Form(True),
):
    source_image, image_id = await upload_to_image(image)
    mask_data = await mask.read()
    try:
        mask_image = Image.open(io.BytesIO(mask_data)).convert("L")
        result = generate_foot_lesions(
            model_name=model_name,
            source_image=source_image,
            mask_image=mask_image,
            steps=steps,
            guidance=guidance,
            strength=strength,
            seed=seed,
            save_output=save_output,
            output_count=output_count,
            dilate=dilate,
            feather=feather,
        )
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    except Exception as exc:
        raise HTTPException(status_code=500, detail=str(exc)) from exc

    return {
        "image_id": image_id,
        "status": result["status"],
        "preview": image_to_data_url(result["preview"]),
        "images": [image_to_data_url(image) for image in result["gallery"]],
        "image_url": storage_object_url(result, "images/result_01.png"),
        "prompts": result["prompts"],
        "saved_paths": result.get("saved_paths", []),
        "metadata_paths": result.get("metadata_paths", []),
        "run_metadata_path": result.get("run_metadata_path"),
        "storage": result.get("storage"),
        "zip_base64": images_to_zip_base64(result["gallery"], f"{model_name}_{image_id}"),
    }
