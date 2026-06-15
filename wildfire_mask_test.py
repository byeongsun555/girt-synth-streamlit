from __future__ import annotations

import io
import json
import os
import random
import time
import zipfile
from contextlib import nullcontext
from functools import lru_cache
from pathlib import Path

import cv2
import numpy as np
import streamlit as st
import streamlit.elements.image as st_image
import torch
from PIL import Image
from diffusers import FluxFillPipeline
from streamlit.elements.lib.image_utils import image_to_url as st_image_to_url
from streamlit_drawable_canvas import st_canvas


def _image_to_url_compat(image, width, clamp, channels, output_format, image_id):
    layout_config = st_image.create_layout_config(width=width)
    return st_image_to_url(image, layout_config, clamp, channels, output_format, image_id)


if not hasattr(st_image, "image_to_url"):
    st_image.image_to_url = _image_to_url_compat


APP_DIR = Path(__file__).resolve().parent
OUTPUT_DIR = APP_DIR / "outputs" / "wildfire_mask_test"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
FLUX_FILL_MODEL_ID = "black-forest-labs/FLUX.1-Fill-dev"
DEFAULT_PROMPT = (
    "Inside the selected wildfire event region, compose a photorealistic wildfire scene. "
    "Automatically distribute active orange flames, dark gray-brown sooty smoke plume, scorched grass, "
    "charred soil, glowing embers, and some partially unburned vegetation. "
    "Smoke should be denser near flames and fade upward. Avoid blue haze, fog, mist, or a uniform flame texture."
)
DEFAULT_NEGATIVE = (
    "fire on road, fire on asphalt, road burning, changed road, changed background, text, watermark, "
    "cartoon flame, repeated flame texture, flat black carpet, solid flame wall, unrealistic uniform fire, "
    "blue smoke, blue haze, fog, mist, clean white vapor, new object, black log, black wall, black bar, "
    "black mound, barrier, pipe, fallen tree trunk, straight horizontal band, rectangular object, "
    "side-by-side smoke and fire, separate smoke-only area, separate fire-only area, split composition, detached smoke"
)
CLASS_ID = 0
CLASS_NAME = "wildfire"
COMPOSITION_LABELS = {
    "balanced": "혼합형",
    "flame_dominant": "불 중심",
    "full_flame": "전체 화염",
    "smoke_dominant": "연기 중심",
}
COMPOSITION_DESCRIPTIONS = {
    "balanced": "불/연기/그을림이 자연스럽게 섞임",
    "flame_dominant": "불꽃이 주로 보이고 연기는 보조",
    "full_flame": "마스크 영역 대부분을 불로 채움",
    "smoke_dominant": "진한 연기와 국소적인 불꽃 중심",
}
COMPOSITION_PROMPT_VARIANTS = {
    "balanced": [
        "Natural mixed wildfire: irregular orange flame pockets, dark sooty smoke, charred grass, ash, embers, and partially visible vegetation.",
        "A realistic uneven burn zone with scattered flames, smoke rising from several points, scorched forest floor, and preserved vegetation gaps.",
        "Photorealistic forest-edge wildfire with active flames, smoke drifting upward, blackened shrubs, glowing embers, and varied burn intensity.",
        "Mixed wildfire event: flames along dry grass, gray-brown smoke above, charred soil below, and unburned green plants at the edges.",
    ],
    "flame_dominant": [
        "Flame-dominant wildfire: bright orange flames spreading across dry grass and shrubs, with smoke kept secondary and behind the flames.",
        "Large active flames on existing vegetation with glowing cores, ember trails, lightly scorched grass texture, and only moderate smoke above the fire line.",
        "Photorealistic flame front moving through forest undergrowth, varied flame heights, intense heat glow, and subtle sooty smoke.",
        "Strong visible flames covering most vegetation in the event region, with natural gaps, burnt grass texture, and small smoke plumes.",
    ],
    "full_flame": [
        "Full wildfire flame coverage: fill nearly the entire selected vegetation region with realistic active flames, varied height, hot cores, embers, and visible grass texture underneath.",
        "A broad natural fire line across the masked vegetation, dense orange flames following the existing shrubs and grass, with smoke only above the flame layer.",
        "Heavy active wildfire over most of the event region, intense flames consuming existing shrubs and grass, glowing embers, ash, and dark smoke rising upward.",
        "Wide realistic forest fire inside the selected vegetation area, non-repeating flame shapes, scorched plants underneath, and hot orange-yellow cores.",
    ],
    "smoke_dominant": [
        "Smoke-dominant wildfire: low orange flames and embers along the vegetation base, with dense dark gray-brown smoke rising directly above each flame source.",
        "Smoldering vegetation fire with visible small flames underneath, charred grass at ground level, and thick smoke vertically connected to the flames.",
        "Forest-edge fire with a continuous low fire line below and dark sooty smoke rising upward from the same burning grass and shrubs.",
        "Heavy wildfire smoke plume inside the event region, but every smoke plume must originate from visible flames, embers, or scorched vegetation directly below it.",
    ],
}

st.set_page_config(page_title="Wildfire Mask Test", layout="wide")


def image_to_png_bytes(image: Image.Image) -> bytes:
    buffer = io.BytesIO()
    image.save(buffer, format="PNG")
    return buffer.getvalue()


def make_seed(seed_value) -> int:
    try:
        seed = int(seed_value)
    except (TypeError, ValueError):
        seed = -1
    if seed < 0:
        return random.randint(0, 2**31 - 1)
    return seed


def autocast_context():
    if DEVICE == "cuda":
        return torch.autocast("cuda", dtype=torch.bfloat16)
    return nullcontext()


def make_zip_bytes(files: dict[str, bytes]) -> bytes:
    buffer = io.BytesIO()
    with zipfile.ZipFile(buffer, "w", compression=zipfile.ZIP_DEFLATED) as archive:
        for filename, data in files.items():
            archive.writestr(filename, data)
    return buffer.getvalue()


def resize_for_canvas(image: Image.Image, max_side: int) -> Image.Image:
    image = image.convert("RGB")
    width, height = image.size
    scale = min(1.0, max_side / max(width, height))
    canvas_size = (max(1, int(width * scale)), max(1, int(height * scale)))
    return image.resize(canvas_size, Image.BICUBIC)


def extract_mask_from_canvas(
    canvas_image_data,
    display_size: tuple[int, int],
    original_size: tuple[int, int],
    dilate: int,
    feather: int,
) -> tuple[Image.Image | None, Image.Image | None]:
    if canvas_image_data is None:
        return None, None

    canvas_arr = np.asarray(canvas_image_data).astype(np.uint8)
    if canvas_arr.ndim != 3 or canvas_arr.shape[2] < 3:
        return None, None

    if canvas_arr.shape[2] >= 4:
        mask_arr = np.where(canvas_arr[:, :, 3] > 0, 255, 0).astype(np.uint8)
    else:
        gray = cv2.cvtColor(canvas_arr[:, :, :3], cv2.COLOR_RGB2GRAY)
        mask_arr = np.where(gray > 0, 255, 0).astype(np.uint8)

    mask_arr = cv2.resize(mask_arr, display_size, interpolation=cv2.INTER_NEAREST)

    if dilate > 0:
        kernel = np.ones((int(dilate), int(dilate)), np.uint8)
        mask_arr = cv2.dilate(mask_arr, kernel, iterations=1)

    binary_display = np.where(mask_arr > 0, 255, 0).astype(np.uint8)
    soft_display = binary_display.copy()
    if feather > 0:
        sigma = max(0.5, float(feather) / 3.0)
        soft_display = cv2.GaussianBlur(soft_display, (0, 0), sigmaX=sigma, sigmaY=sigma)
        soft_display = np.clip(soft_display, 0, 255).astype(np.uint8)

    binary_original = cv2.resize(binary_display, original_size, interpolation=cv2.INTER_NEAREST)
    soft_original = cv2.resize(soft_display, original_size, interpolation=cv2.INTER_LINEAR)
    binary_original = np.where(binary_original > 127, 255, 0).astype(np.uint8)

    return Image.fromarray(binary_original, mode="L"), Image.fromarray(soft_original, mode="L")


def overlay_mask(image: Image.Image, mask: Image.Image, color=(255, 60, 0), alpha=0.38) -> Image.Image:
    image_arr = np.array(image.convert("RGB"), dtype=np.uint8)
    mask_arr = np.array(mask.convert("L"), dtype=np.uint8)
    color_layer = np.zeros_like(image_arr, dtype=np.uint8)
    color_layer[mask_arr > 0] = np.array(color, dtype=np.uint8)
    return Image.fromarray(cv2.addWeighted(image_arr, 1.0, color_layer, alpha, 0.0))


def make_generation_edit_mask(
    label_mask: Image.Image,
    feather: int,
    smoke_headroom_percent: int,
) -> tuple[Image.Image, Image.Image]:
    label_arr = np.array(label_mask.convert("L"), dtype=np.uint8)
    edit_arr = np.where(label_arr > 0, 255, 0).astype(np.uint8)

    ys, xs = np.where(edit_arr > 0)
    if len(xs) and smoke_headroom_percent > 0:
        x_min, x_max = int(xs.min()), int(xs.max())
        y_min, y_max = int(ys.min()), int(ys.max())
        bbox_h = max(1, y_max - y_min + 1)
        headroom_px = max(1, int(bbox_h * smoke_headroom_percent / 100))

        smoke_arr = np.zeros_like(edit_arr)
        for x in range(x_min, x_max + 1):
            column_ys = np.where(edit_arr[:, x] > 0)[0]
            if len(column_ys) == 0:
                continue
            top = int(column_ys.min())
            y0 = max(0, top - headroom_px)
            smoke_arr[y0 : top + 1, x] = 255

        width_kernel = max(11, int((x_max - x_min + 1) * 0.06))
        height_kernel = max(11, int(headroom_px * 0.20))
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (width_kernel | 1, height_kernel | 1))
        smoke_arr = cv2.dilate(smoke_arr, kernel, iterations=1)
        edit_arr = np.maximum(edit_arr, smoke_arr)

    soft_arr = edit_arr.copy()
    if feather > 0:
        sigma = max(0.5, float(feather) / 2.0)
        soft_arr = cv2.GaussianBlur(soft_arr, (0, 0), sigmaX=sigma, sigmaY=sigma)
        soft_arr = np.clip(soft_arr, 0, 255).astype(np.uint8)

    return Image.fromarray(edit_arr, mode="L"), Image.fromarray(soft_arr, mode="L")


def mask_metadata(mask: Image.Image, image_id: str) -> dict:
    mask_arr = np.array(mask.convert("L"), dtype=np.uint8)
    ys, xs = np.where(mask_arr > 0)
    area = int(len(xs))
    width, height = mask.size
    metadata = {
        "image_id": image_id,
        "mask_size": [width, height],
        "mask_area_pixels": area,
        "mask_area_ratio": float(area / max(1, width * height)),
    }
    if area:
        x_min, x_max = int(xs.min()), int(xs.max())
        y_min, y_max = int(ys.min()), int(ys.max())
        metadata.update(
            {
                "bbox_xyxy": [x_min, y_min, x_max, y_max],
                "bbox_xywh": [x_min, y_min, x_max - x_min + 1, y_max - y_min + 1],
            }
        )
    else:
        metadata.update({"bbox_xyxy": None, "bbox_xywh": None})
    return metadata


def mask_contours(mask: Image.Image, min_area: int = 16) -> list[np.ndarray]:
    mask_arr = np.array(mask.convert("L"), dtype=np.uint8)
    contour_mask = np.where(mask_arr > 0, 255, 0).astype(np.uint8)
    contours, _ = cv2.findContours(contour_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    kept = []
    for contour in contours:
        if cv2.contourArea(contour) >= min_area:
            epsilon = 0.003 * cv2.arcLength(contour, True)
            kept.append(cv2.approxPolyDP(contour, epsilon, True))
    return kept


def coco_annotation(mask: Image.Image, image_id: str, file_name: str) -> dict:
    width, height = mask.size
    metadata = mask_metadata(mask, image_id)
    segmentations = []
    for contour in mask_contours(mask):
        flat = contour.reshape(-1, 2).astype(float).flatten().tolist()
        if len(flat) >= 6:
            segmentations.append(flat)

    return {
        "images": [{"id": 1, "file_name": file_name, "width": width, "height": height}],
        "annotations": [
            {
                "id": 1,
                "image_id": 1,
                "category_id": CLASS_ID,
                "segmentation": segmentations,
                "area": metadata["mask_area_pixels"],
                "bbox": metadata["bbox_xywh"],
                "iscrowd": 0,
            }
        ],
        "categories": [{"id": CLASS_ID, "name": CLASS_NAME}],
    }


def yolo_bbox_text(mask: Image.Image) -> str:
    metadata = mask_metadata(mask, image_id="label")
    if not metadata.get("bbox_xywh"):
        return ""
    width, height = mask.size
    x, y, box_w, box_h = metadata["bbox_xywh"]
    x_center = (x + box_w / 2) / width
    y_center = (y + box_h / 2) / height
    return f"{CLASS_ID} {x_center:.6f} {y_center:.6f} {box_w / width:.6f} {box_h / height:.6f}\n"


def yolo_seg_text(mask: Image.Image) -> str:
    width, height = mask.size
    lines = []
    for contour in mask_contours(mask):
        points = contour.reshape(-1, 2)
        if len(points) < 3:
            continue
        normalized = []
        for x, y in points:
            normalized.extend([x / width, y / height])
        lines.append(f"{CLASS_ID} " + " ".join(f"{value:.6f}" for value in normalized))
    return "\n".join(lines) + ("\n" if lines else "")


def mask_bbox(mask: Image.Image, padding: int) -> tuple[int, int, int, int]:
    mask_arr = np.array(mask.convert("L"), dtype=np.uint8)
    ys, xs = np.where(mask_arr > 0)
    if len(xs) == 0:
        raise ValueError("마스크가 비어 있습니다.")
    width, height = mask.size
    x0 = max(0, int(xs.min()) - padding)
    y0 = max(0, int(ys.min()) - padding)
    x1 = min(width, int(xs.max()) + padding + 1)
    y1 = min(height, int(ys.max()) + padding + 1)
    return x0, y0, x1, y1


def resize_to_multiple(image: Image.Image, max_side: int, multiple: int = 16, resample=Image.BICUBIC) -> Image.Image:
    width, height = image.size
    scale = min(1.0, max_side / max(width, height))
    new_w = max(multiple, int(width * scale))
    new_h = max(multiple, int(height * scale))
    new_w = max(multiple, round(new_w / multiple) * multiple)
    new_h = max(multiple, round(new_h / multiple) * multiple)
    return image.resize((new_w, new_h), resample)


@lru_cache(maxsize=1)
def get_flux_fill_pipeline(model_id: str):
    hf_token = os.environ.get("HF_TOKEN") or os.environ.get("HUGGINGFACE_HUB_TOKEN")
    if not hf_token:
        raise RuntimeError("FLUX Fill 모델을 사용하려면 HF_TOKEN 환경 변수가 필요합니다.")
    pipe = FluxFillPipeline.from_pretrained(
        model_id,
        torch_dtype=torch.bfloat16 if DEVICE == "cuda" else torch.float32,
        token=hf_token,
    )
    if DEVICE == "cuda":
        pipe.enable_model_cpu_offload()
        if hasattr(pipe, "vae") and hasattr(pipe.vae, "enable_tiling"):
            pipe.vae.enable_tiling()
    else:
        pipe = pipe.to(DEVICE)
    return pipe


def run_flux_fill(
    source: Image.Image,
    label_mask: Image.Image,
    edit_binary_mask: Image.Image,
    edit_soft_mask: Image.Image,
    prompt: str,
    negative_prompt: str,
    model_id: str,
    steps: int,
    guidance: float,
    seed,
    crop_padding: int,
    crop_max_side: int,
    burn_coverage: int,
    event_intensity: str,
    composition_mode: str,
) -> dict:
    seed = make_seed(seed)
    x0, y0, x1, y1 = mask_bbox(edit_binary_mask, padding=int(crop_padding))
    crop_box = (x0, y0, x1, y1)
    crop_image = source.crop(crop_box)
    crop_binary_mask = edit_binary_mask.crop(crop_box)
    crop_soft_mask = edit_soft_mask.crop(crop_box)

    work_image = resize_to_multiple(crop_image, max_side=int(crop_max_side), multiple=16, resample=Image.BICUBIC)
    work_mask = crop_soft_mask.resize(work_image.size, Image.BILINEAR)
    work_binary_mask = crop_binary_mask.resize(work_image.size, Image.NEAREST)

    if event_intensity == "severe":
        coverage_instruction = (
            f"The mask is the overall wildfire event region, not a solid flame mask. "
            f"About {burn_coverage}% of the region should show active burning effects. "
            "Arrange a severe wildfire with dense flame pockets, dark gray-brown sooty smoke, scorched ground, "
            "embers, and a few remaining vegetation details. Avoid filling the whole region with identical flames. "
        )
    elif event_intensity == "moderate":
        coverage_instruction = (
            f"The mask is the overall wildfire event region, not a solid flame mask. "
            f"About {burn_coverage}% of the region should show active flames, smoke, and scorch marks. "
            "Use natural gaps, uneven intensity, and varied flame height. "
        )
    else:
        coverage_instruction = (
            f"The mask is the overall wildfire event region, not a solid flame mask. "
            f"About {burn_coverage}% of the region should show localized flames, smoke, and light scorch marks. "
        )

    if composition_mode == "full_flame":
        composition_instruction = (
            "Composition mode: full flame coverage. Fill most of the selected vegetation area with active orange flames. "
            "Use varied flame heights, bright flame cores, ember texture, and keep the original grass or shrub structure visible underneath. "
            "Smoke should exist as a natural plume above and behind the flames, but it must not replace half of the fire area. "
        )
    elif composition_mode == "flame_dominant":
        composition_instruction = (
            "Composition mode: flame dominant. Make flames the main visible effect across the region, "
            "with smoke and scorch marks as secondary supporting details. "
        )
    elif composition_mode == "smoke_dominant":
        composition_instruction = (
            "Composition mode: smoke dominant. Build a vertical fire-smoke structure, not a side-by-side split. "
            "Place visible low flames, glowing embers, and scorched grass along the lower vegetation base inside the mask. "
            "Then make dense dark gray-brown smoke rise directly upward from those same flame sources. "
            "Every smoke plume must be physically connected to a visible flame, ember, or scorched patch below it. "
            "Do not put smoke in one part of the mask and fire in a separate distant part. "
        )
    else:
        composition_instruction = (
            "Composition mode: balanced wildfire. Mix active flames, smoke, scorched ground, embers, "
            "and some partially unburned vegetation in a natural distribution. "
        )

    variant_pool = COMPOSITION_PROMPT_VARIANTS.get(composition_mode, COMPOSITION_PROMPT_VARIANTS["balanced"])
    selected_prompt_variant = random.Random(f"{seed}:{composition_mode}").choice(variant_pool)

    final_prompt = (
        "Only edit inside the white mask. Keep all unmasked pixels unchanged. "
        "The white edit mask includes the user wildfire event region plus extra upward space reserved for smoke. "
        "Use the lower vegetation area as the fire source, and use the upper part mainly for smoke that rises from the fire. "
        "Do not fill the entire white mask with grass, shrubs, solid fire, or new vegetation. "
        "Within the event region, the model should decide where to place flames, smoke, scorch, embers, and unburned vegetation. "
        "Place active flames and smoke on vegetation, dry grass, shrubs, and forest floor inside the mask. "
        "If the mask overlaps road or asphalt, keep the road mostly intact with only subtle smoke shadow or embers at the edge. "
        "Do not invent new solid objects. Do not create logs, walls, pipes, mounds, black bars, or straight horizontal bands. "
        "The edited area must look like the existing vegetation is burning, not like an object was placed on top of the image. "
        f"{coverage_instruction}"
        f"{composition_instruction}"
        f"Randomized composition detail: {selected_prompt_variant} "
        f"{prompt} Negative constraints: {negative_prompt}"
    )
    pipe = get_flux_fill_pipeline(model_id)
    generator = torch.Generator(device="cpu").manual_seed(seed)

    with torch.inference_mode(), autocast_context():
        generated_work = pipe(
            prompt=final_prompt,
            image=work_image,
            mask_image=work_mask,
            height=work_image.height,
            width=work_image.width,
            num_inference_steps=int(steps),
            guidance_scale=float(guidance),
            generator=generator,
            max_sequence_length=512,
        ).images[0]

    generated_crop = generated_work.resize(crop_image.size, Image.BICUBIC)
    composite = source.copy()
    composite.paste(generated_crop, crop_box, crop_soft_mask)
    generated_overlay = overlay_mask(composite, label_mask)

    return {
        "result": composite,
        "generated_crop": generated_crop,
        "work_image": work_image,
        "work_mask": work_binary_mask.convert("L"),
        "generation_edit_mask": edit_binary_mask,
        "generation_edit_soft_mask": edit_soft_mask,
        "overlay": generated_overlay,
        "crop_box": crop_box,
        "seed": seed,
        "final_prompt": final_prompt,
        "model_id": model_id,
        "steps": int(steps),
        "guidance": float(guidance),
        "burn_coverage": int(burn_coverage),
        "event_intensity": event_intensity,
        "composition_mode": composition_mode,
        "selected_prompt_variant": selected_prompt_variant,
    }


def save_run(
    original: Image.Image,
    mask: Image.Image,
    soft_mask: Image.Image,
    overlay: Image.Image,
    metadata: dict,
    generated: dict | None = None,
) -> Path:
    run_dir = OUTPUT_DIR / time.strftime("%Y%m%d_%H%M%S")
    run_dir.mkdir(parents=True, exist_ok=True)
    original.save(run_dir / "original.png")
    mask.save(run_dir / "event_region_mask_binary.png")
    soft_mask.save(run_dir / "event_region_mask_soft.png")
    overlay.save(run_dir / "overlay.png")
    coco = coco_annotation(mask, image_id=metadata["image_id"], file_name="generated.png" if generated else "original.png")
    (run_dir / "coco_annotation.json").write_text(json.dumps(coco, ensure_ascii=False, indent=2), encoding="utf-8")
    (run_dir / "yolo_bbox.txt").write_text(yolo_bbox_text(mask), encoding="utf-8")
    (run_dir / "yolo_seg.txt").write_text(yolo_seg_text(mask), encoding="utf-8")
    if generated:
        generated["result"].save(run_dir / "generated.png")
        generated["generated_crop"].save(run_dir / "generated_crop.png")
        generated["work_image"].save(run_dir / "crop_input.png")
        generated["work_mask"].save(run_dir / "crop_event_region_mask.png")
        generated["generation_edit_mask"].save(run_dir / "generation_edit_mask_binary.png")
        generated["generation_edit_soft_mask"].save(run_dir / "generation_edit_mask_soft.png")
        generated["overlay"].save(run_dir / "generated_overlay.png")
        generation_metadata = metadata.get("generation", {})
        metadata = {
            **metadata,
            "generation": {
                **generation_metadata,
                "model_id": generated["model_id"],
                "seed": generated["seed"],
                "steps": generated["steps"],
                "guidance": generated["guidance"],
                "burn_coverage": generated["burn_coverage"],
                "event_intensity": generated["event_intensity"],
                "composition_mode": generated["composition_mode"],
                "selected_prompt_variant": generated["selected_prompt_variant"],
                "crop_box_xyxy": list(generated["crop_box"]),
                "final_prompt": generated["final_prompt"],
                "negative_prompt": generated.get("negative_prompt"),
            },
        }
    (run_dir / "metadata.json").write_text(json.dumps(metadata, ensure_ascii=False, indent=2), encoding="utf-8")
    return run_dir


st.title("Wildfire Mask Test")
st.caption("사용자가 그린 mask를 산불 이벤트 영역으로 사용하고, 모델이 그 안에서 불/연기/그을림을 자동 배치합니다.")

with st.sidebar:
    st.subheader("Mask Settings")
    max_side = st.slider("Canvas Max Side", 512, 1400, 900, 32)
    brush_size = st.slider("Brush Size", 4, 96, 28, 1)
    dilate = st.slider("Mask Dilate", 0, 48, 0, 1)
    feather = st.slider("Soft Mask Feather", 0, 48, 10, 1)
    smoke_headroom = st.slider("Smoke Headroom (%)", 0, 160, 70, 5)
    st.caption("이 mask는 픽셀 단위 불꽃 mask가 아니라 산불 이벤트 영역 라벨로 저장됩니다.")
    st.divider()
    st.subheader("Generation Settings")
    model_id = st.text_input("FLUX Fill Model", value=FLUX_FILL_MODEL_ID)
    prompt = st.text_area("Prompt", value=DEFAULT_PROMPT, height=120)
    negative_prompt = st.text_area("Negative Prompt", value=DEFAULT_NEGATIVE, height=80)
    steps = st.slider("Steps", 10, 80, 42, 1)
    guidance = st.slider("Guidance Scale", 1.0, 100.0, 30.0, 1.0)
    event_intensity = st.selectbox(
        "Event Intensity",
        options=["severe", "moderate", "light"],
        index=0,
        format_func=lambda value: {
            "severe": "강한 산불 - 큰 불/진한 연기",
            "moderate": "중간 산불 - 불/연기/그을림 혼합",
            "light": "약한 산불 - 국소 불꽃 중심",
        }[value],
    )
    if "wildfire_composition_mode" not in st.session_state:
        st.session_state["wildfire_composition_mode"] = "flame_dominant"
    st.markdown("**Fire / Smoke Composition**")
    mode_items = ["balanced", "flame_dominant", "full_flame", "smoke_dominant"]
    for row_modes in (mode_items[:2], mode_items[2:]):
        mode_cols = st.columns(2)
        for mode, mode_col in zip(row_modes, mode_cols):
            selected = st.session_state["wildfire_composition_mode"] == mode
            if mode_col.button(
                COMPOSITION_LABELS[mode],
                key=f"composition_mode_{mode}",
                type="primary" if selected else "secondary",
                width="stretch",
            ):
                st.session_state["wildfire_composition_mode"] = mode
    composition_mode = st.session_state["wildfire_composition_mode"]
    st.caption(COMPOSITION_DESCRIPTIONS[composition_mode])
    burn_coverage = st.slider("Burn Coverage in Event Region (%)", 20, 100, 70, 5)
    seed = st.number_input("Seed (-1 random)", value=-1, step=1)
    crop_padding = st.slider("Crop Padding", 0, 512, 160, 8)
    crop_max_side = st.selectbox("Crop Max Side", options=[512, 768, 1024, 1280], index=1)

uploaded = st.file_uploader("원본 이미지 업로드", type=["png", "jpg", "jpeg", "webp"])

if uploaded is None:
    st.info("원본 이미지를 업로드하면 캔버스 위에 산불 생성 영역을 그릴 수 있습니다.")
    st.stop()

uploaded.seek(0)
source = Image.open(uploaded).convert("RGB")
display = resize_for_canvas(source, max_side=max_side)
display_width, display_height = display.size
image_id = Path(uploaded.name).stem

canvas_col, result_col = st.columns([1.15, 0.85], vertical_alignment="top")

with canvas_col:
    st.subheader("1. 산불 이벤트 영역 그리기")
    canvas_result = st_canvas(
        fill_color="rgba(255, 80, 0, 0.25)",
        stroke_width=int(brush_size),
        stroke_color="#ff3b00",
        background_image=display,
        update_streamlit=True,
        height=display_height,
        width=display_width,
        drawing_mode="freedraw",
        display_toolbar=True,
        key=f"wildfire_mask_canvas_{image_id}_{display_width}_{display_height}",
    )

with result_col:
    st.subheader("2. Mask 확인")
    make_preview = st.button("마스크 생성/미리보기", type="primary", width="stretch")

    if make_preview:
        binary_mask, soft_mask = extract_mask_from_canvas(
            canvas_result.image_data,
            display_size=(display_width, display_height),
            original_size=source.size,
            dilate=int(dilate),
            feather=int(feather),
        )
        if binary_mask is None or not np.count_nonzero(np.array(binary_mask)):
            st.warning("마스크가 비어 있습니다. 캔버스에 산불 이벤트 영역을 먼저 그려주세요.")
        else:
            overlay = overlay_mask(source, binary_mask)
            edit_binary_mask, edit_soft_mask = make_generation_edit_mask(
                binary_mask,
                feather=int(feather),
                smoke_headroom_percent=int(smoke_headroom),
            )
            metadata = mask_metadata(binary_mask, image_id=image_id)
            metadata.update(
                {
                    "source_file": uploaded.name,
                    "source_size": list(source.size),
                    "display_size": [display_width, display_height],
                    "brush_size": int(brush_size),
                    "dilate": int(dilate),
                    "feather": int(feather),
                    "smoke_headroom_percent": int(smoke_headroom),
                    "created_at": time.strftime("%Y-%m-%d %H:%M:%S"),
                }
            )

            st.session_state["wildfire_mask_test"] = {
                "binary_mask": binary_mask,
                "soft_mask": soft_mask,
                "edit_binary_mask": edit_binary_mask,
                "edit_soft_mask": edit_soft_mask,
                "overlay": overlay,
                "metadata": metadata,
                "source": source,
            }

result = st.session_state.get("wildfire_mask_test")
if result:
    binary_mask = result["binary_mask"]
    soft_mask = result["soft_mask"]
    edit_binary_mask = result["edit_binary_mask"]
    edit_soft_mask = result["edit_soft_mask"]
    overlay = result["overlay"]
    metadata = result["metadata"]

    st.success(
        f"Mask 생성 완료 | area={metadata['mask_area_pixels']} px | ratio={metadata['mask_area_ratio']:.4f}"
    )
    if metadata.get("bbox_xyxy"):
        st.caption(f"BBox xyxy: {metadata['bbox_xyxy']}")

    p1, p2, p3 = st.columns(3)
    with p1:
        st.image(binary_mask, caption="Event Region Label Mask - original size", width="stretch")
    with p2:
        st.image(edit_binary_mask, caption="Generation Edit Mask - smoke headroom included", width="stretch")
    with p3:
        st.image(overlay, caption="Overlay Preview - original size", width="stretch")

    st.subheader("3. FLUX Fill 생성 테스트")
    gen_col, save_col, download_col = st.columns(3)
    with gen_col:
        generate_clicked = st.button("마스크 영역 산불 생성", type="primary", width="stretch")

    if generate_clicked:
        try:
            with st.spinner("FLUX Fill로 mask 영역에 산불을 생성 중입니다..."):
                generated = run_flux_fill(
                    source=source,
                    label_mask=binary_mask,
                    edit_binary_mask=edit_binary_mask,
                    edit_soft_mask=edit_soft_mask,
                    prompt=prompt,
                    negative_prompt=negative_prompt,
                    model_id=model_id,
                    steps=steps,
                    guidance=guidance,
                    seed=seed,
                    crop_padding=crop_padding,
                    crop_max_side=crop_max_side,
                    burn_coverage=burn_coverage,
                    event_intensity=event_intensity,
                    composition_mode=composition_mode,
                )
                generated["negative_prompt"] = negative_prompt
                st.session_state["wildfire_mask_test"]["generated"] = generated
        except Exception as exc:
            st.error(str(exc))

    generated = st.session_state.get("wildfire_mask_test", {}).get("generated")
    if generated:
        g1, g2 = st.columns(2)
        with g1:
            st.image(generated["result"], caption="Generated image + original-size mask label", width="stretch")
        with g2:
            st.image(generated["generated_crop"], caption=f"Generated crop | crop_box={generated['crop_box']}", width="stretch")
        st.caption(
            f"Selected mode: {COMPOSITION_LABELS.get(generated['composition_mode'], generated['composition_mode'])} | "
            f"Prompt variant: {generated['selected_prompt_variant']}"
        )

        metadata = {
            **metadata,
            "negative_prompt_or_memo": generated.get("negative_prompt", negative_prompt),
            "generation": {
                "model_id": generated["model_id"],
                "seed": generated["seed"],
                "steps": generated["steps"],
                "guidance": generated["guidance"],
                "burn_coverage": generated["burn_coverage"],
                "event_intensity": generated["event_intensity"],
                "composition_mode": generated["composition_mode"],
                "selected_prompt_variant": generated["selected_prompt_variant"],
                "crop_box_xyxy": list(generated["crop_box"]),
                "final_prompt": generated["final_prompt"],
            },
        }

    with save_col:
        if st.button("서버에 저장", width="stretch"):
            run_dir = save_run(source, binary_mask, soft_mask, overlay, metadata, generated=generated)
            st.success(f"저장 완료: {run_dir}")

    zip_files = {
        "original.png": image_to_png_bytes(source),
        "event_region_mask_binary.png": image_to_png_bytes(binary_mask),
        "event_region_mask_soft.png": image_to_png_bytes(soft_mask),
        "generation_edit_mask_binary.png": image_to_png_bytes(edit_binary_mask),
        "generation_edit_mask_soft.png": image_to_png_bytes(edit_soft_mask),
        "overlay.png": image_to_png_bytes(overlay),
        "metadata.json": json.dumps(metadata, ensure_ascii=False, indent=2).encode("utf-8"),
        "coco_annotation.json": json.dumps(
            coco_annotation(binary_mask, image_id=metadata["image_id"], file_name="generated.png" if generated else "original.png"),
            ensure_ascii=False,
            indent=2,
        ).encode("utf-8"),
        "yolo_bbox.txt": yolo_bbox_text(binary_mask).encode("utf-8"),
        "yolo_seg.txt": yolo_seg_text(binary_mask).encode("utf-8"),
    }
    if generated:
        zip_files.update(
            {
                "generated.png": image_to_png_bytes(generated["result"]),
                "generated_crop.png": image_to_png_bytes(generated["generated_crop"]),
                "crop_input.png": image_to_png_bytes(generated["work_image"]),
                "crop_event_region_mask.png": image_to_png_bytes(generated["work_mask"]),
                "generation_edit_mask_binary.png": image_to_png_bytes(generated["generation_edit_mask"]),
                "generation_edit_mask_soft.png": image_to_png_bytes(generated["generation_edit_soft_mask"]),
                "generated_overlay.png": image_to_png_bytes(generated["overlay"]),
            }
        )
    zip_bytes = make_zip_bytes(zip_files)
    with download_col:
        st.download_button(
            "전체 ZIP 다운로드",
            data=zip_bytes,
            file_name=f"{image_id}_wildfire_mask_test.zip",
            mime="application/zip",
            width="stretch",
        )

    with st.expander("metadata.json"):
        st.json(metadata)
