from __future__ import annotations

import base64
import hashlib
import io
import json
import os
import random
import time
import zipfile
from contextlib import nullcontext
from datetime import datetime, timezone
from functools import lru_cache
from pathlib import Path

import cv2
import numpy as np
import torch
from diffusers import FluxFillPipeline, FluxKontextPipeline, StableDiffusionInpaintPipeline
from PIL import Image

from storage_service import get_storage


APP_DIR = Path(__file__).resolve().parents[2]
PROJECT_ROOT = APP_DIR.parent
FOOT_ROOT = PROJECT_ROOT / "foot_make_dataset"

OUTPUT_ROOT = APP_DIR / "outputs"
WILDFIRE_OUTPUT_DIR = OUTPUT_ROOT / "wildfire"
FOOT_OUTPUT_DIR = OUTPUT_ROOT / "foot"
WILDFIRE_OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
FOOT_OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
TORCH_DTYPE = torch.float16 if DEVICE == "cuda" else torch.float32

FOOT_TARGET_SIZE = 512
FOOT_DEFAULT_STEPS = 50
FOOT_DEFAULT_GUIDANCE = 6.5
FOOT_DEFAULT_STRENGTH = 0.85
FOOT_DEFAULT_DILATE = 5
FOOT_DEFAULT_FEATHER = 4
WILDFIRE_MODEL_ID = "black-forest-labs/FLUX.1-Kontext-dev"
WILDFIRE_FILL_MODEL_ID = "black-forest-labs/FLUX.1-Fill-dev"
WILDFIRE_DEFAULT_STEPS = 34
WILDFIRE_DEFAULT_GUIDANCE = 2.5
WILDFIRE_DEFAULT_FILL_GUIDANCE = 9.0
WILDFIRE_DEFAULT_SEED = -1
WILDFIRE_DEFAULT_OUTPUT_SIZE = 1024
WILDFIRE_DEFAULT_SMOKE_HEADROOM = 30
WILDFIRE_DEFAULT_MASK_DILATE = 2
WILDFIRE_DEFAULT_MASK_FEATHER = 8
WILDFIRE_DEFAULT_PERTURB = 0.40
WILDFIRE_DEFAULT_VEG_REFINE = True
WILDFIRE_DEFAULT_VEG_THRESHOLD = 0.30
WILDFIRE_DEFAULT_VEG_MIN_COVERAGE = 0.15
WILDFIRE_DEFAULT_FIRE_CORE_MIN_RATIO = 0.90
WILDFIRE_DEFAULT_FIRE_CORE_MAX_RATIO = 1.00
WILDFIRE_ENGINE_KONTEXT_HINT = "kontext_hint"
WILDFIRE_ENGINE_FILL_MASK = "fill_mask"
# 기본을 Fill로 두어 마스크 위치를 정확히 따르도록 함. Kontext는 자유 편집이라 위치 어긋남 발생.
WILDFIRE_DEFAULT_ENGINE = WILDFIRE_ENGINE_FILL_MASK
WILDFIRE_PROMPT = "원본 구도와 배경은 유지하고, 도로나 포장면이 아닌 나무 또는 풀숲 내부에 작은 산불과 연기를 자연스럽게 추가해줘."
WILDFIRE_EDIT_GUARDRAIL = (
    "Photorealistic natural daylight photograph. Edit only vegetation in the marked area; "
    "non-vegetation surfaces and unmarked regions stay identical to the original."
)
WILDFIRE_NEGATIVE = (
    "cartoon, illustration, painting, CGI, low quality, blurry, watermark, text, "
    "rectangular fire shape, geometric flame outline, flame wall, fire stripe tracing the mask, "
    "solid black mound, large dark scorched bar, fire on asphalt, fire on road, scorched pavement, "
    "smoke without fire, detached smoke patch, sticker-like flames, neon glow, "
    "single round flame blob, uniform fireball, isolated fire ball, perfect sphere flame, "
    "saturated fluorescent orange, plastic fire texture, video-game fire effect"
)
WILDFIRE_COMPOSITION_LABELS = {
    "balanced": "혼합형",
    "flame_dominant": "불 중심",
    "full_flame": "전체 화염",
    "smoke_dominant": "연기 중심",
}
WILDFIRE_COMPOSITION_PROMPTS = {
    "balanced": [
        "Natural documentary photo of early-stage brush fire: small irregular orange flame pockets in shrubs and dry grass, "
        "thin wispy smoke from each flame, surrounding plants still green and intact, warm soft daylight, sharp realistic detail.",
        "Photorealistic ground-level wildfire: scattered small flame clusters of different heights, rising gray smoke trails, "
        "unburned green grass visible between flames, natural outdoor sunlight, photojournalism style.",
        "Realistic patchy vegetation fire: two or three irregular flame spots on plants and grass, "
        "soft smoke wisps drifting upward, surrounding vegetation intact, warm afternoon light, candid news camera quality.",
        "Authentic wildfire snapshot: small organic flame pockets in dry vegetation with curling tongues, "
        "thin smoke fading into sky, char only beneath flames, intact vegetation nearby, fine photographic detail.",
    ],
    "flame_dominant": [
        "Photorealistic wildfire photo: irregular orange flame tongues of varied heights burning on shrubs and dry grass, "
        "glowing yellow cores, thin smoke above, natural daylight color, sharp realistic detail.",
        "Realistic brush fire: flickering orange flames with pointed tips, ember sparks, "
        "plants visibly burning at the base, soft gray smoke wisps, warm directional sunlight.",
        "Natural wildfire close-up: dynamic curling flame tongues engulfing bushes, hot yellow-white cores fading to red tips, "
        "floating embers, thin smoke, ground-level perspective, photographic texture.",
        "Vivid wildfire in vegetation: clusters of flames with varying heights, yellow-white centers fading to orange-red tips, "
        "gray smoke trails, glowing embers, warm afternoon light, photorealistic detail.",
    ],
    "full_flame": [
        "Photorealistic intense wildfire engulfing brush and grass: many overlapping bright orange flames of varied height, "
        "thick dark smoke billowing upward, scorched ground beneath, glowing embers in the air, "
        "natural directional light, dramatic news photography style.",
        "Realistic photo of a major active wildfire: dense flame tongues spreading across vegetation, "
        "hot yellow-orange centers, thick rising smoke column, floating ash debris, "
        "ground-level perspective, sharp realistic detail.",
        "Natural wildfire scene at peak intensity: waves of flickering orange flames across shrubs and grass, "
        "billowing gray-brown smoke, ember storm, blackened plant bases, "
        "warm intense afternoon light, documentary photo realism.",
        "Photorealistic large wildfire: vegetation overtaken by flame waves of varying height, "
        "rising thick smoke, embers flying, ash floating, intense heat glow, "
        "natural outdoor lighting, real disaster photography quality.",
    ],
    "smoke_dominant": [
        "Photorealistic smoldering wildfire: small low irregular orange flames at the vegetation base with thick gray-brown smoke columns rising directly above them, "
        "charred grass beneath each flame, glowing embers visible, "
        "natural daylight filtering through smoke, atmospheric haze, organic flame tongues never round blobs.",
        "Realistic brush fire photo dominated by smoke: a few small flame pockets of varied size at vegetation base, "
        "dense semi-transparent smoke plumes rising vertically with natural turbulence, sunlight scattering through the smoke, "
        "ground-level perspective, documentary aesthetic, flames match daylight color temperature.",
        "Natural wildfire shot heavy with smoke: low flickering irregular flames nearly hidden in vegetation, "
        "thick dark smoke rising from each individual fire spot, glowing embers underneath, "
        "warm directional sunlight, hazy atmosphere, news camera realism, asymmetric smoke distribution.",
        "Photorealistic smoky vegetation fire: small bright flame tongues at the base with pointed irregular tips, billowing gray smoke columns "
        "rising and gradually fading into the sky, ash particles in the air, "
        "light beams cutting through smoke, soft photographic detail, flames blend naturally with the scene lighting.",
    ],
}

WILDFIRE_ENGINES = {
    WILDFIRE_ENGINE_KONTEXT_HINT: "자연어 편집",
    WILDFIRE_ENGINE_FILL_MASK: "정밀 마스크 Fill",
}
WILDFIRE_COMPOSITION_GUIDANCE = {
    "balanced": "Spread fire as several small organic pockets with green vegetation visible between them.",
    "flame_dominant": "Flames are the main subject; smoke stays light and attached to the flames.",
    "full_flame": "Most vegetation burns with dense varied flames and rising smoke, still organic in shape.",
    "smoke_dominant": "Smoke dominates the scene but rises from clearly visible small flames at the base.",
}

FOOT_MODEL_CONFIGS = {
    "corn": {
        "label": "Corn",
        "model_dir": FOOT_ROOT / "output" / "corn_inpaint_sd15_200_4_5e-6_newcap",
        "prompt_files": [FOOT_ROOT / "corn_captions_llava_med_v1.jsonl"],
        "negative": "blurry, cartoon, drawing, painting, unrealistic, watermark, logo, text",
        "out_subdir": "corn",
        "prompts": [
            "Close-up clinical photograph of the plantar surface of an adult foot, showing a single hard corn located under the second metatarsal head, about 6-7 mm in diameter, with a dense yellow central core and a sharply demarcated hyperkeratotic rim, rough and warty surface texture, mild surrounding callus.",
            "Close-up clinical photo of the sole of the foot with a small plantar corn near the lateral forefoot, roughly 4-5 mm, conical pale-yellow central core, thin elevated rim of compact keratin, surrounding skin slightly erythematous and dry, lesion surface finely rough like sandpaper.",
            "Plantar foot macro image showing a well-defined round corn on the ball of the foot beneath the third toe, about 8 mm, deep central cone-shaped core with darker yellow center, thick raised rim blending into a broader callused area, surface texture distinctly warty and irregular.",
            "Clinical close-up of the plantar heel with a solitary hard corn slightly off-center, 5-6 mm in diameter, central opaque core surrounded by a narrow halo of compact keratin, surrounding skin dry and fissured, corn surface hard and glossy.",
            "Macro clinical photo of the plantar forefoot showing a corn along the weight-bearing line, 5-6 mm, central translucent core with yellow-white coloration, rim narrow but very dense, surface moderately rough, mild erythema at the periphery.",
        ],
    },
    "crack": {
        "label": "Crack",
        "model_dir": FOOT_ROOT / "output" / "crack_inpaint_sd15_200_4_5e-6",
        "prompt_files": [FOOT_ROOT / "crack_captions_llava_med_v1.jsonl"],
        "negative": "blurry, cartoon, drawing, painting, unrealistic, watermark, logo, text",
        "out_subdir": "crack",
        "prompts": [
            "Close-up clinical photograph of a plantar heel with dry skin and a single deep vertical crack crossing the thick callused skin, sharply demarcated edges, mild surrounding dryness.",
            "Macro clinical photo of the heel showing multiple fine superficial skin cracks radiating through a broad yellow callus, no bleeding, surrounding skin dry and flaky.",
            "Plantar heel close-up with a wide central fissure in the callused skin, edges slightly elevated, base shallow and dry, adjacent skin thick and rough.",
            "Clinical dermatology photo of the plantar heel margin with several parallel shallow cracks in thick hyperkeratotic skin, background very dry and scaly.",
            "Macro view of the plantar heel with a single long diagonal fissure through a thick yellow callus, crack edges clean and well-defined, surrounding skin dry.",
        ],
    },
}


def autocast_context(dtype: torch.dtype | None = None):
    if DEVICE == "cuda":
        return torch.autocast("cuda", dtype=dtype or torch.float16)
    return nullcontext()


def make_seed(seed_value) -> int:
    try:
        seed = int(seed_value)
    except (TypeError, ValueError):
        seed = -1
    if seed < 0:
        return random.randint(0, 2**31 - 1)
    return seed


def utc_now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


def clamp_int(value, name: str, min_value: int, max_value: int) -> int:
    try:
        parsed = int(value)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"{name} 값이 올바르지 않습니다.") from exc
    if parsed < min_value or parsed > max_value:
        raise ValueError(f"{name} 값은 {min_value}부터 {max_value} 사이여야 합니다.")
    return parsed


def clamp_float(value, name: str, min_value: float, max_value: float) -> float:
    try:
        parsed = float(value)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"{name} 값이 올바르지 않습니다.") from exc
    if parsed < min_value or parsed > max_value:
        raise ValueError(f"{name} 값은 {min_value:g}부터 {max_value:g} 사이여야 합니다.")
    return parsed


def read_image_bytes(data: bytes) -> tuple[Image.Image, str]:
    image_id = hashlib.md5(data).hexdigest()[:12]
    return Image.open(io.BytesIO(data)).convert("RGB").copy(), image_id


def read_jsonl_prompts(path: Path) -> list[str]:
    prompts: list[str] = []
    if not path.exists():
        return prompts
    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            line = line.strip()
            if not line:
                continue
            try:
                record = json.loads(line)
            except json.JSONDecodeError:
                continue
            text = record.get("caption_core") or record.get("caption") or record.get("text") or record.get("prompt")
            if isinstance(text, str):
                text = " ".join(text.strip().split())
                if len(text) > 20 and text not in prompts:
                    prompts.append(text)
    return prompts


@lru_cache(maxsize=2)
def get_foot_prompt_pool(model_name: str) -> tuple[str, ...]:
    config = FOOT_MODEL_CONFIGS[model_name]
    prompts: list[str] = list(config["prompts"])
    for prompt_file in config.get("prompt_files", []):
        prompts.extend(read_jsonl_prompts(Path(prompt_file)))

    cleaned: list[str] = []
    for prompt in prompts:
        prompt = " ".join(str(prompt).split())
        if prompt and prompt not in cleaned:
            cleaned.append(prompt)
    return tuple(cleaned)


def choose_foot_prompts(model_name: str, count: int, seed: int) -> list[str]:
    prompt_pool = list(get_foot_prompt_pool(model_name))
    if not prompt_pool:
        prompt_pool = list(FOOT_MODEL_CONFIGS[model_name]["prompts"])
    rng = random.Random(seed)
    rng.shuffle(prompt_pool)
    return [prompt_pool[idx % len(prompt_pool)] for idx in range(max(1, int(count)))]


def build_foot_prompt(model_name: str, prompt: str) -> str:
    if model_name == "corn":
        return (
            f"{prompt} The plantar corn must be clearly visible and clinically distinct, "
            "with a pronounced yellow-white hyperkeratotic central core, raised compact rim, "
            "rough hard texture, and strong contrast from surrounding normal skin."
        )
    return prompt


def resize_to_square(image: Image.Image, size: int = FOOT_TARGET_SIZE) -> Image.Image:
    return image.convert("RGB").resize((size, size), Image.BICUBIC)


def overlay_mask(background: Image.Image, mask: Image.Image, color=(255, 0, 0), alpha=0.35) -> Image.Image:
    bg_arr = np.array(background.convert("RGB"), dtype=np.uint8)
    mask_arr = np.array(mask.convert("L"), dtype=np.uint8)
    overlay = np.zeros_like(bg_arr, dtype=np.uint8)
    overlay[mask_arr > 0] = np.array(color, dtype=np.uint8)
    return Image.fromarray(cv2.addWeighted(bg_arr, 1.0, overlay, alpha, 0.0))


def make_wildfire_hint_image(source_image: Image.Image, mask_image: Image.Image, alpha: float = 0.35) -> Image.Image:
    """Kontext 입력용 색상 힌트 합성. alpha 0.35 + 노란 테두리 + 마스크 중심점으로 위치 인지 강화."""
    image = source_image.convert("RGB")
    bg_arr = np.array(image, dtype=np.uint8)
    mask_arr = np.array(mask_image.convert("L").resize(image.size, Image.NEAREST), dtype=np.uint8)
    binary = np.where(mask_arr > 8, 255, 0).astype(np.uint8)
    if not np.count_nonzero(binary):
        return image

    edge_kernel = np.ones((9, 9), np.uint8)
    edge = cv2.subtract(cv2.dilate(binary, edge_kernel, iterations=1), cv2.erode(binary, edge_kernel, iterations=1))
    overlay = bg_arr.copy()
    fill_color = np.array([245, 112, 24], dtype=np.float32)
    edge_color = np.array([255, 210, 64], dtype=np.float32)
    fill_alpha = float(np.clip(alpha, 0.05, 0.55))
    edge_alpha = 0.55
    fill_pixels = binary > 0
    edge_pixels = edge > 0
    overlay[fill_pixels] = (
        bg_arr[fill_pixels].astype(np.float32) * (1.0 - fill_alpha) + fill_color * fill_alpha
    ).astype(np.uint8)
    overlay[edge_pixels] = (
        overlay[edge_pixels].astype(np.float32) * (1.0 - edge_alpha) + edge_color * edge_alpha
    ).astype(np.uint8)

    # 마스크 중심점에 작은 빨간 표시 → Kontext가 합성 위치를 더 강하게 인지
    ys, xs = np.where(binary > 0)
    if xs.size:
        cx = int(np.mean(xs))
        cy = int(np.mean(ys))
        radius = max(4, int(min(image.size) * 0.008))
        cv2.circle(overlay, (cx, cy), radius, (255, 30, 30), thickness=-1, lineType=cv2.LINE_AA)
        cv2.circle(overlay, (cx, cy), radius + 2, (255, 255, 255), thickness=2, lineType=cv2.LINE_AA)
    return Image.fromarray(overlay, mode="RGB")


def process_mask(mask_image: Image.Image, dilate: int = 0, feather: int = 0) -> Image.Image:
    mask_arr = np.array(mask_image.convert("L").resize((FOOT_TARGET_SIZE, FOOT_TARGET_SIZE), Image.NEAREST))
    mask_arr = np.where(mask_arr > 8, 255, 0).astype(np.uint8)

    if dilate > 0:
        kernel = np.ones((int(dilate), int(dilate)), np.uint8)
        mask_arr = cv2.dilate(mask_arr, kernel, iterations=1)

    if feather > 0:
        sigma = max(0.5, float(feather) / 3.0)
        mask_arr = cv2.GaussianBlur(mask_arr, (0, 0), sigmaX=sigma, sigmaY=sigma)
        mask_arr = np.clip(mask_arr, 0, 255).astype(np.uint8)

    return Image.fromarray(mask_arr, mode="L")


def process_wildfire_label_mask(
    mask_image: Image.Image,
    size: tuple[int, int],
    dilate: int = 0,
    feather: int = 0,
) -> Image.Image:
    mask_arr = np.array(mask_image.convert("L").resize(size, Image.NEAREST), dtype=np.uint8)
    mask_arr = np.where(mask_arr > 8, 255, 0).astype(np.uint8)
    if dilate > 0:
        kernel = np.ones((int(dilate), int(dilate)), np.uint8)
        mask_arr = cv2.dilate(mask_arr, kernel, iterations=1)
    if feather > 0:
        sigma = max(0.5, float(feather) / 3.0)
        mask_arr = cv2.GaussianBlur(mask_arr, (0, 0), sigmaX=sigma, sigmaY=sigma)
        mask_arr = np.clip(mask_arr, 0, 255).astype(np.uint8)
    return Image.fromarray(mask_arr, mode="L")


@lru_cache(maxsize=1)
def get_clipseg_pipeline():
    from transformers import CLIPSegForImageSegmentation, CLIPSegProcessor

    processor = CLIPSegProcessor.from_pretrained("CIDAS/clipseg-rd64-refined")
    model = CLIPSegForImageSegmentation.from_pretrained("CIDAS/clipseg-rd64-refined")
    if DEVICE == "cuda":
        model = model.to(DEVICE)
    return processor, model.eval()


def refine_mask_to_vegetation(
    source_image: Image.Image,
    user_mask: Image.Image,
    threshold: float = WILDFIRE_DEFAULT_VEG_THRESHOLD,
    dilate_px: int = 4,
    min_coverage_ratio: float = WILDFIRE_DEFAULT_VEG_MIN_COVERAGE,
) -> tuple[Image.Image, dict]:
    processor, model = get_clipseg_pipeline()
    image = source_image.convert("RGB")
    # Positive: 식생 (max-pool로 한 쿼리라도 강하면 식생 인정)
    positive_queries = [
        "vegetation, trees, shrubs, grass, foliage, brush, leaves",
        "lawn, grass field, mountain slope, forest edge, dry grass, dead grass",
        "bushes, undergrowth, hedge, plants, ground cover",
    ]
    # Negative: 비식생 (도로/건물/물). 식생 점수에서 빼서 false positive 제거
    negative_queries = [
        "road, asphalt, pavement, concrete, sidewalk, paved surface, parking lot",
        "building, wall, roof, structure, fence, signpost",
        "water, river, pond, lake, sky, bare ground, rock, soil patch",
    ]
    all_queries = positive_queries + negative_queries
    inputs = processor(
        text=all_queries,
        images=[image] * len(all_queries),
        return_tensors="pt",
        padding=True,
    )
    if DEVICE == "cuda":
        inputs = {key: value.to(DEVICE) for key, value in inputs.items()}

    with torch.inference_mode():
        outputs = model(**inputs)

    logits = outputs.logits
    if logits.dim() == 2:
        logits = logits.unsqueeze(0)
    seg_all = torch.sigmoid(logits).detach().cpu().numpy().astype(np.float32)

    pos_seg = np.max(seg_all[: len(positive_queries)], axis=0)
    neg_seg = np.max(seg_all[len(positive_queries) :], axis=0)

    # 비식생 점수가 식생 점수보다 명확히 높은 영역을 식생에서 빼냄 (margin 0.05로 보수적 차감)
    refined_score = np.where(neg_seg > pos_seg + 0.05, 0.0, pos_seg).astype(np.float32)

    pos_full = cv2.resize(pos_seg, image.size, interpolation=cv2.INTER_LINEAR)
    neg_full = cv2.resize(neg_seg, image.size, interpolation=cv2.INTER_LINEAR)
    seg_full = cv2.resize(refined_score, image.size, interpolation=cv2.INTER_LINEAR)

    # 동적 threshold: 전체 이미지 식생 비율이 낮으면 threshold 자동 완화
    used_threshold = float(threshold)
    full_coverage_initial = float((seg_full > used_threshold).mean())
    if full_coverage_initial < 0.20:
        used_threshold = max(0.15, used_threshold - 0.10)
        if float((seg_full > used_threshold).mean()) < 0.10:
            used_threshold = max(0.10, used_threshold - 0.05)

    veg_mask = (seg_full > used_threshold).astype(np.uint8) * 255

    # 명확한 비식생(점수 차이가 큰 곳) 강제 제거
    hard_negative = (neg_full > pos_full + 0.10).astype(np.uint8) * 255
    veg_mask = np.where(hard_negative > 0, 0, veg_mask).astype(np.uint8)

    if dilate_px > 0:
        kernel = np.ones((int(dilate_px), int(dilate_px)), np.uint8)
        veg_mask = cv2.dilate(veg_mask, kernel, iterations=1)
        # dilate 후에도 hard_negative는 다시 제거 (도로 침범 방지)
        veg_mask = np.where(hard_negative > 0, 0, veg_mask).astype(np.uint8)

    user_arr = np.array(user_mask.convert("L").resize(image.size, Image.NEAREST), dtype=np.uint8)
    user_binary = np.where(user_arr > 8, 255, 0).astype(np.uint8)
    refined_arr = np.minimum(veg_mask, user_binary)

    user_pixels = int(np.count_nonzero(user_binary))
    refined_pixels = int(np.count_nonzero(refined_arr))
    coverage = (refined_pixels / user_pixels) if user_pixels > 0 else 0.0
    nonveg_in_user = int(np.count_nonzero(np.minimum(hard_negative, user_binary)))
    info = {
        "vegetation_coverage": float(np.count_nonzero(veg_mask)) / float(veg_mask.size),
        "user_mask_pixels": user_pixels,
        "refined_mask_pixels": refined_pixels,
        "coverage_ratio": float(coverage),
        "threshold_requested": float(threshold),
        "threshold_used": used_threshold,
        "dilate_px": int(dilate_px),
        "positive_queries": positive_queries,
        "negative_queries": negative_queries,
        "nonveg_pixels_removed": nonveg_in_user,
    }

    if coverage < float(min_coverage_ratio):
        # Fallback: 정제 결과가 너무 작아도, 비식생 영역은 빼낸 user mask 사용
        # → 도로/건물은 빠지지만 사용자 마스크 형태는 가능한 보존
        user_minus_neg = np.where(hard_negative > 0, 0, user_binary).astype(np.uint8)
        fallback_pixels = int(np.count_nonzero(user_minus_neg))
        if fallback_pixels >= max(1500, int(user_pixels * 0.30)):
            info["applied"] = True
            info["reason"] = (
                f"refined coverage {coverage:.2%} below min {min_coverage_ratio:.0%}; "
                f"using user mask minus detected non-vegetation ({fallback_pixels} px)"
            )
            info["fallback_mode"] = "user_minus_negative"
            return Image.fromarray(user_minus_neg, mode="L"), info
        # 그래도 너무 작으면 원본 user mask 사용
        info["applied"] = False
        info["reason"] = (
            f"refined coverage {coverage:.2%} below min {min_coverage_ratio:.0%}; "
            f"using original user mask (scene vegetation: {full_coverage_initial:.1%})"
        )
        return user_mask, info

    info["applied"] = True
    info["reason"] = "vegetation refinement applied"
    return Image.fromarray(refined_arr, mode="L"), info


def make_wildfire_fire_core_mask(
    label_mask: Image.Image,
    seed: int = 0,
    min_ratio: float = WILDFIRE_DEFAULT_FIRE_CORE_MIN_RATIO,
    max_ratio: float = WILDFIRE_DEFAULT_FIRE_CORE_MAX_RATIO,
) -> Image.Image:
    """사용자 마스크를 거의 그대로 fire core로 사용. 분포 변형은 perturb 단계에서만 수행."""
    label_arr = np.array(label_mask.convert("L"), dtype=np.uint8)
    binary = np.where(label_arr > 8, 255, 0).astype(np.uint8)
    if not np.count_nonzero(binary):
        return Image.fromarray(binary, mode="L")
    return Image.fromarray(binary, mode="L")


def make_wildfire_generation_mask(
    label_mask: Image.Image,
    feather: int = 6,
    smoke_headroom_percent: int = 30,
    smoke_paste_opacity: int = 210,
    perturb_strength: float = 0.40,
    seed: int = 0,
) -> tuple[Image.Image, Image.Image, Image.Image]:
    rng = np.random.default_rng(int(seed) & 0xFFFFFFFF)
    core_mask = make_wildfire_fire_core_mask(label_mask, seed=seed)
    edit_arr = np.array(core_mask.convert("L"), dtype=np.uint8)
    h, w = edit_arr.shape

    # A) 외곽만 약간 흐트러뜨림. 마스크 내부는 절대 잘리지 않게 함 (분리 방지)
    preserved = edit_arr.copy()  # 사용자 마스크 형태 보존용
    if perturb_strength > 0 and np.count_nonzero(edit_arr) > 0:
        band_px = max(3, int(min(h, w) * 0.012 * float(perturb_strength)))
        ring_kernel = np.ones((band_px * 2 + 1, band_px * 2 + 1), np.uint8)
        dilated_for_ring = cv2.dilate(edit_arr, ring_kernel, iterations=1)
        outer_ring = cv2.subtract(dilated_for_ring, edit_arr)

        noise = rng.random((h, w)).astype(np.float32)
        noise = cv2.GaussianBlur(noise, (0, 0), sigmaX=max(2.0, band_px * 0.6), sigmaY=max(2.0, band_px * 0.6))
        threshold = float(np.clip(0.62 - 0.14 * perturb_strength, 0.42, 0.68))
        noise_mask = ((noise > threshold).astype(np.uint8)) * 255
        edit_arr = np.maximum(edit_arr, np.minimum(outer_ring, noise_mask))

        # opening은 작은 가시만 다듬는 정도로 약화
        open_k = max(3, int(min(h, w) * 0.0015)) | 1
        open_kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (open_k, open_k))
        edit_arr = cv2.morphologyEx(edit_arr, cv2.MORPH_OPEN, open_kernel)
        # closing으로 분리된 조각을 다시 연결
        close_k = max(5, int(min(h, w) * 0.008)) | 1
        close_kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (close_k, close_k))
        edit_arr = cv2.morphologyEx(edit_arr, cv2.MORPH_CLOSE, close_kernel)
        # 핵심 보호: 원본 fire core가 잘려나가지 않도록 OR 복원
        edit_arr = np.maximum(edit_arr, preserved)

    paste_arr = np.where(edit_arr > 0, 255, 0).astype(np.uint8)
    ys, xs = np.where(edit_arr > 0)

    # B) Smoke headroom: column 위쪽으로 갈수록 alpha 페이드되는 gradient
    if len(xs) and smoke_headroom_percent > 0:
        x_min, x_max = int(xs.min()), int(xs.max())
        y_min, y_max = int(ys.min()), int(ys.max())
        bbox_h = max(1, y_max - y_min + 1)
        headroom_px = max(1, int(bbox_h * smoke_headroom_percent / 100))

        smoke_alpha_f = np.zeros_like(edit_arr, dtype=np.float32)
        for x in range(x_min, x_max + 1):
            col_nonzero = np.where(edit_arr[:, x] > 0)[0]
            if col_nonzero.size == 0:
                continue
            top = int(col_nonzero.min())
            y_start = max(0, top - headroom_px)
            ramp_h = top - y_start
            if ramp_h <= 0:
                continue
            # top에 가까운 곳은 진하고, 위로 갈수록 부드럽게 사라짐 (1.4 거듭제곱 = 부드러운 페이드)
            ratios = np.linspace(0.0, 1.0, ramp_h, dtype=np.float32)
            ramp_values = 255.0 * (1.0 - ratios) ** 1.4
            smoke_alpha_f[y_start:top, x] = np.maximum(smoke_alpha_f[y_start:top, x], ramp_values)

        # 가로/세로로 자연스럽게 퍼지게 dilate + blur
        width_kernel = (max(9, int((x_max - x_min + 1) * 0.035)) | 1)
        height_kernel = (max(9, int(headroom_px * 0.14)) | 1)
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (width_kernel, height_kernel))
        smoke_alpha_u8 = smoke_alpha_f.clip(0, 255).astype(np.uint8)
        smoke_alpha_u8 = cv2.dilate(smoke_alpha_u8, kernel, iterations=1)
        blur_sigma = max(3.0, headroom_px * 0.14)
        smoke_alpha_u8 = cv2.GaussianBlur(smoke_alpha_u8, (0, 0), sigmaX=blur_sigma, sigmaY=blur_sigma)

        # Smoke는 낮은 강도의 soft mask로만 열어 두어, 회색 띠처럼 통째로 다시 그려지는 것을 줄인다.
        smoke_edit = np.minimum(smoke_alpha_u8, 135).astype(np.uint8)
        edit_arr = np.maximum(edit_arr, smoke_edit)
        smoke_paste = np.minimum(smoke_alpha_u8, int(smoke_paste_opacity))
        paste_arr = np.maximum(paste_arr, smoke_paste)

    # 외곽 페더링
    soft_arr = edit_arr.copy()
    soft_paste_arr = paste_arr.copy()
    if feather > 0:
        sigma = max(0.5, float(feather) / 2.0)
        soft_arr = cv2.GaussianBlur(soft_arr, (0, 0), sigmaX=sigma, sigmaY=sigma)
        soft_arr = np.clip(soft_arr, 0, 255).astype(np.uint8)
        soft_paste_arr = cv2.GaussianBlur(soft_paste_arr, (0, 0), sigmaX=sigma, sigmaY=sigma)
        soft_paste_arr = np.clip(soft_paste_arr, 0, 255).astype(np.uint8)

    return (
        Image.fromarray(edit_arr, mode="L"),
        Image.fromarray(soft_arr, mode="L"),
        Image.fromarray(soft_paste_arr, mode="L"),
    )


def mask_bbox(mask: Image.Image, padding: int) -> tuple[int, int, int, int]:
    mask_arr = np.array(mask.convert("L"), dtype=np.uint8)
    ys, xs = np.where(mask_arr > 0)
    if len(xs) == 0:
        raise ValueError("마스크가 비어 있습니다. 산불 생성 영역을 먼저 그려주세요.")
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


def mask_has_pixels(mask: Image.Image | None) -> bool:
    if mask is None:
        return False
    return bool(np.count_nonzero(np.array(mask.convert("L"))))


def image_to_png_bytes(image: Image.Image) -> bytes:
    buffer = io.BytesIO()
    image.convert("RGB").save(buffer, format="PNG")
    return buffer.getvalue()


def image_to_data_url(image: Image.Image) -> str:
    encoded = base64.b64encode(image_to_png_bytes(image)).decode("ascii")
    return f"data:image/png;base64,{encoded}"


def images_to_zip_base64(images: list[Image.Image], prefix: str) -> str:
    buffer = io.BytesIO()
    with zipfile.ZipFile(buffer, "w", compression=zipfile.ZIP_DEFLATED) as archive:
        for idx, image in enumerate(images, start=1):
            archive.writestr(f"{prefix}_{idx:02d}.png", image_to_png_bytes(image))
    return base64.b64encode(buffer.getvalue()).decode("ascii")


def upload_artifacts(run_id: str, prefix: str, artifacts: dict[str, Image.Image | dict | Path]) -> dict:
    storage = get_storage()
    storage_info = {
        "enabled": storage.enabled,
        "available": False,
        "bucket": storage.bucket,
        "prefix": prefix,
        "objects": {},
        "urls": {},
        "error": None,
    }
    if not storage.enabled:
        return storage_info

    try:
        storage.ensure_bucket()
        storage_info["available"] = True
        for name, artifact in artifacts.items():
            object_name = f"{prefix}/{name}"
            if isinstance(artifact, Image.Image):
                uploaded = storage.put_image(object_name, artifact)
            elif isinstance(artifact, Path):
                uploaded = storage.put_file(object_name, artifact)
            else:
                uploaded = storage.put_json(object_name, artifact)
            if uploaded:
                storage_info["objects"][name] = uploaded
                storage_info["urls"][name] = storage.presigned_get_url(uploaded)
    except Exception as exc:
        storage.last_error = str(exc)
        storage_info["error"] = f"{run_id} MinIO 저장 실패: {exc}"
    return storage_info


def choose_wildfire_variant(composition_mode: str, seed: int) -> str:
    mode = composition_mode if composition_mode in WILDFIRE_COMPOSITION_PROMPTS else "flame_dominant"
    variants = WILDFIRE_COMPOSITION_PROMPTS[mode]
    return random.Random(f"{seed}:{mode}").choice(variants)


def build_wildfire_prompt(
    user_prompt: str,
    composition_mode: str,
    seed: int,
    negative_prompt: str = "",
) -> tuple[str, str, str]:
    clean_prompt = " ".join((user_prompt or "").split())
    mode = composition_mode if composition_mode in WILDFIRE_COMPOSITION_PROMPTS else "flame_dominant"
    selected_variant = choose_wildfire_variant(mode, seed)
    composition_instruction = WILDFIRE_COMPOSITION_GUIDANCE[mode]
    # 시각 어휘를 앞쪽에 배치 (T5는 앞쪽 토큰을 더 강하게 반영)
    final_prompt = (
        f"{selected_variant} {composition_instruction} "
        f"{WILDFIRE_EDIT_GUARDRAIL}"
    )
    if clean_prompt:
        final_prompt += f" Additional context: {clean_prompt}"
    if negative_prompt:
        # 긴 negative는 T5 토큰 낭비 → 핵심 200자만 사용
        clean_neg = " ".join(negative_prompt.split())
        if len(clean_neg) > 220:
            clean_neg = clean_neg[:220].rsplit(",", 1)[0]
        final_prompt += f" Avoid: {clean_neg}"
    return final_prompt, mode, selected_variant


def build_wildfire_kontext_hint_prompt(
    user_prompt: str,
    composition_mode: str,
    seed: int,
) -> tuple[str, str, str]:
    clean_prompt = " ".join((user_prompt or "").split())
    mode = composition_mode if composition_mode in WILDFIRE_COMPOSITION_PROMPTS else "flame_dominant"
    selected_variant = choose_wildfire_variant(mode, seed)
    final_prompt = (
        f"{selected_variant} "
        "The translucent orange overlay and red center dot in the input photo mark the exact location where the wildfire must appear. "
        "Place flames inside that marked region ONLY. Do not add flames anywhere else in the scene. "
        "Remove all overlay color, the red dot, and the yellow outline from the final image. "
        "Add the wildfire only on real vegetation (grass, shrubs, leaves, tree branches) within or immediately near the marked area. "
        "Skip any non-vegetation surface such as road, asphalt, water, sky, buildings, or bare ground inside the marked area — "
        "those pixels stay identical to the original photo. "
        "Flames must look physically integrated: they emerge between leaves and grass, "
        "lighting and color match the surrounding scene, leaves partially occlude flame edges, "
        "and thin uneven smoke rises softly from the flame sources. "
        "Preserve the exact camera framing, aspect ratio, road shape, trees, and background composition. "
        "Avoid sticker overlays, neon glow, geometric flame shapes, and burn patterns that match the marked region outline."
    )
    if clean_prompt:
        final_prompt += f" Additional context: {clean_prompt}"
    return final_prompt, mode, selected_variant


def normalize_wildfire_engine(engine: str | None) -> str:
    return engine if engine in WILDFIRE_ENGINES else WILDFIRE_DEFAULT_ENGINE


def save_wildfire_artifacts(
    result: Image.Image,
    source_image: Image.Image,
    user_prompt: str,
    final_prompt: str,
    negative_prompt: str,
    seed: int,
    steps: int,
    guidance: float,
    output_size: int,
    composition_mode: str,
    selected_prompt_variant: str,
    model_id: str,
    label_mask: Image.Image | None = None,
    edit_mask: Image.Image | None = None,
    crop_padding: int | None = None,
    mask_dilate: int = 0,
    mask_feather: int = 0,
    fill_guidance: float | None = None,
    smoke_headroom: int | None = None,
    perturb_strength: float | None = None,
    mask_refine_info: dict | None = None,
    wildfire_engine: str = WILDFIRE_DEFAULT_ENGINE,
) -> tuple[Path, Path, dict]:
    timestamp = time.strftime("%Y%m%d_%H%M%S")
    run_id = f"wildfire_{timestamp}_seed{seed}"
    image_path = WILDFIRE_OUTPUT_DIR / f"{run_id}.png"
    metadata_path = WILDFIRE_OUTPUT_DIR / f"{run_id}.json"
    result.save(image_path)
    label_mask_path = None
    edit_mask_path = None
    if label_mask is not None:
        label_mask_path = WILDFIRE_OUTPUT_DIR / f"{run_id}_event_region_mask.png"
        label_mask.save(label_mask_path)
    if edit_mask is not None:
        edit_mask_path = WILDFIRE_OUTPUT_DIR / f"{run_id}_generation_edit_mask.png"
        edit_mask.save(edit_mask_path)
    metadata = {
        "run_id": run_id,
        "kind": "wildfire",
        "created_at": utc_now_iso(),
        "model": model_id,
        "wildfire_engine": wildfire_engine,
        "seed": seed,
        "steps": int(steps),
        "guidance": float(guidance),
        "fill_guidance": float(fill_guidance) if fill_guidance is not None else None,
        "output_size": int(output_size),
        "crop_padding": int(crop_padding) if crop_padding is not None else None,
        "mask_dilate": int(mask_dilate),
        "mask_feather": int(mask_feather),
        "smoke_headroom": int(smoke_headroom) if smoke_headroom is not None else None,
        "perturb_strength": float(perturb_strength) if perturb_strength is not None else None,
        "composition_mode": composition_mode,
        "composition_label": WILDFIRE_COMPOSITION_LABELS.get(composition_mode, composition_mode),
        "selected_prompt_variant": selected_prompt_variant,
        "source_size": list(source_image.size),
        "user_prompt": user_prompt,
        "final_prompt": final_prompt,
        "negative_prompt": negative_prompt,
        "saved_image": str(image_path),
        "event_region_mask": str(label_mask_path) if label_mask_path else None,
        "generation_edit_mask": str(edit_mask_path) if edit_mask_path else None,
        "mask_refinement": mask_refine_info,
    }
    artifacts: dict[str, Image.Image | dict | Path] = {
        "image.png": result,
        "metadata.json": metadata,
    }
    if label_mask is not None:
        artifacts["event_region_mask.png"] = label_mask
    if edit_mask is not None:
        artifacts["generation_edit_mask.png"] = edit_mask
    storage_info = upload_artifacts(run_id, f"wildfire/{run_id}", artifacts)
    metadata["storage"] = storage_info
    metadata["storage_image"] = storage_info["objects"].get("image.png")
    metadata["storage_metadata"] = storage_info["objects"].get("metadata.json")
    if storage_info["objects"].get("event_region_mask.png"):
        metadata["storage_event_region_mask"] = storage_info["objects"]["event_region_mask.png"]
    if storage_info["objects"].get("generation_edit_mask.png"):
        metadata["storage_generation_edit_mask"] = storage_info["objects"]["generation_edit_mask.png"]
    metadata_path.write_text(json.dumps(metadata, ensure_ascii=False, indent=2), encoding="utf-8")
    if storage_info["available"]:
        metadata_object = f"wildfire/{run_id}/metadata.json"
        try:
            get_storage().put_json(metadata_object, metadata)
            storage_info["objects"]["metadata.json"] = metadata_object
            storage_info["urls"]["metadata.json"] = get_storage().presigned_get_url(metadata_object)
        except Exception as exc:
            storage_info["error"] = f"{run_id} MinIO 메타데이터 갱신 실패: {exc}"
    return image_path, metadata_path, storage_info


@lru_cache(maxsize=1)
def get_wildfire_pipeline():
    hf_token = os.environ.get("HF_TOKEN") or os.environ.get("HUGGINGFACE_HUB_TOKEN")
    if not hf_token:
        raise RuntimeError("HF_TOKEN 환경 변수가 필요합니다. FLUX 모델 접근 동의 후 토큰을 설정해주세요.")
    pipe = FluxKontextPipeline.from_pretrained(
        WILDFIRE_MODEL_ID,
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


@lru_cache(maxsize=1)
def get_wildfire_fill_pipeline():
    hf_token = os.environ.get("HF_TOKEN") or os.environ.get("HUGGINGFACE_HUB_TOKEN")
    if not hf_token:
        raise RuntimeError("HF_TOKEN 환경 변수가 필요합니다. FLUX Fill 모델 접근 동의 후 토큰을 설정해주세요.")
    pipe = FluxFillPipeline.from_pretrained(
        WILDFIRE_FILL_MODEL_ID,
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


@lru_cache(maxsize=2)
def get_foot_pipeline(model_name: str):
    config = FOOT_MODEL_CONFIGS[model_name]
    pipe = StableDiffusionInpaintPipeline.from_pretrained(
        str(config["model_dir"]),
        torch_dtype=TORCH_DTYPE,
        safety_checker=None,
    )
    if DEVICE != "cuda":
        pipe.enable_attention_slicing()
    return pipe.to(DEVICE)


def generate_wildfire_image(
    source_image: Image.Image,
    mask_image: Image.Image | None,
    prompt: str,
    negative_prompt: str,
    steps: int,
    guidance: float,
    seed,
    output_size: int,
    save_output: bool,
    fill_guidance: float = WILDFIRE_DEFAULT_FILL_GUIDANCE,
    composition_mode: str = "flame_dominant",
    smoke_headroom: int = WILDFIRE_DEFAULT_SMOKE_HEADROOM,
    mask_dilate: int = WILDFIRE_DEFAULT_MASK_DILATE,
    mask_feather: int = WILDFIRE_DEFAULT_MASK_FEATHER,
    vegetation_refine: bool = WILDFIRE_DEFAULT_VEG_REFINE,
    vegetation_threshold: float = WILDFIRE_DEFAULT_VEG_THRESHOLD,
    vegetation_min_coverage: float = WILDFIRE_DEFAULT_VEG_MIN_COVERAGE,
    wildfire_engine: str = WILDFIRE_DEFAULT_ENGINE,
):
    steps = clamp_int(steps, "steps", 10, 80)
    guidance = clamp_float(guidance, "guidance", 1.0, 10.0)
    fill_guidance = clamp_float(fill_guidance, "fill_guidance", 1.0, 50.0)
    output_size = clamp_int(output_size, "output_size", 512, 1536)
    smoke_headroom = clamp_int(smoke_headroom, "smoke_headroom", 0, 100)
    mask_dilate = clamp_int(mask_dilate, "mask_dilate", 0, 32)
    mask_feather = clamp_int(mask_feather, "mask_feather", 0, 32)
    vegetation_threshold = clamp_float(vegetation_threshold, "vegetation_threshold", 0.1, 0.7)
    vegetation_min_coverage = clamp_float(vegetation_min_coverage, "vegetation_min_coverage", 0.0, 1.0)
    wildfire_engine = normalize_wildfire_engine(wildfire_engine)
    seed = make_seed(seed)
    init_image = source_image.convert("RGB")
    generator = torch.Generator(device="cpu").manual_seed(seed)
    final_prompt, normalized_composition_mode, selected_prompt_variant = build_wildfire_prompt(
        prompt,
        composition_mode=composition_mode,
        seed=seed,
        negative_prompt=negative_prompt,
    )

    model_id = WILDFIRE_MODEL_ID
    label_mask = None
    edit_mask = None
    crop_padding = None
    effective_guidance = guidance
    mask_refine_info = None
    if mask_image is not None and mask_has_pixels(mask_image):
        if vegetation_refine:
            try:
                mask_image, mask_refine_info = refine_mask_to_vegetation(
                    source_image=init_image,
                    user_mask=mask_image,
                    threshold=vegetation_threshold,
                    min_coverage_ratio=vegetation_min_coverage,
                )
            except Exception as exc:
                mask_refine_info = {"applied": False, "reason": f"error: {exc}"}
        label_mask = process_wildfire_label_mask(
            mask_image,
            init_image.size,
            dilate=mask_dilate,
            feather=mask_feather,
        )
        if wildfire_engine == WILDFIRE_ENGINE_KONTEXT_HINT:
            model_id = WILDFIRE_MODEL_ID
            effective_guidance = guidance
            edit_mask, soft_edit_mask, paste_mask = make_wildfire_generation_mask(
                label_mask,
                smoke_headroom_percent=int(smoke_headroom),
                perturb_strength=WILDFIRE_DEFAULT_PERTURB,
                seed=seed,
            )
            short_side = min(init_image.size)
            crop_padding = max(100, int(short_side * 0.28))
            x0, y0, x1, y1 = mask_bbox(edit_mask, padding=crop_padding)
            crop_box = (x0, y0, x1, y1)
            crop_image = init_image.crop(crop_box)
            crop_hint_mask = label_mask.crop(crop_box)
            crop_paste_mask = paste_mask.crop(crop_box)
            final_prompt, normalized_composition_mode, selected_prompt_variant = build_wildfire_kontext_hint_prompt(
                prompt,
                composition_mode=composition_mode,
                seed=seed,
            )
            hint_image = make_wildfire_hint_image(crop_image, crop_hint_mask)
            work_image = resize_to_multiple(hint_image, max_side=int(output_size), multiple=16, resample=Image.BICUBIC)
            pipe = get_wildfire_pipeline()
            with torch.inference_mode(), autocast_context(torch.bfloat16 if DEVICE == "cuda" else None):
                result = pipe(
                    prompt=final_prompt,
                    image=work_image,
                    height=work_image.height,
                    width=work_image.width,
                    guidance_scale=float(effective_guidance),
                    num_inference_steps=int(steps),
                    generator=generator,
                    max_area=work_image.width * work_image.height,
                    max_sequence_length=512,
                ).images[0]
            generated_crop = result.resize(crop_image.size, Image.BICUBIC)
            result = init_image.copy()
            result.paste(generated_crop, crop_box, crop_paste_mask)
        else:
            model_id = WILDFIRE_FILL_MODEL_ID
            effective_guidance = fill_guidance
            edit_mask, soft_edit_mask, paste_mask = make_wildfire_generation_mask(
                label_mask,
                smoke_headroom_percent=int(smoke_headroom),
                perturb_strength=WILDFIRE_DEFAULT_PERTURB,
                seed=seed,
            )
            short_side = min(init_image.size)
            crop_padding = max(60, int(short_side * 0.20))
            x0, y0, x1, y1 = mask_bbox(edit_mask, padding=crop_padding)
            crop_box = (x0, y0, x1, y1)
            crop_image = init_image.crop(crop_box)
            crop_soft_mask = soft_edit_mask.crop(crop_box)
            crop_paste_mask = paste_mask.crop(crop_box)
            work_image = resize_to_multiple(crop_image, max_side=int(output_size), multiple=16, resample=Image.BICUBIC)
            work_mask = crop_soft_mask.resize(work_image.size, Image.BILINEAR)
            fill_prompt = (
                f"{final_prompt} "
                "Real flames burning on the vegetation inside the marked area, "
                "with organic flickering flame tongues of varied heights, "
                "warm orange and yellow colors blending naturally with the daylight, "
                "thin smoke rising softly from each flame into the sky, "
                "individual plant leaves and grass blades still visible between and around the flames, "
                "subtle char only at the immediate flame base."
            )
            pipe = get_wildfire_fill_pipeline()
            with torch.inference_mode(), autocast_context(torch.bfloat16 if DEVICE == "cuda" else None):
                generated_work = pipe(
                    prompt=fill_prompt,
                    image=work_image,
                    mask_image=work_mask,
                    height=work_image.height,
                    width=work_image.width,
                    guidance_scale=float(effective_guidance),
                    num_inference_steps=int(steps),
                    generator=generator,
                    max_sequence_length=512,
                ).images[0]
            generated_crop = generated_work.resize(crop_image.size, Image.BICUBIC)
            result = init_image.copy()
            result.paste(generated_crop, crop_box, crop_paste_mask)
            final_prompt = fill_prompt
    else:
        pipe = get_wildfire_pipeline()
        with torch.inference_mode(), autocast_context(torch.bfloat16 if DEVICE == "cuda" else None):
            result = pipe(
                prompt=final_prompt,
                image=init_image,
                guidance_scale=float(guidance),
                num_inference_steps=int(steps),
                generator=generator,
                max_area=int(output_size) * int(output_size),
                max_sequence_length=512,
            ).images[0]

    saved_path = None
    meta_path = None
    storage_info = None
    if save_output:
        image_path, metadata_path, storage_info = save_wildfire_artifacts(
            result=result,
            source_image=init_image,
            user_prompt=prompt,
            final_prompt=final_prompt,
            negative_prompt=negative_prompt,
            seed=seed,
            steps=steps,
            guidance=effective_guidance,
            output_size=output_size,
            composition_mode=normalized_composition_mode,
            selected_prompt_variant=selected_prompt_variant,
            model_id=model_id,
            label_mask=label_mask,
            edit_mask=edit_mask,
            crop_padding=crop_padding,
            mask_dilate=mask_dilate if label_mask is not None else 0,
            mask_feather=mask_feather if label_mask is not None else 0,
            fill_guidance=fill_guidance if label_mask is not None else None,
            smoke_headroom=smoke_headroom if label_mask is not None else None,
            perturb_strength=WILDFIRE_DEFAULT_PERTURB if label_mask is not None else None,
            mask_refine_info=mask_refine_info,
            wildfire_engine=wildfire_engine,
        )
        saved_path = str(image_path)
        meta_path = str(metadata_path)

    return {
        "status": (
            f"산불 이미지 생성 완료 | {WILDFIRE_COMPOSITION_LABELS.get(normalized_composition_mode, normalized_composition_mode)} "
            f"| seed={seed} | "
            f"{WILDFIRE_ENGINES.get(wildfire_engine, wildfire_engine) if label_mask is not None else f'max_area={output_size}x{output_size}'}"
        ),
        "result": result,
        "event_region_mask": label_mask,
        "generation_edit_mask": edit_mask,
        "saved_path": saved_path,
        "metadata_path": meta_path,
        "storage": storage_info,
        "guidance": effective_guidance,
        "fill_guidance": fill_guidance if label_mask is not None else None,
        "wildfire_engine": wildfire_engine,
        "crop_padding": crop_padding,
        "composition_mode": normalized_composition_mode,
        "composition_label": WILDFIRE_COMPOSITION_LABELS.get(normalized_composition_mode, normalized_composition_mode),
        "selected_prompt_variant": selected_prompt_variant,
        "mask_refinement": mask_refine_info,
    }


def preview_foot_mask(source_image: Image.Image, mask_image: Image.Image, dilate: int, feather: int):
    dilate = clamp_int(dilate, "dilate", 0, 32)
    feather = clamp_int(feather, "feather", 0, 32)
    image_512 = resize_to_square(source_image)
    mask_512 = process_mask(mask_image, dilate=dilate, feather=feather)
    if not mask_has_pixels(mask_512):
        raise ValueError("마스크가 비어 있습니다. 합성할 영역을 먼저 그려주세요.")
    return {"input": image_512, "mask": mask_512, "preview": overlay_mask(image_512, mask_512)}


def generate_foot_lesions(
    model_name: str,
    source_image: Image.Image,
    mask_image: Image.Image,
    steps: int,
    guidance: float,
    strength: float,
    seed,
    save_output: bool,
    output_count: int,
    dilate: int = FOOT_DEFAULT_DILATE,
    feather: int = FOOT_DEFAULT_FEATHER,
):
    if model_name not in FOOT_MODEL_CONFIGS:
        raise ValueError("지원하지 않는 병변 모델입니다.")

    steps = clamp_int(steps, "steps", 10, 100)
    guidance = clamp_float(guidance, "guidance", 1.0, 15.0)
    strength = clamp_float(strength, "strength", 0.1, 1.0)
    output_count = clamp_int(output_count, "output_count", 1, 20)
    dilate = clamp_int(dilate, "dilate", 0, 32)
    feather = clamp_int(feather, "feather", 0, 32)

    config = FOOT_MODEL_CONFIGS[model_name]
    seed = make_seed(seed)
    image_512 = resize_to_square(source_image)
    mask_512 = process_mask(mask_image, dilate=dilate, feather=feather)
    if not mask_has_pixels(mask_512):
        raise ValueError("마스크가 비어 있습니다. 마스크 미리보기를 먼저 확인해주세요.")

    pipe = get_foot_pipeline(model_name)
    outputs = []
    saved_paths = []
    metadata_paths = []
    model_output_dir = FOOT_OUTPUT_DIR / config["out_subdir"]
    model_output_dir.mkdir(parents=True, exist_ok=True)
    timestamp = time.strftime("%Y%m%d_%H%M%S")
    run_id = f"{model_name}_{timestamp}_seed{seed}"
    run_metadata_path = model_output_dir / f"{run_id}_summary.json"
    storage_prefix = f"foot/{model_name}/{run_id}"

    selected_prompts = choose_foot_prompts(model_name, output_count, seed)
    applied_prompts = [build_foot_prompt(model_name, prompt) for prompt in selected_prompts]
    run_records = []
    for idx, prompt in enumerate(applied_prompts, start=1):
        image_seed = seed + idx
        generator = torch.Generator(device=DEVICE).manual_seed(image_seed)
        with torch.inference_mode(), autocast_context():
            result = pipe(
                prompt=prompt,
                negative_prompt=config["negative"],
                image=image_512,
                mask_image=mask_512,
                strength=float(strength),
                num_inference_steps=int(steps),
                guidance_scale=float(guidance),
                generator=generator,
            ).images[0]

        output_path = model_output_dir / f"{model_name}_{timestamp}_p{idx:02d}_seed{image_seed}.png"
        metadata_path = model_output_dir / f"{model_name}_{timestamp}_p{idx:02d}_seed{image_seed}.json"
        result.save(output_path)
        metadata = {
            "run_id": run_id,
            "kind": "foot",
            "created_at": utc_now_iso(),
            "model_name": model_name,
            "model_label": config["label"],
            "model_dir": str(config["model_dir"]),
            "index": idx,
            "seed": image_seed,
            "base_seed": seed,
            "steps": int(steps),
            "guidance": float(guidance),
            "strength": float(strength),
            "dilate": int(dilate),
            "feather": int(feather),
            "negative_prompt": config["negative"],
            "prompt": prompt,
            "base_prompt": selected_prompts[idx - 1],
            "source_size": list(source_image.size),
            "working_size": [FOOT_TARGET_SIZE, FOOT_TARGET_SIZE],
            "saved_image": str(output_path),
        }
        metadata_path.write_text(json.dumps(metadata, ensure_ascii=False, indent=2), encoding="utf-8")
        saved_paths.append(str(output_path))
        metadata_paths.append(str(metadata_path))
        run_records.append(metadata)
        outputs.append(result)

    run_metadata = {
        "run_id": run_id,
        "kind": "foot",
        "created_at": utc_now_iso(),
        "model_name": model_name,
        "model_label": config["label"],
        "model_dir": str(config["model_dir"]),
        "base_seed": seed,
        "output_count": len(outputs),
        "steps": int(steps),
        "guidance": float(guidance),
        "strength": float(strength),
        "dilate": int(dilate),
        "feather": int(feather),
        "negative_prompt": config["negative"],
        "source_size": list(source_image.size),
        "working_size": [FOOT_TARGET_SIZE, FOOT_TARGET_SIZE],
        "items": run_records,
    }
    preview = overlay_mask(image_512, mask_512)
    storage_artifacts: dict[str, Image.Image | dict | Path] = {
        "input.png": image_512,
        "mask.png": mask_512,
        "preview.png": preview,
        "metadata.json": run_metadata,
    }
    for idx, (output, item_metadata) in enumerate(zip(outputs, run_records), start=1):
        storage_artifacts[f"images/result_{idx:02d}.png"] = output
        storage_artifacts[f"metadata/result_{idx:02d}.json"] = item_metadata
    storage_info = upload_artifacts(run_id, storage_prefix, storage_artifacts) if save_output else None
    if storage_info:
        run_metadata["storage"] = storage_info
        run_metadata["storage_preview_image"] = storage_info["objects"].get("preview.png")
        # 데이터셋 리스트 대표 이미지를 result_01.png (첫 번째 합성 결과)로 지정
        run_metadata["storage_image"] = storage_info["objects"].get("images/result_01.png")
        run_metadata["storage_metadata"] = storage_info["objects"].get("metadata.json")
        for idx, item_metadata in enumerate(run_records, start=1):
            image_key = f"images/result_{idx:02d}.png"
            metadata_key = f"metadata/result_{idx:02d}.json"
            item_metadata["storage_image"] = storage_info["objects"].get(image_key)
            item_metadata["storage_metadata"] = storage_info["objects"].get(metadata_key)
    run_metadata_path.write_text(json.dumps(run_metadata, ensure_ascii=False, indent=2), encoding="utf-8")
    if save_output and storage_info and storage_info["available"]:
        try:
            get_storage().put_json(f"{storage_prefix}/metadata.json", run_metadata)
        except Exception as exc:
            storage_info["error"] = f"{run_id} MinIO 메타데이터 갱신 실패: {exc}"

    return {
        "status": f"{config['label']} 합성 완료 | {len(outputs)}장 | seed={seed}",
        "input": image_512,
        "mask": mask_512,
        "preview": preview,
        "gallery": outputs,
        "prompts": applied_prompts,
        "saved_paths": saved_paths,
        "metadata_paths": metadata_paths,
        "run_metadata_path": str(run_metadata_path),
        "storage": storage_info,
    }
