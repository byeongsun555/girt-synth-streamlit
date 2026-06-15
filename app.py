from __future__ import annotations

import hashlib
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
from diffusers import FluxKontextPipeline, StableDiffusionInpaintPipeline
from streamlit.elements.lib.image_utils import image_to_url as st_image_to_url
from streamlit_drawable_canvas import st_canvas


def _image_to_url_compat(image, width, clamp, channels, output_format, image_id):
    layout_config = st_image.create_layout_config(width=width)
    return st_image_to_url(image, layout_config, clamp, channels, output_format, image_id)


if not hasattr(st_image, "image_to_url"):
    st_image.image_to_url = _image_to_url_compat


st.set_page_config(
    page_title="Synthetic Data Service",
    page_icon="synthetic",
    layout="wide",
)


def apply_app_style():
    st.markdown(
        """
        <style>
        .stApp {
            background: #f7f8fa;
        }
        .block-container {
            padding-top: 1.5rem;
            max-width: 1440px;
        }
        .app-hero {
            background: #ffffff;
            border: 1px solid #d8dde6;
            border-radius: 8px;
            padding: 22px 26px;
            margin-bottom: 18px;
            box-shadow: 0 10px 28px rgba(31, 41, 55, 0.06);
        }
        .app-hero h1 {
            margin: 0 0 6px 0;
            font-size: 2rem;
            line-height: 1.2;
        }
        .app-hero p {
            margin: 0;
            color: #64748b;
            font-size: 0.98rem;
        }
        .metric-strip {
            display: grid;
            grid-template-columns: repeat(4, minmax(0, 1fr));
            gap: 10px;
            margin: 12px 0 20px 0;
        }
        .metric-item {
            background: #ffffff;
            border: 1px solid #d8dde6;
            border-radius: 8px;
            padding: 12px 14px;
        }
        .metric-label {
            color: #64748b;
            font-size: 0.78rem;
            margin-bottom: 4px;
        }
        .metric-value {
            color: #111827;
            font-size: 0.95rem;
            font-weight: 700;
            overflow-wrap: anywhere;
        }
        h1, h2, h3 {
            color: #1f2937;
            letter-spacing: 0;
        }
        [data-testid="stTabs"] [role="tablist"] {
            gap: 8px;
            border-bottom: 1px solid #d8dde6;
        }
        [data-testid="stTabs"] [role="tab"] {
            border-radius: 7px 7px 0 0;
            padding: 10px 16px;
            background: #eef1f5;
        }
        [data-testid="stTabs"] [aria-selected="true"] {
            background: #ffffff;
            border: 1px solid #d8dde6;
            border-bottom: 1px solid #ffffff;
        }
        div[data-testid="stFileUploader"] section {
            border: 1px dashed #aeb7c4;
            border-radius: 8px;
            background: #ffffff;
        }
        div.stButton > button {
            border-radius: 7px;
            min-height: 42px;
            font-weight: 600;
        }
        div[data-testid="stImage"] img {
            border-radius: 8px;
            border: 1px solid #d8dde6;
            background: #ffffff;
        }
        .stAlert {
            border-radius: 8px;
        }
        @media (max-width: 900px) {
            .metric-strip {
                grid-template-columns: repeat(2, minmax(0, 1fr));
            }
        }
        </style>
        """,
        unsafe_allow_html=True,
    )


def render_hero():
    st.markdown(
        f"""
        <div class="app-hero">
            <h1>Synthetic Data Service</h1>
            <p>산불 장면 변환과 발 병변 인페인팅을 한 화면에서 시연하고 결과를 저장합니다.</p>
        </div>
        <div class="metric-strip">
            <div class="metric-item">
                <div class="metric-label">Compute</div>
                <div class="metric-value">{DEVICE.upper()}</div>
            </div>
            <div class="metric-item">
                <div class="metric-label">Wildfire Model</div>
                <div class="metric-value">FLUX Kontext</div>
            </div>
            <div class="metric-item">
                <div class="metric-label">Foot Models</div>
                <div class="metric-value">Corn / Crack</div>
            </div>
            <div class="metric-item">
                <div class="metric-label">Output</div>
                <div class="metric-value">Saved to outputs/</div>
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )


APP_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = APP_DIR.parent
FOOT_ROOT = PROJECT_ROOT / "foot_make_dataset"

OUTPUT_ROOT = APP_DIR / "outputs"
WILDFIRE_OUTPUT_DIR = OUTPUT_ROOT / "wildfire"
FOOT_OUTPUT_DIR = OUTPUT_ROOT / "foot"
WILDFIRE_OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
FOOT_OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
TORCH_DTYPE = torch.float16 if DEVICE == "cuda" else torch.float32

WILDFIRE_MODEL_ID = "black-forest-labs/FLUX.1-Kontext-dev"
WILDFIRE_PROMPT = "원본 구도와 배경은 유지하고, 도로나 포장면이 아닌 나무 또는 풀숲 내부에 작은 산불과 연기를 자연스럽게 추가해줘."
WILDFIRE_EDIT_GUARDRAIL = (
    "Minimal photo edit. Keep road and asphalt unchanged; no fire, smoke, glow, or burn marks on road. "
    "Add small realistic wildfire only in existing trees, bushes, shrubs, or grass. Preserve camera, layout, and background."
)
WILDFIRE_NEGATIVE = (
    "different scene, changed camera angle, changed road, moved trees, new background, crop, zoom, "
    "fire on road, fire on pavement, road burning, flames on asphalt, smoke from pavement, burning lane, "
    "blurry, low quality, cartoon, illustration, text, watermark"
)

FOOT_TARGET_SIZE = 512
FOOT_DEFAULT_STEPS = 50
FOOT_DEFAULT_GUIDANCE = 6.5
FOOT_DEFAULT_STRENGTH = 0.85
FOOT_DEFAULT_DILATE = 5
FOOT_DEFAULT_FEATHER = 4
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


def maybe_load_image(path: Path) -> Image.Image | None:
    if not path.exists():
        return None
    return Image.open(path).convert("RGB")


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
                text = text.strip()
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


def uploaded_image(uploaded_file, fallback: Image.Image | None = None) -> tuple[Image.Image | None, str]:
    if uploaded_file is None:
        return fallback, "sample" if fallback is not None else "empty"
    uploaded_file.seek(0)
    data = uploaded_file.getvalue()
    image_id = hashlib.md5(data).hexdigest()[:12]
    uploaded_file.seek(0)
    image = Image.open(uploaded_file).convert("RGB").copy()
    return image, image_id


def clear_foot_preview_state():
    st.session_state.pop("foot_preview_result", None)
    st.session_state.pop("foot_result", None)


def resize_to_square(image: Image.Image, size: int = FOOT_TARGET_SIZE) -> Image.Image:
    return image.convert("RGB").resize((size, size), Image.BICUBIC)


def overlay_mask(background: Image.Image, mask: Image.Image, color=(255, 0, 0), alpha=0.35) -> Image.Image:
    bg_arr = np.array(background.convert("RGB"), dtype=np.uint8)
    mask_arr = np.array(mask.convert("L"), dtype=np.uint8)
    overlay = np.zeros_like(bg_arr, dtype=np.uint8)
    overlay[mask_arr > 0] = np.array(color, dtype=np.uint8)
    return Image.fromarray(cv2.addWeighted(bg_arr, 1.0, overlay, alpha, 0.0))


def image_to_png_bytes(image: Image.Image) -> bytes:
    buffer = io.BytesIO()
    image.convert("RGB").save(buffer, format="PNG")
    return buffer.getvalue()


def images_to_zip_bytes(images: list[Image.Image], prefix: str) -> bytes:
    buffer = io.BytesIO()
    with zipfile.ZipFile(buffer, "w", compression=zipfile.ZIP_DEFLATED) as archive:
        for idx, image in enumerate(images, start=1):
            archive.writestr(f"{prefix}_{idx:02d}.png", image_to_png_bytes(image))
    return buffer.getvalue()


def build_wildfire_prompt(user_prompt: str) -> str:
    clean_prompt = " ".join((user_prompt or "").split())
    if not clean_prompt:
        clean_prompt = WILDFIRE_PROMPT
    return (
        f"{WILDFIRE_EDIT_GUARDRAIL}\n\n"
        f"User request, possibly Korean: {clean_prompt}"
    )


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
) -> tuple[Path, Path]:
    timestamp = time.strftime("%Y%m%d_%H%M%S")
    run_id = f"wildfire_{timestamp}_seed{seed}"
    image_path = WILDFIRE_OUTPUT_DIR / f"{run_id}.png"
    metadata_path = WILDFIRE_OUTPUT_DIR / f"{run_id}.json"
    result.save(image_path)
    metadata = {
        "run_id": run_id,
        "model": WILDFIRE_MODEL_ID,
        "seed": seed,
        "steps": int(steps),
        "guidance": float(guidance),
        "output_size": int(output_size),
        "source_size": list(source_image.size),
        "user_prompt": user_prompt,
        "final_prompt": final_prompt,
        "negative_prompt": negative_prompt,
        "saved_image": str(image_path),
    }
    metadata_path.write_text(json.dumps(metadata, ensure_ascii=False, indent=2), encoding="utf-8")
    return image_path, metadata_path


def extract_mask_from_canvas(canvas_image_data, size: tuple[int, int], dilate: int = 0, feather: int = 0):
    if canvas_image_data is None:
        return None

    canvas_arr = np.asarray(canvas_image_data).astype(np.uint8)
    if canvas_arr.ndim != 3 or canvas_arr.shape[2] < 3:
        return None

    if canvas_arr.shape[2] >= 4:
        mask_arr = np.where(canvas_arr[:, :, 3] > 0, 255, 0).astype(np.uint8)
    else:
        gray = cv2.cvtColor(canvas_arr[:, :, :3], cv2.COLOR_RGB2GRAY)
        mask_arr = np.where(gray > 0, 255, 0).astype(np.uint8)

    if dilate > 0:
        kernel = np.ones((int(dilate), int(dilate)), np.uint8)
        mask_arr = cv2.dilate(mask_arr, kernel, iterations=1)

    if feather > 0:
        sigma = max(0.5, float(feather) / 3.0)
        mask_arr = cv2.GaussianBlur(mask_arr, (0, 0), sigmaX=sigma, sigmaY=sigma)
        mask_arr = np.clip(mask_arr, 0, 255).astype(np.uint8)

    return Image.fromarray(mask_arr, mode="L").resize(size, Image.NEAREST)


def mask_has_pixels(mask: Image.Image | None) -> bool:
    if mask is None:
        return False
    return bool(np.count_nonzero(np.array(mask.convert("L"))))


@lru_cache(maxsize=1)
def get_wildfire_pipeline():
    hf_token = os.environ.get("HF_TOKEN") or os.environ.get("HUGGINGFACE_HUB_TOKEN")
    if not hf_token:
        raise RuntimeError(
            "FLUX.1-Kontext-dev는 gated model입니다. Hugging Face에서 모델 접근 동의를 완료한 뒤 "
            "HF_TOKEN 환경 변수를 설정해주세요."
        )
    pipe = FluxKontextPipeline.from_pretrained(
        WILDFIRE_MODEL_ID,
        torch_dtype=torch.bfloat16 if DEVICE == "cuda" else torch.float32,
        token=hf_token,
    )
    if DEVICE == "cuda":
        pipe.enable_model_cpu_offload()
        if hasattr(pipe, "enable_vae_tiling"):
            pipe.enable_vae_tiling()
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
    if DEVICE == "cuda":
        return pipe.to(DEVICE)
    pipe.enable_attention_slicing()
    return pipe.to(DEVICE)


def generate_wildfire_image(
    source_image: Image.Image,
    prompt: str,
    negative_prompt: str,
    strength: float,
    steps: int,
    guidance: float,
    seed,
    output_size: int,
    save_output: bool,
):
    seed = make_seed(seed)
    init_image = source_image.convert("RGB")
    generator = torch.Generator(device="cpu").manual_seed(seed)
    pipe = get_wildfire_pipeline()
    final_prompt = build_wildfire_prompt(prompt)

    with torch.inference_mode(), autocast_context(torch.bfloat16 if DEVICE == "cuda" else None):
        result = pipe(
            prompt=final_prompt,
            image=init_image,
            negative_prompt=negative_prompt,
            guidance_scale=float(guidance),
            num_inference_steps=int(steps),
            generator=generator,
            max_area=int(output_size) * int(output_size),
            max_sequence_length=512,
        ).images[0]

    image_path, metadata_path = save_wildfire_artifacts(
        result=result,
        source_image=init_image,
        user_prompt=prompt,
        final_prompt=final_prompt,
        negative_prompt=negative_prompt,
        seed=seed,
        steps=steps,
        guidance=guidance,
        output_size=output_size,
    )

    return {
        "status": f"산불 이미지 생성 완료 | seed={seed} | max_area={output_size}x{output_size}",
        "result": result,
        "saved_path": str(image_path),
        "metadata_path": str(metadata_path),
    }


def generate_foot_lesions(
    model_name: str,
    source_image: Image.Image,
    mask: Image.Image,
    steps: int,
    guidance: float,
    strength: float,
    seed,
    save_output: bool,
    output_count: int,
):
    config = FOOT_MODEL_CONFIGS[model_name]
    seed = make_seed(seed)
    image_512 = resize_to_square(source_image)
    mask_512 = mask.convert("L").resize((FOOT_TARGET_SIZE, FOOT_TARGET_SIZE), Image.NEAREST)
    preview = overlay_mask(image_512, mask_512)

    pipe = get_foot_pipeline(model_name)
    outputs = []
    saved_paths = []
    metadata_paths = []
    model_output_dir = FOOT_OUTPUT_DIR / config["out_subdir"]
    model_output_dir.mkdir(parents=True, exist_ok=True)
    timestamp = time.strftime("%Y%m%d_%H%M%S")
    run_id = f"{model_name}_{timestamp}_seed{seed}"
    run_metadata_path = model_output_dir / f"{run_id}_summary.json"

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
            "model_name": model_name,
            "model_label": config["label"],
            "model_dir": str(config["model_dir"]),
            "index": idx,
            "seed": image_seed,
            "base_seed": seed,
            "steps": int(steps),
            "guidance": float(guidance),
            "strength": float(strength),
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
        "model_name": model_name,
        "model_label": config["label"],
        "model_dir": str(config["model_dir"]),
        "base_seed": seed,
        "output_count": len(outputs),
        "steps": int(steps),
        "guidance": float(guidance),
        "strength": float(strength),
        "negative_prompt": config["negative"],
        "source_size": list(source_image.size),
        "working_size": [FOOT_TARGET_SIZE, FOOT_TARGET_SIZE],
        "items": run_records,
    }
    run_metadata_path.write_text(json.dumps(run_metadata, ensure_ascii=False, indent=2), encoding="utf-8")

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
    }


def render_wildfire_tab():
    heading_col, model_col = st.columns([1.0, 1.2])
    with heading_col:
        st.subheader("산불 이미지 생성")
    with model_col:
        st.caption(f"Model: {WILDFIRE_MODEL_ID}")
    st.info("원본 이미지를 업로드한 뒤 프롬프트를 실행하면 산불 상황 이미지로 변환합니다.")
    uploaded = st.file_uploader(
        "원본 이미지 업로드",
        type=["png", "jpg", "jpeg", "webp"],
        key="wildfire_upload",
    )
    source_image, image_id = uploaded_image(uploaded)
    if source_image is None:
        st.info("이미지를 업로드하면 image + prompt 방식으로 산불 이미지를 생성할 수 있습니다.")
        return

    with st.container(border=True):
        st.markdown("**생성 설정**")
        prompt = st.text_area("Prompt", value=WILDFIRE_PROMPT, height=100)
        negative = st.text_area("Negative Prompt", value=WILDFIRE_NEGATIVE, height=80)
        c1, c2, c3 = st.columns(3)
        with c1:
            output_size = st.selectbox("Output Area", options=[768, 1024, 1344], index=1)
        with c2:
            steps = st.slider("Steps", 10, 80, 24, 1)
            guidance = st.slider("Guidance Scale", 1.0, 10.0, 2.5, 0.1)
        with c3:
            seed = st.number_input("Seed (-1 random)", value=1234, step=1)
            st.caption("생성 결과는 outputs/wildfire/에 자동 저장됩니다.")
        run_clicked = st.button("산불 이미지 생성", type="primary", width="stretch")

    if run_clicked:
        try:
            with st.spinner("산불 이미지를 생성 중입니다..."):
                st.session_state["wildfire_result"] = generate_wildfire_image(
                    source_image=source_image,
                    prompt=prompt,
                    negative_prompt=negative,
                    strength=1.0,
                    steps=steps,
                    guidance=guidance,
                    seed=seed,
                    output_size=int(output_size),
                    save_output=True,
                )
                st.session_state["wildfire_result"]["image_id"] = image_id
        except Exception as exc:
            st.error(str(exc))

    result = st.session_state.get("wildfire_result")
    if result and result.get("image_id") == image_id:
        st.success(result["status"])
        with st.container(border=True):
            st.markdown("**생성 결과**")
            st.image(result["result"], caption="생성 결과", width="stretch")
            if result.get("saved_path"):
                st.caption(f"저장 위치: {result['saved_path']}")
            st.download_button(
                "생성 이미지 다운로드",
                data=image_to_png_bytes(result["result"]),
                file_name=f"wildfire_{image_id}.png",
                mime="image/png",
                width="stretch",
            )


def render_foot_tab():
    heading_col, model_col = st.columns([1.0, 0.8])
    with heading_col:
        st.subheader("발 병변 인페인팅")
    with model_col:
        model_name = st.selectbox(
            "병변 모델",
            options=list(FOOT_MODEL_CONFIGS.keys()),
            format_func=lambda name: FOOT_MODEL_CONFIGS[name]["label"],
            index=0,
            on_change=clear_foot_preview_state,
        )

    uploaded = st.file_uploader(
        "발 원본 이미지 업로드",
        type=["png", "jpg", "jpeg", "webp"],
        key="foot_upload",
        on_change=clear_foot_preview_state,
    )
    source_image, image_id = uploaded_image(uploaded)
    if source_image is None:
        st.info("발 이미지를 업로드하면 마스크를 그리고 병변을 합성할 수 있습니다.")
        return

    working_image = resize_to_square(source_image)

    canvas_col, control_col = st.columns([1.15, 0.85], vertical_alignment="top")
    with canvas_col:
        with st.container(border=True):
            st.markdown("**마스크 캔버스**")
            canvas_result = st_canvas(
                fill_color="rgba(255, 255, 255, 0.25)",
                stroke_width=18,
                stroke_color="#ffffff",
                background_image=working_image,
                update_streamlit=True,
                height=FOOT_TARGET_SIZE,
                width=FOOT_TARGET_SIZE,
                drawing_mode="freedraw",
                display_toolbar=True,
                key=f"foot_canvas_{model_name}_{image_id}",
            )

    with control_col:
        with st.container(border=True):
            st.markdown("**합성 설정**")
            steps = st.slider("Steps", 10, 100, FOOT_DEFAULT_STEPS, 1, key="foot_steps")
            guidance = st.slider("Guidance Scale", 1.0, 15.0, FOOT_DEFAULT_GUIDANCE, 0.5, key="foot_guidance")
            strength = st.slider("Strength", 0.1, 1.0, FOOT_DEFAULT_STRENGTH, 0.05, key="foot_strength")
            output_count = st.slider("생성 수", 1, 20, 5, 1, key="foot_output_count")
            seed = st.number_input("Seed (-1 random)", value=777, step=1, key="foot_seed")
            dilate = st.slider("Mask Dilate", 0, 32, FOOT_DEFAULT_DILATE, 1, key="foot_dilate")
            feather = st.slider("Mask Feather", 0, 32, FOOT_DEFAULT_FEATHER, 1, key="foot_feather")
            st.caption("생성 결과는 outputs/foot/에 자동 저장됩니다.")
            st.caption(f"프롬프트 풀: {len(get_foot_prompt_pool(model_name))}개")
            preview_clicked = st.button("마스크 미리보기", width="stretch", key="foot_preview")
            run_clicked = st.button("발 병변 합성 실행", type="primary", width="stretch", key="foot_run")

    if preview_clicked:
        mask = extract_mask_from_canvas(
            canvas_result.image_data,
            size=(FOOT_TARGET_SIZE, FOOT_TARGET_SIZE),
            dilate=int(dilate),
            feather=int(feather),
        )
        if not mask_has_pixels(mask):
            st.warning("마스크가 비어 있습니다. 흰색 브러시로 영역을 먼저 그려주세요.")
        else:
            st.session_state["foot_preview_result"] = {
                "status": "마스크 미리보기 생성 완료",
                "image_id": image_id,
                "model_name": model_name,
                "input": working_image,
                "mask": mask,
                "preview": overlay_mask(working_image, mask),
            }
            st.session_state.pop("foot_result", None)

    if run_clicked:
        preview = st.session_state.get("foot_preview_result")
        if not preview or preview.get("image_id") != image_id or preview.get("model_name") != model_name:
            st.warning("먼저 마스크를 그리고 `마스크 미리보기`를 완료한 다음 합성데이터를 생성해주세요.")
        else:
            with st.spinner(f"{FOOT_MODEL_CONFIGS[model_name]['label']} 병변을 합성 중입니다..."):
                st.session_state["foot_result"] = generate_foot_lesions(
                    model_name=model_name,
                    source_image=source_image,
                    mask=preview["mask"],
                    steps=steps,
                    guidance=guidance,
                    strength=strength,
                    seed=seed,
                    save_output=True,
                    output_count=output_count,
                )
                st.session_state["foot_result"]["image_id"] = image_id
                st.session_state["foot_result"]["model_name"] = model_name

    preview = st.session_state.get("foot_preview_result")
    if preview and preview.get("image_id") == image_id and preview.get("model_name") == model_name:
        st.success(preview["status"])
        with st.container(border=True):
            st.markdown("**마스크 확인**")
            c1, c2 = st.columns(2)
            with c1:
                st.image(preview["mask"], caption="생성된 마스크", width="stretch")
            with c2:
                st.image(preview["preview"], caption="오버레이 미리보기", width="stretch")

    result = st.session_state.get("foot_result")
    if result and result.get("image_id") == image_id and result.get("model_name") == model_name:
        st.success(result["status"])
        if result.get("gallery"):
            with st.container(border=True):
                st.markdown("**합성 결과**")
                st.image(
                    result["gallery"],
                    caption=[f"result {i + 1}" for i in range(len(result["gallery"]))],
                    width="stretch",
                )
                download_cols = st.columns([1, 1])
                with download_cols[0]:
                    st.download_button(
                        "전체 ZIP 다운로드",
                        data=images_to_zip_bytes(result["gallery"], f"{model_name}_{image_id}"),
                        file_name=f"{model_name}_{image_id}_synthetic.zip",
                        mime="application/zip",
                        width="stretch",
                    )
                with download_cols[1]:
                    st.download_button(
                        "첫 번째 이미지 다운로드",
                        data=image_to_png_bytes(result["gallery"][0]),
                        file_name=f"{model_name}_{image_id}_result_01.png",
                        mime="image/png",
                        width="stretch",
                    )
                with st.expander("사용된 프롬프트"):
                    for idx, prompt in enumerate(result.get("prompts", []), start=1):
                        st.markdown(f"{idx}. {prompt}")
                with st.expander("저장 위치"):
                    for path in result.get("saved_paths", []):
                        st.markdown(path)


def main():
    apply_app_style()
    render_hero()

    wildfire_tab, foot_tab = st.tabs(["산불 이미지 생성", "발 병변 인페인팅"])
    with wildfire_tab:
        render_wildfire_tab()
    with foot_tab:
        render_foot_tab()


if __name__ == "__main__":
    main()
