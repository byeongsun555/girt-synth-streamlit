export type DefaultsResponse = {
  wildfire: {
    model: string;
    prompt: string;
    negative_prompt: string;
    steps: number;
    guidance: number;
    fill_guidance: number;
    seed: number;
    output_size: number;
    smoke_headroom: number;
    mask_dilate: number;
    mask_feather: number;
    vegetation_refine: boolean;
    vegetation_threshold: number;
    vegetation_min_coverage: number;
    wildfire_engine: string;
    wildfire_engines: Record<string, string>;
    composition_mode: string;
    composition_modes: Record<string, string>;
  };
  foot: {
    steps: number;
    guidance: number;
    strength: number;
    seed: number;
    count: number;
    dilate: number;
    feather: number;
    models: Record<string, { label: string; prompt_count: number }>;
  };
};

export type WildfireResult = {
  image_id: string;
  status: string;
  image: string;
  image_url?: string | null;
  event_region_mask?: string | null;
  generation_edit_mask?: string | null;
  saved_path?: string;
  metadata_path?: string;
  storage?: StorageInfo | null;
  composition_mode?: string;
  composition_label?: string;
  selected_prompt_variant?: string;
  wildfire_engine?: string;
  mask_refinement?: {
    applied: boolean;
    reason: string;
    vegetation_coverage?: number;
    user_mask_pixels?: number;
    refined_mask_pixels?: number;
    coverage_ratio?: number;
    threshold?: number;
    dilate_px?: number;
  } | null;
};

export type FootPreviewResult = {
  image_id: string;
  mask: string;
  preview: string;
};

export type FootResult = {
  image_id: string;
  status: string;
  preview: string;
  images: string[];
  image_url?: string | null;
  prompts: string[];
  saved_paths?: string[];
  metadata_paths?: string[];
  run_metadata_path?: string;
  storage?: StorageInfo | null;
  zip_base64: string;
};

export type StorageInfo = {
  enabled: boolean;
  available: boolean;
  bucket: string;
  endpoint?: string;
  prefix?: string;
  objects?: Record<string, string>;
  urls?: Record<string, string | null>;
  error?: string | null;
  last_error?: string | null;
};

export type DatasetItem = {
  run_id: string;
  kind: "wildfire" | "foot" | string;
  label: string;
  created_at?: string | null;
  seed?: number | string | null;
  model?: string | null;
  object_prefix: string;
  image_url?: string | null;
  metadata_url?: string | null;
  summary?: string | null;
};

export type RecentDatasetsResponse = {
  items: DatasetItem[];
  storage: StorageInfo;
};

async function readJson<T>(response: Response): Promise<T> {
  const data = await response.json().catch(() => ({}));
  if (!response.ok) {
    const message = typeof data.detail === "string" ? data.detail : "요청 처리 중 오류가 발생했습니다.";
    throw new Error(message);
  }
  return data as T;
}

export async function getDefaults() {
  const response = await fetch("/api/defaults");
  return readJson<DefaultsResponse>(response);
}

export async function getStorageStatus() {
  const response = await fetch("/api/storage/status");
  return readJson<StorageInfo>(response);
}

export async function getRecentDatasets(limit = 40) {
  const response = await fetch(`/api/datasets/recent?limit=${encodeURIComponent(String(limit))}`);
  return readJson<RecentDatasetsResponse>(response);
}

export async function generateWildfire(payload: {
  image: File;
  mask?: Blob | null;
  prompt?: string;
  compositionMode: string;
  steps: number;
  guidance: number;
  fillGuidance: number;
  seed: number;
  outputSize: number;
  smokeHeadroom: number;
  maskDilate: number;
  maskFeather: number;
  vegetationRefine?: boolean;
  vegetationThreshold?: number;
  vegetationMinCoverage?: number;
  wildfireEngine?: string;
  saveOutput: boolean;
}) {
  const form = new FormData();
  form.append("image", payload.image);
  if (payload.mask) form.append("mask", payload.mask, "wildfire_mask.png");
  if (payload.prompt) form.append("prompt", payload.prompt);
  form.append("composition_mode", payload.compositionMode);
  form.append("steps", String(payload.steps));
  form.append("guidance", String(payload.guidance));
  form.append("fill_guidance", String(payload.fillGuidance));
  form.append("seed", String(payload.seed));
  form.append("output_size", String(payload.outputSize));
  form.append("smoke_headroom", String(payload.smokeHeadroom));
  form.append("mask_dilate", String(payload.maskDilate));
  form.append("mask_feather", String(payload.maskFeather));
  form.append("vegetation_refine", String(payload.vegetationRefine ?? true));
  form.append("vegetation_threshold", String(payload.vegetationThreshold ?? 0.35));
  form.append("vegetation_min_coverage", String(payload.vegetationMinCoverage ?? 0.15));
  form.append("wildfire_engine", payload.wildfireEngine ?? "kontext_hint");
  form.append("save_output", String(payload.saveOutput));
  const response = await fetch("/api/wildfire/generate", { method: "POST", body: form });
  return readJson<WildfireResult>(response);
}

export async function previewFootMask(payload: {
  image: File;
  mask: Blob;
  dilate: number;
  feather: number;
}) {
  const form = new FormData();
  form.append("image", payload.image);
  form.append("mask", payload.mask, "mask.png");
  form.append("dilate", String(payload.dilate));
  form.append("feather", String(payload.feather));
  const response = await fetch("/api/foot/preview", { method: "POST", body: form });
  return readJson<FootPreviewResult>(response);
}

export async function generateFoot(payload: {
  image: File;
  mask: Blob;
  modelName: string;
  steps: number;
  guidance: number;
  strength: number;
  seed: number;
  outputCount: number;
  dilate: number;
  feather: number;
  saveOutput: boolean;
}) {
  const form = new FormData();
  form.append("image", payload.image);
  form.append("mask", payload.mask, "mask.png");
  form.append("model_name", payload.modelName);
  form.append("steps", String(payload.steps));
  form.append("guidance", String(payload.guidance));
  form.append("strength", String(payload.strength));
  form.append("seed", String(payload.seed));
  form.append("output_count", String(payload.outputCount));
  form.append("dilate", String(payload.dilate));
  form.append("feather", String(payload.feather));
  form.append("save_output", String(payload.saveOutput));
  const response = await fetch("/api/foot/generate", { method: "POST", body: form });
  return readJson<FootResult>(response);
}

export function downloadDataUrl(dataUrl: string, filename: string) {
  const link = document.createElement("a");
  link.href = dataUrl;
  link.download = filename;
  document.body.appendChild(link);
  link.click();
  link.remove();
}

export function downloadBase64(base64: string, filename: string, mime = "application/zip") {
  const bytes = Uint8Array.from(atob(base64), (char) => char.charCodeAt(0));
  const blob = new Blob([bytes], { type: mime });
  const url = URL.createObjectURL(blob);
  const link = document.createElement("a");
  link.href = url;
  link.download = filename;
  document.body.appendChild(link);
  link.click();
  link.remove();
  URL.revokeObjectURL(url);
}
