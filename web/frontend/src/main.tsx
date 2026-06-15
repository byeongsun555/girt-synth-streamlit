import { Tabs } from "@base-ui/react/tabs";
import {
  Activity,
  Database,
  Download,
  ExternalLink,
  Flame,
  ImagePlus,
  Layers3,
  Loader2,
  Paintbrush,
  RefreshCw,
  Server,
  Sparkles,
  Wand2,
} from "lucide-react";
import { useEffect, useMemo, useState } from "react";
import { createRoot } from "react-dom/client";
import { MaskCanvas } from "./components/MaskCanvas";
import {
  DefaultsResponse,
  DatasetItem,
  FootPreviewResult,
  FootResult,
  RecentDatasetsResponse,
  StorageInfo,
  WildfireResult,
  downloadBase64,
  downloadDataUrl,
  generateFoot,
  generateWildfire,
  getDefaults,
  getRecentDatasets,
  previewFootMask,
} from "./lib/api";
import "./styles.css";

type BusyState = "idle" | "wildfire" | "foot-preview" | "foot-generate";
type WildfireCompositionMode = "balanced" | "flame_dominant" | "full_flame" | "smoke_dominant";

const WILDFIRE_COMPOSITION_OPTIONS: Array<{
  value: WildfireCompositionMode;
  label: string;
  description: string;
}> = [
  { value: "balanced", label: "혼합형", description: "불꽃, 연기, 그을림, 불씨를 균형 있게 생성" },
  { value: "flame_dominant", label: "불 중심", description: "주황색 불꽃을 가장 강하게 생성하고 연기는 보조" },
  { value: "full_flame", label: "전체 화염", description: "마스크 안 식생 대부분을 진한 화염으로 채움" },
  { value: "smoke_dominant", label: "연기 중심", description: "작은 불꽃과 연결된 짙은 연기를 우선 생성" },
];
const MASK_COLOR_PRESETS = ["#ef4444", "#f97316", "#eab308", "#22c55e", "#06b6d4", "#3b82f6", "#a855f7", "#ec4899"];

const DEFAULT_FOOT_STEPS = 50;
const DEFAULT_FOOT_GUIDANCE = 6.5;
const DEFAULT_FOOT_STRENGTH = 0.85;
const DEFAULT_FOOT_DILATE = 5;
const DEFAULT_FOOT_FEATHER = 4;
const DEFAULT_WILDFIRE_FILL_GUIDANCE = 9;
const DEFAULT_WILDFIRE_MASK_DILATE = 4;
const DEFAULT_WILDFIRE_MASK_FEATHER = 2;
const DEFAULT_WILDFIRE_VEG_THRESHOLD = 0.30;
const DEFAULT_WILDFIRE_ENGINE = "fill_mask";

function fileUrl(file: Blob | null) {
  if (!file) return null;
  return URL.createObjectURL(file);
}

function App() {
  const [defaults, setDefaults] = useState<DefaultsResponse | null>(null);
  const [busy, setBusy] = useState<BusyState>("idle");
  const [datasetsBusy, setDatasetsBusy] = useState(false);
  const [error, setError] = useState<string | null>(null);
  const [datasetError, setDatasetError] = useState<string | null>(null);
  const [activeTab, setActiveTab] = useState<"wildfire" | "foot">("wildfire");
  const [storageStatus, setStorageStatus] = useState<StorageInfo | null>(null);
  const [recentDatasets, setRecentDatasets] = useState<DatasetItem[]>([]);

  const [wildfireFile, setWildfireFile] = useState<File | null>(null);
  const [wildfireMaskBlob, setWildfireMaskBlob] = useState<Blob | null>(null);
  const [wildfireBrushSize, setWildfireBrushSize] = useState(28);
  const [wildfireMaskColor, setWildfireMaskColor] = useState("#ef4444");
  const [wildfirePrompt, setWildfirePrompt] = useState("");
  const [wildfireComposition, setWildfireComposition] = useState<WildfireCompositionMode>("flame_dominant");
  const [wildfireSteps, setWildfireSteps] = useState(24);
  const [wildfireGuidance, setWildfireGuidance] = useState(2.5);
  const [wildfireFillGuidance, setWildfireFillGuidance] = useState(DEFAULT_WILDFIRE_FILL_GUIDANCE);
  const [wildfireSeed, setWildfireSeed] = useState(-1);
  const [wildfireSize, setWildfireSize] = useState(1024);
  const [wildfireSmokeHeadroom, setWildfireSmokeHeadroom] = useState(30);
  const [wildfireMaskDilate, setWildfireMaskDilate] = useState(DEFAULT_WILDFIRE_MASK_DILATE);
  const [wildfireMaskFeather, setWildfireMaskFeather] = useState(DEFAULT_WILDFIRE_MASK_FEATHER);
  const [wildfireVegRefine, setWildfireVegRefine] = useState(true);
  const [wildfireVegThreshold, setWildfireVegThreshold] = useState(DEFAULT_WILDFIRE_VEG_THRESHOLD);
  const [wildfireEngine, setWildfireEngine] = useState(DEFAULT_WILDFIRE_ENGINE);
  const [wildfireResult, setWildfireResult] = useState<WildfireResult | null>(null);

  const [footFile, setFootFile] = useState<File | null>(null);
  const [maskBlob, setMaskBlob] = useState<Blob | null>(null);
  const [footModel, setFootModel] = useState("corn");
  const [brushSize, setBrushSize] = useState(22);
  const [footSteps, setFootSteps] = useState(DEFAULT_FOOT_STEPS);
  const [footGuidance, setFootGuidance] = useState(DEFAULT_FOOT_GUIDANCE);
  const [footStrength, setFootStrength] = useState(DEFAULT_FOOT_STRENGTH);
  const [footSeed, setFootSeed] = useState(777);
  const [footCount, setFootCount] = useState(5);
  const [dilate, setDilate] = useState(DEFAULT_FOOT_DILATE);
  const [feather, setFeather] = useState(DEFAULT_FOOT_FEATHER);
  const [footPreview, setFootPreview] = useState<FootPreviewResult | null>(null);
  const [footResult, setFootResult] = useState<FootResult | null>(null);

  const wildfirePreview = useMemo(() => fileUrl(wildfireFile), [wildfireFile]);
  const wildfireMaskPreview = useMemo(() => fileUrl(wildfireMaskBlob), [wildfireMaskBlob]);
  const footPreviewUrl = useMemo(() => fileUrl(footFile), [footFile]);

  useEffect(() => {
    loadDefaults();
    loadRecentDatasets();
  }, []);

  useEffect(() => {
    return () => {
      if (wildfirePreview) URL.revokeObjectURL(wildfirePreview);
      if (wildfireMaskPreview) URL.revokeObjectURL(wildfireMaskPreview);
      if (footPreviewUrl) URL.revokeObjectURL(footPreviewUrl);
    };
  }, [wildfirePreview, wildfireMaskPreview, footPreviewUrl]);

  async function loadDefaults() {
    let lastError: Error | null = null;
    for (let attempt = 0; attempt < 3; attempt += 1) {
      try {
        const data = await getDefaults();
        setDefaults(data);
        setWildfireComposition((data.wildfire.composition_mode || "flame_dominant") as WildfireCompositionMode);
        setWildfireSteps(data.wildfire.steps);
        setWildfireGuidance(data.wildfire.guidance);
        setWildfireFillGuidance(data.wildfire.fill_guidance || DEFAULT_WILDFIRE_FILL_GUIDANCE);
        setWildfireSeed(data.wildfire.seed);
        setWildfireSize(data.wildfire.output_size);
        setWildfireSmokeHeadroom(data.wildfire.smoke_headroom || 30);
        setWildfireMaskDilate(data.wildfire.mask_dilate ?? DEFAULT_WILDFIRE_MASK_DILATE);
        setWildfireMaskFeather(data.wildfire.mask_feather ?? DEFAULT_WILDFIRE_MASK_FEATHER);
        setWildfireVegRefine(data.wildfire.vegetation_refine ?? true);
        setWildfireVegThreshold(data.wildfire.vegetation_threshold ?? DEFAULT_WILDFIRE_VEG_THRESHOLD);
        setWildfireEngine(data.wildfire.wildfire_engine ?? DEFAULT_WILDFIRE_ENGINE);
        setFootSteps(data.foot.steps || DEFAULT_FOOT_STEPS);
        setFootGuidance(data.foot.guidance || DEFAULT_FOOT_GUIDANCE);
        setFootStrength(data.foot.strength || DEFAULT_FOOT_STRENGTH);
        setFootSeed(data.foot.seed);
        setFootCount(data.foot.count);
        setDilate(data.foot.dilate || DEFAULT_FOOT_DILATE);
        setFeather(data.foot.feather || DEFAULT_FOOT_FEATHER);
        setError(null);
        return;
      } catch (err) {
        lastError = err instanceof Error ? err : new Error("기본 설정을 불러오지 못했습니다.");
        await new Promise((resolve) => window.setTimeout(resolve, 600));
      }
    }
    setError(lastError?.message ?? "기본 설정을 불러오지 못했습니다.");
  }

  async function loadRecentDatasets() {
    setDatasetsBusy(true);
    try {
      const data: RecentDatasetsResponse = await getRecentDatasets(40);
      setRecentDatasets(data.items);
      setStorageStatus(data.storage);
      setDatasetError(null);
    } catch (err) {
      setDatasetError(err instanceof Error ? err.message : "저장된 합성데이터를 불러오지 못했습니다.");
    } finally {
      setDatasetsBusy(false);
    }
  }

  async function runWildfire() {
    if (!wildfireFile) {
      setError("산불 이미지 생성을 위해 원본 이미지를 업로드해주세요.");
      return;
    }
    setError(null);
    setBusy("wildfire");
    try {
      const result = await generateWildfire({
        image: wildfireFile,
        mask: wildfireMaskBlob,
        prompt: wildfirePrompt || undefined,
        compositionMode: wildfireComposition,
        steps: wildfireSteps,
        guidance: wildfireGuidance,
        fillGuidance: wildfireFillGuidance,
        seed: wildfireSeed,
        outputSize: wildfireSize,
        smokeHeadroom: wildfireSmokeHeadroom,
        maskDilate: wildfireMaskDilate,
        maskFeather: wildfireMaskFeather,
        vegetationRefine: wildfireVegRefine,
        vegetationThreshold: wildfireVegThreshold,
        wildfireEngine,
        saveOutput: true,
      });
      setWildfireResult(result);
      loadRecentDatasets();
    } catch (err) {
      setError(err instanceof Error ? err.message : "산불 이미지 생성에 실패했습니다.");
    } finally {
      setBusy("idle");
    }
  }

  async function runFootPreview() {
    if (!footFile || !maskBlob) {
      setError("발 이미지 업로드 후 캔버스에 마스크를 그려주세요.");
      return;
    }
    setError(null);
    setBusy("foot-preview");
    try {
      const result = await previewFootMask({ image: footFile, mask: maskBlob, dilate, feather });
      setFootPreview(result);
      setFootResult(null);
    } catch (err) {
      setError(err instanceof Error ? err.message : "마스크 미리보기에 실패했습니다.");
    } finally {
      setBusy("idle");
    }
  }

  async function runFootGenerate() {
    if (!footFile || !maskBlob || !footPreview) {
      setError("마스크 미리보기를 먼저 완료한 뒤 합성데이터를 생성해주세요.");
      return;
    }
    setError(null);
    setBusy("foot-generate");
    try {
      const result = await generateFoot({
        image: footFile,
        mask: maskBlob,
        modelName: footModel,
        steps: footSteps,
        guidance: footGuidance,
        strength: footStrength,
        seed: footSeed,
        outputCount: footCount,
        dilate,
        feather,
        saveOutput: true,
      });
      setFootResult(result);
      loadRecentDatasets();
    } catch (err) {
      setError(err instanceof Error ? err.message : "발 병변 합성에 실패했습니다.");
    } finally {
      setBusy("idle");
    }
  }

  const footModels = defaults?.foot.models ?? {};

  return (
    <main className="min-h-screen bg-[#f6f7f9] text-slate-900">
      <section className="border-b border-slate-200 bg-white">
        <div className="mx-auto flex max-w-[1440px] flex-col gap-8 px-6 py-8 lg:flex-row lg:items-end lg:justify-between">
          <div className="max-w-3xl">
            <div className="mb-4 inline-flex items-center gap-2 rounded-lg border border-emerald-200 bg-emerald-50 px-3 py-2 text-sm font-semibold text-emerald-800">
              <Sparkles className="h-4 w-4" />
              Synthetic Data Service
            </div>
            <h1 className="text-4xl font-bold tracking-normal text-slate-950 md:text-5xl">
              합성데이터 생성 서비스
            </h1>
            <p className="mt-4 max-w-2xl text-base leading-7 text-slate-600">
              업로드, 생성 설정, 마스크 미리보기, 결과 다운로드까지 한 화면에서 처리합니다.
            </p>
          </div>
          <div className="grid min-w-[360px] grid-cols-2 gap-3">
            <Metric icon={<Activity />} label="Runtime" value="GPU API" />
            <Metric icon={<Flame />} label="Wildfire" value="FLUX Kontext" />
            <Metric icon={<Layers3 />} label="Foot" value="Corn / Crack" />
            <Metric
              icon={<Download />}
              label="Export"
              value={storageStatus?.available ? "MinIO / ZIP" : "Local / ZIP"}
            />
          </div>
        </div>
      </section>

      <section className="mx-auto max-w-[1440px] px-6 py-6">
        {error && (
          <div className="mb-5 rounded-lg border border-red-200 bg-red-50 px-4 py-3 text-sm font-semibold text-red-700">
            {error}
          </div>
        )}

        <Tabs.Root
          value={activeTab}
          onValueChange={(value) => setActiveTab(value as "wildfire" | "foot")}
          className="space-y-5"
        >
          <Tabs.List className="inline-flex rounded-lg border border-slate-200 bg-white p-1 shadow-sm">
            <Tabs.Tab
              value="wildfire"
              className={`inline-flex h-11 items-center gap-2 rounded-md px-5 text-sm font-bold transition ${
                activeTab === "wildfire"
                  ? "bg-orange-600 text-white shadow-sm"
                  : "bg-transparent text-slate-600 hover:bg-orange-50 hover:text-orange-700"
              }`}
            >
              <Flame className="h-4 w-4" />
              산불 이미지 생성
            </Tabs.Tab>
            <Tabs.Tab
              value="foot"
              className={`inline-flex h-11 items-center gap-2 rounded-md px-5 text-sm font-bold transition ${
                activeTab === "foot"
                  ? "bg-emerald-600 text-white shadow-sm"
                  : "bg-transparent text-slate-600 hover:bg-emerald-50 hover:text-emerald-700"
              }`}
            >
              <Paintbrush className="h-4 w-4" />
              발 병변 인페인팅
            </Tabs.Tab>
          </Tabs.List>

          <Tabs.Panel value="wildfire">
            <div className="grid gap-5 xl:grid-cols-[420px_1fr]">
              <Panel title="생성 설정" icon={<Wand2 />}>
                <FileDrop
                  label="원본 이미지"
                  file={wildfireFile}
                  onChange={(file) => {
                    setWildfireFile(file);
                    setWildfireMaskBlob(null);
                    setWildfireResult(null);
                  }}
                />
                <MaskCanvas
                  file={wildfireFile}
                  brushSize={wildfireBrushSize}
                  onMaskChange={setWildfireMaskBlob}
                  emptyText="산불 원본 이미지를 업로드하면 산불이 생성될 영역을 그릴 수 있습니다."
                  maskColor={wildfireMaskColor}
                />
                <div>
                  <div className="mb-2 text-sm font-bold text-slate-700">Mask Color</div>
                  <div className="flex flex-wrap items-center gap-2">
                    {MASK_COLOR_PRESETS.map((color) => (
                      <button
                        key={color}
                        type="button"
                        onClick={() => setWildfireMaskColor(color)}
                        className={`h-8 w-8 rounded-full border transition ${
                          wildfireMaskColor === color
                            ? "border-slate-950 ring-2 ring-slate-300"
                            : "border-slate-200 hover:ring-2 hover:ring-slate-200"
                        }`}
                        style={{ backgroundColor: color }}
                        title={color}
                      />
                    ))}
                    <input
                      type="color"
                      value={wildfireMaskColor}
                      onChange={(event) => setWildfireMaskColor(event.target.value)}
                      className="h-8 w-12 cursor-pointer rounded-md border border-slate-200 bg-white p-1"
                      title="마스크 색상 직접 선택"
                    />
                    <span className="text-xs font-semibold text-slate-500">{wildfireMaskColor}</span>
                  </div>
                </div>
                <div>
                  <div className="mb-2 text-sm font-bold text-slate-700">Mask Shape</div>
                  <div className="grid grid-cols-3 gap-3">
                    <NumberInput
                      label="Brush"
                      value={wildfireBrushSize}
                      min={8}
                      max={80}
                      onChange={setWildfireBrushSize}
                    />
                    <NumberInput
                      label="Dilate"
                      value={wildfireMaskDilate}
                      min={0}
                      max={32}
                      onChange={setWildfireMaskDilate}
                    />
                    <NumberInput
                      label="Feather"
                      value={wildfireMaskFeather}
                      min={0}
                      max={32}
                      onChange={setWildfireMaskFeather}
                    />
                  </div>
                </div>
                <div className="space-y-3 rounded-lg border border-slate-200 bg-slate-50 p-3">
                  <Select
                    label="생성 방식"
                    value={wildfireEngine}
                    onChange={setWildfireEngine}
                    options={Object.entries(defaults?.wildfire.wildfire_engines ?? {
                      kontext_hint: "자연어 편집",
                      fill_mask: "정밀 마스크 Fill",
                    })}
                  />
                  <p className="text-xs font-semibold leading-5 text-slate-500">
                    <strong>정밀 마스크 Fill</strong>: 마스크 위치에 정확히 합성합니다. 마스크가 명확할 때 권장 (기본). <br />
                    <strong>자연어 편집</strong>: 색칠 영역을 위치 힌트로만 사용해 장면 전체에서 자연스럽게 합성하지만 위치가 어긋날 수 있습니다.
                  </p>
                  <label className="flex items-center gap-2">
                    <input
                      type="checkbox"
                      checked={wildfireVegRefine}
                      onChange={(event) => setWildfireVegRefine(event.target.checked)}
                      className="h-4 w-4 rounded border-slate-300 text-orange-600 focus:ring-orange-500"
                    />
                    <span className="text-sm font-bold text-slate-700">식생 영역 자동 정제</span>
                  </label>
                  <p className="text-xs font-semibold leading-5 text-slate-500">
                    마스크 안에서 나무와 풀 영역만 자동 추출해 사용합니다. 정제 후 마스크가 너무 작으면 원본을 그대로 사용합니다.
                  </p>
                  {wildfireVegRefine && (
                    <NumberInput
                      label="식생 감지 임계값"
                      value={wildfireVegThreshold}
                      min={0.1}
                      max={0.7}
                      step={0.05}
                      onChange={setWildfireVegThreshold}
                    />
                  )}
                </div>
                <label className="block">
                  <span className="mb-2 block text-sm font-bold text-slate-700">
                    추가 지시 (선택)
                  </span>
                  <textarea
                    value={wildfirePrompt}
                    onChange={(e) => setWildfirePrompt(e.target.value)}
                    placeholder="예: 선택한 영역에 화염과 연기를 균형있게 생성해줘"
                    rows={2}
                    className="w-full resize-none rounded-lg border border-slate-200 bg-white px-3 py-2 text-sm font-semibold text-slate-700 outline-none transition focus:border-slate-500 focus:ring-4 focus:ring-slate-100"
                  />
                </label>
                <div>
                  <div className="mb-2 text-sm font-bold text-slate-700">Fire / Smoke Composition</div>
                  <div className="grid grid-cols-2 gap-2">
                    {WILDFIRE_COMPOSITION_OPTIONS.map((option) => {
                      const selected = wildfireComposition === option.value;
                      return (
                        <button
                          key={option.value}
                          type="button"
                          onClick={() => setWildfireComposition(option.value)}
                          className={`rounded-lg border px-3 py-3 text-left transition ${
                            selected
                              ? "border-orange-500 bg-orange-50 text-orange-900 ring-2 ring-orange-100"
                              : "border-slate-200 bg-white text-slate-700 hover:border-orange-200 hover:bg-orange-50"
                          }`}
                        >
                          <span className="block text-sm font-extrabold">{option.label}</span>
                          <span className="mt-1 block text-xs font-semibold leading-5 text-slate-500">
                            {option.description}
                          </span>
                        </button>
                      );
                    })}
                  </div>
                </div>
                <div className="grid grid-cols-2 gap-3">
                  <NumberInput label="Steps" value={wildfireSteps} min={10} max={80} onChange={setWildfireSteps} />
                  <NumberInput
                    label="Guidance (T2I)"
                    value={wildfireGuidance}
                    min={1}
                    max={10}
                    step={0.1}
                    hint="자연어 편집(Kontext)에서만 사용"
                    onChange={setWildfireGuidance}
                  />
                  <NumberInput
                    label="Fill Guidance (Inpaint)"
                    value={wildfireFillGuidance}
                    min={1}
                    max={50}
                    step={1}
                    onChange={setWildfireFillGuidance}
                  />
                  <NumberInput label="Seed" value={wildfireSeed} min={-1} onChange={setWildfireSeed} />
                  <NumberInput
                    label="Smoke Area"
                    value={wildfireSmokeHeadroom}
                    min={0}
                    max={100}
                    onChange={setWildfireSmokeHeadroom}
                  />
                  <Select
                    label="Output Area"
                    value={String(wildfireSize)}
                    onChange={(value) => setWildfireSize(Number(value))}
                    options={[
                      ["768", "768 x 768"],
                      ["1024", "1024 x 1024"],
                      ["1344", "1344 x 1344"],
                    ]}
                  />
                </div>
                <PrimaryButton busy={busy !== "idle"} onClick={runWildfire}>
                  산불 이미지 생성
                </PrimaryButton>
              </Panel>

              <Panel title="이미지 확인" icon={<ImagePlus />}>
                <div className="grid gap-4 lg:grid-cols-2">
                  <ImageFrame src={wildfirePreview} label="업로드 이미지" />
                  <ImageFrame src={wildfireResult?.image ?? wildfireResult?.image_url ?? null} label="생성 결과" />
                  <ImageFrame src={wildfireResult?.event_region_mask ?? wildfireMaskPreview} label="마스크 이미지" />
                  <ImageFrame src={wildfireResult?.generation_edit_mask ?? null} label="확장된 생성용 마스크" />
                </div>
                {wildfireResult && (
                  <ResultBar status={wildfireResult.status}>
                    {wildfireResult.mask_refinement && (
                      <p className="text-xs font-semibold text-slate-600">
                        {wildfireResult.mask_refinement.applied
                          ? `식생 정제 적용 (${Math.round((wildfireResult.mask_refinement.coverage_ratio ?? 0) * 100)}% 유지)`
                          : `정제 미적용: ${wildfireResult.mask_refinement.reason}`}
                      </p>
                    )}
                    <button
                      type="button"
                      onClick={() => downloadDataUrl(wildfireResult.image, `wildfire_${wildfireResult.image_id}.png`)}
                      className="inline-flex h-11 items-center gap-2 rounded-lg bg-orange-600 px-4 text-sm font-bold text-white shadow-sm hover:bg-orange-700"
                    >
                      <Download className="h-4 w-4" />
                      생성 이미지 다운로드
                    </button>
                  </ResultBar>
                )}
              </Panel>
            </div>
          </Tabs.Panel>

          <Tabs.Panel value="foot">
            <div className="grid gap-5 xl:grid-cols-[600px_1fr]">
              <Panel title="마스크 캔버스" icon={<Paintbrush />}>
                <FileDrop
                  label="발 원본 이미지"
                  file={footFile}
                  onChange={(file) => {
                    setFootFile(file);
                    setMaskBlob(null);
                    setFootPreview(null);
                    setFootResult(null);
                  }}
                />
                <MaskCanvas file={footFile} brushSize={brushSize} onMaskChange={setMaskBlob} />
                <div className="grid grid-cols-3 gap-3">
                  <NumberInput label="Brush" value={brushSize} min={8} max={64} onChange={setBrushSize} />
                  <NumberInput
                    label="Dilate"
                    value={dilate}
                    min={0}
                    max={32}
                    onChange={(value) => {
                      setDilate(value);
                      setFootPreview(null);
                      setFootResult(null);
                    }}
                  />
                  <NumberInput
                    label="Feather"
                    value={feather}
                    min={0}
                    max={32}
                    onChange={(value) => {
                      setFeather(value);
                      setFootPreview(null);
                      setFootResult(null);
                    }}
                  />
                </div>
                <PrimaryButton busy={busy !== "idle"} onClick={runFootPreview}>
                  마스크 미리보기
                </PrimaryButton>
              </Panel>

              <div className="space-y-5">
                <Panel title="합성 설정" icon={<Layers3 />}>
                  <div className="grid grid-cols-2 gap-3">
                    <Select
                      label="병변 모델"
                      value={footModel}
                      onChange={(value) => {
                        setFootModel(value);
                        setFootPreview(null);
                        setFootResult(null);
                      }}
                      options={Object.entries(footModels).map(([key, model]) => [
                        key,
                        `${model.label} (${model.prompt_count})`,
                      ])}
                    />
                    <NumberInput label="생성 수" value={footCount} min={1} max={20} onChange={setFootCount} />
                    <NumberInput label="Steps" value={footSteps} min={10} max={100} onChange={setFootSteps} />
                    <NumberInput
                      label="Guidance"
                      value={footGuidance}
                      min={1}
                      max={15}
                      step={0.5}
                      onChange={setFootGuidance}
                    />
                    <NumberInput
                      label="Strength"
                      value={footStrength}
                      min={0.1}
                      max={1}
                      step={0.05}
                      onChange={setFootStrength}
                    />
                    <NumberInput label="Seed" value={footSeed} min={-1} onChange={setFootSeed} />
                  </div>
                  <PrimaryButton busy={busy !== "idle"} onClick={runFootGenerate}>
                    발 병변 합성 실행
                  </PrimaryButton>
                </Panel>

                <Panel title="마스크 확인" icon={<ImagePlus />}>
                  <div className="grid gap-4 md:grid-cols-2">
                    <ImageFrame src={footPreview?.mask ?? null} label="생성된 마스크" />
                    <ImageFrame src={footPreview?.preview ?? footPreviewUrl} label="오버레이 미리보기" />
                  </div>
                </Panel>
              </div>
            </div>

            {footResult && (
              <Panel title="합성 결과" icon={<Sparkles />} className="mt-5">
                <ResultBar status={footResult.status}>
                  <button
                    type="button"
                    onClick={() => downloadBase64(footResult.zip_base64, `${footModel}_${footResult.image_id}.zip`)}
                    className="inline-flex h-11 items-center gap-2 rounded-lg bg-emerald-600 px-4 text-sm font-bold text-white shadow-sm hover:bg-emerald-700"
                  >
                    <Download className="h-4 w-4" />
                    전체 합성데이터 다운로드
                  </button>
                </ResultBar>
                <div className="grid gap-4 md:grid-cols-2 xl:grid-cols-4">
                  {footResult.images.map((image, index) => (
                    <div key={image.slice(0, 40) + index} className="overflow-hidden rounded-lg border border-slate-200 bg-white">
                      <img src={image} alt={`합성 결과 ${index + 1}`} className="aspect-square w-full object-cover" />
                      <div className="flex items-center justify-between px-3 py-2">
                        <span className="text-sm font-bold text-slate-700">result {index + 1}</span>
                        <button
                          type="button"
                          onClick={() => downloadDataUrl(image, `${footModel}_${footResult.image_id}_${index + 1}.png`)}
                          className="inline-flex h-8 items-center gap-1 rounded-md border border-slate-200 px-2 text-xs font-bold text-slate-600 hover:bg-slate-50"
                        >
                          <Download className="h-3.5 w-3.5" />
                          PNG
                        </button>
                      </div>
                    </div>
                  ))}
                </div>
                <details className="rounded-lg border border-slate-200 bg-slate-50 p-4">
                  <summary className="cursor-pointer text-sm font-bold text-slate-700">사용된 프롬프트</summary>
                  <ol className="mt-3 space-y-2 text-sm leading-6 text-slate-600">
                    {footResult.prompts.map((prompt, index) => (
                      <li key={`${prompt}-${index}`}>{index + 1}. {prompt}</li>
                    ))}
                  </ol>
                </details>
              </Panel>
            )}
          </Tabs.Panel>
        </Tabs.Root>

        <DatasetShelf items={recentDatasets} busy={datasetsBusy} error={datasetError} onRefresh={loadRecentDatasets} />
      </section>
    </main>
  );
}

function DatasetShelf({
  items,
  busy,
  error,
  onRefresh,
}: {
  items: DatasetItem[];
  busy: boolean;
  error: string | null;
  onRefresh: () => void;
}) {
  return (
    <section className="mt-6 rounded-lg border border-slate-200 bg-white p-5 shadow-sm">
      <div className="mb-4 flex flex-col gap-3 md:flex-row md:items-center md:justify-between">
        <div className="flex items-center gap-2">
          <div className="grid h-9 w-9 place-items-center rounded-lg bg-slate-100 text-slate-700">
            <Database className="h-5 w-5" />
          </div>
          <div>
            <h2 className="text-lg font-bold tracking-normal text-slate-950">저장된 합성데이터</h2>
            <p className="mt-1 text-xs font-semibold text-slate-500">MinIO에 저장된 최근 결과를 바로 확인합니다.</p>
          </div>
        </div>
        <button
          type="button"
          onClick={onRefresh}
          className="inline-flex h-10 items-center justify-center gap-2 rounded-lg border border-slate-200 bg-white px-3 text-sm font-bold text-slate-700 transition hover:bg-slate-50"
        >
          <RefreshCw className={`h-4 w-4 ${busy ? "animate-spin" : ""}`} />
          목록 갱신
        </button>
      </div>
      {error && (
        <div className="mb-4 rounded-lg border border-orange-200 bg-orange-50 px-4 py-3 text-sm font-semibold text-orange-900">
          {error}
        </div>
      )}
      {items.length === 0 ? (
        <div className="grid min-h-[160px] place-items-center rounded-lg border border-dashed border-slate-200 bg-slate-50 px-6 text-center text-sm font-semibold text-slate-400">
          저장된 항목이 아직 없습니다.
        </div>
      ) : (
        <div className="grid gap-4 md:grid-cols-2 xl:grid-cols-4">
          {items.map((item) => (
            <article key={item.object_prefix} className="overflow-hidden rounded-lg border border-slate-200 bg-white">
              <div className="grid aspect-square place-items-center bg-slate-100">
                {item.image_url ? (
                  <img src={item.image_url} alt={item.run_id} className="h-full w-full object-cover" />
                ) : (
                  <Server className="h-8 w-8 text-slate-300" />
                )}
              </div>
              <div className="space-y-3 px-3 py-3">
                <div>
                  <div className="text-sm font-bold text-slate-950">{item.label}</div>
                  <div className="mt-1 truncate text-xs font-semibold text-slate-500">{item.run_id}</div>
                </div>
                <div className="flex flex-wrap gap-2 text-xs font-bold text-slate-500">
                  <span className="rounded-md bg-slate-100 px-2 py-1">{item.kind}</span>
                  {item.seed !== undefined && item.seed !== null && (
                    <span className="rounded-md bg-slate-100 px-2 py-1">seed {item.seed}</span>
                  )}
                </div>
                <div className="flex flex-wrap gap-2">
                  {item.image_url && (
                    <a
                      href={item.image_url}
                      target="_blank"
                      rel="noreferrer"
                      className="inline-flex h-8 items-center gap-1 rounded-md border border-slate-200 px-2 text-xs font-bold text-slate-600 hover:bg-slate-50"
                    >
                      <ExternalLink className="h-3.5 w-3.5" />
                      이미지
                    </a>
                  )}
                  {item.metadata_url && (
                    <a
                      href={item.metadata_url}
                      target="_blank"
                      rel="noreferrer"
                      className="inline-flex h-8 items-center gap-1 rounded-md border border-slate-200 px-2 text-xs font-bold text-slate-600 hover:bg-slate-50"
                    >
                      <ExternalLink className="h-3.5 w-3.5" />
                      메타데이터
                    </a>
                  )}
                </div>
              </div>
            </article>
          ))}
        </div>
      )}
    </section>
  );
}

function Metric({ icon, label, value }: { icon: React.ReactNode; label: string; value: string }) {
  return (
    <div className="rounded-lg border border-slate-200 bg-slate-50 p-4">
      <div className="mb-3 h-5 w-5 text-slate-500">{icon}</div>
      <div className="text-xs font-bold uppercase tracking-normal text-slate-500">{label}</div>
      <div className="mt-1 text-sm font-bold text-slate-950">{value}</div>
    </div>
  );
}

function Panel({
  title,
  icon,
  children,
  className = "",
}: {
  title: string;
  icon: React.ReactNode;
  children: React.ReactNode;
  className?: string;
}) {
  return (
    <section className={`rounded-lg border border-slate-200 bg-white p-5 shadow-sm ${className}`}>
      <div className="mb-4 flex items-center gap-2">
        <div className="grid h-9 w-9 place-items-center rounded-lg bg-slate-100 text-slate-700">{icon}</div>
        <h2 className="text-lg font-bold tracking-normal text-slate-950">{title}</h2>
      </div>
      <div className="space-y-4">{children}</div>
    </section>
  );
}

function FileDrop({ label, file, onChange }: { label: string; file: File | null; onChange: (file: File | null) => void }) {
  return (
    <label className="block">
      <span className="mb-2 block text-sm font-bold text-slate-700">{label}</span>
      <input
        type="file"
        accept="image/png,image/jpeg,image/webp"
        onChange={(event) => onChange(event.target.files?.[0] ?? null)}
        className="block w-full cursor-pointer rounded-lg border border-dashed border-slate-300 bg-slate-50 px-3 py-3 text-sm text-slate-600 file:mr-3 file:rounded-md file:border-0 file:bg-slate-950 file:px-3 file:py-2 file:text-sm file:font-bold file:text-white"
      />
      {file && <span className="mt-2 block text-xs font-semibold text-slate-500">{file.name}</span>}
    </label>
  );
}

function NumberInput({
  label,
  value,
  min,
  max,
  step = 1,
  hint,
  onChange,
}: {
  label: string;
  value: number;
  min?: number;
  max?: number;
  step?: number;
  hint?: string;
  onChange: (value: number) => void;
}) {
  return (
    <label className="block">
      <span className="mb-2 block text-sm font-bold text-slate-700">{label}</span>
      <input
        type="number"
        value={value}
        min={min}
        max={max}
        step={step}
        onChange={(event) => {
          const raw = event.target.value;
          if (raw === "") return;
          const val = Number(raw);
          if (!Number.isNaN(val)) onChange(val);
        }}
        className="h-11 w-full rounded-lg border border-slate-200 bg-white px-3 text-sm font-semibold text-slate-700 outline-none transition focus:border-slate-500 focus:ring-4 focus:ring-slate-100"
      />
      {hint && <span className="mt-1 block text-xs font-semibold leading-5 text-slate-500">{hint}</span>}
    </label>
  );
}

function Select({
  label,
  value,
  onChange,
  options,
}: {
  label: string;
  value: string;
  onChange: (value: string) => void;
  options: Array<[string, string]>;
}) {
  return (
    <label className="block">
      <span className="mb-2 block text-sm font-bold text-slate-700">{label}</span>
      <select
        value={value}
        onChange={(event) => onChange(event.target.value)}
        className="h-11 w-full rounded-lg border border-slate-200 bg-white px-3 text-sm font-semibold text-slate-700 outline-none transition focus:border-slate-500 focus:ring-4 focus:ring-slate-100"
      >
        {options.map(([optionValue, labelText]) => (
          <option key={optionValue} value={optionValue}>
            {labelText}
          </option>
        ))}
      </select>
    </label>
  );
}

function PrimaryButton({ children, busy, onClick }: { children: React.ReactNode; busy: boolean; onClick: () => void }) {
  return (
    <button
      type="button"
      disabled={busy}
      onClick={onClick}
      className="inline-flex h-12 w-full items-center justify-center gap-2 rounded-lg bg-slate-950 px-5 text-sm font-bold text-white transition hover:bg-slate-800 disabled:cursor-not-allowed disabled:bg-slate-400"
    >
      {busy ? <Loader2 className="h-4 w-4 animate-spin" /> : <Sparkles className="h-4 w-4" />}
      {children}
    </button>
  );
}

function ImageFrame({ src, label }: { src: string | null | undefined; label: string }) {
  return (
    <figure className="overflow-hidden rounded-lg border border-slate-200 bg-slate-50">
      <div className="grid aspect-[4/3] place-items-center bg-slate-100">
        {src ? (
          <img src={src} alt={label} className="h-full w-full object-contain" />
        ) : (
          <div className="px-5 text-center text-sm font-semibold text-slate-400">이미지 대기 중</div>
        )}
      </div>
      <figcaption className="border-t border-slate-200 bg-white px-3 py-2 text-sm font-bold text-slate-600">
        {label}
      </figcaption>
    </figure>
  );
}

function ResultBar({ status, children }: { status: string; children: React.ReactNode }) {
  return (
    <div className="flex flex-col gap-3 rounded-lg border border-emerald-200 bg-emerald-50 px-4 py-3 md:flex-row md:items-center md:justify-between">
      <span className="text-sm font-bold text-emerald-800">{status}</span>
      <div className="flex flex-wrap gap-2">{children}</div>
    </div>
  );
}

createRoot(document.getElementById("root")!).render(<App />);
