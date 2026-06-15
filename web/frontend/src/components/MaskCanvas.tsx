import { Eraser, RotateCcw } from "lucide-react";
import { useEffect, useRef, useState } from "react";

type MaskCanvasProps = {
  file: File | null;
  brushSize: number;
  onMaskChange: (mask: Blob | null) => void;
  emptyText?: string;
  maskColor?: string;
};

const CANVAS_SIZE = 512;

export function MaskCanvas({ file, brushSize, onMaskChange, emptyText, maskColor = "#ef4444" }: MaskCanvasProps) {
  const displayRef = useRef<HTMLCanvasElement | null>(null);
  const maskRef = useRef<HTMLCanvasElement | null>(null);
  const imageRef = useRef<HTMLImageElement | null>(null);
  const [isDrawing, setIsDrawing] = useState(false);
  const [objectUrl, setObjectUrl] = useState<string | null>(null);

  useEffect(() => {
    if (!file) {
      setObjectUrl(null);
      clearCanvas();
      return;
    }
    const url = URL.createObjectURL(file);
    setObjectUrl(url);
    return () => URL.revokeObjectURL(url);
  }, [file]);

  useEffect(() => {
    if (!objectUrl) return;
    const image = new Image();
    image.onload = () => {
      imageRef.current = image;
      drawBackground();
      clearMaskOnly();
      emitMask();
    };
    image.src = objectUrl;
  }, [objectUrl]);

  useEffect(() => {
    drawBackground();
  }, [maskColor]);

  function drawBackground() {
    const canvas = displayRef.current;
    const image = imageRef.current;
    if (!canvas || !image) return;
    const ctx = canvas.getContext("2d");
    if (!ctx) return;
    ctx.clearRect(0, 0, CANVAS_SIZE, CANVAS_SIZE);
    ctx.drawImage(image, 0, 0, CANVAS_SIZE, CANVAS_SIZE);
    drawMaskOverlay();
  }

  function drawMaskOverlay() {
    const display = displayRef.current;
    const mask = maskRef.current;
    if (!display || !mask) return;
    const ctx = display.getContext("2d");
    if (!ctx) return;
    const overlay = document.createElement("canvas");
    overlay.width = CANVAS_SIZE;
    overlay.height = CANVAS_SIZE;
    const overlayCtx = overlay.getContext("2d");
    if (!overlayCtx) return;
    overlayCtx.drawImage(mask, 0, 0);
    overlayCtx.globalCompositeOperation = "source-in";
    overlayCtx.fillStyle = maskColor;
    overlayCtx.fillRect(0, 0, CANVAS_SIZE, CANVAS_SIZE);
    ctx.save();
    ctx.globalAlpha = 0.36;
    ctx.globalCompositeOperation = "source-over";
    ctx.drawImage(overlay, 0, 0);
    ctx.restore();
  }

  function clearMaskOnly() {
    const mask = maskRef.current;
    if (!mask) return;
    const ctx = mask.getContext("2d");
    ctx?.clearRect(0, 0, CANVAS_SIZE, CANVAS_SIZE);
  }

  function clearCanvas() {
    const display = displayRef.current;
    const displayCtx = display?.getContext("2d");
    displayCtx?.clearRect(0, 0, CANVAS_SIZE, CANVAS_SIZE);
    clearMaskOnly();
    onMaskChange(null);
  }

  function pointerPosition(event: React.PointerEvent<HTMLCanvasElement>) {
    const rect = event.currentTarget.getBoundingClientRect();
    const x = ((event.clientX - rect.left) / rect.width) * CANVAS_SIZE;
    const y = ((event.clientY - rect.top) / rect.height) * CANVAS_SIZE;
    return { x, y };
  }

  function drawPoint(x: number, y: number) {
    const mask = maskRef.current;
    if (!mask) return;
    const ctx = mask.getContext("2d");
    if (!ctx) return;
    ctx.fillStyle = "#ffffff";
    ctx.beginPath();
    ctx.arc(x, y, brushSize / 2, 0, Math.PI * 2);
    ctx.fill();
    drawBackground();
  }

  function handlePointerDown(event: React.PointerEvent<HTMLCanvasElement>) {
    if (!file) return;
    event.currentTarget.setPointerCapture(event.pointerId);
    setIsDrawing(true);
    const { x, y } = pointerPosition(event);
    drawPoint(x, y);
  }

  function handlePointerMove(event: React.PointerEvent<HTMLCanvasElement>) {
    if (!isDrawing || !file) return;
    const { x, y } = pointerPosition(event);
    drawPoint(x, y);
  }

  function handlePointerUp(event: React.PointerEvent<HTMLCanvasElement>) {
    if (!isDrawing) return;
    setIsDrawing(false);
    event.currentTarget.releasePointerCapture(event.pointerId);
    emitMask();
  }

  function emitMask() {
    const mask = maskRef.current;
    if (!mask) return;
    mask.toBlob((blob) => onMaskChange(blob), "image/png");
  }

  return (
    <div className="space-y-3">
      <div className="relative mx-auto aspect-square w-full max-w-[512px] overflow-hidden rounded-lg border border-slate-200 bg-slate-950">
        {!file && (
          <div className="absolute inset-0 grid place-items-center px-8 text-center text-sm text-slate-400">
            {emptyText ?? "이미지를 업로드하면 512 x 512 캔버스에서 합성 영역을 그릴 수 있습니다."}
          </div>
        )}
        <canvas
          ref={displayRef}
          width={CANVAS_SIZE}
          height={CANVAS_SIZE}
          className="h-full w-full touch-none"
          onPointerDown={handlePointerDown}
          onPointerMove={handlePointerMove}
          onPointerUp={handlePointerUp}
          onPointerCancel={handlePointerUp}
        />
        <canvas ref={maskRef} width={CANVAS_SIZE} height={CANVAS_SIZE} className="hidden" />
      </div>
      <button
        type="button"
        onClick={() => {
          clearMaskOnly();
          drawBackground();
          onMaskChange(null);
        }}
        className="inline-flex h-10 items-center gap-2 rounded-lg border border-slate-200 bg-white px-3 text-sm font-semibold text-slate-700 transition hover:bg-slate-50"
        title="마스크 초기화"
      >
        <RotateCcw className="h-4 w-4" />
        초기화
      </button>
      <span className="ml-2 inline-flex h-10 items-center gap-2 text-sm text-slate-500">
        <Eraser className="h-4 w-4" />
        선택한 색상으로 합성 영역을 표시합니다.
      </span>
    </div>
  );
}
