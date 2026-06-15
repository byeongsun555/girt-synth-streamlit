from __future__ import annotations

import io
import json
import os
from datetime import datetime, timezone, timedelta
from functools import lru_cache
from pathlib import Path
from typing import Any
from urllib.parse import quote, urlsplit, urlunsplit

from PIL import Image

try:
    from minio import Minio
    from minio.error import S3Error
except ImportError:  # Local static checks can run before the backend image is rebuilt.
    Minio = None
    S3Error = Exception


DEFAULT_BUCKET = "synthetic-data"


def _env_bool(name: str, default: bool = False) -> bool:
    value = os.environ.get(name)
    if value is None:
        return default
    return value.strip().lower() in {"1", "true", "yes", "on"}


def _content_type(path: str) -> str:
    if path.endswith(".png"):
        return "image/png"
    if path.endswith(".json"):
        return "application/json"
    if path.endswith(".zip"):
        return "application/zip"
    return "application/octet-stream"


def object_proxy_url(object_name: str | None) -> str | None:
    if not object_name:
        return None
    return f"/api/storage/object?object_name={quote(object_name, safe='')}"


class ObjectStorage:
    def __init__(self) -> None:
        self.enabled = _env_bool("MINIO_ENABLED", True)
        self.endpoint = os.environ.get("MINIO_ENDPOINT", "minio:9000")
        self.public_endpoint = os.environ.get("MINIO_PUBLIC_ENDPOINT", "localhost:9000")
        self.access_key = os.environ.get("MINIO_ACCESS_KEY", "minioadmin")
        self.secret_key = os.environ.get("MINIO_SECRET_KEY", "minioadmin")
        self.bucket = os.environ.get("MINIO_BUCKET", DEFAULT_BUCKET)
        self.secure = _env_bool("MINIO_SECURE", False)
        self.presign_expiry = timedelta(hours=int(os.environ.get("MINIO_PRESIGN_HOURS", "24")))
        self._client: Minio | None = None
        self.last_error: str | None = None

    @property
    def client(self):
        if Minio is None:
            raise RuntimeError("minio 패키지가 설치되어 있지 않습니다. backend 이미지를 다시 빌드해주세요.")
        if self._client is None:
            self._client = Minio(
                self.endpoint,
                access_key=self.access_key,
                secret_key=self.secret_key,
                secure=self.secure,
            )
        return self._client

    def status(self) -> dict[str, Any]:
        if not self.enabled:
            return {
                "enabled": False,
                "available": False,
                "bucket": self.bucket,
                "endpoint": self.public_endpoint,
                "last_error": self.last_error,
            }
        try:
            self.ensure_bucket()
            available = True
            self.last_error = None
        except Exception as exc:
            available = False
            self.last_error = str(exc)
        return {
            "enabled": True,
            "available": available,
            "bucket": self.bucket,
            "endpoint": self.public_endpoint,
            "last_error": self.last_error,
        }

    def ensure_bucket(self) -> None:
        if not self.enabled:
            return
        found = self.client.bucket_exists(self.bucket)
        if not found:
            try:
                self.client.make_bucket(self.bucket)
            except S3Error as exc:
                if getattr(exc, "code", "") != "BucketAlreadyOwnedByYou":
                    raise

    def put_bytes(self, object_name: str, data: bytes, content_type: str | None = None) -> str | None:
        if not self.enabled:
            return None
        self.ensure_bucket()
        self.client.put_object(
            self.bucket,
            object_name,
            io.BytesIO(data),
            length=len(data),
            content_type=content_type or _content_type(object_name),
        )
        return object_name

    def put_json(self, object_name: str, payload: dict[str, Any]) -> str | None:
        data = json.dumps(payload, ensure_ascii=False, indent=2).encode("utf-8")
        return self.put_bytes(object_name, data, content_type="application/json")

    def put_image(self, object_name: str, image: Image.Image) -> str | None:
        buffer = io.BytesIO()
        image.convert("RGB").save(buffer, format="PNG")
        return self.put_bytes(object_name, buffer.getvalue(), content_type="image/png")

    def put_file(self, object_name: str, path: Path) -> str | None:
        if not self.enabled:
            return None
        data = path.read_bytes()
        return self.put_bytes(object_name, data, content_type=_content_type(object_name))

    def get_object_bytes(self, object_name: str) -> tuple[bytes, str]:
        if not self.enabled:
            raise RuntimeError("MinIO 저장소가 비활성화되어 있습니다.")
        if not object_name or object_name.startswith("/") or ".." in Path(object_name).parts:
            raise ValueError("잘못된 객체 경로입니다.")
        response = None
        try:
            response = self.client.get_object(self.bucket, object_name)
            return response.read(), _content_type(object_name)
        finally:
            if response is not None:
                response.close()
                response.release_conn()

    def presigned_get_url(self, object_name: str | None) -> str | None:
        if not self.enabled or not object_name:
            return None
        try:
            url = self.client.presigned_get_object(self.bucket, object_name, expires=self.presign_expiry)
            return self._public_url(url)
        except S3Error as exc:
            self.last_error = str(exc)
            return None

    def _public_url(self, url: str) -> str:
        if not self.public_endpoint or self.public_endpoint == self.endpoint:
            return url
        parts = urlsplit(url)
        public_parts = urlsplit(self.public_endpoint if "://" in self.public_endpoint else f"{parts.scheme}://{self.public_endpoint}")
        return urlunsplit((public_parts.scheme, public_parts.netloc, parts.path, parts.query, parts.fragment))

    def list_dataset_items(self, limit: int = 50) -> list[dict[str, Any]]:
        if not self.enabled:
            return []
        self.ensure_bucket()
        objects = [
            obj
            for obj in self.client.list_objects(self.bucket, recursive=True)
            if obj.object_name and obj.object_name.endswith("metadata.json")
        ]
        objects.sort(key=lambda obj: obj.last_modified or datetime.min.replace(tzinfo=timezone.utc), reverse=True)

        items: list[dict[str, Any]] = []
        for obj in objects[: max(1, min(limit, 200))]:
            response = None
            try:
                response = self.client.get_object(self.bucket, obj.object_name)
                metadata = json.loads(response.read().decode("utf-8"))
            except Exception as exc:
                self.last_error = str(exc)
                continue
            finally:
                if response is not None:
                    try:
                        response.close()
                        response.release_conn()
                    except Exception:
                        pass

            object_prefix = str(Path(obj.object_name).parent)
            image_object = metadata.get("storage_image") or metadata.get("storage_preview_image")
            if not image_object:
                image_object = f"{object_prefix}/image.png"
            items.append(
                {
                    "run_id": metadata.get("run_id") or object_prefix,
                    "kind": metadata.get("kind") or object_prefix.split("/", 1)[0],
                    "label": metadata.get("model_label")
                    or metadata.get("composition_label")
                    or metadata.get("model_name")
                    or metadata.get("kind")
                    or "synthetic",
                    "created_at": metadata.get("created_at") or (obj.last_modified.isoformat() if obj.last_modified else None),
                    "seed": metadata.get("base_seed", metadata.get("seed")),
                    "model": metadata.get("model") or metadata.get("model_name"),
                    "object_prefix": object_prefix,
                    "image_url": object_proxy_url(image_object),
                    "metadata_url": object_proxy_url(obj.object_name),
                    "summary": metadata.get("summary"),
                }
            )
        return items


@lru_cache(maxsize=1)
def get_storage() -> ObjectStorage:
    return ObjectStorage()
