"""Client gọi API LLM dùng chung cho các pipeline.

File này quản lý header xác thực, endpoint fallback, gọi chat completions,
liệt kê model và chuẩn hóa lỗi API để pipeline không phải gọi requests trực tiếp.
"""

import json
import time
from typing import Any
from urllib.parse import urljoin

import requests

from .config import AppConfig


DEFAULT_MODELS_ENDPOINTS = ["/v1/models", "/models"]
DEFAULT_CHAT_COMPLETIONS_ENDPOINTS = ["/v1/chat/completions", "/chat/completions"]


class ApiError(RuntimeError):
    def __init__(self, message: str, status_code: int | None = None):
        super().__init__(message)
        self.status_code = status_code


class LLMClient:
    def __init__(self, config: AppConfig):
        self.config = config

    def auth_headers(self) -> dict[str, str]:
        return {
            "Authorization": f"Bearer {self.config.api_key}",
            "Content-Type": "application/json",
        }

    def endpoint_url(self, endpoint: str) -> str:
        return urljoin(self.config.base_url, endpoint.lstrip("/"))

    def configured_endpoints(self, custom_endpoint: str | None, defaults: list[str]) -> list[str]:
        if custom_endpoint:
            return [custom_endpoint]
        return defaults

    def list_models(self) -> list[str]:
        data, _endpoint = self.get_json_with_fallback(
            self.configured_endpoints(self.config.models_endpoint, DEFAULT_MODELS_ENDPOINTS)
        )
        items = data.get("data", data) if isinstance(data, dict) else data
        if not isinstance(items, list):
            raise ApiError("Response models không chứa danh sách model.")

        model_ids = []
        for item in items:
            if isinstance(item, dict) and item.get("id"):
                model_ids.append(str(item["id"]))
            elif isinstance(item, str):
                model_ids.append(item)

        if not model_ids:
            raise ApiError("Không tìm thấy model id trong response models.")
        return sorted(model_ids)

    def chat_completion(self, model: str, messages: list[dict[str, str]], temperature: float = 0) -> dict[str, Any]:
        payload = {
            "model": model,
            "messages": messages,
            "temperature": temperature,
        }
        started_at = time.perf_counter()
        data, endpoint = self.post_json_with_fallback(
            self.configured_endpoints(self.config.chat_completions_endpoint, DEFAULT_CHAT_COMPLETIONS_ENDPOINTS),
            payload,
            timeout_seconds=(
                max(1, int(getattr(self.config, "gemma_request_timeout_seconds", 30)))
                if model == self.config.primary_judge_model
                else None
            ),
        )
        latency_seconds = round(time.perf_counter() - started_at, 3)
        return {
            "model": model,
            "endpoint": endpoint,
            "latency_seconds": latency_seconds,
            "raw_response": data,
            "content": self.extract_chat_content(data),
        }

    def get_json_with_fallback(self, endpoints: list[str]) -> tuple[Any, str]:
        last_error: ApiError | None = None
        attempt_errors = []
        for endpoint in endpoints:
            url = self.endpoint_url(endpoint)
            try:
                response = requests.get(url, headers=self.auth_headers(), timeout=self.config.request_timeout_seconds)
            except requests.RequestException as exc:
                last_error = ApiError(f"{endpoint}: {exc}")
                attempt_errors.append(str(last_error))
                continue

            if response.ok:
                try:
                    return response.json(), endpoint
                except ValueError as exc:
                    last_error = ApiError(f"{endpoint}: response không phải JSON hợp lệ: {exc}", response.status_code)
                    attempt_errors.append(str(last_error))
                    continue

            last_error = ApiError(
                f"{endpoint}: HTTP {response.status_code}: {self.short_api_message(response)}",
                response.status_code,
            )
            attempt_errors.append(str(last_error))

        if attempt_errors:
            raise ApiError("Đã thử các endpoint: " + " | ".join(attempt_errors), last_error.status_code if last_error else None)
        raise ApiError("Chưa thử endpoint nào.")

    def post_json_with_fallback(
        self,
        endpoints: list[str],
        payload: dict[str, Any],
        *,
        timeout_seconds: int | None = None,
    ) -> tuple[Any, str]:
        last_error: ApiError | None = None
        attempt_errors = []
        deadline = time.perf_counter() + timeout_seconds if timeout_seconds is not None else None
        for endpoint in endpoints:
            request_timeout = self.config.request_timeout_seconds
            if deadline is not None:
                request_timeout = deadline - time.perf_counter()
                if request_timeout <= 0:
                    last_error = ApiError(f"Đã hết timeout tổng {timeout_seconds}s trước endpoint {endpoint}.")
                    attempt_errors.append(str(last_error))
                    break
            url = self.endpoint_url(endpoint)
            try:
                response = requests.post(
                    url,
                    headers=self.auth_headers(),
                    json=payload,
                    timeout=request_timeout,
                )
            except requests.RequestException as exc:
                last_error = ApiError(f"{endpoint}: {exc}")
                attempt_errors.append(str(last_error))
                continue

            if response.ok:
                try:
                    return response.json(), endpoint
                except ValueError as exc:
                    last_error = ApiError(f"{endpoint}: response không phải JSON hợp lệ: {exc}", response.status_code)
                    attempt_errors.append(str(last_error))
                    continue

            last_error = ApiError(
                f"{endpoint}: HTTP {response.status_code}: {self.short_api_message(response)}",
                response.status_code,
            )
            attempt_errors.append(str(last_error))

        if attempt_errors:
            raise ApiError("Đã thử các endpoint: " + " | ".join(attempt_errors), last_error.status_code if last_error else None)
        raise ApiError("Chưa thử endpoint nào.")

    @staticmethod
    def short_api_message(response: requests.Response) -> str:
        try:
            body = response.json()
        except ValueError:
            text = response.text.strip()
            return text[:300] if text else response.reason

        if isinstance(body, dict):
            error = body.get("error")
            if isinstance(error, dict):
                return str(error.get("message") or error.get("type") or error)[:300]
            if error:
                return str(error)[:300]
            message = body.get("message")
            if message:
                return str(message)[:300]
        return json.dumps(body, ensure_ascii=False)[:300]

    @staticmethod
    def extract_chat_content(response_data: Any) -> str:
        try:
            choice = response_data["choices"][0]
            message = choice.get("message") or {}
            if "content" in message:
                content = message["content"]
                if isinstance(content, list):
                    return "".join(part.get("text", "") for part in content if isinstance(part, dict))
                return str(content)
            if "text" in choice:
                return str(choice["text"])
        except (KeyError, IndexError, TypeError):
            pass
        return json.dumps(response_data, ensure_ascii=False)
