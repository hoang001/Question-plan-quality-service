"""JSON Pointer va JSON Patch helper an toan, khong mutate object goc."""

from __future__ import annotations

from copy import deepcopy
from typing import Any


class JsonPointerError(ValueError):
    """Loi truy cap hoac apply JSON Pointer/Patch."""


def decode_pointer_token(token: str) -> str:
    return token.replace("~1", "/").replace("~0", "~")


def encode_pointer_token(token: str) -> str:
    return token.replace("~", "~0").replace("/", "~1")


def parse_json_pointer(pointer: str) -> list[str]:
    if pointer == "":
        return []
    if not isinstance(pointer, str) or not pointer.startswith("/"):
        raise JsonPointerError(f"JSON Pointer khong hop le: {pointer!r}")
    return [decode_pointer_token(token) for token in pointer.split("/")[1:]]


def get_by_json_pointer(obj: Any, pointer: str) -> Any:
    current = obj
    for token in parse_json_pointer(pointer):
        if isinstance(current, dict):
            if token not in current:
                raise JsonPointerError(f"Path khong ton tai: {pointer}")
            current = current[token]
            continue
        if isinstance(current, list):
            if token == "-":
                raise JsonPointerError(f"Khong the get path list append token: {pointer}")
            try:
                index = int(token)
            except ValueError as exc:
                raise JsonPointerError(f"List index khong hop le trong path: {pointer}") from exc
            if index < 0 or index >= len(current):
                raise JsonPointerError(f"List index vuot gioi han trong path: {pointer}")
            current = current[index]
            continue
        raise JsonPointerError(f"Khong the truy cap tiep path: {pointer}")
    return current


def _get_parent(target: Any, pointer: str) -> tuple[Any, str]:
    tokens = parse_json_pointer(pointer)
    if not tokens:
        raise JsonPointerError("Khong the thao tac parent tren root pointer.")
    parent_pointer = "/" + "/".join(encode_pointer_token(token) for token in tokens[:-1]) if len(tokens) > 1 else ""
    return get_by_json_pointer(target, parent_pointer), tokens[-1]


def set_by_json_pointer(obj: Any, pointer: str, value: Any) -> Any:
    target = deepcopy(obj)
    if pointer == "":
        return deepcopy(value)
    parent, token = _get_parent(target, pointer)
    if isinstance(parent, dict):
        if token not in parent:
            raise JsonPointerError(f"Path khong ton tai de replace: {pointer}")
        parent[token] = deepcopy(value)
        return target
    if isinstance(parent, list):
        try:
            index = int(token)
        except ValueError as exc:
            raise JsonPointerError(f"List index khong hop le trong path: {pointer}") from exc
        if index < 0 or index >= len(parent):
            raise JsonPointerError(f"List index vuot gioi han de replace: {pointer}")
        parent[index] = deepcopy(value)
        return target
    raise JsonPointerError(f"Parent khong the replace: {pointer}")


def add_by_json_pointer(obj: Any, pointer: str, value: Any) -> Any:
    target = deepcopy(obj)
    if pointer == "":
        return deepcopy(value)
    parent, token = _get_parent(target, pointer)
    if isinstance(parent, dict):
        parent[token] = deepcopy(value)
        return target
    if isinstance(parent, list):
        if token == "-":
            parent.append(deepcopy(value))
            return target
        try:
            index = int(token)
        except ValueError as exc:
            raise JsonPointerError(f"List index khong hop le trong path: {pointer}") from exc
        if index < 0 or index > len(parent):
            raise JsonPointerError(f"List index vuot gioi han de add: {pointer}")
        parent.insert(index, deepcopy(value))
        return target
    raise JsonPointerError(f"Parent khong the add: {pointer}")


def remove_by_json_pointer(obj: Any, pointer: str) -> Any:
    target = deepcopy(obj)
    if pointer == "":
        raise JsonPointerError("Khong ho tro remove root object.")
    parent, token = _get_parent(target, pointer)
    if isinstance(parent, dict):
        if token not in parent:
            raise JsonPointerError(f"Path khong ton tai de remove: {pointer}")
        del parent[token]
        return target
    if isinstance(parent, list):
        try:
            index = int(token)
        except ValueError as exc:
            raise JsonPointerError(f"List index khong hop le trong path: {pointer}") from exc
        if index < 0 or index >= len(parent):
            raise JsonPointerError(f"List index vuot gioi han de remove: {pointer}")
        del parent[index]
        return target
    raise JsonPointerError(f"Parent khong the remove: {pointer}")


def apply_json_patch(obj: Any, patches: list[dict[str, Any]]) -> Any:
    target = deepcopy(obj)
    if not isinstance(patches, list):
        raise JsonPointerError("patches phai la list.")
    for patch in patches:
        if not isinstance(patch, dict):
            raise JsonPointerError("Moi patch phai la object.")
        op = patch.get("op")
        path = patch.get("path")
        if op not in {"replace", "add", "remove"}:
            raise JsonPointerError(f"Patch op khong ho tro: {op!r}")
        if not isinstance(path, str):
            raise JsonPointerError("Patch path phai la string JSON Pointer.")
        if op == "replace":
            target = set_by_json_pointer(target, path, patch.get("value"))
        elif op == "add":
            target = add_by_json_pointer(target, path, patch.get("value"))
        elif op == "remove":
            target = remove_by_json_pointer(target, path)
    return target
