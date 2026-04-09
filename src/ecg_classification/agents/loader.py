from __future__ import annotations

from pathlib import Path

from ecg_classification.agents.schema import AgentSpec


def load_agent_spec(path: str | Path) -> AgentSpec:
    agent_path = Path(path)
    content = agent_path.read_text(encoding="utf-8").strip()
    metadata, body = _split_frontmatter(content)

    return AgentSpec(
        name=str(metadata.get("name", agent_path.stem)),
        role=str(metadata.get("role", "")),
        objective=str(metadata.get("objective", "")),
        model=_optional_str(metadata.get("model")),
        temperature=_optional_float(metadata.get("temperature")),
        handoff_to=_optional_str(metadata.get("handoff_to")),
        tools=tuple(metadata.get("tools", [])),
        system_prompt=body.strip(),
        source_path=agent_path,
    )


def _split_frontmatter(content: str) -> tuple[dict[str, object], str]:
    if not content.startswith("---\n"):
        return {}, content

    parts = content.split("\n---\n", 1)
    if len(parts) != 2:
        raise ValueError("Frontmatter malformado: delimitador final '---' nao encontrado.")

    raw_frontmatter = parts[0][4:]
    body = parts[1]
    return _parse_frontmatter(raw_frontmatter), body


def _parse_frontmatter(raw_frontmatter: str) -> dict[str, object]:
    metadata: dict[str, object] = {}
    current_list_key: str | None = None

    for raw_line in raw_frontmatter.splitlines():
        line = raw_line.rstrip()
        if not line:
            continue
        if line.startswith("  - ") or line.startswith("- "):
            if current_list_key is None:
                raise ValueError("Item de lista encontrado sem chave de lista ativa no frontmatter.")
            metadata.setdefault(current_list_key, [])
            value = line.split("- ", 1)[1].strip()
            casted = _cast_scalar(value)
            assert isinstance(metadata[current_list_key], list)
            metadata[current_list_key].append(casted)
            continue
        if ":" not in line:
            raise ValueError(f"Linha invalida no frontmatter: {line}")
        key, raw_value = line.split(":", 1)
        key = key.strip()
        value = raw_value.strip()
        if value == "":
            metadata[key] = []
            current_list_key = key
            continue
        metadata[key] = _cast_scalar(value)
        current_list_key = None

    return metadata


def _cast_scalar(value: str) -> object:
    lowered = value.lower()
    if lowered in {"null", "none"}:
        return None
    if lowered in {"true", "false"}:
        return lowered == "true"
    try:
        if "." in value:
            return float(value)
        return int(value)
    except ValueError:
        return value.strip("\"'")


def _optional_str(value: object) -> str | None:
    if value is None:
        return None
    return str(value)


def _optional_float(value: object) -> float | None:
    if value is None:
        return None
    return float(value)
