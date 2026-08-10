"""Dataset format converters.

Converts between common fine-tuning data formats so users can
bring any format and Forge normalizes it internally.
"""

from __future__ import annotations

from typing import Any, Callable, Dict, List


def sharegpt_to_openai(record: Dict[str, Any]) -> Dict[str, Any]:
    """Convert ShareGPT format to OpenAI-chat format.

    ShareGPT: {"conversations": [{"from": "human", "value": "..."}, {"from": "gpt", "value": "..."}]}
    OpenAI:   {"messages": [{"role": "user", "content": "..."}, {"role": "assistant", "content": "..."}]}
    """
    role_map = {
        "human": "user",
        "gpt": "assistant",
        "system": "system",
    }
    messages = []
    for turn in record.get("conversations", []):
        role = role_map.get(turn.get("from", ""), turn.get("from", "user"))
        content = turn.get("value", "")
        messages.append({"role": role, "content": content})
    return {"messages": messages}


def alpaca_to_openai(record: Dict[str, Any]) -> Dict[str, Any]:
    """Convert Alpaca format to OpenAI-chat format.

    Alpaca: {"instruction": "...", "input": "...", "output": "..."}
    """
    messages = []
    if record.get("system"):
        messages.append({"role": "system", "content": record["system"]})

    instruction = record.get("instruction", "")
    input_text = record.get("input", "")
    if input_text:
        user_content = f"{instruction}\n\n{input_text}"
    else:
        user_content = instruction
    messages.append({"role": "user", "content": user_content})
    messages.append({"role": "assistant", "content": record.get("output", "")})
    return {"messages": messages}


def completion_to_openai(record: Dict[str, Any]) -> Dict[str, Any]:
    """Convert simple prompt/completion to OpenAI-chat format.

    Input: {"prompt": "...", "completion": "..."} or {"text": "..."}
    """
    if "text" in record:
        # Split on common delimiters
        text = record["text"]
        for sep in ["\n\nAssistant:", "\n\n### Response:", "\n\nA:"]:
            if sep in text:
                parts = text.split(sep, 1)
                return {
                    "messages": [
                        {"role": "user", "content": parts[0].strip()},
                        {"role": "assistant", "content": parts[1].strip()},
                    ]
                }
        # No clear split — treat as single-turn
        return {"messages": [{"role": "user", "content": text}]}

    prompt = record.get("prompt", record.get("instruction", ""))
    completion = record.get("completion", record.get("response", record.get("output", "")))
    return {
        "messages": [
            {"role": "user", "content": prompt},
            {"role": "assistant", "content": completion},
        ]
    }


def openai_to_sharegpt(record: Dict[str, Any]) -> Dict[str, Any]:
    """Convert OpenAI-chat format to ShareGPT format."""
    role_map = {"user": "human", "assistant": "gpt", "system": "system"}
    conversations = []
    for msg in record.get("messages", []):
        conversations.append({
            "from": role_map.get(msg.get("role", ""), msg.get("role", "")),
            "value": msg.get("content", ""),
        })
    return {"conversations": conversations}


def openai_to_alpaca(record: Dict[str, Any]) -> Dict[str, Any]:
    """Convert OpenAI-chat format to Alpaca format."""
    messages = record.get("messages", [])
    result: Dict[str, Any] = {"instruction": "", "input": "", "output": ""}

    for msg in messages:
        role = msg.get("role", "")
        content = msg.get("content", "")
        if role == "system":
            result["system"] = content
        elif role == "user":
            result["instruction"] = content
        elif role == "assistant":
            result["output"] = content
            break  # Take first assistant response

    return result


# Registry of converters: (source_format, target_format) -> converter_fn
CONVERTERS: Dict[tuple[str, str], Callable[[Dict[str, Any]], Dict[str, Any]]] = {
    ("sharegpt", "openai"): sharegpt_to_openai,
    ("alpaca", "openai"): alpaca_to_openai,
    ("completion", "openai"): completion_to_openai,
    ("openai", "sharegpt"): openai_to_sharegpt,
    ("openai", "alpaca"): openai_to_alpaca,
}


def convert_record(
    record: Dict[str, Any],
    source_format: str,
    target_format: str,
) -> Dict[str, Any]:
    """Convert a single record between formats.

    Args:
        record: The data record to convert.
        source_format: Source format name.
        target_format: Target format name.

    Returns:
        Converted record.

    Raises:
        ValueError: If no converter is available for the format pair.
    """
    if source_format == target_format:
        return record

    key = (source_format, target_format)
    if key in CONVERTERS:
        return CONVERTERS[key](record)

    # Try two-hop via OpenAI as the canonical intermediate
    if source_format != "openai" and target_format != "openai":
        to_openai = CONVERTERS.get((source_format, "openai"))
        from_openai = CONVERTERS.get(("openai", target_format))
        if to_openai and from_openai:
            return from_openai(to_openai(record))

    raise ValueError(
        f"No converter available from '{source_format}' to '{target_format}'. "
        f"Available: {list(CONVERTERS.keys())}"
    )


def convert_dataset(
    dataset: Any,
    source_format: str,
    target_format: str,
) -> Any:
    """Convert an entire HuggingFace Dataset between formats.

    Args:
        dataset: HuggingFace Dataset.
        source_format: Source format.
        target_format: Target format.

    Returns:
        New Dataset with converted records.
    """
    if source_format == target_format:
        return dataset

    def _convert_batch(batch: Dict[str, list]) -> Dict[str, list]:
        # Reconstruct records from columnar batch
        keys = list(batch.keys())
        n = len(batch[keys[0]])
        results: Dict[str, list] = {}

        for i in range(n):
            record = {k: batch[k][i] for k in keys}
            converted = convert_record(record, source_format, target_format)
            for k, v in converted.items():
                if k not in results:
                    results[k] = []
                results[k].append(v)

        return results

    return dataset.map(
        _convert_batch,
        batched=True,
        remove_columns=dataset.column_names,
        desc=f"Converting {source_format} → {target_format}",
    )
