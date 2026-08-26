"""Tests for the data pipeline — loader, formats, quality."""

import json
import tempfile

import pytest

from forge.data.formats import (
    alpaca_to_openai,
    completion_to_openai,
    convert_record,
    openai_to_sharegpt,
    sharegpt_to_openai,
)
from forge.data.loader import detect_format, get_stats
from forge.data.quality import compute_quality_report


class TestFormatDetection:
    """Test auto-detection of dataset formats."""

    def test_detect_sharegpt(self) -> None:
        with tempfile.NamedTemporaryFile(mode="w", suffix=".jsonl", delete=False) as f:
            record = {
                "conversations": [
                    {"from": "human", "value": "Hello"},
                    {"from": "gpt", "value": "Hi there!"},
                ]
            }
            f.write(json.dumps(record) + "\n")
            f.flush()
            assert detect_format(f.name) == "sharegpt"

    def test_detect_openai(self) -> None:
        with tempfile.NamedTemporaryFile(mode="w", suffix=".jsonl", delete=False) as f:
            record = {
                "messages": [
                    {"role": "user", "content": "Hello"},
                    {"role": "assistant", "content": "Hi!"},
                ]
            }
            f.write(json.dumps(record) + "\n")
            f.flush()
            assert detect_format(f.name) == "openai"

    def test_detect_alpaca(self) -> None:
        with tempfile.NamedTemporaryFile(mode="w", suffix=".jsonl", delete=False) as f:
            record = {"instruction": "Translate", "input": "Hello", "output": "Hola"}
            f.write(json.dumps(record) + "\n")
            f.flush()
            assert detect_format(f.name) == "alpaca"

    def test_detect_preference(self) -> None:
        with tempfile.NamedTemporaryFile(mode="w", suffix=".jsonl", delete=False) as f:
            record = {"prompt": "Tell me a joke", "chosen": "Why...", "rejected": "I don't..."}
            f.write(json.dumps(record) + "\n")
            f.flush()
            assert detect_format(f.name) == "preference"

    def test_detect_completion(self) -> None:
        with tempfile.NamedTemporaryFile(mode="w", suffix=".jsonl", delete=False) as f:
            record = {"text": "Hello world, this is a test."}
            f.write(json.dumps(record) + "\n")
            f.flush()
            assert detect_format(f.name) == "completion"


class TestFormatConversion:
    """Test format conversions."""

    def test_sharegpt_to_openai(self) -> None:
        record = {
            "conversations": [
                {"from": "human", "value": "What is 2+2?"},
                {"from": "gpt", "value": "4"},
            ]
        }
        result = sharegpt_to_openai(record)
        assert "messages" in result
        assert len(result["messages"]) == 2
        assert result["messages"][0]["role"] == "user"
        assert result["messages"][1]["role"] == "assistant"

    def test_alpaca_to_openai(self) -> None:
        record = {"instruction": "Translate to Spanish", "input": "Hello", "output": "Hola"}
        result = alpaca_to_openai(record)
        assert result["messages"][0]["role"] == "user"
        assert "Hello" in result["messages"][0]["content"]
        assert result["messages"][1]["content"] == "Hola"

    def test_completion_to_openai_with_text(self) -> None:
        record = {"text": "Question\n\nAssistant: Answer here"}
        result = completion_to_openai(record)
        assert result["messages"][0]["role"] == "user"
        assert result["messages"][1]["role"] == "assistant"

    def test_openai_to_sharegpt_roundtrip(self) -> None:
        original = {
            "conversations": [
                {"from": "human", "value": "Hi"},
                {"from": "gpt", "value": "Hello!"},
            ]
        }
        openai_format = sharegpt_to_openai(original)
        back = openai_to_sharegpt(openai_format)
        assert back["conversations"][0]["from"] == "human"
        assert back["conversations"][1]["from"] == "gpt"

    def test_convert_record_same_format(self) -> None:
        record = {"messages": [{"role": "user", "content": "Hi"}]}
        result = convert_record(record, "openai", "openai")
        assert result == record

    def test_convert_record_two_hop(self) -> None:
        """Test conversion via OpenAI intermediate (sharegpt → alpaca)."""
        record = {
            "conversations": [
                {"from": "human", "value": "Explain gravity"},
                {"from": "gpt", "value": "Gravity is a force..."},
            ]
        }
        result = convert_record(record, "sharegpt", "alpaca")
        assert "instruction" in result
        assert "output" in result

    def test_convert_unknown_raises(self) -> None:
        with pytest.raises(ValueError, match="No converter available"):
            convert_record({}, "nonexistent", "also_nonexistent")


class TestDataQuality:
    """Test data quality analysis."""

    def test_quality_report_basic(self) -> None:
        with tempfile.NamedTemporaryFile(mode="w", suffix=".jsonl", delete=False) as f:
            for i in range(10):
                record = {"text": f"Sample text number {i} with some content to analyze."}
                f.write(json.dumps(record) + "\n")
            f.flush()

            report = compute_quality_report(f.name)
            assert report["num_sampled"] == 10
            assert "text" in report["columns"]
            assert report["quality_score"] >= 0
            assert report["quality_score"] <= 100

    def test_quality_detects_duplicates(self) -> None:
        with tempfile.NamedTemporaryFile(mode="w", suffix=".jsonl", delete=False) as f:
            for _ in range(5):
                f.write(json.dumps({"text": "Exact same content"}) + "\n")
            f.flush()

            report = compute_quality_report(f.name)
            assert report["exact_duplicates"] == 4  # 5 total - 1 unique = 4 dupes
            assert report["duplicate_ratio"] > 0

    def test_quality_empty_file(self) -> None:
        with tempfile.NamedTemporaryFile(mode="w", suffix=".jsonl", delete=False) as f:
            f.flush()
            report = compute_quality_report(f.name)
            assert report.get("num_samples", 0) == 0 or "error" in report

    def test_get_stats(self) -> None:
        with tempfile.NamedTemporaryFile(mode="w", suffix=".jsonl", delete=False) as f:
            for i in range(5):
                f.write(json.dumps({"prompt": f"Question {i}", "completion": f"Answer {i}"}) + "\n")
            f.flush()

            stats = get_stats(f.name)
            assert stats["num_samples"] == 5
            assert stats["format"] == "completion"
