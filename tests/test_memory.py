from pathlib import Path
from src.memory import load_threat_intel


def test_load_threat_intel_returns_none_when_missing(tmp_path):
    result = load_threat_intel(1, memory_dir=tmp_path)
    assert result is None


def test_load_threat_intel_returns_content_when_present(tmp_path):
    brief = "THREAT_INTEL: Level 1 → Level 2\nFRAUD_PATTERNS_FOUND: smishing"
    (tmp_path / "level_1_intel.md").write_text(brief)
    result = load_threat_intel(1, memory_dir=tmp_path)
    assert result == brief


def test_load_threat_intel_reads_correct_level(tmp_path):
    (tmp_path / "level_1_intel.md").write_text("level 1 intel")
    (tmp_path / "level_2_intel.md").write_text("level 2 intel")
    assert load_threat_intel(1, memory_dir=tmp_path) == "level 1 intel"
    assert load_threat_intel(2, memory_dir=tmp_path) == "level 2 intel"
