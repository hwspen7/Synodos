"""One-time functional code layout migration (requires a clean Git worktree)."""
from __future__ import annotations
from collections import Counter
from pathlib import Path
import ast
import re
import subprocess

ROOT = Path(__file__).resolve().parents[1]
SPEECH = {
    "asr": "engine whisper qwen qwen_bridge qwen_worker hybrid multilingual_windows streaming",
    "audio": "acoustic_guard acoustic_timeline adaptive_audio audio_windows dual_vad enhancement recording silence_retention",
    "diarization": "adaptive_cluster analysis diarization ecapa_shadow_worker nemotron_activity nemotron_stream offline_speaker offline_speaker_cli overlap overlap_content_risk phase1b_diarization_models separation speaker_backends spherical_cluster",
    "captions": "asr_stage_trace caption_acoustic_alignment commit_boundary commit_guard content_first_projection corrections coverage_retention events gap_recovery quality_guard quality_profile text_format translation turn_integrity turn_split utterance_units vocabulary",
}
CHANNELS = set("corroborated_speaker_channels stable_speaker_channels retroactive_channel_recovery fused_short_turn_recovery short_turn_identity provisional_speakers rapid_speaker_fragmentation speaker_partition_evidence".split())
IDENTITY = set("identity_consensus identity_matcher registered_identity speaker_birth_guard speaker_cluster_birth speaker_evidence speaker_fallback speaker_resolution speaker_shadow speaker_stability speaker_tasks".split())
NAMES = dict(line.split(":") for line in """
phase07_audit:session_quality_audit
phase07_benchmark:session_quality_benchmark
phase09_acceptance:dialogue_acceptance
phase09_speaker_benchmark:dialogue_speaker_benchmark
phase12_release_gate:integrated_release_gate
phase1_fixed195_text_gate:fixed195_text_gate
phase1_ten_language_gate:language_gate
phase1_text_acceptance:text_acceptance
phase1a_compare:asr_pair_comparison
phase1a_quality_pair:asr_quality_pair
phase1b_compare:diarization_pair_comparison
phase1b_diarization_benchmark:diarization_benchmark
phase1b_pyannote_ami10:pyannote_ami_benchmark
phase1b_qwen_shadow:qwen_shadow_review
phase1b_reference_pressure:reference_pressure
phase1b_replay_manifest:replay_manifest
phase1b_temporal_overlap:temporal_overlap
phase2_build_trials:identity_trial_builder
phase2_cross_session_acceptance:cross_session_acceptance
phase1b_diarization_models:model_candidates
check_phase12_integrity:check_integrated_integrity
run_phase12_acceptance:run_integrated_acceptance
run_phase1_closure:run_content_closure
run_phase1_engineering:run_engineering_acceptance
run_phase1a_ablation:run_asr_ablation
run_phase1b_compare:run_diarization_comparison
doctor_06c:speech_runtime_doctor
asr_benchmark_06c:asr_runtime_benchmark
test_audio_frames_v2:test_audio_frames
test_model_profile_06a:test_model_profile
test_ollama_runtime_06b:test_ollama_runtime
test_phase2_manual_review:test_manual_review
test_phase2_verified_enrollment:test_verified_enrollment
test_phase2_identity_consensus:test_identity_consensus
test_phase2_identity_matcher:test_identity_matcher
test_phase2_speaker_store_integrity:test_speaker_store_integrity
test_phase3_index_cas:test_retrieval_index_cas
test_phase4_evidence_qa:test_evidence_qa
test_commit_guard_08:test_commit_guard
test_commit_integration_08:test_commit_integration
test_wav_checkpoint_08:test_wav_checkpoint
test_acoustic_turns_09:test_acoustic_turns
test_dual_vad_04e:test_dual_vad
test_quality_guard_04b:test_quality_guard_baseline
test_quality_guard_04d:test_quality_guard_recovery
""".strip().splitlines())
DOC_NAMES = dict(line.split(":") for line in """
foundation_02:infrastructure_foundation
stability_patch_01:runtime_stability
phase08_repair_status:recovery_status
phase09_six_issues:dialogue_reliability
phase12_integrated_upgrade:integrated_upgrade
phase12_research_release:research_release
phase12_roadmap_code_audit_20261104:architecture_code_audit_20261104
phase1_final_ami_ci_closure:ami_ci_closure
phase1_product_acceptance:product_acceptance
phase1_text_acceptance:text_acceptance
phase1b_pyannote_ami10_runbook:pyannote_ami_runbook
""".strip().splitlines())

def rename(name: str) -> str:
    if name in NAMES:
        return NAMES[name]
    if name.startswith("test_") and name[5:] in NAMES:
        return "test_" + NAMES[name[5:]]
    m = re.match(r"^(test_)?phase(?:\d+[a-z]?)[_-]?(.*)$", name)
    return (m[1] or "") + (m[2] or "runtime") if m else name

def tool_group(name: str) -> str:
    if name.startswith("ami_") or name == "prepare_ami_individual_headsets":
        return "evaluation/ami"
    if name.startswith(("fixed10_", "fixed120_", "fixed150_", "fixed195_", "prepare_meeting_reference_", "prepare_real_meeting_", "prepare_audio_review_")) or name == "freeze_fleurs_asr_corpus":
        return "evaluation/corpora"
    if name.startswith("speaker_") or name == "repair_speaker_scope":
        return "evaluation/speakers"
    if name.startswith(("asr_", "phase1a_", "phase1b_")) or "whisper" in name or "qwen" in name:
        return "evaluation/asr"
    if name.startswith("phase") or any(k in name for k in ("acceptance", "release_gate", "integrity", "knowledge_contract")):
        return "validation"
    if name.startswith(("doctor_", "diagnose_", "evaluate_", "meeting_audio_review_", "live_caption_")):
        return "diagnostics"
    return "maintenance"

def quality_group(stem: str) -> str:
    name = stem.removeprefix("test_")
    if name.startswith("ami_"):
        return "ami"
    if name.startswith(("speaker_", "nemotron_", "ecapa_", "adaptive_speaker_", "spherical_speaker_", "short_speaker_", "multi_speaker_", "acoustic_late_")):
        return "speakers"
    if any(k in name for k in ("asr", "whisper", "qwen", "language", "fleurs", "text_acceptance", "text_gate", "quality_pair", "temporal_overlap", "reference_pressure", "pair_comparison", "diarization")):
        return "recognition"
    return "integrity"

def destination(path: str) -> str:
    p = Path(path)
    stem, ext = p.stem, p.suffix
    if path.startswith(("src/", "public/", "tests/frontend/")):
        return path
    if path.startswith("speech/") and ext == ".py" and stem != "__init__":
        for group, members in SPEECH.items():
            if stem in members.split():
                return "speech/" + group + "/" + rename(stem) + ".py"
    if path.startswith("sessions/") and ext == ".py":
        if stem in CHANNELS: return "sessions/channels/" + stem + ".py"
        if stem in IDENTITY: return "sessions/identity/" + stem + ".py"
    if path.startswith("phase07/"): return "runtime_quality/" + p.name
    for old, new in (("tests/phase06c/", "tests/runtime_compat/"), ("tests/phase07/", "tests/runtime_quality/"), ("tests/phase09/", "tests/dialogue/")):
        if path.startswith(old): return new + p.name
    if path.startswith("tests/quality/") and ext == ".py" and stem != "__init__":
        return "tests/quality/" + quality_group(stem) + "/" + rename(stem) + ".py"
    if path.startswith("tests/") and ext == ".py":
        return str(p.with_name(rename(stem) + ext))
    if path.startswith("tests/windows/") and ext.lower() == ".ps1":
        return str(p.with_name(rename(stem) + ext))
    if path.startswith("tools/") and p.parent == Path("tools"):
        if ext == ".py": return "tools/" + tool_group(stem) + "/" + rename(stem) + ".py"
        if ext.lower() == ".ps1": return "tools/" + rename(stem) + ext
    if path == "docs/phase07.env.example": return "config/runtime_quality.env.example"
    if path == "config/phase06c.env.example": return "config/speech_runtime.env.example"
    if path.startswith("docs/") and ext == ".md":
        name = DOC_NAMES.get(stem, rename(stem))
        if stem.startswith(("ami_", "fixed", "phase1b_")) or "benchmark" in stem or "model" in stem:
            folder = "research"
        elif stem.startswith(("phase", "meeting_knowledge_", "live_caption_", "real_meeting_", "context_correction")) or "acceptance" in stem:
            folder = "validation"
        elif stem in ("backend_layout", "foundation_02", "85_point_architecture", "speaker_stage1_architecture", "phase12_roadmap_code_audit_20261104"):
            folder = "architecture"
        else:
            folder = "guides"
        return "docs/" + folder + "/" + name + ".md"
    return path

def replace_all(content: str, values: dict[str, str]) -> str:
    # A single regex pass prevents rewriting already-rewritten import paths.
    keys = sorted(values, key=len, reverse=True)
    pattern = re.compile("|".join(
        r"(?<![\w])" + re.escape(key) + r"(?![\w])" for key in keys
    ))
    content = pattern.sub(lambda match: values[match.group(0)], content)
    # Legacy 'from tools import module' imports cannot follow a file move
    # through dotted/path replacement alone.
    content = content.replace(
        "from tools import meeting_knowledge_synthetic as acceptance",
        "from tools.maintenance import meeting_knowledge_synthetic as acceptance",
    )
    return content

def main() -> None:
    if (ROOT / "docs/architecture/layout_migrated.md").exists():
        print("Already migrated")
        return
    files = [x for x in subprocess.check_output(["git", "ls-files", "-z"], cwd=ROOT).decode().split("\0") if x]
    mapping = {old: destination(old) for old in files}
    duplicates = [name for name, count in Counter(mapping.values()).items() if count != 1]
    if duplicates: raise RuntimeError("Destination collisions: " + str(duplicates))
    # Version markers belong in metadata, not permanent source paths.
    bad_names = [
        p for p in mapping.values()
        if not p.startswith("tests/frontend/")
        and re.search(r"(?:^|[/_-])phase\d|(?:^|[_.-])v\d+(?:[_.-]|$)", p, re.I)
    ]
    if bad_names:
        raise RuntimeError("Unnormalized source paths: " + repr(bad_names))
    replacements = {
        "phase07.": "runtime_quality.", "phase07/": "runtime_quality/",
        "tests.phase06c.": "tests.runtime_compat.", "tests.phase07.": "tests.runtime_quality.",
        "tests.phase09.": "tests.dialogue.",
        "tests/phase06c/": "tests/runtime_compat/", "tests/phase07/": "tests/runtime_quality/",
        "tests/phase09/": "tests/dialogue/"
    }
    for old, new in mapping.items():
        if old == new: continue
        replacements[old] = new
        replacements[old.replace("/", "\\\\")] = new.replace("/", "\\\\")
        if old.endswith(".py"):
            replacements[old[:-3].replace("/", ".")] = new[:-3].replace("/", ".")
    updated = {}
    for old, new in mapping.items():
        data = (ROOT / old).read_bytes()
        if old.endswith((".py", ".md", ".yml", ".yaml", ".ps1", ".toml", ".cmd", ".txt", ".json", ".example")):
            try: content = data.decode("utf-8")
            except UnicodeDecodeError: updated[new] = data; continue
            if old != new and old.endswith(".py"):
                try:
                    relative = [n.lineno for n in ast.walk(ast.parse(content)) if isinstance(n, ast.ImportFrom) and n.level]
                except SyntaxError: relative = []
                if relative: raise RuntimeError("Relative import in moved " + old + ": " + str(relative))
            content = replace_all(content, replacements)
            if old != new and old.endswith(".py") and "__file__" in content and old.startswith(("speech/", "tools/")):
                gain = new.count("/") - old.count("/")
                if gain:
                    content = re.sub(r"(Path\(__file__\)(?:\.resolve\(\))?\.parents\[)(\d+)(\])",
                        lambda m: m[1] + str(int(m[2]) + gain) + m[3], content)
            updated[new] = content.encode("utf-8")
        else: updated[new] = data
    for old, new in mapping.items():
        if old != new: (ROOT / old).unlink()
    for name, data in updated.items():
        target = ROOT / name
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_bytes(data)
    for name in updated:
        if name.endswith(".py") and name.startswith(("speech/", "sessions/", "tools/", "tests/", "runtime_quality/")):
            parent = (ROOT / name).parent
            if not (parent / "__init__.py").exists():
                (parent / "__init__.py").write_text('"""Functional module package."""\n', encoding="utf-8")
    for name in (".github/workflows/ci.yml", "tools/run_engineering_acceptance.ps1"):
        f = ROOT / name
        if f.exists():
            content = f.read_text()
            content = re.sub(r"(^[^\n]*\bcompileall\b[^\n]*)", lambda m: m[1].replace(" phase07 ", " runtime_quality "), content, flags=re.M)
            f.write_text(content)
    note = ROOT / "docs/architecture/layout_migrated.md"
    note.parent.mkdir(parents=True, exist_ok=True)
    note.write_text(
        "# Functional source layout\n\n"
        "Speech is organized by ASR, acoustics, diarization, and captions.\n"
        "Session identity and acoustic channel logic have separate packages.\n"
        "Tools are grouped by evaluation domain and validation purpose.\n"
        "Tests use functional labels rather than milestone names.\n"
        "Frontend Vue/TypeScript and public assets are untouched.\n"
        "Historical protocol schema identifiers and frozen corpus names remain unchanged.\n"
    )
    print("Moved", sum(old != new for old, new in mapping.items()), "tracked files")

if __name__ == "__main__":
    main()
