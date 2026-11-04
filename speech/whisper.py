"""Faster Whisper engine; multisegment, multilingual and word timestamp capable."""
from __future__ import annotations
import threading
import os
import numpy as np
from speech.dual_vad import crosscheck_speech_regions
from speech.contracts import Recognition
from speech.quality_guard import verify_utterances, speech_intervals
from speech.gap_recovery import recover_missing_voice
from speech.utterance_units import split_utterances


class WhisperEngine:
    def __init__(self, model: str, device: str = "auto", compute_type: str = "default"):
        self.model_id, self.device, self.compute_type = model, device, compute_type
        self._model = None
        self._load_lock = threading.Lock()
        self._infer_lock = threading.Lock()

    def _get_model(self):
        with self._load_lock:
            if self._model is None:
                if self.device != "cpu":
                    from speech.cuda_runtime import prepare_windows_cuda_runtime
                    prepare_windows_cuda_runtime()
                from faster_whisper import WhisperModel
                self._model = WhisperModel(self.model_id, device=self.device, compute_type=self.compute_type)
            return self._model

    def recognize(self, audio: np.ndarray, language: str = "auto", terms: str = "") -> Recognition:
        if len(audio) == 0:
            return Recognition(engine="whisper")
        from faster_whisper.vad import VadOptions, get_speech_timestamps

        samples = np.asarray(audio, dtype=np.float32)
        options = VadOptions(
            min_speech_duration_ms=200,
            min_silence_duration_ms=650,
            speech_pad_ms=180,
        )
        # We already run Silero once and verify every accepted utterance
        # against those same regions. A second internal VAD pass may silently
        # remove quiet / interrupted words or change time alignment.
        kwargs = {
            "language": None if language in ("auto", "und", "") else language,
            "multilingual": language in ("auto", "und", ""),
            "beam_size": 5,
            "word_timestamps": True,
            "vad_filter": False,
            "no_speech_threshold": 0.5,
            "log_prob_threshold": -1.0,
            "compression_ratio_threshold": 2.4,
            "hallucination_silence_threshold": 1.2,
            "condition_on_previous_text": False,
            "temperature": 0,
        }
        if terms.strip():
            kwargs["hotwords"] = terms[:1500]
        # Preserve acoustically verified segment words when the model's
        # forced word alignment is incomplete. This avoids deleting genuine
        # meeting content merely because word timestamps disagree.
        preserve_disputed = os.environ.get(
            "MEETING_WHISPER_PRESERVE_UNALIGNED_TEXT", "1"
        ).strip().lower() in ("1", "true", "yes")
        gap_recovery = os.environ.get(
            "MEETING_WHISPER_GAP_RECOVERY", "1"
        ).strip().lower() in ("1", "true", "yes")
        with self._infer_lock:
            speech_regions = get_speech_timestamps(
                samples, options, sampling_rate=16000
            )
            if os.environ.get(
                "MEETING_WHISPER_DUAL_VAD", "0"
            ).strip().lower() in ("1", "true", "yes"):
                speech_regions = crosscheck_speech_regions(samples, speech_regions)
            if not speech_regions:
                return Recognition(engine="whisper", speech_detected=False)
            intervals_ms = speech_intervals(
                speech_regions, round(len(samples) * 1000 / 16000)
            )

            window_lid = (
                language in ("auto", "und", "")
                and os.getenv("MEETING_WHISPER_WINDOW_LID", "1").strip().lower()
                in ("1", "true", "yes")
            )
            if window_lid:
                from speech.multilingual_windows import recognize_windows
                recognized = recognize_windows(
                    samples, speech_regions,
                    lambda chunk, **kw: self._get_model().transcribe(chunk, **kw),
                    dict(kwargs, multilingual=True),
                    preserve_disputed_text=preserve_disputed,
                )
                primary = recognized.utterances
                detected = recognized.language
                probability = recognized.confidence
            else:
                # The iterator is lazy, and must be exhausted under the lock.
                segments, info = self._get_model().transcribe(samples, **kwargs)
                pieces = list(segments)
                detected = (
                    language if language not in ("auto", "und", "")
                    else getattr(info, "language", None) or "und"
                )
                probability = getattr(info, "language_probability", None)
                primary = verify_utterances(
                    pieces, speech_regions,
                    round(len(samples) * 1000 / 16000), detected,
                    preserve_disputed_text=preserve_disputed,
                )

            # Bounded actual ASR retry for a VAD-positive interval omitted
            # from accepted speech. No text is fabricated, and the original
            # transcription is retained if retry fails.
            if gap_recovery:
                primary, recovery = recover_missing_voice(
                    samples, speech_regions, primary,
                    lambda chunk, **kw: self._get_model().transcribe(chunk, **kw),
                    kwargs, language=language,
                )
            else:
                recovery = {
                    "uncovered_speech_intervals": (),
                    "recovery_attempted": 0,
                    "recovery_confirmed": 0,
                    "recovery_failed": 0,
                    "disabled": True,
                }
        return Recognition(
            utterances=split_utterances(primary),
            language=detected,
            confidence=probability,
            engine="whisper",
            speech_detected=True,
            speech_intervals_ms=intervals_ms,
            recovery_diagnostics=recovery,
        )
