import os
os.environ.setdefault("OMP_NUM_THREADS", "4")
os.environ.setdefault("MKL_NUM_THREADS", "4")
os.environ.setdefault("OPENBLAS_NUM_THREADS", "4")
os.environ.setdefault("VECLIB_MAXIMUM_THREADS", "4")
os.environ.setdefault("NUMEXPR_NUM_THREADS", "4")

import argparse
import sys
import time
import numpy as np
import av
from faster_whisper import WhisperModel, decode_audio
from nlu_parser import denoise_audio


def detect_stereo_channels(audio_path):
    """
    Checks if the audio file has 2 independent stereo channels (e.g. 2 speakers on separate channels).
    Returns (is_stereo, left_channel, right_channel).
    """
    try:
        with av.open(audio_path) as container:
            for s in container.streams:
                if s.type == "audio" and s.channels >= 2:
                    left, right = decode_audio(audio_path, split_stereo=True)
                    check_len = min(len(left), len(right), 16000 * 30)
                    if check_len > 0:
                        corr = np.corrcoef(left[:check_len], right[:check_len])[0, 1]
                        # Low correlation means channels have distinct audio (different speakers)
                        if corr < 0.85:
                            return True, left, right
                    return False, left, None
    except Exception:
        pass
    return False, None, None


def cluster_mono_speakers(audio, segments, speaker1="I told", speaker2="They told"):
    """
    Clusters mono audio segments into 2 speakers using acoustic features (RMS, ZCR, Spectral Centroid).
    """
    if not segments:
        return []
    if len(segments) == 1:
        return [(segments[0].start, segments[0].end, speaker1, segments[0].text.strip())]

    sr = 16000
    features = []
    valid_indices = []
    for i, s in enumerate(segments):
        start_samp = int(s.start * sr)
        end_samp = min(int(s.end * sr), len(audio))
        chunk = audio[start_samp:end_samp]
        if len(chunk) < 320:
            continue
        rms = float(np.sqrt(np.mean(chunk**2)))
        zcr = float(np.mean(np.abs(np.diff(np.sign(chunk)))) / 2)
        spec = np.abs(np.fft.rfft(chunk))
        freqs = np.fft.rfftfreq(len(chunk), 1 / sr)
        centroid = float(np.sum(freqs * spec) / (np.sum(spec) + 1e-9))
        features.append([rms, zcr, centroid])
        valid_indices.append(i)

    if len(features) < 2:
        return [(s.start, s.end, speaker1, s.text.strip()) for s in segments]

    features = np.array(features)
    mean = np.mean(features, axis=0)
    std = np.std(features, axis=0) + 1e-9
    norm = (features - mean) / std

    # 2-means clustering
    c0 = norm[np.argmin(norm[:, 2])]
    c1 = norm[np.argmax(norm[:, 2])]
    labels = np.zeros(len(features), dtype=int)
    for _ in range(15):
        d0 = np.sum((norm - c0) ** 2, axis=1)
        d1 = np.sum((norm - c1) ** 2, axis=1)
        new_labels = (d1 < d0).astype(int)
        if np.all(new_labels == labels):
            break
        labels = new_labels
        if np.sum(labels == 0) > 0:
            c0 = np.mean(norm[labels == 0], axis=0)
        if np.sum(labels == 1) > 0:
            c1 = np.mean(norm[labels == 1], axis=0)

    first_label = labels[0]
    label_map = {
        idx: (speaker1 if lab == first_label else speaker2)
        for idx, lab in zip(valid_indices, labels)
    }

    out = []
    for i, s in enumerate(segments):
        spk = label_map.get(i, speaker1)
        out.append((s.start, s.end, spk, s.text.strip()))
    return out


def alternate_speakers_by_pause(segments, speaker1="I told", speaker2="They told", pause_threshold=0.8):
    """
    Alternates speaker turns when there is a noticeable pause between utterances.
    """
    out = []
    current_spk = speaker1
    other_spk = speaker2
    prev_end = None
    for s in segments:
        if prev_end is not None and (s.start - prev_end) >= pause_threshold:
            current_spk, other_spk = other_spk, current_spk
        out.append((s.start, s.end, current_spk, s.text.strip()))
        prev_end = s.end
    return out


def format_dialogue(dialogue_events):
    """
    Groups consecutive segments from the same speaker and formats as a conversation.
    """
    if not dialogue_events:
        return ""

    grouped = []
    current_spk = None
    current_texts = []

    for _, _, speaker, text in dialogue_events:
        if not text:
            continue
        if speaker != current_spk:
            if current_spk is not None and current_texts:
                grouped.append((current_spk, " ".join(current_texts)))
            current_spk = speaker
            current_texts = [text]
        else:
            current_texts.append(text)

    if current_spk is not None and current_texts:
        grouped.append((current_spk, " ".join(current_texts)))

    lines = []
    for speaker, text in grouped:
        lines.append(f"{speaker}: {text}")
    return "\n".join(lines)


def main():
    parser = argparse.ArgumentParser(
        description="Run faster-whisper speech recognition with speaker dialogue support."
    )
    parser.add_argument(
        "--audio",
        "-a",
        type=str,
        default="tests/data/stereo_diarization.wav",
        help="Path to audio file (default: tests/data/stereo_diarization.wav)",
    )
    parser.add_argument(
        "--model",
        "-m",
        type=str,
        default="tiny",
        help="Model size (e.g. tiny, base, small, medium, large-v3) (default: tiny)",
    )
    parser.add_argument(
        "--device",
        "-d",
        type=str,
        default="cpu",
        choices=["cpu", "cuda", "auto"],
        help="Device to run inference on (default: cpu)",
    )
    parser.add_argument(
        "--compute_type",
        "-c",
        type=str,
        default="int8",
        help="Quantization (int8, float32, float16) (default: int8)",
    )
    parser.add_argument(
        "--beam_size",
        "-b",
        type=int,
        default=5,
        help="Beam size (default: 5)",
    )
    parser.add_argument(
        "--language",
        "-l",
        type=str,
        default=None,
        help="Language code (optional, e.g. 'en', 'es')",
    )
    parser.add_argument(
        "--speaker1",
        type=str,
        default="I told",
        help="Label for Speaker 1 (default: 'I told')",
    )
    parser.add_argument(
        "--speaker2",
        type=str,
        default="They told",
        help="Label for Speaker 2 (default: 'They told')",
    )
    parser.add_argument(
        "--diarize_mode",
        type=str,
        default="auto",
        choices=["auto", "stereo", "acoustic", "pause", "none"],
        help="Diarization strategy: 'auto' (detect stereo or acoustic), 'stereo', 'acoustic', 'pause', or 'none'",
    )

    args = parser.parse_args()

    print(f"Loading faster-whisper model '{args.model}' on {args.device} ({args.compute_type})...")
    start_load = time.time()
    try:
        model = WhisperModel(args.model, device=args.device, compute_type=args.compute_type, cpu_threads=4)
    except RuntimeError as _e:
        if "mkl_malloc" in str(_e).lower() and args.model != "tiny":
            print(f"[Warning] MKL memory allocation error on '{args.model}'. Falling back to 'tiny'...")
            model = WhisperModel("tiny", device=args.device, compute_type=args.compute_type, cpu_threads=2)
        else:
            raise
    print(f"Model loaded in {time.time() - start_load:.2f} seconds.")

    print(f"Transcribing: {args.audio}")
    start_transcribe = time.time()

    dialogue_events = []

    # Check for independent stereo channels
    is_stereo, left, right = detect_stereo_channels(args.audio)

    if (args.diarize_mode == "stereo" or args.diarize_mode == "auto") and is_stereo:
        print(f"-> Detected 2-channel conversation (Stereo Diarization)")
        print(f"   Channel 1: '{args.speaker1}' | Channel 2: '{args.speaker2}'")

        left = denoise_audio(left)
        right = denoise_audio(right)

        left_segs, info_l = model.transcribe(
            left,
            beam_size=args.beam_size,
            language=args.language,
            vad_filter=True,
            vad_parameters=dict(min_silence_duration_ms=450, speech_pad_ms=250, threshold=0.45),
            no_speech_threshold=0.6,
            compression_ratio_threshold=2.4,
            temperature=[0.0, 0.2, 0.4],
        )
        right_segs, info_r = model.transcribe(
            right,
            beam_size=args.beam_size,
            language=args.language,
            vad_filter=True,
            vad_parameters=dict(min_silence_duration_ms=450, speech_pad_ms=250, threshold=0.45),
            no_speech_threshold=0.6,
            compression_ratio_threshold=2.4,
            temperature=[0.0, 0.2, 0.4],
        )

        for s in left_segs:
            dialogue_events.append((s.start, s.end, args.speaker1, s.text.strip()))
        for s in right_segs:
            dialogue_events.append((s.start, s.end, args.speaker2, s.text.strip()))

        dialogue_events.sort(key=lambda x: x[0])
        detected_lang = info_l.language
        lang_prob = info_l.language_probability
        duration = max(len(left), len(right)) / 16000.0

    else:
        # Mono or single-channel audio
        audio_data = left if left is not None else decode_audio(args.audio)
        audio_data = denoise_audio(audio_data)
        duration = len(audio_data) / 16000.0

        segments, info = model.transcribe(
            audio_data,
            beam_size=args.beam_size,
            language=args.language,
            vad_filter=True,
            vad_parameters=dict(min_silence_duration_ms=450, speech_pad_ms=250, threshold=0.45),
            no_speech_threshold=0.6,
            compression_ratio_threshold=2.4,
            temperature=[0.0, 0.2, 0.4],
        )
        segments = list(segments)
        detected_lang = info.language
        lang_prob = info.language_probability

        if args.diarize_mode == "pause":
            dialogue_events = alternate_speakers_by_pause(
                segments, speaker1=args.speaker1, speaker2=args.speaker2
            )
        elif args.diarize_mode in ("auto", "acoustic"):
            dialogue_events = cluster_mono_speakers(
                audio_data, segments, speaker1=args.speaker1, speaker2=args.speaker2
            )
        else:
            dialogue_events = [(s.start, s.end, args.speaker1, s.text.strip()) for s in segments]

    transcribe_time = time.time() - start_transcribe

    print("\n--- Audio Information ---")
    print(f"Detected language: {detected_lang} (probability: {lang_prob:.2%})")
    print(f"Duration: {duration:.2f}s")

    print("\n--- Timestamped Dialogue ---")
    for start, end, speaker, text in dialogue_events:
        print(f"[{start:6.2f}s -> {end:6.2f}s] {speaker}: {text}")

    print("\n--- Conversation Messages ---")
    formatted_convo = format_dialogue(dialogue_events)
    print(formatted_convo)
    print(f"\nCompleted in {transcribe_time:.2f} seconds.")


if __name__ == "__main__":
    main()
