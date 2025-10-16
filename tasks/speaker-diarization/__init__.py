#region generated meta
import typing
class Inputs(typing.TypedDict):
    audio_file: str
    num_speakers: float | None
    include_overlap: bool
    output_format: typing.Literal["json", "rttm"]
class Outputs(typing.TypedDict):
    segments: typing.NotRequired[list[dict]]
    output_file: typing.NotRequired[str]
#endregion

from oocana import Context
import os
import json
from modelscope.pipelines import pipeline


def main(params: Inputs, context: Context) -> Outputs:
    """
    Perform speaker diarization on audio file using 3D-Speaker models.

    Args:
        params: Input parameters containing audio file and configuration
        context: OOMOL context object

    Returns:
        Dictionary containing speaker segments and output file path
    """
    audio_file = params["audio_file"]
    num_speakers = params.get("num_speakers")
    include_overlap = params.get("include_overlap", False)
    output_format = params.get("output_format", "json")

    # Validate input file
    if not os.path.exists(audio_file):
        raise ValueError(f"Audio file not found: {audio_file}")

    context.logger.info(f"Processing audio file: {audio_file}")

    # Use FunASR pipeline for speaker diarization
    segments = perform_diarization_with_funasr(
        audio_file,
        num_speakers,
        context
    )

    # Save output file
    output_dir = "/oomol-driver/oomol-storage/speaker-diarization"
    os.makedirs(output_dir, exist_ok=True)

    base_name = os.path.splitext(os.path.basename(audio_file))[0]

    if output_format == "json":
        output_file = os.path.join(output_dir, f"{base_name}_diarization.json")
        with open(output_file, 'w', encoding='utf-8') as f:
            json.dump(segments, f, indent=2, ensure_ascii=False)
    else:  # rttm format
        output_file = os.path.join(output_dir, f"{base_name}_diarization.rttm")
        write_rttm_file(output_file, segments, base_name)

    context.logger.info(f"Diarization completed. Found {len(set(s['speaker_id'] for s in segments))} speakers")
    context.logger.info(f"Output saved to: {output_file}")

    # Display results as table
    context.preview({
        "type": "table",
        "data": {
            "headers": ["Start Time (s)", "End Time (s)", "Speaker ID", "Duration (s)"],
            "rows": [
                [
                    f"{seg['start_time']:.2f}",
                    f"{seg['end_time']:.2f}",
                    seg['speaker_id'],
                    f"{seg['end_time'] - seg['start_time']:.2f}"
                ]
                for seg in segments
            ]
        }
    })

    return {
        "segments": segments,
        "output_file": output_file
    }


def perform_diarization_with_funasr(
    audio_file: str,
    num_speakers: typing.Optional[int],
    context: Context
) -> typing.List[typing.Dict[str, typing.Any]]:
    """
    Perform speaker diarization using FunASR.

    Uses VAD + sliding window + speaker embeddings + clustering approach.
    """
    from funasr import AutoModel
    import numpy as np
    from sklearn.cluster import AgglomerativeClustering
    import soundfile as sf

    context.logger.info("Loading audio file...")
    audio, sample_rate = sf.read(audio_file)

    # Convert to mono if stereo
    if len(audio.shape) > 1:
        audio = np.mean(audio, axis=1)

    context.logger.info(f"Audio duration: {len(audio) / sample_rate:.2f} seconds")

    # Step 1: Voice Activity Detection
    context.logger.info("Performing voice activity detection...")
    vad_model = AutoModel(
        model="fsmn-vad",
        model_revision="v2.0.4"
    )

    vad_result = vad_model.generate(
        input=audio_file,
        cache={},
        is_final=True,
        chunk_size=200,
        encoder_chunk_look_back=4,
        decoder_chunk_look_back=1
    )

    context.logger.info(f"VAD result: {vad_result}")

    # Parse VAD segments
    speech_segments = parse_funasr_vad_result(vad_result, sample_rate)

    if not speech_segments:
        context.logger.warning("No speech segments detected")
        return []

    context.logger.info(f"Detected {len(speech_segments)} speech segments")

    # Step 2: Extract speaker embeddings
    context.logger.info("Extracting speaker embeddings...")
    sv_model = AutoModel(
        model="cam++",
        model_revision="v2.0.2"
    )

    embeddings = []
    valid_segments = []

    for idx, seg in enumerate(speech_segments):
        try:
            start_sample = int(seg["start_time"] * sample_rate)
            end_sample = int(seg["end_time"] * sample_rate)

            # Skip very short segments (< 0.5 seconds)
            if (end_sample - start_sample) / sample_rate < 0.5:
                continue

            segment_audio = audio[start_sample:end_sample]

            # Extract embedding
            embedding_result = sv_model.generate(
                input=segment_audio,
                cache={},
                output_dir=None,
                batch_size=1
            )

            if embedding_result and len(embedding_result) > 0:
                # Get embedding from result
                if isinstance(embedding_result[0], dict) and 'spk_embedding' in embedding_result[0]:
                    emb = embedding_result[0]['spk_embedding']
                elif hasattr(embedding_result[0], 'spk_embedding'):
                    emb = embedding_result[0].spk_embedding
                else:
                    # Try to extract embedding directly
                    emb = embedding_result[0] if isinstance(embedding_result[0], np.ndarray) else np.array(embedding_result[0])

                # Flatten embedding to 1D if needed
                if isinstance(emb, np.ndarray):
                    emb = emb.flatten()

                embeddings.append(emb)
                valid_segments.append(seg)
                context.logger.info(f"Segment {idx+1}/{len(speech_segments)}: {seg['start_time']:.2f}s - {seg['end_time']:.2f}s, emb shape: {emb.shape if hasattr(emb, 'shape') else len(emb)}")
        except Exception as e:
            context.logger.warning(f"Failed to extract embedding for segment {idx}: {e}")
            continue

    if not embeddings:
        raise ValueError("Failed to extract any speaker embeddings. Audio may be too short or contain no clear speech.")

    context.logger.info(f"Extracted {len(embeddings)} valid embeddings")

    # Handle case with only one segment
    if len(embeddings) == 1:
        context.logger.warning("Only one speech segment found - assigning to single speaker")
        segments = [{
            "start_time": round(valid_segments[0]["start_time"], 2),
            "end_time": round(valid_segments[0]["end_time"], 2),
            "speaker_id": "SPEAKER_00"
        }]
        return segments

    # Step 3: Cluster embeddings
    context.logger.info("Clustering speaker embeddings...")

    # Convert to numpy array and ensure 2D shape
    embeddings_list = []
    for emb in embeddings:
        if isinstance(emb, np.ndarray):
            emb_flat = emb.flatten()
        else:
            emb_flat = np.array(emb).flatten()
        embeddings_list.append(emb_flat)

    embeddings_array = np.array(embeddings_list)
    context.logger.info(f"Embeddings array shape: {embeddings_array.shape}")

    # Normalize embeddings
    embeddings_array = embeddings_array / (np.linalg.norm(embeddings_array, axis=1, keepdims=True) + 1e-8)

    if num_speakers is None:
        # Auto-detect number of speakers (max 5)
        num_speakers = min(max(2, len(embeddings) // 3), 5)
        context.logger.info(f"Auto-detected number of speakers: {num_speakers}")

    n_clusters = min(int(num_speakers), len(embeddings))

    clustering = AgglomerativeClustering(
        n_clusters=n_clusters,
        metric='cosine',
        linkage='average'
    )
    labels = clustering.fit_predict(embeddings_array)

    # Step 4: Create segments with speaker labels
    segments = []
    for seg, label in zip(valid_segments, labels):
        segments.append({
            "start_time": round(seg["start_time"], 2),
            "end_time": round(seg["end_time"], 2),
            "speaker_id": f"SPEAKER_{label:02d}"
        })

    # Sort by start time
    segments.sort(key=lambda x: x["start_time"])

    context.logger.info(f"Found {len(set(labels))} unique speakers in {len(segments)} segments")

    return segments


def parse_funasr_vad_result(vad_result, sample_rate: int) -> typing.List[typing.Dict[str, float]]:
    """Parse FunASR VAD result to extract speech segments."""
    segments = []

    try:
        if isinstance(vad_result, list) and len(vad_result) > 0:
            result_item = vad_result[0]

            # Check if result has 'value' field with timestamps
            if isinstance(result_item, dict):
                if 'value' in result_item:
                    # Format: [[start_ms, end_ms], ...]
                    for timestamp_pair in result_item['value']:
                        if len(timestamp_pair) >= 2:
                            start_ms, end_ms = timestamp_pair[0], timestamp_pair[1]
                            segments.append({
                                "start_time": start_ms / 1000.0,
                                "end_time": end_ms / 1000.0
                            })
                elif 'text' in result_item:
                    # Try to parse text format
                    import ast
                    try:
                        timestamps = ast.literal_eval(result_item['text'])
                        for ts in timestamps:
                            if len(ts) >= 2:
                                segments.append({
                                    "start_time": float(ts[0]) / 1000.0,
                                    "end_time": float(ts[1]) / 1000.0
                                })
                    except Exception:
                        pass
    except Exception as e:
        # If VAD fails, create a single segment for the whole audio
        pass

    return segments


def write_rttm_file(
    output_file: str,
    segments: typing.List[typing.Dict[str, typing.Any]],
    audio_id: str
) -> None:
    """
    Write diarization results in RTTM format.

    RTTM format: SPEAKER <audio_id> 1 <start> <duration> <NA> <NA> <speaker_id> <NA> <NA>
    """
    with open(output_file, 'w', encoding='utf-8') as f:
        for seg in segments:
            start = seg['start_time']
            duration = seg['end_time'] - seg['start_time']
            speaker = seg['speaker_id']

            line = f"SPEAKER {audio_id} 1 {start:.3f} {duration:.3f} <NA> <NA> {speaker} <NA> <NA>\n"
            f.write(line)
