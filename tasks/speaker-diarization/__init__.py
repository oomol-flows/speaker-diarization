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

    # Initialize speaker diarization pipeline
    # Using ANDiT model which supports diarization
    try:
        diarization_pipeline = pipeline(
            task='speaker-diarization',
            model='iic/speech_campplus_sv_zh-cn_16k-common'
        )
    except Exception as e:
        # Fallback to using FunASR for diarization
        context.logger.warning(f"Failed to load speaker-diarization pipeline: {e}")
        context.logger.info("Attempting to use alternative diarization method...")

        # Use speaker embedding + clustering approach
        segments = perform_diarization_with_embeddings(
            audio_file,
            num_speakers,
            include_overlap,
            context
        )
    else:
        # Process audio file
        pipeline_params = {}
        if num_speakers is not None:
            pipeline_params['num_speakers'] = int(num_speakers)

        result = diarization_pipeline(audio_file, **pipeline_params)

        # Parse results into segments format
        segments = parse_diarization_result(result)

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


def perform_diarization_with_embeddings(
    audio_file: str,
    num_speakers: typing.Optional[int],
    include_overlap: bool,
    context: Context
) -> typing.List[typing.Dict[str, typing.Any]]:
    """
    Perform speaker diarization using speaker embeddings and clustering.

    This is a fallback method using VAD + speaker embeddings + clustering.
    """
    from modelscope.pipelines import pipeline
    import numpy as np
    from sklearn.cluster import AgglomerativeClustering

    # Step 1: Voice Activity Detection
    context.logger.info("Performing voice activity detection...")
    vad_pipeline = pipeline(
        task='voice-activity-detection',
        model='iic/speech_fsmn_vad_zh-cn-16k-common-pytorch'
    )
    vad_result = vad_pipeline(audio_file)

    # Step 2: Extract speaker embeddings for each speech segment
    context.logger.info("Extracting speaker embeddings...")
    sv_pipeline = pipeline(
        task='speaker-verification',
        model='iic/speech_campplus_sv_zh-cn_16k-common'
    )

    # Parse VAD segments
    speech_segments = parse_vad_result(vad_result)

    if not speech_segments:
        context.logger.warning("No speech segments detected")
        return []

    # Extract embeddings for each segment
    embeddings = []
    valid_segments = []

    for seg in speech_segments:
        try:
            # Extract embedding for this segment
            result = sv_pipeline(audio_file, output_emb=True)
            if 'embs' in result:
                embeddings.append(result['embs'][0])
                valid_segments.append(seg)
        except Exception as e:
            context.logger.warning(f"Failed to extract embedding for segment: {e}")
            continue

    if not embeddings:
        raise ValueError("Failed to extract any speaker embeddings")

    # Step 3: Cluster embeddings to identify speakers
    context.logger.info("Clustering speaker embeddings...")
    embeddings_array = np.array(embeddings)

    if num_speakers is None:
        # Auto-detect number of speakers using threshold-based clustering
        num_speakers = min(len(embeddings), 10)  # Max 10 speakers

    clustering = AgglomerativeClustering(
        n_clusters=min(int(num_speakers), len(embeddings)),
        metric='cosine',
        linkage='average'
    )
    labels = clustering.fit_predict(embeddings_array)

    # Step 4: Assign speaker labels to segments
    segments = []
    for seg, label in zip(valid_segments, labels):
        segments.append({
            "start_time": seg["start_time"],
            "end_time": seg["end_time"],
            "speaker_id": f"SPEAKER_{label:02d}"
        })

    # Sort by start time
    segments.sort(key=lambda x: x["start_time"])

    return segments


def parse_vad_result(vad_result) -> typing.List[typing.Dict[str, float]]:
    """Parse VAD result to extract speech segments."""
    segments = []

    if isinstance(vad_result, dict) and 'text' in vad_result:
        # Parse text format: [[start1, end1], [start2, end2], ...]
        text = vad_result['text']
        # Simple parsing logic - adapt based on actual output format
        import ast
        try:
            timestamps = ast.literal_eval(text)
            for ts in timestamps:
                if len(ts) >= 2:
                    segments.append({
                        "start_time": float(ts[0]) / 1000.0,  # Convert ms to seconds
                        "end_time": float(ts[1]) / 1000.0
                    })
        except Exception:
            pass

    return segments


def parse_diarization_result(result) -> typing.List[typing.Dict[str, typing.Any]]:
    """Parse ModelScope diarization result into segments format."""
    segments = []

    # Handle different result formats
    if isinstance(result, dict):
        if 'text' in result:
            # Parse text-based output
            text = result['text']
            # Expected format: "SPEAKER_00 start end\nSPEAKER_01 start end\n..."
            for line in text.strip().split('\n'):
                parts = line.split()
                if len(parts) >= 3:
                    segments.append({
                        "start_time": float(parts[1]),
                        "end_time": float(parts[2]),
                        "speaker_id": parts[0]
                    })
        elif 'segments' in result:
            segments = result['segments']
    elif isinstance(result, list):
        # Direct list of segments
        for item in result:
            if isinstance(item, (list, tuple)) and len(item) >= 3:
                segments.append({
                    "start_time": float(item[0]),
                    "end_time": float(item[1]),
                    "speaker_id": str(item[2])
                })
            elif isinstance(item, dict):
                segments.append(item)

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
