# Speaker Diarization Toolkit

## What is This Project?

This is a speaker diarization toolkit built for the OOMOL platform. Speaker diarization is the process of automatically identifying and labeling "who spoke when" in an audio recording.

**In simple terms**: Upload an audio file with multiple speakers, and this tool will tell you which speaker spoke at what time, even if you don't know how many speakers are in the recording.

## Use Cases

This toolkit is perfect for:

- **Meeting Transcription**: Identify different speakers in recorded meetings or conference calls
- **Interview Analysis**: Separate interviewer and interviewee voices with precise timestamps
- **Podcast Processing**: Label different speakers in multi-person podcast episodes
- **Legal & Medical Documentation**: Create speaker-attributed transcripts of consultations or depositions
- **Research & Analysis**: Study conversation patterns and speaking time distribution among participants

## Available Blocks

### Speaker Diarization Block

The main block that performs speaker identification and labeling on audio files.

**What it does:**
- Analyzes audio files to identify different speakers
- Provides precise timestamps for when each speaker starts and stops talking
- Can automatically detect the number of speakers or work with a pre-specified number
- Supports overlap detection for when multiple people speak simultaneously
- Outputs results in easy-to-read formats (JSON or RTTM)

**Inputs:**
- **Audio File**: Your audio recording (supports WAV, MP3, FLAC, M4A formats)
- **Number of Speakers** (optional): If you know how many speakers are in the recording, specify it here for better accuracy. Leave empty for automatic detection.
- **Include Overlap** (optional): Enable this to detect when multiple speakers talk at the same time
- **Output Format**: Choose between JSON (easy to read) or RTTM (standard research format)

**Outputs:**
- **Speaker Segments**: A detailed list showing when each speaker talked, including:
  - Start time (in seconds)
  - End time (in seconds)
  - Speaker ID (e.g., "SPEAKER_00", "SPEAKER_01")
  - Duration of each speaking segment
- **Output File**: A saved file containing all the diarization results in your chosen format

## How to Use

### Getting Started

1. **Open OOMOL Platform**: Launch your OOMOL workspace
2. **Create a Workflow**: Start a new workflow or use an existing one
3. **Add the Speaker Diarization Block**: Drag the "Speaker Diarization" block into your workflow
4. **Upload Your Audio**: Select an audio file you want to analyze
5. **Configure Settings** (optional):
   - Specify the number of speakers if known
   - Enable overlap detection if needed
   - Choose your preferred output format
6. **Run the Workflow**: Click run and wait for processing to complete
7. **View Results**: Check the output table showing all speaker segments with timestamps

### Example Workflow

**Scenario**: You have a recorded team meeting with 4 people and want to know who spoke when.

1. Add the Speaker Diarization block
2. Upload your meeting audio file (e.g., "team-meeting.mp3")
3. Set "Number of Speakers" to 4
4. Keep "Include Overlap" as false (unless people frequently interrupt each other)
5. Choose "json" as output format for easy reading
6. Run the workflow
7. Review the results table showing each speaker's segments

## Output Examples

### JSON Format Output
```json
[
  {
    "start_time": 0.5,
    "end_time": 5.2,
    "speaker_id": "SPEAKER_00"
  },
  {
    "start_time": 5.8,
    "end_time": 12.3,
    "speaker_id": "SPEAKER_01"
  }
]
```

### Visual Table Output
The block automatically displays results in an easy-to-read table:

| Start Time (s) | End Time (s) | Speaker ID | Duration (s) |
|---------------|--------------|------------|--------------|
| 0.50          | 5.20         | SPEAKER_00 | 4.70         |
| 5.80          | 12.30        | SPEAKER_01 | 6.50         |

## Technical Details (For Reference)

- **AI Model**: Uses advanced 3D-Speaker models from ModelScope
- **Supported Audio Formats**: WAV, MP3, FLAC, M4A
- **Processing Method**: Voice Activity Detection + Speaker Embeddings + Clustering
- **Output Storage**: Results are saved to `/oomol-driver/oomol-storage/speaker-diarization/`

## Frequently Asked Questions

**Q: How accurate is the speaker identification?**
A: The accuracy depends on audio quality, number of speakers, and how distinct their voices are. Clear recordings with well-separated speakers yield the best results.

**Q: Can it identify speakers by name?**
A: No, it labels speakers as "SPEAKER_00", "SPEAKER_01", etc. You'll need to match these labels to actual names based on the timestamps and context.

**Q: What if I don't know how many speakers are in the audio?**
A: Leave the "Number of Speakers" field empty, and the system will automatically detect it.

**Q: How long does processing take?**
A: Processing time varies based on audio length and quality. Typically, it takes a few minutes for a 30-minute audio file.

**Q: What audio quality works best?**
A: Clear recordings with minimal background noise work best. Each speaker should be audible and distinct.

## Support & Feedback

For issues, questions, or feature requests, please contact your OOMOL platform administrator or refer to the OOMOL documentation.

## Version

Current Version: 0.0.1

---

**Note**: This toolkit is designed for the OOMOL platform and requires OOMOL to run. It cannot be used as a standalone application.