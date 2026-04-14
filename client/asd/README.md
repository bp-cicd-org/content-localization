# ASD (Active Speaker Detection) Client

This client provides a standalone interface for the Active Speaker Detection (ASD) NIM service.

## Overview

The ASD client processes both video and audio input, then detects active speakers in the video,
outputting speaker info data in CSV format. This is useful for:

- Identifying who is speaking in a video
- Extracting speaker bounding boxes for further processing
- Analyzing speaker activity patterns

## Usage

### Basic Usage

```bash
python client/asd/app.py
```

This will use the default settings:
- Input video: `assets/sample_video_streamable.mp4`
- Input audio: `assets/sample_audio.wav`
- Diarization: `assets/diarization.json` (ElevenLabs format)
- ASD server: `localhost:50055`
- Output speaker info file: `assets/asd_speaker_info.csv`
- Video chunk size: 1 MB
- Audio chunk size: 1.0 seconds

### Custom Parameters

```bash
python client/asd/app.py \
    --input-mp4 assets/sample_video_streamable.mp4 \
    --input-audio assets/sample_audio.wav \
    --diarization-file assets/diarization.json \
    --diarization-format elevenlabs \
    --asd-server localhost:50055 \
    --output-speaker-info assets/asd_speaker_info.csv \
    --chunk-size-video-bytes 32768 \
    --chunk-size-audio-secs 0.5
```

## Command Line Arguments

| Argument | Default | Description |
|----------|---------|-------------|
| `--asd-server` | `localhost:50055` | Address and port of the ASD NIM gRPC service |
| `--input-mp4` | `assets/sample_video_streamable.mp4` | Path to input video file (streamable MP4 only) |
| `--input-audio` | `assets/sample_audio.wav` | Path to input audio file (WAV by default; MP3 allowed with `--asd-input-audio-codec MP3`) |
| `--chunk-size-video-bytes` | `1048576` (1 MB) | Chunk size for streaming video |
| `--chunk-size-audio-secs` | `1.0` | Chunk size for streaming audio in seconds |
| `--output-speaker-info` | `assets/asd_speaker_info.csv` | Path to output speaker info file (CSV format) |
| `--diarization-file` | `None` | Optional diarization JSON/CSV file for speaker segments |
| `--diarization-format` | `elevenlabs` | Diarization format (`flat`, `riva`, `elevenlabs`, `elevenlabs-studio`, `camb`) |

## Diarization

Diarization data is a critical input for ASD. It provides speaker segmentation — who is speaking
and when — which ASD uses to associate detected faces with the correct speakers. Without
diarization, ASD cannot reliably map audio activity to the right face in multi-speaker scenes.

### Sample Diarization Data

A sample diarization file is included at `assets/diarization.json` in ElevenLabs STT JSON format.
Use it with the default sample video:

```bash
python client/asd/app.py \
    --diarization-file assets/diarization.json \
    --diarization-format elevenlabs
```

### Supported Diarization Formats

| Format | Flag | Description |
|--------|------|-------------|
| ElevenLabs | `elevenlabs` | Native ElevenLabs STT JSON with word-level timestamps and speaker IDs |
| ElevenLabs Studio | `elevenlabs-studio` | CSV export from ElevenLabs Dubbing Studio (`speaker`, `start_time`, `end_time`, `transcription`) |
| RIVA | `riva` | Native RIVA ASR diarization JSON (`results[].alternatives[].words[]`) |
| Camb AI | `camb` | Camb AI transcription JSON with segment-level speaker labels |
| Flat | `flat` | Simple JSON list with `start_time`, `end_time`, `speaker_id`, and optional `word`/`language_code` |

### Generating Diarization Data

Several helper scripts are provided to generate diarization files from external STT/ASR services:

- `scripts/el_diarize.py` — Generate diarization using the ElevenLabs STT API
- `scripts/riva_parakeet_diarize.py` — Generate diarization using the RIVA Parakeet ASR NIM
- `scripts/camb_diarize.py` — Generate diarization using the Camb AI transcription API

## Output Format

The client outputs a CSV file with the following columns:

- `chunk`: The video chunk number
- `x`: X coordinate of the speaker bounding box
- `y`: Y coordinate of the speaker bounding box  
- `width`: Width of the speaker bounding box
- `height`: Height of the speaker bounding box

If no speaker is detected in a chunk, the coordinates will be (0, 0, 0, 0).

## Requirements

- Python 3.12+
- gRPC
- The ASD NIM service must be running and accessible
- Input audio must be WAV by default (or MP3 with `--asd-input-audio-codec MP3`)
- Input video must be in streamable MP4 format

## Error Handling

The client includes proper error handling for:
- gRPC connection issues
- File I/O errors
- Service unavailability

If you encounter the "Exception iterating requests!" error, this typically indicates:
1. The ASD NIM service is not running
2. Network connectivity issues
3. The service is overloaded or encountering internal errors

## Integration

This client is designed to work independently but can also be integrated into larger workflows. The
speaker info data it produces can be used by:

- The LipSync client for speaker-aware lip synchronization
- Video analysis pipelines
- Speaker tracking applications

## Troubleshooting

1. **Service not found**: Ensure the ASD NIM service is running on the specified port
2. **Video format issues**: Convert your video to streamable MP4 format using the provided script
3. **Permission errors**: Ensure you have write permissions for the output directory
4. **Memory issues**: Try reducing the chunk size if processing large videos 