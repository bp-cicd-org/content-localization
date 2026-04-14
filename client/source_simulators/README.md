# Source Simulators

This directory contains shared audio and video processing utilities used by all client applications for file I/O operations.

## Overview

The source simulators provide standardized interfaces for reading and writing audio and video files across all client applications. They handle:
- **Audio Processing**: WAV and MP3 file reading/writing
- **Video Processing**: MP4 file reading/writing with streaming support
- **Chunk-based Processing**: Efficient handling of large files
- **Format Validation**: Input/output format verification

## Components

### Core Modules

#### `base.py`
**Base classes for file simulators**
- `BaseFileSimulator` - Abstract base class for file simulators
- Common functionality for file operations
- Error handling and validation utilities

#### `audio.py`
**Audio source and sink simulators**
- `AudioSourceSimulator` - Reads audio files (WAV/MP3) in chunks
- `AudioSinkSimulator` - Writes audio data to WAV files
- Audio format conversion and validation
- Chunk-based streaming support

#### `file.py`
**Generic file source simulator**
- `FileSourceSimulator` - Streams any file as raw bytes in fixed-size chunks
- Treats files as opaque byte streams without format parsing
- Useful for non-WAV audio formats (e.g., MP3) or any binary file
- Compatible with `AudioSourceSimulator` interface via `read()` method

#### `video.py`
**Video source and sink simulators**
- `VideoSourceSimulator` - Reads video files (MP4) in chunks
- `VideoSinkSimulator` - Writes video data to MP4 files
- Streaming video support with proper metadata handling
- Video format validation and conversion

## Usage

### Audio Processing

```python
from source_simulators.audio import AudioSourceSimulator, AudioSinkSimulator

# Read audio file
audio_source = AudioSourceSimulator("input.wav")
for chunk in audio_source:
    # Process audio chunk
    process_audio(chunk)

# Write audio file
audio_sink = AudioSinkSimulator("output.wav")
audio_sink.write_audio_data(audio_chunk)
```

### File Processing (Non-WAV Audio)

```python
from source_simulators.file import FileSourceSimulator

# Stream any file as raw bytes (useful for MP3 or other formats)
source = FileSourceSimulator(file_path="translated.mp3")
for chunk in source:
    # Process raw byte chunk
    process_chunk(chunk)
```

The Controller client uses `FileSourceSimulator` internally for streaming
non-WAV translated audio files when `--translated-audio` is provided.

### Video Processing

```python
from source_simulators.video import VideoSourceSimulator, VideoSinkSimulator

# Read video file
video_source = VideoSourceSimulator("input.mp4")
for chunk in video_source:
    # Process video chunk
    process_video(chunk)

# Write video file
video_sink = VideoSinkSimulator("output.mp4")
video_sink.write_video_data(video_chunk)
```

## Features

### Audio Support
- **Input Formats**: WAV, MP3
- **Output Format**: WAV
- **Chunk-based Processing**: Configurable chunk sizes
- **Format Validation**: Automatic format detection and validation
- **Metadata Handling**: Preserves audio metadata

### Video Support
- **Input Format**: MP4 (streamable preferred)
- **Output Format**: MP4
- **Streaming Support**: Optimized for streaming video files
- **Metadata Preservation**: Maintains video metadata and properties
- **Chunk-based Processing**: Efficient handling of large video files

### Common Features
- **Error Handling**: Robust error handling and recovery
- **Memory Efficiency**: Chunk-based processing for large files
- **Format Validation**: Input/output format verification
- **Logging**: Comprehensive logging for debugging
- **Thread Safety**: Safe for concurrent access

## Integration

The source simulators are used by all client applications:

- **Controller Client**: Audio and video processing for complete pipeline; `FileSourceSimulator` for non-WAV translated audio
- **Direct Client**: Audio and video processing for direct service communication
- **S2S Client**: Audio processing for speech-to-speech translation
- **LipSync Client**: Audio and video processing for lip synchronization
- **ASD Client**: Video processing for speaker detection

## File Format Requirements

### Audio Files
- **WAV**: Uncompressed PCM audio (recommended)
- **MP3**: Compressed audio (supported for input)
- **Sample Rates**: 8kHz, 16kHz, 44.1kHz, 48kHz
- **Channels**: Mono or stereo

### Video Files
- **MP4**: H.264 encoded video (streamable preferred)
- **Resolution**: Any standard resolution
- **Frame Rate**: Any standard frame rate
- **Streaming**: `moov` atom at start of file for optimal streaming

## Performance Considerations

### Chunk Size Optimization
- **Audio**: 0.5-2 seconds for optimal processing
- **Video**: 32KB-128KB for optimal streaming
- **Memory Usage**: Larger chunks use more memory but reduce overhead

### File Format Optimization
- **Streamable MP4**: Use `ffmpeg -movflags +faststart` for optimal streaming
- **WAV Audio**: Uncompressed for best quality and processing speed
- **File Size**: Consider chunk sizes based on available memory

## Error Handling

The simulators include comprehensive error handling for:
- **File Not Found**: Graceful handling of missing files
- **Format Errors**: Invalid file format detection and reporting
- **I/O Errors**: Network and disk I/O error recovery
- **Memory Errors**: Out-of-memory condition handling
- **Validation Errors**: Input/output data validation

## Troubleshooting

### Common Issues

1. **File not found**: Check file paths and permissions
2. **Format errors**: Verify file format and convert if necessary
3. **Memory issues**: Reduce chunk sizes for large files
4. **Streaming issues**: Convert video to streamable format
5. **Permission errors**: Check read/write permissions

### Performance Issues

1. **Slow processing**: Optimize chunk sizes for your use case
2. **High memory usage**: Reduce chunk sizes or use streaming
3. **File format issues**: Convert to recommended formats

## Example Workflows

### Audio Processing Pipeline
```python
from source_simulators.audio import AudioSourceSimulator, AudioSinkSimulator

# Process audio file
source = AudioSourceSimulator("input.wav")
sink = AudioSinkSimulator("output.wav")

for chunk in source:
    # Process audio chunk (e.g., translate with S2S)
    processed_chunk = process_audio_chunk(chunk)
    sink.write_audio_data(processed_chunk)
```

### Video Processing Pipeline
```python
from source_simulators.video import VideoSourceSimulator, VideoSinkSimulator

# Process video file
source = VideoSourceSimulator("input.mp4")
sink = VideoSinkSimulator("output.mp4")

for chunk in source:
    # Process video chunk (e.g., lip sync with LipSync)
    processed_chunk = process_video_chunk(chunk)
    sink.write_video_data(processed_chunk)
```

The source simulators provide a robust foundation for all client applications, ensuring consistent and efficient file I/O operations across the entire content localization pipeline.
