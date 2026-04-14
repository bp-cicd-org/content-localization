#!/bin/bash
# Run all clients against the sample video with EL diarization (English -> German).
# Usage: bash scripts/run_all_clients.sh [OPTIONS] [OUTPUT_DIR] [TARGET_IP]
#   --no-background-audio  — skip background audio in LipSync
#   OUTPUT_DIR             — output directory (default: outputs)
#   TARGET_IP              — server IP/hostname (default: localhost)

set -euo pipefail

# Parse flags
NO_BACKGROUND_AUDIO=false
while [[ "${1:-}" == --* ]]; do
    case "$1" in
        --no-background-audio) NO_BACKGROUND_AUDIO=true; shift ;;
        *) echo "Unknown option: $1"; exit 1 ;;
    esac
done

cd /localhome/local-ragavv/Code-Packages/h4m-ebu
source .venv/bin/activate
set -a && source .env && set +a
export PYTHONPATH="${PYTHONPATH:-}:${PWD}:${PWD}/src:${PWD}/client:${PWD}/protos/generated"

OUT="${1:-outputs}"
TARGET="${2:-localhost}"

AUDIO="assets/sample_audio.wav"
VIDEO_NON_STREAMABLE="assets/sample_video.mp4"
VIDEO="assets/sample_video_streamable.mp4"
DIARIZATION="assets/diarization.json"

mkdir -p "$OUT"

run_step() {
    local label="$1"
    shift
    echo ""
    echo "================================================================"
    echo "  $label"
    echo "================================================================"
    local start=$SECONDS
    "$@"
    local elapsed=$(( SECONDS - start ))
    echo ">>> $label completed in ${elapsed}s"
    echo ""
}

# --- Generate Streamable Video ---
run_step "Generate Streamable Video" \
    sh scripts/convert_to_streamable_mp4.sh "$VIDEO_NON_STREAMABLE" "$VIDEO"

# --- Generate Audio ---
run_step "Generate Audio" \
    sh scripts/extract_audio_from_videos.sh "$VIDEO" "$AUDIO"

# --- Run Diarization -- 
run_step "Diarization" \
    python scripts/el_diarize.py \
        --input-file "$AUDIO" \
        --output-file "$DIARIZATION"
    
# --- S2S Client ---
run_step "S2S Client" \
    python client/s2s/app.py \
        --s2s-server "$TARGET:50050" \
        --input-audio "$AUDIO" \
        --output-audio "$OUT/sample_audio_output_s2s_client.mp3" \
        --latency-plot "$OUT/s2s_latency.png" \
        --source-language en --target-language de

# --- ASD Client ---
run_step "ASD Client" \
    python client/asd/app.py \
        --asd-server "$TARGET:50055" \
        --input-mp4 "$VIDEO" \
        --input-audio "$AUDIO" \
        --output-speaker-info "assets/asd_speaker_info_from_asd.csv" \
        --diarization-file "$DIARIZATION" \
        --diarization-format elevenlabs

# --- LipSync Client (uses S2S output + ASD speaker info) ---
# Background audio skipped for standalone LipSync: the source WAV (16 kHz) and
# the S2S output (44.1 kHz MP3) have different sample rates. The controller and
# direct clients handle resampling internally.
run_step "LipSync Client" \
    python client/lipsync/app.py \
        --video-input "$VIDEO" \
        --audio-input "$OUT/sample_audio_output_s2s_client.mp3" \
        --speaker-info-input "assets/asd_speaker_info_from_asd.csv" \
        --output "$OUT/lipsync_output.mp4" \
        --lipsync-input-audio-codec MP3

# --- Controller Client ---
run_step "Controller Client" \
    python client/controller/app.py \
        --controller-server "$TARGET:50056" \
        --input-audio "$AUDIO" \
        --input-mp4 "$VIDEO" \
        --output-mp4 "$OUT/controller_output.mp4" \
        --diarization-file "$DIARIZATION" \
        --diarization-format elevenlabs \
        --source-language en --target-language de

# --- Direct Client ---
run_step "Direct Client" \
    python client/direct/app.py \
        --s2s-server "$TARGET:50050" \
        --asd-server "$TARGET:50055" \
        --lipsync-server "$TARGET:50054" \
        --input-audio "$AUDIO" \
        --input-mp4 "$VIDEO" \
        --output-mp4 "$OUT/direct_output.mp4" \
        --output-audio "$OUT/direct_audio_output.mp3" \
        --diarization-file "$DIARIZATION" \
        --diarization-format elevenlabs \
        --source-language en --target-language de

# --- Summary ---
echo ""
echo "================================================================"
echo "  ALL CLIENTS COMPLETE - Output Summary"
echo "================================================================"
echo ""
echo "Outputs:"
ls -lh "$OUT/"
