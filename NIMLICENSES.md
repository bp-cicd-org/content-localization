# NVIDIA NIM Licenses

This project uses the following NVIDIA NIM (NVIDIA Inference Microservices) containers. Each NIM is subject to its own license terms as described below.

## LipSync NIM


|                 |                                                                                                                                                                                                                                                                                           |
| --------------- | ----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| **Name**        | NVIDIA LipSync NIM                                                                                                                                                                                                                                                                        |
| **Image**       | `nvcr.io/nim/nvidia/lipsync:1.2.0`                                                                                                                                                                                                                                                       |
| **NGC**         | [https://catalog.ngc.nvidia.com/orgs/nim/teams/nvidia/containers/lipsync](https://catalog.ngc.nvidia.com/orgs/nim/teams/nvidia/containers/lipsync)                                                                                                                                        |
| **License**     | [NVIDIA Software License Agreement](https://www.nvidia.com/en-us/agreements/enterprise-software/nvidia-software-license-agreement/), [Product-Specific Terms for NVIDIA AI Products](https://www.nvidia.com/en-us/agreements/enterprise-software/product-specific-terms-for-ai-products/) |
| **Description** | Synchronizes lip movements in video with translated audio.                                                                                                                                                                                                                                |


## Active Speaker Detection (ASD) NIM


|                 |                                                                                                                                                                                                                                                                                           |
| --------------- | ----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| **Name**        | NVIDIA Active Speaker Detection NIM                                                                                                                                                                                                                                                       |
| **Image**       | `nvcr.io/nim/nvidia/active-speaker-detection:1.0.0`                                                                                                                                                                                                                                      |
| **NGC**         | [https://catalog.ngc.nvidia.com/orgs/nim/teams/nvidia/containers/active-speaker-detection](https://catalog.ngc.nvidia.com/orgs/nim/teams/nvidia/containers/active-speaker-detection)                                                                                                      |
| **License**     | [NVIDIA Software License Agreement](https://www.nvidia.com/en-us/agreements/enterprise-software/nvidia-software-license-agreement/), [Product-Specific Terms for NVIDIA AI Products](https://www.nvidia.com/en-us/agreements/enterprise-software/product-specific-terms-for-ai-products/) |
| **Description** | Detects which speakers are actively speaking in a video for speaker-aware dubbing.                                                                                                                                                                                                        |


## RIVA ASR (Automatic Speech Recognition) NIM


|                 |                                                                                                                                                                                                                                                                                           |
| --------------- | ----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| **Name**        | NVIDIA RIVA ASR NIM (Canary 1B)                                                                                                                                                                                                                                                           |
| **Image**       | `nvcr.io/nim/nvidia/riva-asr:latest`                                                                                                                                                                                                                                                      |
| **NGC**         | [https://catalog.ngc.nvidia.com/orgs/nim/teams/nvidia/containers/riva-asr](https://catalog.ngc.nvidia.com/orgs/nim/teams/nvidia/containers/riva-asr)                                                                                                                                      |
| **License**     | [NVIDIA Software License Agreement](https://www.nvidia.com/en-us/agreements/enterprise-software/nvidia-software-license-agreement/), [Product-Specific Terms for NVIDIA AI Products](https://www.nvidia.com/en-us/agreements/enterprise-software/product-specific-terms-for-ai-products/) |
| **Description** | Performs automatic speech recognition (ASR) for transcription and diarization using the Canary 1B model. Used in RIVA-based S2S profiles.                                                                                                                                                 |


## RIVA TTS (Text-to-Speech) NIM


|                 |                                                                                                                                                                                                                                                                                           |
| --------------- | ----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| **Name**        | NVIDIA RIVA Magpie TTS Zero-Shot NIM                                                                                                                                                                                                                                                      |
| **Image**       | `nvcr.io/nim/nvidia/magpie-tts-zeroshot:latest`                                                                                                                                                                                                                                           |
| **NGC**         | [https://catalog.ngc.nvidia.com/orgs/nim/teams/nvidia/containers/magpie-tts-zeroshot](https://catalog.ngc.nvidia.com/orgs/nim/teams/nvidia/containers/magpie-tts-zeroshot)                                                                                                                |
| **License**     | [NVIDIA Software License Agreement](https://www.nvidia.com/en-us/agreements/enterprise-software/nvidia-software-license-agreement/), [Product-Specific Terms for NVIDIA AI Products](https://www.nvidia.com/en-us/agreements/enterprise-software/product-specific-terms-for-ai-products/) |
| **Description** | Performs zero-shot text-to-speech synthesis with voice cloning for translated audio generation. Used in RIVA-based S2S profiles.                                                                                                                                                          |


---

**Note**: These NIMs are not distributed as part of this open-source project. They are pulled from NVIDIA NGC at deployment time and require an NGC API key (`NGC_API_KEY`). Users must accept the applicable license terms when accessing these containers from NGC.
