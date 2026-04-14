.. _configuration:

=============
Configuration
=============

All services are configured via environment variables set in
``configs/*.env`` files and passed through ``docker-compose.yml``.
Secrets (API keys) go in a ``.env`` file at the repo root, which is
git-ignored.

**Timeouts and intervals:** All environment variables that specify a
timeout or interval (e.g. ``HEALTH_CHECK_TIMEOUT``, ``BUFFER_POLL_TIMEOUT``,
``CONTROLLER_CLEANUP_TIMEOUT``, ``S2S_CLEANUP_TIMEOUT``,
``S2S_EL_DUBBING_POLL_INTERVAL``, ``S2S_EL_KEEPALIVE_INTERVAL``) use
**seconds** unless otherwise documented.

Shared Configuration
--------------------

These variables are used by **both** the Controller and S2S services.

.. list-table::
   :header-rows: 1
   :widths: 35 10 10 45

   * - Variable
     - Type
     - Default
     - Description
   * - ``HEALTH_CHECK_TIMEOUT``
     - float
     - ``5.0``
     - Seconds for HTTP and gRPC health-check requests before timeout.
   * - ``BUFFER_POLL_TIMEOUT``
     - float
     - ``0.1``
     - Seconds between poll attempts in ``RequestIteratorFromBuffer``.
       Controls how often buffer-backed iterators check for new items
       or exhaustion.

Controller Service
------------------

Basic
~~~~~

.. list-table::
   :header-rows: 1
   :widths: 40 10 10 40

   * - Variable
     - Type
     - Default
     - Description
   * - ``CONTROLLER_GRPC_API_PORT``
     - int
     - ``50056``
     - gRPC listen port.
   * - ``CONTROLLER_MAX_CONCURRENCY``
     - int
     - ``1``
     - Maximum concurrent requests.
   * - ``CONTROLLER_LOG_LEVEL``
     - str
     - ``INFO``
     - Logging level.
   * - ``CONTROLLER_GRPC_CONCURRENCY_MODE``
     - str
     - ``threading``
     - gRPC concurrency mode (``threading`` or ``asyncio``).
   * - ``CONTROLLER_GRPC_THREADS_PER_PROCESS``
     - int
     - ``1``
     - Worker threads per gRPC process.

Service Endpoints
~~~~~~~~~~~~~~~~~

.. list-table::
   :header-rows: 1
   :widths: 30 10 15 45

   * - Variable
     - Type
     - Default
     - Description
   * - ``S2S_SERVER``
     - str
     - (set by compose)
     - S2S service ``host:port``. Optional when running in
       bypass-S2S-only mode.
   * - ``ASD_SERVER``
     - str
     - (set by compose)
     - ASD NIM service ``host:port``. Not required when
       ``bypass_asd=True`` in ``ContentLocalizationConfig``.
   * - ``LIPSYNC_SERVER``
     - str
     - (set by compose)
     - LipSync NIM service ``host:port``.

Processing and Timeouts
~~~~~~~~~~~~~~~~~~~~~~~

.. list-table::
   :header-rows: 1
   :widths: 40 10 10 40

   * - Variable
     - Type
     - Default
     - Description
   * - ``S2S_SERVICE``
     - str
     - ``EL_DUBBING``
     - S2S backend (``EL_DUBBING`` or ``RIVA_TRANSACTIONAL``).
   * - ``CONTROLLER_CONFIG_POLL_TIMEOUT``
     - float
     - ``5.0``
     - Seconds to wait for per-request config messages
       (``controller_config``, ``asd_config``, ``lipsync_config``)
       before treating them as absent.
   * - ``CONTROLLER_CLEANUP_TIMEOUT``
     - float
     - ``10.0``
     - Seconds to wait for the deserializer thread and client threads
       to finish during per-request cleanup.

Debug
~~~~~

.. list-table::
   :header-rows: 1
   :widths: 40 10 10 40

   * - Variable
     - Type
     - Default
     - Description
   * - ``CONTROLLER_DEBUG_PORT``
     - int
     - ``5678``
     - VS Code debugpy listen port.
   * - ``CONTROLLER_VS_CODE_DEBUG``
     - int
     - ``0``
     - Set to ``1`` to enable debugpy wait-for-client mode.

Profiling
~~~~~~~~~

.. list-table::
   :header-rows: 1
   :widths: 40 10 10 40

   * - Variable
     - Type
     - Default
     - Description
   * - ``CONTROLLER_PROFILER``
     - int
     - ``0``
     - Set to ``1`` to enable the profiling framework.
   * - ``CONTROLLER_PROFILER_TYPE``
     - str
     - ``cprofiler``
     - Profiler backend (``cprofiler`` or ``yappi``).
   * - ``CONTROLLER_METRIC_TRACKER``
     - int
     - ``0``
     - Set to ``1`` to enable metric tracking.

S2S Service
-----------

Basic
~~~~~

.. list-table::
   :header-rows: 1
   :widths: 40 10 10 40

   * - Variable
     - Type
     - Default
     - Description
   * - ``S2S_GRPC_API_PORT``
     - int
     - ``50050``
     - gRPC listen port.
   * - ``S2S_LOG_LEVEL``
     - str
     - ``INFO``
     - Logging level.
   * - ``S2S_SAMPLE_RATE_HZ``
     - int
     - ``16000``
     - Input audio sample rate (Hz).
   * - ``S2S_MESSAGE_SIZE``
     - int
     - ``67108864``
     - Maximum gRPC message size (bytes).

Language
~~~~~~~~

.. list-table::
   :header-rows: 1
   :widths: 40 10 10 40

   * - Variable
     - Type
     - Default
     - Description
   * - ``S2S_DEFAULT_SOURCE_LANGUAGE``
     - str
     - ``auto``
     - Source language code (``auto`` for ElevenLabs auto-detect).
   * - ``S2S_DEFAULT_TARGET_LANGUAGE``
     - str
     - ``es``
     - Target language code.

Timeouts
~~~~~~~~

.. list-table::
   :header-rows: 1
   :widths: 40 10 10 40

   * - Variable
     - Type
     - Default
     - Description
   * - ``S2S_CLEANUP_TIMEOUT``
     - float
     - ``1.0``
     - Seconds to wait for ASR, segmentizer, and TTS sub-pipeline
       threads during cleanup.

ElevenLabs
~~~~~~~~~~

These apply only when ``S2S_SERVICE=EL_DUBBING``.

.. list-table::
   :header-rows: 1
   :widths: 40 10 10 40

   * - Variable
     - Type
     - Default
     - Description
   * - ``S2S_EL_DUBBING_POLL_INTERVAL``
     - int
     - ``10``
     - Seconds between ElevenLabs dubbing status checks.
   * - ``S2S_EL_DUBBING_MAX_ATTEMPTS``
     - int
     - ``120``
     - Maximum dubbing status poll attempts before timeout
       (total max wait = interval x attempts, default 20 minutes).
   * - ``S2S_EL_KEEPALIVE_INTERVAL``
     - int
     - ``1``
     - Seconds between keepalive pings sent to the client while
       waiting for ElevenLabs dubbing to complete.

ASD NIM
-------

.. list-table::
   :header-rows: 1
   :widths: 40 10 10 40

   * - Variable
     - Type
     - Default
     - Description
   * - ``ASD_GRPC_API_PORT``
     - int
     - ``50055``
     - gRPC listen port.
   * - ``ASD_LOG_LEVEL``
     - str
     - ``INFO``
     - Logging level.

LipSync NIM
-----------

.. list-table::
   :header-rows: 1
   :widths: 40 10 10 40

   * - Variable
     - Type
     - Default
     - Description
   * - ``LIPSYNC_NIM_GRPC_API_PORT``
     - int
     - ``50054``
     - gRPC listen port.
   * - ``LIPSYNC_LOG_LEVEL``
     - str
     - ``INFO``
     - Logging level.
   * - ``LIPSYNC_DEBUG_MODE``
     - int
     - ``0``
     - Set to ``1`` to enable LipSync NIM debug mode.

Secrets
-------

These must be set in a ``.env`` file at the repository root (never
committed to version control).

.. list-table::
   :header-rows: 1
   :widths: 30 70

   * - Variable
     - Description
   * - ``LIPSYNC_API_KEY``
     - NGC API key for LipSync NIM. Mapped to ``NGC_API_KEY`` inside the
       LipSync container by ``docker-compose.yml``.
   * - ``ASD_API_KEY``
     - NGC API key for ASD NIM. Mapped to ``NGC_API_KEY`` inside the
       ASD container by ``docker-compose.yml``.
   * - ``AST_API_KEY``
     - NGC API key for RIVA ASR NIM (RIVA profiles only). Mapped to
       ``NGC_API_KEY`` inside the AST container by ``docker-compose.yml``.
   * - ``TTS_API_KEY``
     - NGC API key for RIVA TTS NIM (RIVA profiles only). Mapped to
       ``NGC_API_KEY`` inside the TTS container by ``docker-compose.yml``.
   * - ``ELEVENLABS_API_KEY``
     - ElevenLabs API key (required when ``S2S_SERVICE=EL_DUBBING``).
