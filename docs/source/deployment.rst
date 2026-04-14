.. _deployment:

Deployment
==========

This section covers deploying the Content Localization Blueprint services for both development and production environments.

Docker Compose Profiles
------------------------

The blueprint uses Docker Compose profiles to configure different service combinations for various use cases.

Available Profiles
~~~~~~~~~~~~~~~~~~

.. list-table::
   :header-rows: 1
   :widths: 20 8 12 12 8 10 12 10 30

   * - Profile
     - S2S
     - ASR (RIVA)
     - TTS (RIVA)
     - ASD
     - LipSync
     - Controller
     - Demo App
     - Description
   * - ``default``
     - ✓
     - ✓
     - ✓
     - ✓
     - ✓
     - ✓
     - ✓
     - All services (for testing)
   * - ``third-party-s2s``
     - ✓
     - \-
     - \-
     - \-
     - \-
     - \-
     - \-
     - S2S only with ElevenLabs/CambAI
   * - ``riva``
     - ✓
     - ✓
     - ✓
     - \-
     - \-
     - \-
     - \-
     - S2S with RIVA ASR/TTS
   * - ``lipsync``
     - \-
     - \-
     - \-
     - \-
     - ✓
     - \-
     - \-
     - LipSync only
   * - ``third-party-s2s-lipsync``
     - ✓
     - \-
     - \-
     - \-
     - ✓
     - \-
     - \-
     - S2S (ElevenLabs/CambAI) + LipSync
   * - ``riva-lipsync``
     - ✓
     - ✓
     - ✓
     - \-
     - ✓
     - \-
     - \-
     - S2S (RIVA) + LipSync
   * - ``third-party-s2s-asd-lipsync``
     - ✓
     - \-
     - \-
     - ✓
     - ✓
     - \-
     - \-
     - Full pipeline with ElevenLabs/CambAI
   * - ``riva-asd-lipsync``
     - ✓
     - ✓
     - ✓
     - ✓
     - ✓
     - \-
     - \-
     - Full pipeline with RIVA
   * - ``asd``
     - \-
     - \-
     - \-
     - ✓
     - \-
     - \-
     - \-
     - Active Speaker Detection only
   * - ``controller-third-party-s2s``
     - ✓
     - \-
     - \-
     - ✓
     - ✓
     - ✓
     - \-
     - Orchestrated pipeline (ElevenLabs/CambAI)
   * - ``controller-riva``
     - ✓
     - ✓
     - ✓
     - ✓
     - ✓
     - ✓
     - \-
     - Orchestrated pipeline (RIVA)
   * - ``demo-app``
     - ✓
     - \-
     - \-
     - \-
     - \-
     - \-
     - ✓
     - S2S + Web Demo App
   * - ``demo-app-third-party-s2s``
     - ✓
     - \-
     - \-
     - ✓
     - ✓
     - ✓
     - ✓
     - Full stack with Web Demo (ElevenLabs/CambAI)
   * - ``demo-app-riva``
     - ✓
     - ✓
     - ✓
     - ✓
     - ✓
     - ✓
     - ✓
     - Full stack with Web Demo (RIVA)

Usage Examples
~~~~~~~~~~~~~~

ElevenLabs/CambAI with full pipeline and demo app:

.. code-block:: bash

   docker compose --profile demo-app-third-party-s2s \
       --env-file configs/elevenlabs.env \
       --env-file .env \
       up --build

RIVA with full pipeline and demo app:

.. code-block:: bash

   docker compose --profile demo-app-riva \
       --env-file configs/riva.env \
       --env-file .env \
       up --build

Controller orchestration with ElevenLabs/CambAI:

.. code-block:: bash

   docker compose --profile controller-third-party-s2s \
       --env-file configs/elevenlabs.env \
       --env-file .env \
       up --build

Controller orchestration with RIVA:

.. code-block:: bash

   docker compose --profile controller-riva \
       --env-file configs/riva.env \
       --env-file .env \
       up --build

Profile Selection Guide
~~~~~~~~~~~~~~~~~~~~~~~

* **For Development/Testing**: Use ``demo-app-third-party-s2s`` or ``demo-app-riva`` for the full stack with web interface
* **For Production with ElevenLabs/CambAI**: Use ``controller-third-party-s2s`` for orchestrated processing
* **For Production with RIVA**: Use ``controller-riva`` for orchestrated processing
* **For Service Testing**: Use individual profiles like ``third-party-s2s``, ``riva``, ``lipsync``, or ``asd``

First-Time Deployment
---------------------

For first-time deployment, use the deploy scripts to verify each service individually before deploying the full stack.

Deploy ASR Service
~~~~~~~~~~~~~~~~~~

Deploy the RIVA ASR service with the Canary model:

.. code-block:: bash

   ./scripts/deploy_asr_canary.sh

This will:

* Download the Canary 1B ASR model to ``volumes/models/ast-canary/``
* Start the RIVA ASR container on ports 8003 (HTTP) and 50053 (gRPC)
* Verify the service is running correctly

**Note:** Model download may take several minutes. Press Ctrl+C to stop once verified.

Deploy TTS Service
~~~~~~~~~~~~~~~~~~

Deploy the zero-shot TTS service:

.. code-block:: bash

   ./scripts/deploy_tts_zeroshot.sh

This downloads the Magpie Zero-Shot model to ``volumes/models/tts-zeroshot/`` and requires ``TTS_API_KEY`` environment variable.

Deploy LipSync Service
~~~~~~~~~~~~~~~~~~~~~~

Deploy the LipSync service:

.. code-block:: bash

   ./scripts/deploy_lipsync.sh

This will:

* Download the LipSync models to ``volumes/models/lipsync/``
* Start the LipSync container on ports 8000 (HTTP) and 8001 (gRPC)
* Requires ``LIPSYNC_API_KEY`` environment variable

Deploy ASD Service
~~~~~~~~~~~~~~~~~~

Deploy the Active Speaker Detection (ASD) NIM service:

.. code-block:: bash

   ./scripts/deploy_asd.sh

This will:

* Deploy ASD NIM container with GPU support
* Start ASD container on ports 8005 (HTTP) and 50055 (gRPC)
* Mount model cache at ``volumes/models/asd/``
* Requires ``ASD_API_KEY`` environment variable

Deployment Notes
~~~~~~~~~~~~~~~~

* Each deploy script runs in interactive mode (``-it``) and will occupy your terminal
* Run each script in a separate terminal or stop (Ctrl+C) before running the next
* These scripts are for **verification only** - use docker compose for production deployments

Service Management
------------------

Starting Services
~~~~~~~~~~~~~~~~~

.. code-block:: bash

   # Default profile: Full stack with ElevenLabs/CambAI and demo app
   docker compose --profile demo-app-third-party-s2s \
       --env-file configs/elevenlabs.env \
       --env-file .env \
       up --build

Stopping Services
~~~~~~~~~~~~~~~~~

.. code-block:: bash

   # Stop all services
   docker compose down

   # Stop and remove volumes (clean state)
   docker compose down -v

Viewing Logs
~~~~~~~~~~~~

Real-Time Log Viewing
^^^^^^^^^^^^^^^^^^^^^

.. code-block:: bash

   # View logs from all services (follow mode)
   docker compose logs -f

   # View logs from specific service
   docker compose logs -f s2s
   docker compose logs -f controller
   docker compose logs -f lipsync

   # View last 100 lines of logs
   docker compose logs --tail=100

Copy Logs to Files
^^^^^^^^^^^^^^^^^^

Use the log copy script to save logs to local files:

.. code-block:: bash

   # Copy all service logs
   ./scripts/copy_docker_logs.sh

   # Copy specific service logs
   ./scripts/copy_docker_logs.sh s2s
   ./scripts/copy_docker_logs.sh controller

This creates log files in ``./logs/``:

* ``./logs/s2s.log`` - Speech-to-Speech service logs
* ``./logs/ast.log`` - ASR (RIVA) service logs
* ``./logs/tts.log`` - TTS (RIVA) service logs
* ``./logs/lipsync.log`` - LipSync service logs
* ``./logs/asd.log`` - Active Speaker Detection logs
* ``./logs/controller.log`` - Controller orchestration logs

Benefits of Copying Logs
^^^^^^^^^^^^^^^^^^^^^^^^^

* Persist logs even after containers are stopped
* Easy to share with team members for debugging
* Can be archived or uploaded to issue trackers
* Includes line counts and helpful status messages

