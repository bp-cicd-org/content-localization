.. _logging:

Docker Service Logging
======================

This section describes the logging configuration for the Docker services in this project.

Overview
--------

Each Docker service writes its logs to independent log files in the ``./logs/`` directory.
This provides better log management and debugging capabilities compared to the default Docker logging driver.

Log Files
---------

The following log files are created when `scripts/copy_docker_logs.sh` is run:

* ``./logs/s2s.log`` - Speech-to-Speech service logs
* ``./logs/ast.log`` - Riva ASR (Automatic Speech Recognition) service logs  
* ``./logs/tts.log`` - Riva TTS (Text-to-Speech) service logs
* ``./logs/lipsync.log`` - Lip Sync service logs
* ``./logs/asd.log`` - Active Speaker Detection service logs
* ``./logs/controller.log`` - Controller service logs

Log Management Scripts
----------------------

Two convenience scripts are provided to manage log files:

* ``./scripts/copy_docker_logs.sh`` - Copy logs from Docker's default location to ``./logs/`` directory

Usage
^^^^^

**copy_docker_logs.sh:**
.. code-block:: bash

    # Copy logs for all services
    ./scripts/copy_docker_logs.sh

    # Copy logs for specific service
    ./scripts/copy_docker_logs.sh s2s
    ./scripts/copy_docker_logs.sh ast
    ./scripts/copy_docker_logs.sh tts
    ./scripts/copy_docker_logs.sh lipsync

Manual Log Access
-----------------

You can also access logs directly:

.. code-block:: bash

    # View specific service logs
    cat ./logs/s2s.log
    cat ./logs/ast.log
    cat ./logs/tts.log
    cat ./logs/lipsync.log

    # Follow specific service logs
    tail -f ./logs/s2s.log

    # Search logs
    rg "ERROR" ./logs/s2s.log

Docker Compose Logs
-------------------

You can still use standard Docker Compose commands to view logs:

.. code-block:: bash

    # View logs for all services
    docker compose logs

    # View logs for specific service
    docker compose logs speech-to-speech
    docker compose logs riva-ast
    docker compose logs riva-tts
    docker compose logs lipsync

    # Follow logs
    docker compose logs -f

Troubleshooting
---------------

If log files are not being created:

1. Ensure the ``./logs/`` directory exists and has write permissions
2. Check that the Docker volumes are properly mounted
3. Verify the logging driver is supported by your Docker version
4. Check Docker daemon logs for any logging-related errors
