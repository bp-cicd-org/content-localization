# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Test configuration and utilities for model tests."""

from pathlib import Path
from typing import Any

import numpy as np
import pytest


class ModelTestConfig:
    """Configuration for model tests."""

    # Model paths
    MODEL_PATHS = {
        "scrfd": "volumes/models/scrfd_10g-local.trt",
        "arcface": "volumes/models/arcface_mobilefacenet_600k-local.trt",
        "auraface": "volumes/models/auraface_glintr100-local.trt",
        "nv_facedetect": "volumes/models/nv_facedetect_sdk-local.trt",
    }

    # Test image configurations
    TEST_IMAGE_SIZES = [(480, 640), (640, 480), (800, 600), (1024, 768)]

    # Performance thresholds
    PERFORMANCE_THRESHOLDS = {
        "detection_time_ms": 1000,  # Maximum detection time in milliseconds
        "recognition_time_ms": 500,  # Maximum recognition time in milliseconds
        "memory_leak_mb": 100,  # Maximum memory leak in MB
    }

    # Test parameters
    CONFIDENCE_THRESHOLDS = [0.1, 0.3, 0.5, 0.7, 0.9]
    IOU_THRESHOLDS = [0.1, 0.3, 0.5, 0.7, 0.9]

    @classmethod
    def get_model_path(cls, model_name: str) -> Path | None:
        """Get the path to a model file."""
        if model_name not in cls.MODEL_PATHS:
            return None

        path = Path(cls.MODEL_PATHS[model_name])
        return path if path.exists() else None

    @classmethod
    def skip_if_model_not_found(cls, model_name: str) -> pytest.MarkDecorator:
        """Create a pytest mark to skip tests if model is not found."""
        return pytest.mark.skipif(
            cls.get_model_path(model_name) is None,
            reason=f"Model file not found: {cls.MODEL_PATHS.get(model_name, 'Unknown')}",
        )

    @classmethod
    def get_test_images(cls) -> list[np.ndarray]:
        """Generate test images of different sizes."""
        images = []

        for height, width in cls.TEST_IMAGE_SIZES:
            # Create a simple test image
            image = np.zeros((height, width, 3), dtype=np.uint8)

            # Add some random content
            image = np.random.randint(0, 255, (height, width, 3), dtype=np.uint8)

            # Add a face-like region in the center
            center_x, center_y = width // 2, height // 2
            radius = min(width, height) // 4

            # Draw a simple face
            cv2.circle(image, (center_x, center_y), radius, (255, 255, 255), -1)
            cv2.circle(
                image, (center_x - radius // 3, center_y - radius // 3), radius // 8, (0, 0, 0), -1
            )
            cv2.circle(
                image, (center_x + radius // 3, center_y - radius // 3), radius // 8, (0, 0, 0), -1
            )

            images.append(image)

        return images

    @classmethod
    def get_test_landmarks(cls) -> list[np.ndarray]:
        """Generate test landmarks for different image sizes."""
        landmarks_list = []

        for height, width in cls.TEST_IMAGE_SIZES:
            center_x, center_y = width // 2, height // 2
            radius = min(width, height) // 4

            landmarks = np.array(
                [
                    [center_x - radius // 3, center_y - radius // 3],  # Left eye
                    [center_x + radius // 3, center_y - radius // 3],  # Right eye
                    [center_x, center_y],  # Nose
                    [center_x - radius // 4, center_y + radius // 3],  # Left mouth corner
                    [center_x + radius // 4, center_y + radius // 3],  # Right mouth corner
                ],
                dtype=np.float32,
            )

            landmarks_list.append(landmarks)

        return landmarks_list


class ModelTestUtils:
    """Utility functions for model tests."""

    @staticmethod
    def create_mock_backend() -> Any:
        """Create a mock backend for testing."""
        from unittest.mock import Mock

        mock_backend = Mock()
        mock_backend.forward.return_value = [
            np.random.rand(1, 2, 40, 40),  # scores
            np.random.rand(1, 4, 40, 40),  # bboxes
            np.random.rand(1, 10, 40, 40),  # keypoints
        ]
        return mock_backend

    @staticmethod
    def create_mock_recognizer_backend() -> Any:
        """Create a mock backend for recognizer testing."""
        from unittest.mock import Mock

        mock_backend = Mock()
        mock_backend.forward.return_value = np.random.rand(1, 512)
        return mock_backend

    @staticmethod
    def validate_bboxes(bboxes: np.ndarray, image_shape: tuple) -> None:
        """Validate bounding box format and values."""
        if bboxes is None or len(bboxes) == 0:
            return

        # Check shape
        assert bboxes.shape[1] == 5, (
            f"Expected 5 columns (x1, y1, x2, y2, conf), got {bboxes.shape[1]}"
        )

        # Check confidence scores
        assert np.all(bboxes[:, 4] >= 0) and np.all(bboxes[:, 4] <= 1), (
            "Confidence scores must be in [0, 1]"
        )

        # Check bounding box coordinates
        height, width = image_shape[:2]
        assert np.all(bboxes[:, 0] >= 0) and np.all(bboxes[:, 0] <= width), (
            "x1 coordinates out of bounds"
        )
        assert np.all(bboxes[:, 1] >= 0) and np.all(bboxes[:, 1] <= height), (
            "y1 coordinates out of bounds"
        )
        assert np.all(bboxes[:, 2] >= 0) and np.all(bboxes[:, 2] <= width), (
            "x2 coordinates out of bounds"
        )
        assert np.all(bboxes[:, 3] >= 0) and np.all(bboxes[:, 3] <= height), (
            "y2 coordinates out of bounds"
        )

        # Check that x2 > x1 and y2 > y1
        assert np.all(bboxes[:, 2] > bboxes[:, 0]), "x2 must be greater than x1"
        assert np.all(bboxes[:, 3] > bboxes[:, 1]), "y2 must be greater than y1"

    @staticmethod
    def validate_keypoints(keypoints: np.ndarray, image_shape: tuple) -> None:
        """Validate keypoints format and values."""
        if keypoints is None or len(keypoints) == 0:
            return

        # Check shape (5 landmarks * 2 coordinates = 10)
        assert keypoints.shape[1] == 10, (
            f"Expected 10 coordinates (5 landmarks * 2), got {keypoints.shape[1]}"
        )

        # Check coordinates are within image bounds
        height, width = image_shape[:2]
        x_coords = keypoints[:, 0::2]  # Even indices (0, 2, 4, 6, 8)
        y_coords = keypoints[:, 1::2]  # Odd indices (1, 3, 5, 7, 9)

        assert np.all(x_coords >= 0) and np.all(x_coords <= width), (
            "Keypoint x coordinates out of bounds"
        )
        assert np.all(y_coords >= 0) and np.all(y_coords <= height), (
            "Keypoint y coordinates out of bounds"
        )

    @staticmethod
    def validate_embedding(embedding: np.ndarray) -> None:
        """Validate embedding format and values."""
        # Check shape
        assert embedding.shape == (512,), f"Expected embedding shape (512,), got {embedding.shape}"

        # Check data type
        assert embedding.dtype == np.float32, f"Expected float32 dtype, got {embedding.dtype}"

        # Check for NaN values
        assert not np.any(np.isnan(embedding)), "Embedding contains NaN values"

        # Check for infinite values
        assert not np.any(np.isinf(embedding)), "Embedding contains infinite values"

        # Check that embedding is not all zeros
        assert not np.all(embedding == 0), "Embedding is all zeros"

    @staticmethod
    def measure_performance(
        func: callable, *args, num_runs: int = 10, **kwargs
    ) -> dict[str, float]:
        """Measure performance of a function."""
        import time

        # Warmup
        for _ in range(3):
            func(*args, **kwargs)

        # Benchmark
        times = []
        for _ in range(num_runs):
            start_time = time.perf_counter()
            result = func(*args, **kwargs)
            end_time = time.perf_counter()
            times.append((end_time - start_time) * 1000)  # Convert to ms

        return {
            "mean": np.mean(times),
            "std": np.std(times),
            "min": np.min(times),
            "max": np.max(times),
            "median": np.median(times),
            "result": result,
        }

    @staticmethod
    def measure_memory_usage(func: callable, *args, **kwargs) -> dict[str, float]:
        """Measure memory usage of a function."""
        try:
            import gc

            import psutil

            process = psutil.Process()

            # Get initial memory
            gc.collect()
            initial_memory = process.memory_info().rss / 1024 / 1024  # MB

            # Run function
            result = func(*args, **kwargs)

            # Get memory after function
            after_func_memory = process.memory_info().rss / 1024 / 1024  # MB

            # Cleanup
            del result
            gc.collect()

            # Get memory after cleanup
            after_cleanup_memory = process.memory_info().rss / 1024 / 1024  # MB

            return {
                "initial_mb": initial_memory,
                "after_func_mb": after_func_memory,
                "after_cleanup_mb": after_cleanup_memory,
                "func_memory_mb": after_func_memory - initial_memory,
                "memory_leak_mb": after_cleanup_memory - initial_memory,
                "result": result,
            }

        except ImportError:
            return {"error": "psutil not available"}


# Import cv2 for the test image generation
try:
    import cv2
except ImportError:
    # If cv2 is not available, create a simple mock
    class MockCV2:
        @staticmethod
        def circle(img, center, radius, color, thickness):
            # Simple mock implementation
            pass

    cv2 = MockCV2()


# Pytest markers for different test categories
pytest_plugins = []


# Define custom markers
def pytest_configure(config):
    """Configure pytest with custom markers."""
    config.addinivalue_line("markers", "unit: mark test as a unit test")
    config.addinivalue_line("markers", "integration: mark test as an integration test")
    config.addinivalue_line("markers", "performance: mark test as a performance test")
    config.addinivalue_line("markers", "memory: mark test as a memory usage test")
    config.addinivalue_line("markers", "slow: mark test as slow running")
