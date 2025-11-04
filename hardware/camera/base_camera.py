"""
Abstract base class for camera implementations.
Defines the common interface for all camera types.
"""

import multiprocessing as mp
import threading
from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Any, Dict, Optional, Tuple

import numpy as np

from common.logging_manager import get_logger


@dataclass
class CameraFrame:
    """Container for camera frame data"""

    color_image: Optional[np.ndarray] = None
    depth_image: Optional[np.ndarray] = None
    timestamp: float = 0.0


class CameraConfig:
    """Base configuration for cameras"""

    def __init__(
        self,
        camera_name: str = "camera",
        fps: int = 30,
        color_resolution: Tuple[int, int] = (640, 480),
        depth_resolution: Tuple[int, int] = (640, 480),
        enable_depth: bool = True,
        enable_color: bool = True,
        save_intrinsics: Optional[bool] = False,
        **kwargs,
    ):
        self.camera_name = camera_name
        self.fps = fps
        self.color_resolution = color_resolution
        self.depth_resolution = depth_resolution
        self.enable_depth = enable_depth
        self.enable_color = enable_color
        self.save_intrinsics = save_intrinsics
        # Store any additional config parameters
        for key, value in kwargs.items():
            setattr(self, key, value)

    def __repr__(self):
        return (
            f"camera_name={self.camera_name}\n"
            f"fps={self.fps}\n"
            f"color_resolution={self.color_resolution}\n"
            f"depth_resolution={self.depth_resolution}\n"
            f"enable_depth={self.enable_depth}\n"
            f"enable_color={self.enable_color}\n"
            f"save_intrinsics={self.save_intrinsics}"
        )


class BaseCamera(ABC):
    """Abstract base class for all camera implementations"""

    def __init__(self, config: CameraConfig):
        self.config = config
        self.logger = get_logger(self.config.camera_name)
        self._is_running = False
        self._frame_count = 0
        self.intrinsics: Optional[Dict[str, Any]] = None

    @abstractmethod
    def start(self) -> bool:
        """
        Start the camera.

        Returns:
            bool: True if started successfully, False otherwise
        """
        pass

    @abstractmethod
    def stop(self):
        """Stop the camera and cleanup resources"""
        pass

    @abstractmethod
    def _initialize_camera(self) -> bool:
        """Initialize the camera hardware"""
        pass

    @abstractmethod
    def _cleanup_camera(self):
        """Cleanup camera resources"""
        pass

    @abstractmethod
    def _capture_frame(self) -> Optional[CameraFrame]:
        """Capture a single frame"""
        pass

    @abstractmethod
    def get_frame(self) -> Optional[CameraFrame]:
        """
        Get the latest frame from the camera.

        Returns:
            CameraFrame or None if no frame available
        """
        pass

    @abstractmethod
    def get_intrinsics(self) -> Optional[Dict[str, Any]]:
        """
        Get camera intrinsic parameters.

        Returns:
            Dictionary with intrinsic parameters or None
        """
        pass

    @abstractmethod
    def is_connected(self) -> bool:
        """
        Check if camera is connected and available.

        Returns:
            bool: True if connected, False otherwise
        """
        pass

    @property
    def is_running(self) -> bool:
        """Check if camera is currently running"""
        return self._is_running

    @property
    def frame_count(self) -> int:
        """Get total number of frames captured"""
        return self._frame_count


class SimpleCamera(BaseCamera):
    """Simple synchronous camera implementation"""

    def __init__(self, config: CameraConfig):
        super().__init__(config)

    def start(self) -> bool:
        """Start the camera"""
        if self._is_running:
            self.logger.warning("Camera already running")
            return True

        if not self._initialize_camera():
            return False

        self._is_running = True
        self.logger.info("Camera started")
        return True

    def stop(self):
        """Stop the camera"""
        if not self._is_running:
            return

        self._cleanup_camera()
        self._is_running = False
        self.logger.info("Camera stopped")

    def get_frame(self) -> Optional[CameraFrame]:
        """Get a frame from the camera"""
        if not self._is_running:
            return None

        frame = self._capture_frame()
        if frame:
            self._frame_count += 1
        return frame


class ThreadedCamera(BaseCamera):
    """Base class for threaded camera implementations"""

    def __init__(self, config: CameraConfig):
        super().__init__(config)
        self._thread: Optional[threading.Thread] = None
        self._stop_event = threading.Event()
        self._frame_lock = threading.Lock()
        self._latest_frame: Optional[CameraFrame] = None

    def start(self) -> bool:
        """Start the camera in a separate thread"""
        if self._is_running:
            self.logger.warning("Camera already running")
            return True

        if not self._initialize_camera():
            return False

        self._stop_event.clear()
        self._thread = threading.Thread(target=self._capture_loop, daemon=True)
        self._thread.start()
        self._is_running = True

        self.logger.info("Camera started in threaded mode")
        return True

    def stop(self):
        """Stop the camera thread"""
        if not self._is_running:
            return

        self._stop_event.set()
        if self._thread and self._thread.is_alive():
            self._thread.join(timeout=2.0)

        self._cleanup_camera()
        self._is_running = False
        self.logger.info("Camera stopped")

    def get_frame(self) -> Optional[CameraFrame]:
        """Get the latest frame (thread-safe)"""
        with self._frame_lock:
            return self._latest_frame

    def _capture_loop(self):
        """Main capture loop running in separate thread"""
        while not self._stop_event.is_set():
            frame = self._capture_frame()
            if frame:
                with self._frame_lock:
                    self._latest_frame = frame
                    self._frame_count += 1


class ProcessCamera(BaseCamera):
    """Base class for process-based camera implementations"""

    def __init__(self, config: CameraConfig):
        super().__init__(config)
        self._process: Optional[mp.Process] = None
        self._stop_event = mp.Event()
        self._ready_event = mp.Event()
        self._frame_queue = mp.Queue(maxsize=2)  # Small buffer

    def start(self) -> bool:
        """Start the camera in a separate process"""
        if self._is_running:
            self.logger.warning("Camera already running")
            return True

        self._stop_event.clear()
        self._ready_event.clear()

        self._process = mp.Process(
            target=self._process_main, name=f"{self.config.camera_name}_process"
        )
        self._process.start()

        # Wait for process to be ready
        if self._ready_event.wait(timeout=5.0):
            self._is_running = True
            self.logger.info("Camera started in process mode")
            return True
        else:
            self.logger.error("Camera failed to start within timeout")
            self.stop()
            return False

    def stop(self):
        """Stop the camera process"""
        if not self._is_running:
            return

        self._stop_event.set()
        if self._process and self._process.is_alive():
            self._process.join(timeout=3.0)
            if self._process.is_alive():
                self._process.terminate()
                self._process.join(timeout=1.0)

        self._is_running = False
        self.logger.info("Camera process stopped")

    def get_frame(self) -> Optional[CameraFrame]:
        """Get the latest frame from the process queue"""
        try:
            # Get the most recent frame, discarding older ones
            frame = None
            while not self._frame_queue.empty():
                frame = self._frame_queue.get_nowait()
            return frame
        except Exception:
            return None

    def _process_main(self):
        """Main function running in separate process"""
        try:
            if not self._initialize_camera():
                return

            self._ready_event.set()

            while not self._stop_event.is_set():
                frame = self._capture_frame()
                if frame:
                    try:
                        # Only keep the latest frame in queue
                        while not self._frame_queue.empty():
                            try:
                                self._frame_queue.get_nowait()
                            except Exception:
                                break
                        self._frame_queue.put_nowait(frame)
                        self._frame_count += 1
                    except Exception:
                        pass  # Queue full, skip frame

        except Exception as e:
            self.logger.error(f"Error in camera process: {e}")
        finally:
            self._cleanup_camera()
