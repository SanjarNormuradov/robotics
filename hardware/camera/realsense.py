"""
RealSense camera implementations.
Provides Simple, Threaded, and Process versions of the RealSense camera.
"""

import time
from typing import Any, Dict, Optional
import omegaconf

import numpy as np

try:
    import pyrealsense2 as rs
except ImportError:
    rs = None
    print("Warning: pyrealsense2 not installed. RealSense cameras will not work.")

from .base_camera import (
    CameraConfig,
    CameraFrame,
    ProcessCamera,
    SimpleCamera,
    ThreadedCamera,
)


class RealSenseConfig(CameraConfig):
    """Configuration specific to RealSense cameras"""

    def __init__(
        self,
        camera_name: str = "realsense",
        fps: int = 30,
        color_resolution: tuple = (640, 480),
        depth_resolution: tuple = (640, 480),
        enable_depth: bool = True,
        enable_color: bool = True,
        save_intrinsics: Optional[bool] = False,
        serial_number: Optional[str] = None,
        depth_range: tuple = (0.1, 3.0),  # meters
        color_exposure: Optional[int] = None,
        color_gain: Optional[int] = None,
        color_white_balance: Optional[int] = None,
        depth_exposure: Optional[int] = None,
        depth_gain: Optional[int] = None,
        **kwargs,
    ):
        super().__init__(
            camera_name=camera_name,
            fps=fps,
            color_resolution=color_resolution,
            depth_resolution=depth_resolution,
            enable_depth=enable_depth,
            enable_color=enable_color,
            save_intrinsics=save_intrinsics,
            **kwargs,
        )
        self.serial_number = serial_number
        self.depth_range = depth_range
        self.color_exposure = color_exposure
        self.color_gain = color_gain
        self.color_white_balance = color_white_balance
        self.depth_exposure = depth_exposure
        self.depth_gain = depth_gain

    def __repr__(self):
        dict_repr = super().__repr__()
        return (
            f"{dict_repr}\n"
            f"serial_number={self.serial_number}\n"
            f"depth_range={self.depth_range}\n"
            f"color_exposure={self.color_exposure}\n"
            f"color_gain={self.color_gain}\n"
            f"color_white_balance={self.color_white_balance}\n"
            f"depth_exposure={self.depth_exposure}\n"
            f"depth_gain={self.depth_gain}"
        )


class RealSenseBase:
    """
    Base class for RealSense cameras that implements common functionality.
    This class provides the RealSense-specific implementations of abstract methods.
    """

    def __init__(self):
        # RealSense-specific attributes
        self.pipeline = None
        self.pipeline_config = None
        self.align = None

    def _initialize_camera(self) -> bool:
        """Initialize the RealSense camera hardware"""
        if rs is None:
            self.logger.error("pyrealsense2 not available")
            return False

        try:
            self.pipeline = rs.pipeline()
            self.pipeline_config = rs.config()

            # Configure streams
            if self.config.enable_color:
                self.pipeline_config.enable_stream(
                    rs.stream.color,
                    self.config.color_resolution[0],
                    self.config.color_resolution[1],
                    rs.format.bgr8,
                    self.config.fps,
                )

            if self.config.enable_depth:
                self.pipeline_config.enable_stream(
                    rs.stream.depth,
                    self.config.depth_resolution[0],
                    self.config.depth_resolution[1],
                    rs.format.z16,
                    self.config.fps,
                )

            # Enable specific device if serial number provided
            if hasattr(self.config, "serial_number") and self.config.serial_number:
                self.pipeline_config.enable_device(self.config.serial_number)
            else:
                serial_numbers = self.get_serial_numbers()
                if serial_numbers:
                    first_serial = list(serial_numbers.values())[0]
                    self.logger.info(
                        f"No serial number specified, using first detected device: {first_serial}"
                    )
                    self.pipeline_config.enable_device(first_serial)

            # Start pipeline
            self.pipeline.start(self.pipeline_config)

            # Setup alignment (align depth to color)
            if self.config.enable_color and self.config.enable_depth:
                self.align = rs.align(rs.stream.color)
            else:
                self.align = None

            try:
                self.set_options(
                    color_exposure=self.config.color_exposure,
                    color_gain=self.config.color_gain,
                    color_white_balance=self.config.color_white_balance,
                    depth_exposure=self.config.depth_exposure,
                    depth_gain=self.config.depth_gain,
                )
            except Exception as e:
                self.logger.warning(
                    f"Failed to set exposure/gain/white balance for color/depth sensors: {e}"
                )
            time.sleep(5.0)  # Allow settings to take effect

            # Store intrinsics if needed
            if hasattr(self.config, "save_intrinsics") and self.config.save_intrinsics:
                self.intrinsics = self.get_intrinsics()

            self.logger.info("RealSense camera initialized successfully")
            return True

        except Exception as e:
            self.logger.error(f"Failed to initialize RealSense camera: {e}")
            return False

    def _cleanup_camera(self):
        """Cleanup RealSense camera resources"""
        if self.pipeline:
            try:
                self.pipeline.stop()
            except Exception as e:
                self.logger.warning(f"Error stopping RealSense pipeline: {e}")
            finally:
                self.pipeline = None
                self.pipeline_config = None
                self.align = None

        self.logger.info("RealSense camera cleanup completed")

    def _capture_frame(self) -> Optional[CameraFrame]:
        """Capture a single frame from RealSense camera"""
        if not self.pipeline:
            return None

        try:
            # Wait for frames with short timeout for responsiveness
            frames = self.pipeline.wait_for_frames(timeout_ms=100)

            # Align frames if both streams enabled
            if self.align:
                frames = self.align.process(frames)

            frame = CameraFrame()
            frame.timestamp = time.time()

            # Get color frame
            if self.config.enable_color:
                color_frame = frames.get_color_frame()
                if color_frame:
                    frame.color_image = np.asanyarray(color_frame.get_data())

            # Get depth frame
            if self.config.enable_depth:
                depth_frame = frames.get_depth_frame()
                if depth_frame:
                    frame.depth_image = np.asanyarray(depth_frame.get_data())

            if frame.color_image is not None or frame.depth_image is not None:
                return frame

        except Exception as e:
            # Don't log every timeout, only unexpected errors
            if "timeout" not in str(e).lower():
                self.logger.warning(f"Failed to capture RealSense frame: {e}")

        return None

    def get_intrinsics(self) -> Optional[Dict[str, Any]]:
        """Get camera intrinsics from active pipeline"""
        if not self.pipeline:
            return None

        try:
            profile = self.pipeline.get_active_profile()
            intrinsics = {}

            if self.config.enable_color:
                color_stream = profile.get_stream(rs.stream.color)
                color_intrinsics = (
                    color_stream.as_video_stream_profile().get_intrinsics()
                )
                intrinsics["color"] = {
                    "fx": color_intrinsics.fx,
                    "fy": color_intrinsics.fy,
                    "cx": color_intrinsics.ppx,
                    "cy": color_intrinsics.ppy,
                    "width": color_intrinsics.width,
                    "height": color_intrinsics.height,
                    "distortion": color_intrinsics.coeffs,
                }

            if self.config.enable_depth:
                depth_stream = profile.get_stream(rs.stream.depth)
                depth_intrinsics = (
                    depth_stream.as_video_stream_profile().get_intrinsics()
                )
                intrinsics["depth"] = {
                    "fx": depth_intrinsics.fx,
                    "fy": depth_intrinsics.fy,
                    "cx": depth_intrinsics.ppx,
                    "cy": depth_intrinsics.ppy,
                    "width": depth_intrinsics.width,
                    "height": depth_intrinsics.height,
                    "distortion": depth_intrinsics.coeffs,
                }

            return intrinsics

        except Exception as e:
            self.logger.warning(f"Failed to get RealSense intrinsics: {e}")
            return None

    def is_connected(self) -> bool:
        """Check if RealSense camera is connected"""
        if rs is None:
            return False

        try:
            context = rs.context()
            devices = context.query_devices()

            if hasattr(self.config, "serial_number") and self.config.serial_number:
                for device in devices:
                    if (
                        device.get_info(rs.camera_info.serial_number)
                        == self.config.serial_number
                    ):
                        return True
                return False
            else:
                return len(devices) > 0

        except Exception:
            return False

    def set_options(
        self,
        color_exposure: Optional[int] = None,
        color_gain: Optional[int] = None,
        color_white_balance: Optional[int] = None,
        depth_exposure: Optional[int] = None,
        depth_gain: Optional[int] = None,
    ) -> bool:
        """Set the exposure time of the RealSense camera in microseconds"""
        if not self.pipeline:
            return False

        try:
            profile = self.pipeline.get_active_profile()

            try:
                color_sensor = profile.get_device().first_color_sensor()
                # report global time
                # https://github.com/IntelRealSense/librealsense/pull/3909
                color_sensor.set_option(rs.option.global_time_enabled, 1)
                if color_white_balance is None:
                    # auto exposure
                    self.color_sensor.set_option(
                        rs.option.enable_auto_white_balance, 1.0
                    )
                else:
                    # manual exposure
                    self.color_sensor.set_option(
                        rs.option.enable_auto_white_balance, 0.0
                    )
                    self.color_sensor.set_option(
                        rs.option.white_balance, color_white_balance
                    )
                if color_exposure is None and color_gain is None:
                    color_sensor.set_option(rs.option.enable_auto_exposure, 1)
                else:
                    color_sensor.set_option(rs.option.enable_auto_exposure, 0)
                    if color_exposure is not None:
                        color_sensor.set_option(rs.option.exposure, color_exposure)
                    if color_gain is not None:
                        color_sensor.set_option(rs.option.gain, color_gain)
            except Exception as e:
                self.logger.warning(f"Failed to get color sensor: {e}")

            try:
                depth_sensor = profile.get_device().first_depth_sensor()
                if depth_exposure is None and depth_gain is None:
                    depth_sensor.set_option(rs.option.enable_auto_exposure, 1)
                else:
                    depth_sensor.set_option(rs.option.enable_auto_exposure, 0)
                    if depth_exposure is not None:
                        depth_sensor.set_option(rs.option.exposure, depth_exposure)
                    if depth_gain is not None:
                        depth_sensor.set_option(rs.option.gain, depth_gain)
            except Exception as e:
                self.logger.warning(f"Failed to get depth sensor: {e}")

            return True
        except Exception as e:
            self.logger.warning(f"Failed to set RealSense exposure: {e}")
            return False

    def get_serial_numbers(self) -> Optional[Dict[str, str]]:
        """Get the serial numbers of the connected RealSense camera"""
        if rs is None:
            return None

        try:
            context = rs.context()
            devices = context.query_devices()

            serials = {}
            for device in devices:
                name = device.get_info(rs.camera_info.name)
                serial = device.get_info(rs.camera_info.serial_number)
                serials[name] = serial

            return serials

        except Exception as e:
            self.logger.warning(f"Failed to get RealSense serial numbers: {e}")
            return None


class RealSenseSimple(RealSenseBase, SimpleCamera):
    """Simple synchronous RealSense implementation"""

    def __init__(self, config: RealSenseConfig):
        SimpleCamera.__init__(self, config)
        RealSenseBase.__init__(self)


class RealSenseThreaded(RealSenseBase, ThreadedCamera):
    """Threaded RealSense implementation"""

    def __init__(self, config: RealSenseConfig):
        ThreadedCamera.__init__(self, config)
        RealSenseBase.__init__(self)


class RealSenseProcess(RealSenseBase, ProcessCamera):
    """Process-based RealSense implementation"""

    def __init__(self, config: RealSenseConfig):
        ProcessCamera.__init__(self, config)
        RealSenseBase.__init__(self)
