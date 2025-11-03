"""
ROS Camera Publisher for RealSense cameras.
Combines camera functionality with ROS Node capabilities.
"""

from typing import Optional

import rclpy
from cv_bridge import CvBridge
from rclpy.node import Node
from sensor_msgs.msg import CameraInfo, Image

from hardware.camera.base_camera import CameraFrame
from hardware.camera.realsense import (
    RealSenseConfig,
    RealSenseD405Simple,
    RealSenseD405Threaded,
)


class CameraPublisher(Node):
    """
    Base ROS camera publisher that combines camera functionality with ROS Node.
    This class inherits from ROS Node and uses composition for camera functionality.
    """

    def __init__(
        self,
        camera_name: str = "camera",
        camera_type: str = "simple",  # "simple", "threaded"
        publish_rate: float = 30.0,
        **camera_kwargs,
    ):
        super().__init__(f"{camera_name}_publisher")

        self.camera_name = camera_name
        self.camera_type = camera_type
        self.publish_rate = publish_rate

        # Initialize CV Bridge for image conversion
        self.cv_bridge = CvBridge()

        # Create camera configuration
        self.camera_config = RealSenseConfig(camera_name=camera_name, **camera_kwargs)

        # Initialize camera based on type
        self.camera = self._create_camera()

        # ROS Publishers
        self.color_publisher = None
        self.depth_publisher = None
        self.color_info_publisher = None
        self.depth_info_publisher = None

        # Setup publishers
        self._setup_publishers()

        # Timer for publishing
        self.timer = None

        self.get_logger().info(f"Camera publisher initialized: {camera_name}")

    def _create_camera(self):
        """Create the appropriate camera instance"""
        if self.camera_type == "simple":
            return RealSenseD405Simple(self.camera_config)
        elif self.camera_type == "threaded":
            return RealSenseD405Threaded(self.camera_config)
        else:
            raise ValueError(f"Unsupported camera type: {self.camera_type}")

    def _setup_publishers(self):
        """Setup ROS publishers based on camera configuration"""
        if self.camera_config.enable_color:
            self.color_publisher = self.create_publisher(
                Image, f"/{self.camera_name}/color/image_raw", 10
            )
            self.color_info_publisher = self.create_publisher(
                CameraInfo, f"/{self.camera_name}/color/camera_info", 10
            )

        if self.camera_config.enable_depth:
            self.depth_publisher = self.create_publisher(
                Image, f"/{self.camera_name}/depth/image_raw", 10
            )
            self.depth_info_publisher = self.create_publisher(
                CameraInfo, f"/{self.camera_name}/depth/camera_info", 10
            )

    def start(self) -> bool:
        """Start the camera and publishing"""
        if not self.camera.start():
            self.get_logger().error("Failed to start camera")
            return False

        # Start publishing timer
        timer_period = 1.0 / self.publish_rate
        self.timer = self.create_timer(timer_period, self._publish_callback)

        self.get_logger().info("Camera publisher started")
        return True

    def stop(self):
        """Stop the camera and publishing"""
        if self.timer:
            self.timer.cancel()
            self.timer = None

        self.camera.stop()
        self.get_logger().info("Camera publisher stopped")

    def _publish_callback(self):
        """Timer callback to publish camera data"""
        frame = self.camera.get_frame()
        if frame is None:
            return

        timestamp = self.get_clock().now().to_msg()

        # Publish color image
        if frame.color_image is not None and self.color_publisher is not None:
            self._publish_color_image(frame, timestamp)

        # Publish depth image
        if frame.depth_image is not None and self.depth_publisher is not None:
            self._publish_depth_image(frame, timestamp)

    def _publish_color_image(self, frame: CameraFrame, timestamp):
        """Publish color image and camera info"""
        try:
            # Convert to ROS Image message
            color_msg = self.cv_bridge.cv2_to_imgmsg(frame.color_image, encoding="bgr8")
            color_msg.header.stamp = timestamp
            color_msg.header.frame_id = f"{self.camera_name}_color_optical_frame"

            self.color_publisher.publish(color_msg)

            # Publish camera info if available
            if (
                frame.intrinsics
                and "color" in frame.intrinsics
                and self.color_info_publisher is not None
            ):
                color_info = self._create_camera_info_msg(
                    frame.intrinsics["color"],
                    timestamp,
                    f"{self.camera_name}_color_optical_frame",
                )
                self.color_info_publisher.publish(color_info)

        except Exception as e:
            self.get_logger().warning(f"Failed to publish color image: {e}")

    def _publish_depth_image(self, frame: CameraFrame, timestamp):
        """Publish depth image and camera info"""
        try:
            # Convert to ROS Image message (depth is typically 16UC1)
            depth_msg = self.cv_bridge.cv2_to_imgmsg(
                frame.depth_image, encoding="16UC1"
            )
            depth_msg.header.stamp = timestamp
            depth_msg.header.frame_id = f"{self.camera_name}_depth_optical_frame"

            self.depth_publisher.publish(depth_msg)

            # Publish camera info if available
            if (
                frame.intrinsics
                and "depth" in frame.intrinsics
                and self.depth_info_publisher is not None
            ):
                depth_info = self._create_camera_info_msg(
                    frame.intrinsics["depth"],
                    timestamp,
                    f"{self.camera_name}_depth_optical_frame",
                )
                self.depth_info_publisher.publish(depth_info)

        except Exception as e:
            self.get_logger().warning(f"Failed to publish depth image: {e}")

    def _create_camera_info_msg(
        self, intrinsics: dict, timestamp, frame_id: str
    ) -> CameraInfo:
        """Create CameraInfo message from intrinsics"""
        info_msg = CameraInfo()
        info_msg.header.stamp = timestamp
        info_msg.header.frame_id = frame_id

        info_msg.width = intrinsics["width"]
        info_msg.height = intrinsics["height"]

        # Camera matrix (3x3)
        info_msg.k = [
            intrinsics["fx"],
            0.0,
            intrinsics["cx"],
            0.0,
            intrinsics["fy"],
            intrinsics["cy"],
            0.0,
            0.0,
            1.0,
        ]

        # Projection matrix (3x4) - same as K for monocular
        info_msg.p = [
            intrinsics["fx"],
            0.0,
            intrinsics["cx"],
            0.0,
            0.0,
            intrinsics["fy"],
            intrinsics["cy"],
            0.0,
            0.0,
            0.0,
            1.0,
            0.0,
        ]

        # Distortion coefficients
        if "distortion" in intrinsics:
            info_msg.d = list(intrinsics["distortion"])

        # Distortion model
        info_msg.distortion_model = "plumb_bob"

        return info_msg

    @property
    def is_running(self) -> bool:
        """Check if camera is running"""
        return self.camera.is_running

    @property
    def is_connected(self) -> bool:
        """Check if camera is connected"""
        return self.camera.is_connected()


class RealSenseCameraPublisher(CameraPublisher):
    """
    Specialized RealSense camera publisher with additional RealSense-specific features.
    """

    def __init__(
        self,
        camera_name: str = "realsense_d405",
        camera_type: str = "threaded",  # Default to threaded for better performance
        publish_rate: float = 30.0,
        serial_number: Optional[str] = None,
        fps: int = 30,
        color_resolution: tuple = (640, 480),
        depth_resolution: tuple = (640, 480),
        enable_depth: bool = True,
        enable_color: bool = True,
        **kwargs,
    ):
        super().__init__(
            camera_name=camera_name,
            camera_type=camera_type,
            publish_rate=publish_rate,
            serial_number=serial_number,
            fps=fps,
            color_resolution=color_resolution,
            depth_resolution=depth_resolution,
            enable_depth=enable_depth,
            enable_color=enable_color,
            **kwargs,
        )

        self.get_logger().info(
            f"RealSense camera publisher initialized with "
            f"serial: {serial_number}, type: {camera_type}"
        )

    def get_camera_info(self) -> dict:
        """Get detailed camera information"""
        info = {
            "camera_name": self.camera_name,
            "camera_type": self.camera_type,
            "serial_number": self.camera_config.serial_number,
            "fps": self.camera_config.fps,
            "color_resolution": self.camera_config.color_resolution,
            "depth_resolution": self.camera_config.depth_resolution,
            "enable_color": self.camera_config.enable_color,
            "enable_depth": self.camera_config.enable_depth,
            "is_running": self.is_running,
            "is_connected": self.is_connected,
            "frame_count": self.camera.frame_count,
        }

        # Add intrinsics if available
        intrinsics = self.camera.get_intrinsics()
        if intrinsics:
            info["intrinsics"] = intrinsics

        return info


def main(args=None):
    """Main function to run the camera publisher as a standalone node"""
    rclpy.init(args=args)

    try:
        # Create camera publisher with default parameters
        camera_publisher = RealSenseCameraPublisher(
            camera_name="realsense_d405",
            camera_type="threaded",
            fps=30,
            enable_color=True,
            enable_depth=True,
        )

        if camera_publisher.start():
            rclpy.spin(camera_publisher)
        else:
            camera_publisher.get_logger().error("Failed to start camera publisher")

    except KeyboardInterrupt:
        pass
    except Exception as e:
        print(f"Error: {e}")
    finally:
        if "camera_publisher" in locals():
            camera_publisher.stop()
            camera_publisher.destroy_node()
        rclpy.shutdown()


if __name__ == "__main__":
    main()
