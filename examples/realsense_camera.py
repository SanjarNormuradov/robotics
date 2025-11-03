"""
Example usage of the refactored RealSense D405 camera classes.
This demonstrates how the new multiple inheritance design works.
"""

from hardware.camera.realsense import (
    RealSenseConfig,
    RealSenseD405Process,
    RealSenseD405Simple,
    RealSenseD405Threaded,
)


def main():
    """Example usage of the different RealSense implementations"""

    # Create configuration
    config = RealSenseConfig(
        camera_name="test_camera",
        fps=30,
        color_resolution=(640, 480),
        depth_resolution=(640, 480),
        enable_color=True,
        enable_depth=True,
        save_intrinsics=True,
        serial_number=None,  # Use any available camera
    )

    # Test Simple implementation
    print("Testing Simple implementation...")
    simple_camera = RealSenseD405Simple(config)

    if simple_camera.is_connected():
        print("Camera is connected")

        if simple_camera.start():
            print("Simple camera started successfully")

            # Get a few frames
            for i in range(5):
                frame = simple_camera.get_frame()
                if frame:
                    print(
                        f"Frame {i}: Color: {frame.color_image is not None}, "
                        f"Depth: {frame.depth_image is not None}, "
                        f"Timestamp: {frame.timestamp}"
                    )
                else:
                    print(f"Frame {i}: No frame received")

            # Get intrinsics
            intrinsics = simple_camera.get_intrinsics()
            if intrinsics:
                print(f"Intrinsics keys: {list(intrinsics.keys())}")

            simple_camera.stop()
            print("Simple camera stopped")
        else:
            print("Failed to start simple camera")
    else:
        print("No RealSense camera connected")

    # Test Threaded implementation
    print("\nTesting Threaded implementation...")
    threaded_camera = RealSenseD405Threaded(config)

    if threaded_camera.is_connected():
        if threaded_camera.start():
            print("Threaded camera started successfully")

            import time

            # Let it capture for a bit
            time.sleep(1.0)

            # Get a few frames
            for i in range(5):
                frame = threaded_camera.get_frame()
                if frame:
                    print(
                        f"Frame {i}: Color: {frame.color_image is not None}, "
                        f"Depth: {frame.depth_image is not None}, "
                        f"Frame count: {threaded_camera.frame_count}"
                    )
                else:
                    print(f"Frame {i}: No frame received")
                time.sleep(0.1)

            threaded_camera.stop()
            print("Threaded camera stopped")
        else:
            print("Failed to start threaded camera")

    # Test Process implementation
    print("\nTesting Process implementation...")
    process_camera = RealSenseD405Process(config)

    if process_camera.is_connected():
        if process_camera.start():
            print("Process camera started successfully")

            import time

            # Let it capture for a bit
            time.sleep(1.0)

            # Get a few frames
            for i in range(5):
                frame = process_camera.get_frame()
                if frame:
                    print(
                        f"Frame {i}: Color: {frame.color_image is not None}, "
                        f"Depth: {frame.depth_image is not None}"
                    )
                else:
                    print(f"Frame {i}: No frame received")
                time.sleep(0.1)

            process_camera.stop()
            print("Process camera stopped")
        else:
            print("Failed to start process camera")


if __name__ == "__main__":
    main()
