import os
import time

import cv2

from common.image_visualizer import ImageVisualizer
from common.logging_manager import get_logger, setup_logging
from hardware.camera.realsense import RealSenseConfig, RealSenseThreaded


def main(args):
    save_dir = args.save_dir
    os.makedirs(save_dir, exist_ok=True)
    setup_logging(log_dir="logs", log_file_prefix="camera_recording_")
    config = RealSenseConfig(
        camera_name=args.camera_name,
        fps=args.fps,
        color_resolution=(args.color_width, args.color_height),
        depth_resolution=(args.depth_width, args.depth_height),
        serial_number=args.serial_number,
        enable_color=args.enable_color,
        enable_depth=args.enable_depth,
    )
    camera = RealSenseThreaded(config)
    visualizer = ImageVisualizer(
        window_name="Camera Feed",
        custom_text="Camera Frame",
        window_size=(args.color_width, args.color_height),
        text_color=(0, 255, 255),  # Yellow text
        text_size=0.8,
    )
    logger = get_logger("camera_recording")

    if not camera.start():
        logger.error("Failed to start camera")
        return
    save_images = False
    image_count = 0
    try:
        while True:
            frame = camera.get_frame()
            if frame and frame.color_image is not None:
                key = visualizer.show_image(frame.color_image, wait_key=False)
                if key == ord("q") or key == 27:  # ESC or 'q' key to exit
                    logger.info("Exiting...")
                    break
                elif key == ord("s"):
                    save_images = not save_images
                    logger.info(
                        f"Image saving {'enabled' if save_images else 'disabled'}"
                    )
                if save_images:
                    filename = os.path.join(save_dir, f"frame_{image_count}.png")
                    cv2.imwrite(filename, frame.color_image)
                    image_count += 1
                time.sleep(0.001)
    finally:
        camera.stop()
        visualizer.close()


def parse_args():
    import argparse

    parser = argparse.ArgumentParser(description="Test RealSense D405 Camera Recording")
    parser.add_argument(
        "--save-dir",
        type=str,
        default="data/images",
        help="Directory to save recordings",
    )
    parser.add_argument(
        "--camera-name", type=str, default="my_camera", help="Name of the camera"
    )
    parser.add_argument("--fps", type=int, default=60, help="Frames per second")
    parser.add_argument(
        "--color-width", type=int, default=640, help="Color resolution width"
    )
    parser.add_argument(
        "--color-height", type=int, default=480, help="Color resolution height"
    )
    parser.add_argument(
        "--depth-width", type=int, default=640, help="Depth resolution width"
    )
    parser.add_argument(
        "--depth-height", type=int, default=480, help="Depth resolution height"
    )
    parser.add_argument(
        "--serial-number", type=str, default="", help="Camera serial number"
    )
    parser.add_argument(
        "--enable-color", action="store_true", help="Enable color stream"
    )
    parser.add_argument(
        "--enable-depth", action="store_true", help="Enable depth stream"
    )
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    main(args)
