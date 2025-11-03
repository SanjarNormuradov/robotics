"""
Simplified image visualization utility for displaying a single image with custom text.
"""

from typing import Optional, Tuple

import cv2
import numpy as np


class ImageVisualizer:
    """
    A simplified class for visualizing a single image with custom text using OpenCV.
    """

    def __init__(
        self,
        window_name: str = "Image",
        custom_text: str = "Frame",
        window_size: Optional[Tuple[int, int]] = None,
        text_color: Tuple[int, int, int] = (255, 255, 255),
        text_size: float = 0.6,
    ):
        """
        Initialize the image visualizer.

        Args:
            window_name: Name of the OpenCV window
            custom_text: Custom text to display on the image
            window_size: Optional (width, height) to resize images
            text_color: RGB color for the text (default: white)
            text_size: Font size for the text (default: 0.6)
        """
        self.window_name = window_name
        self.custom_text = custom_text
        self.window_size = window_size
        self.text_color = text_color
        self.text_size = text_size
        self.frame_count = 0

    def _add_text_overlay(self, img: np.ndarray) -> np.ndarray:
        """
        Add custom text overlay to the image.

        Args:
            img: Input image (BGR format for OpenCV)

        Returns:
            Image with text overlay
        """
        try:
            # Create a copy to avoid modifying the original
            img_with_text = img.copy()

            # Add semi-transparent background for text
            overlay = img_with_text.copy()

            # Calculate text position and background size
            text_with_frame = f"{self.custom_text} {self.frame_count}"
            text_size = cv2.getTextSize(
                text_with_frame, cv2.FONT_HERSHEY_SIMPLEX, self.text_size, 2
            )[0]

            # Add background rectangle
            cv2.rectangle(
                overlay, (10, 10), (text_size[0] + 20, text_size[1] + 20), (0, 0, 0), -1
            )

            # Blend the overlay with the image
            cv2.addWeighted(overlay, 0.7, img_with_text, 0.3, 0, img_with_text)

            # Add the text
            cv2.putText(
                img_with_text,
                text_with_frame,
                (15, 15 + text_size[1]),
                cv2.FONT_HERSHEY_SIMPLEX,
                self.text_size,
                self.text_color,
                2,
            )

            return img_with_text

        except Exception as e:
            print(f"Error adding text overlay: {e}")
            return img

    def show_image(
        self, image: np.ndarray, add_text: bool = True, wait_key: bool = True
    ) -> int:
        """
        Display an image in the OpenCV window.

        Args:
            image: Input image (can be RGB or BGR)
            add_text: Whether to add the custom text overlay
            wait_key: Whether to wait for a key press

        Returns:
            Key code pressed (if wait_key=True), otherwise 0
        """
        try:
            # Make a copy to avoid modifying the original
            display_image = image.copy()

            # Resize image if window size is specified
            if self.window_size is not None:
                display_image = cv2.resize(display_image, self.window_size)

            # Add text overlay if requested
            if add_text:
                display_image = self._add_text_overlay(display_image)
                self.frame_count += 1

            # Display the image
            cv2.imshow(self.window_name, display_image)

            # Wait for key press if requested
            if wait_key:
                return cv2.waitKey(0) & 0xFF
            else:
                return cv2.waitKey(1) & 0xFF

        except Exception as e:
            print(f"Error displaying image: {e}")
            return -1

    def update_text(self, new_text: str):
        """
        Update the custom text to display on images.

        Args:
            new_text: New custom text
        """
        self.custom_text = new_text

    def reset_frame_count(self):
        """Reset the frame counter to 0."""
        self.frame_count = 0

    def close(self):
        """Close the OpenCV window."""
        cv2.destroyWindow(self.window_name)

    @staticmethod
    def close_all():
        """Close all OpenCV windows."""
        cv2.destroyAllWindows()


def show_image(
    image: np.ndarray,
    window_name: str = "Image",
    text: Optional[str] = None,
    wait_key: bool = True,
) -> int:
    """
    Simple function to quickly display an image with optional text.

    Args:
        image: Input image to display
        window_name: Name of the window
        text: Optional text to display on the image
        wait_key: Whether to wait for key press

    Returns:
        Key code pressed (if wait_key=True), otherwise 0
    """
    if text is None:
        cv2.imshow(window_name, image)
    else:
        visualizer = ImageVisualizer(window_name=window_name, custom_text=text)
        return visualizer.show_image(image, add_text=True, wait_key=wait_key)

    if wait_key:
        return cv2.waitKey(0) & 0xFF
    else:
        return cv2.waitKey(1) & 0xFF


# Example usage
if __name__ == "__main__":
    # Create a sample image
    sample_image = np.zeros((480, 640, 3), dtype=np.uint8)
    sample_image[:] = (50, 50, 150)  # Dark red background

    # Create visualizer
    visualizer = ImageVisualizer(
        window_name="Test Window", custom_text="Test Image", window_size=(800, 600)
    )

    # Show the image
    print("Press any key to continue...")
    key = visualizer.show_image(sample_image)
    print(f"Key pressed: {chr(key) if key != 27 else 'ESC'}")

    # Update text and show again
    visualizer.update_text("Updated Text")
    key = visualizer.show_image(sample_image)

    # Clean up
    visualizer.close()
