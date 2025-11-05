from PIL import Image, ImageOps, ImageEnhance
from pathlib import Path
import numpy as np
import cv2
import random
from typing import Optional, Tuple

SUPPORTED_EXTENSION_LIST = [".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff"]


def adjust_brightness_contrast(image, brightness_delta=0, contrast_scale=1.0):
    """
    Adjust brightness and contrast of an image.

    Args:
        image: PIL Image object
        brightness_delta: Amount to add to brightness (-100 to 100)
        contrast_scale: Factor to scale contrast (0.5 to 2.0)

    Returns:
        PIL Image with adjusted brightness and contrast
    """
    if brightness_delta == 0 and contrast_scale == 1.0:
        return image

    print(f"Applying brightness/contrast: B={brightness_delta}, C={contrast_scale:.2f}")

    # Apply brightness adjustment
    if brightness_delta != 0:
        enhancer = ImageEnhance.Brightness(image)
        # Convert delta to multiplier (0 = black, 1 = original, 2 = very bright)
        brightness_factor = 1.0 + (brightness_delta / 100.0)
        brightness_factor = max(0.0, brightness_factor)  # Ensure non-negative
        image = enhancer.enhance(brightness_factor)

    # Apply contrast adjustment
    if contrast_scale != 1.0:
        enhancer = ImageEnhance.Contrast(image)
        image = enhancer.enhance(contrast_scale)

    return image


def apply_gamma_correction(image, gamma=1.0):
    """
    Apply gamma correction to an image.

    Args:
        image: PIL Image object
        gamma: Gamma value (0.5 = darker, 1.0 = no change, 2.0 = brighter)

    Returns:
        PIL Image with gamma correction applied
    """
    if gamma == 1.0:
        return image

    print(f"Applying gamma correction: γ={gamma:.2f}")

    # Convert to numpy array
    img_array = np.array(image, dtype=np.float32)

    # Normalize to 0-1 range
    img_array = img_array / 255.0

    # Apply gamma correction
    img_corrected = np.power(img_array, gamma)

    # Convert back to 0-255 range
    img_corrected = (img_corrected * 255).astype(np.uint8)

    return Image.fromarray(img_corrected)


def adjust_hsv(image, hsv_delta: Tuple[float, float, float]):
    """
    Adjust HSV (fixed/random variations in color properties).

    Args:
        image: PIL Image object
        hsv_delta: Tuple of (hue_delta, saturation_delta, value_delta)

    Returns:
        PIL Image with color jittering applied
    """
    hue_delta, saturation_delta, value_delta, is_random = hsv_delta
    is_random = bool(is_random)
    print(
        f"Applying HSV adjustments: H={hue_delta}, S={saturation_delta:.2f}, V={value_delta:.2f}"
    )
    image = image.convert("RGB")
    # Convert PIL RGB to OpenCV format
    img_array = np.array(image)

    if is_random:
        hue_delta = random.uniform(-hue_delta, hue_delta)
        saturation_delta = random.uniform(-saturation_delta, saturation_delta)
        value_delta = random.uniform(-value_delta, value_delta)

    # Only process if we have a 3-channel RGB image
    if len(img_array.shape) == 3 and img_array.shape[2] == 3:
        # Convert RGB to BGR for OpenCV
        img_bgr = cv2.cvtColor(img_array, cv2.COLOR_RGB2BGR)
        # Convert to HSV
        hsv = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2HSV).astype(np.float32)

        # Adjust hue (wrap around 0-360)
        hsv[:, :, 0] = (hsv[:, :, 0] + hue_delta) % 360
        # Adjust saturation
        hsv[:, :, 1] = np.clip(hsv[:, :, 1] * (1.0 + saturation_delta), 0, 255)
        # Adjust value
        hsv[:, :, 2] = np.clip(hsv[:, :, 2] * (1.0 + value_delta), 0, 255)

        # Convert back to BGR then RGB
        hsv = hsv.astype(np.uint8)
        img_bgr_adjusted = cv2.cvtColor(hsv, cv2.COLOR_HSV2BGR)
        img_rgb_adjusted = cv2.cvtColor(img_bgr_adjusted, cv2.COLOR_BGR2RGB)

        return Image.fromarray(img_rgb_adjusted, "RGB")
    else:
        print(
            f"Warning: Cannot process image with shape {img_array.shape}, returning original"
        )
        return image


def process_images(args):
    src = Path(args.src_dir)
    file_list = sorted(src.iterdir())
    if args.src_name_prefix:
        file_list = [f for f in file_list if f.name.startswith(args.src_name_prefix)]
    if args.src_num_files > 0:
        if args.src_start_index >= 0:
            file_list = file_list[
                args.src_start_index : args.src_start_index + args.src_num_files
            ]
        else:
            file_list = file_list[: args.src_num_files]

    dst = Path(args.dst_dir)
    dst.mkdir(parents=True, exist_ok=True)
    num_zeros = 4
    start_num = 0
    if args.rename_files:
        if args.dst_name_suffix:
            _suff = args.dst_name_suffix.split("_")
            if len(_suff) != 2:
                raise ValueError("dst_name_suffix format is incorrect.")
            suffix_type = _suff[0]
            start_num = int(_suff[1])
            if "x" in suffix_type:
                num_zeros = int(suffix_type.split("x")[1])

    for fid, fname in enumerate(file_list):
        if fname.suffix.lower() not in SUPPORTED_EXTENSION_LIST:
            continue

        if args.rename_files:
            new_fname = f"{args.dst_name_prefix}_{start_num + fid:0{num_zeros+len(str(start_num))}d}{fname.suffix}"
        else:
            new_fname = fname.name
        out_fpath = dst / new_fname
        in_ext = fname.suffix.lower()[1:]  # without dot
        out_ext = in_ext
        width, height = args.resize if args.resize else (640, 480)

        if args.change_ext:
            out_ext = args.change_ext.lower()
            assert out_ext in ["jpg", "png", "bmp", "tiff"], "Unsupported target format"
            # Save as JPG (lossy, no alpha)
            out_fpath = out_fpath.with_suffix(f".{out_ext}")

        message = ""
        if args.resize:
            message += f" resize {fname} to {width}x{height};"
        if args.change_ext:
            message += f" convert from {in_ext} to {out_ext};"
        if args.adjust_brightness_contrast:
            if args.random_brightness_contrast:
                message += f" apply random brightness/contrast;"
            else:
                message += f" adjust brightness/contrast (B:{args.brightness_delta}, C:{args.contrast_scale:.2f});"
        if args.apply_gamma:
            if args.random_gamma:
                message += f" apply random gamma correction;"
            else:
                message += f" apply gamma correction (γ:{args.gamma:.2f});"
        if args.adjust_hsv:
            message += f" adjust HSV;"
        message += f" save to {out_fpath}"

        if args.dry_run:
            print(f"[Dry Run] Would {message}")
            continue

        out_img = Image.open(fname)
        # Fix orientation via EXIF if needed
        out_img = ImageOps.exif_transpose(out_img)
        # If source is CMYK, convert to RGB
        if out_img.mode == "CMYK":
            out_img = out_img.convert("RGB")

        if args.resize:
            out_img = out_img.resize((width, height))

        # # Apply brightness/contrast adjustments if specified
        # if args.adjust_brightness_contrast:
        #     if args.random_brightness_contrast:
        #         brightness_delta = random.randint(-30, 30)
        #         contrast_scale = random.uniform(0.7, 1.5)
        #     else:
        #         brightness_delta = args.brightness_delta
        #         contrast_scale = args.contrast_scale
        #     out_img = adjust_brightness_contrast(
        #         out_img, brightness_delta, contrast_scale
        #     )

        # # Apply gamma correction if specified
        # if args.apply_gamma:
        #     if args.random_gamma:
        #         gamma = random.uniform(0.7, 1.5)
        #     else:
        #         gamma = args.gamma
        #     out_img = apply_gamma_correction(out_img, gamma)

        # Apply HSV adjustments if specified
        if args.adjust_hsv:
            out_img = adjust_hsv(
                out_img,
                args.adjust_hsv,
            )

        if out_ext == "jpg":
            out_img.save(out_fpath, quality=95)
        elif out_ext == "png":
            out_img.save(out_fpath, optimize=True)
        else:
            out_img.save(out_fpath)

        print(f"{fname.stem}: {message}")


def parse_args():
    import argparse

    parser = argparse.ArgumentParser(
        description="Preprocess images: fix orientation, convert color modes, save as JPG."
    )
    parser.add_argument(
        "--src_dir",
        type=str,
        required=True,
        help="Source directory containing images to preprocess.",
    )
    parser.add_argument(
        "--dst_dir",
        type=str,
        required=True,
        help="Destination directory to save preprocessed images.",
    )
    parser.add_argument(
        "--src_name_prefix",
        type=str,
        default="",
        help="If specified, only process files starting with this prefix.",
    )
    parser.add_argument(
        "--src_num_files",
        type=int,
        default=-1,
        help="If specified (>0), only process this number of files from the source directory.",
    )
    parser.add_argument(
        "--src_start_index",
        type=int,
        default=-1,
        help="If specified (>0), only process files starting from this index.",
    )
    # Arguments for destination naming
    parser.add_argument(
        "--rename_files",
        action="store_true",
        help="If set, rename files in the destination directory according to specified prefix/suffix.",
    )
    parser.add_argument(
        "--dst_name_prefix",
        type=str,
        default="",
        help="If specified, save files with this prefix. Otherwise, original names are used.",
    )
    parser.add_argument(
        "--dst_name_suffix",
        type=str,
        default="",
        help="If specified, save files with this suffix. Otherwise, suffix is ascending order starting from 0 with 4 preceding zeros."
        "Available: "
        "0x1_0, 0x2_0, etc - number of zeros with order starting from 0, e.g. (0x3_0) 'img_0000', 'img_0001'"
        "0x1_1, 0x2_1, etc - number of zeros with order starting from 1, e.g. (0x3_1) 'img_0001', 'img_0002'"
        "0x1_50, 0x2_50, etc - number of zeros with order starting from 50, e.g. (0x3_50) 'img_00050', 'img_00051'",
    )
    parser.add_argument(
        "--change_ext",
        type=str,
        help="Specify input and output format conversion, e.g. 'png'. Available formats: jpg, png, bmp, tiff.",
    )
    parser.add_argument(
        "--resize",
        type=int,
        nargs=2,
        metavar=("WIDTH", "HEIGHT"),
        help="Resize images to the specified width and height.",
    )
    parser.add_argument(
        "--dry_run",
        action="store_true",
        help="If set, only print actions without saving files.",
    )

    # Brightness/Contrast adjustment arguments
    parser.add_argument(
        "--adjust_brightness_contrast",
        action="store_true",
        help="If set, adjust brightness and contrast of the images.",
    )
    parser.add_argument(
        "--random_brightness_contrast",
        action="store_true",
        help="If set, apply random brightness/contrast adjustments.",
    )
    parser.add_argument(
        "--brightness_delta",
        type=int,
        default=0,
        help="Brightness adjustment delta (-100 to 100). Only used if --random_brightness_contrast is not set.",
    )
    parser.add_argument(
        "--contrast_scale",
        type=float,
        default=1.0,
        help="Contrast scale factor (0.5 to 2.0). Only used if --random_brightness_contrast is not set.",
    )

    # Gamma correction arguments
    parser.add_argument(
        "--apply_gamma",
        action="store_true",
        help="If set, apply gamma correction to the images.",
    )
    parser.add_argument(
        "--random_gamma",
        action="store_true",
        help="If set, apply random gamma correction.",
    )
    parser.add_argument(
        "--gamma",
        type=float,
        default=1.0,
        help="Gamma correction value (0.5 to 2.0). Only used if --random_gamma is not set.",
    )

    # Color jittering arguments
    parser.add_argument(
        "--adjust_hsv",
        type=float,
        nargs=4,
        default=(0.0, 0.0, 0.0, 1.0),
        metavar=("hue_delta", "SATURATION_DELTA", "VALUE_DELTA", "IS_RANDOM"),
        help="If set, apply delta to HSV values of the images."
        "hue_delta: maximum hue adjustment in degrees (0-180). "
        "SATURATION_DELTA: maximum saturation scale adjustment (0.0-1.0). "
        "VALUE_DELTA: maximum value scale adjustment (0.0-1.0). "
        "IS_RANDOM: whether to apply random adjustments (0 or 1).",
    )

    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    process_images(args)
