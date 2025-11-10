from PIL import Image, ImageOps, ImageEnhance
from pathlib import Path
import numpy as np
import cv2
import random
from typing import Optional, Tuple

SUPPORTED_EXTENSION_LIST = [".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff"]
RANDOM_BRIGHTNESS_CONTRAST = [25.0, 0.25, 1.0]
RANDOM_GAMMA = [0.25, 1.0]
RANDOM_HSV = [10.0, 0.25, 0.25, 1.0]


def adjust_brightness_contrast(image, bc_delta: Tuple[float, float, float]):
    """
    Adjust brightness and contrast of an image.

    Args:
        image: PIL Image object
        bc_delta: Tuple of (brightness_delta, contrast_delta, is_random)

    Returns:
        PIL Image with adjusted brightness and contrast
    """
    b_delta, c_delta, is_random = bc_delta
    if b_delta == 0.0 and c_delta == 0.0:
        return image

    is_random = bool(is_random)
    if is_random:
        b_delta = random.randint(-abs(int(b_delta)), abs(int(b_delta)))
        c_delta = random.uniform(-abs(c_delta), abs(c_delta))
    b_scale = max(0.0, 1.0 + (b_delta / 100.0))
    c_scale = max(0.0, 1.0 + c_delta)
    print(f"Applying brightness/contrast: B={b_scale:.2f}, C={c_scale:.2f}")

    # Apply brightness adjustment
    enhancer = ImageEnhance.Brightness(image)
    image = enhancer.enhance(b_scale)
    # Apply contrast adjustment
    enhancer = ImageEnhance.Contrast(image)
    image = enhancer.enhance(c_scale)

    return image


def apply_gamma(image, gamma: Tuple[float, float]):
    """
    Apply gamma correction to an image.

    Args:
        image: PIL Image object
        gamma[0]: gamma delta (-0.5 = brighter, 0.0 = no change, 0.5 = darker)
        gamma[1]: whether gamma[0] sets max/min for random

    Returns:
        PIL Image with gamma correction applied
    """
    gamma_delta, is_random = gamma
    if gamma_delta == 0.0:
        return image

    is_random = bool(is_random)
    if is_random:
        gamma_delta = random.uniform(-abs(gamma_delta), abs(gamma_delta))
    _gamma = max(0.0, 1.0 + gamma_delta)
    print(f"Applying gamma correction: γ={_gamma:.2f}")

    img_array = np.array(image, dtype=np.float32)
    # Normalize to 0-1 range
    img_array = img_array / 255.0
    # Apply gamma correction
    img_corrected = np.power(img_array, _gamma)
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
    h_delta, s_delta, v_delta, is_random = hsv_delta
    if h_delta == 0.0 and s_delta == 0.0 and v_delta == 0.0:
        return image
    is_random = bool(is_random)

    image = image.convert("RGB")
    img_array = np.array(image)

    if is_random:
        h_delta, s_delta, v_delta = np.abs((h_delta, s_delta, v_delta))
        h_delta = random.uniform(-h_delta, h_delta)
        s_delta = random.uniform(-s_delta, s_delta)
        v_delta = random.uniform(-v_delta, v_delta)
    print(
        f"Applying HSV adjustments: H={h_delta:.2f}, S={s_delta:.2f}, V={v_delta:.2f}"
    )

    # Only process if we have a 3-channel RGB image
    if len(img_array.shape) == 3 and img_array.shape[2] == 3:
        img_bgr = cv2.cvtColor(img_array, cv2.COLOR_RGB2BGR)
        hsv = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2HSV).astype(np.float32)

        # Adjust hue (wrap around 0-360)
        hsv[:, :, 0] = (hsv[:, :, 0] + h_delta) % 360
        # Adjust saturation
        hsv[:, :, 1] = np.clip(hsv[:, :, 1] * (1.0 + s_delta), 0, 255)
        # Adjust value
        hsv[:, :, 2] = np.clip(hsv[:, :, 2] * (1.0 + v_delta), 0, 255)

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
        if np.any(args.adjust_brightness_contrast[:2]):
            message += f" adjust brightness/contrast;"
        if args.random_brightness_contrast:
            message += f" adjust brightness/contrast randomly;"
        if args.apply_gamma[0] != 0.0:
            message += f" apply gamma correction;"
        if args.random_gamma:
            message += f" apply gamma randomly;"
        if np.any(args.adjust_hsv[:3]):
            message += f" adjust HSV;"
        if args.random_hsv:
            message += f" adjust HSV randomly;"
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

        # Apply brightness/contrast adjustments
        if args.adjust_brightness_contrast:
            out_img = adjust_brightness_contrast(
                out_img, args.adjust_brightness_contrast
            )
        if args.random_brightness_contrast or args.random_augments:
            out_img = adjust_brightness_contrast(out_img, RANDOM_BRIGHTNESS_CONTRAST)

        # Apply gamma correction
        if args.apply_gamma:
            out_img = apply_gamma(out_img, args.apply_gamma)
        if args.random_gamma or args.random_augments:
            out_img = apply_gamma(out_img, RANDOM_GAMMA)

        # Apply HSV adjustments
        if args.adjust_hsv:
            out_img = adjust_hsv(
                out_img,
                args.adjust_hsv,
            )
        if args.random_hsv or args.random_augments:
            out_img = adjust_hsv(out_img, RANDOM_HSV)

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
        help="If set, only print actions without execution.",
    )
    # Brightness/Contrast adjustment arguments
    parser.add_argument(
        "--adjust_brightness_contrast",
        type=float,
        nargs=3,
        default=(0.0, 0.0, 1.0),
        metavar=("BRIGHTNESS_DELTA", "CONTRAST_DELTA", "IS_RANDOM"),
        help="If set, apply brightness and contrast adjustments."
        "BRIGHTNESS_DELTA: (-100 to 100); -25 darker, 0 no change, 25 brighter."
        "CONTRAST_DELTA: (-1.0, 1.0); -0.5 lower, 0 no change, 0.5 higher."
        "IS_RANDOM: whether *_DELTA sets max/min value for random.",
    )
    parser.add_argument(
        "--random_brightness_contrast",
        action="store_true",
        help="If set, use default random brightness/contrast adjustments.",
    )
    # Gamma correction arguments
    parser.add_argument(
        "--apply_gamma",
        type=float,
        nargs=2,
        default=(0.0, 1.0),
        metavar=("GAMMA_DELTA", "IS_RANDOM"),
        help="If set, apply gamma correction."
        "GAMMA_DELTA: (-1.0, 1.0); -0.5 brighter, 0 no change, 0.5 darker."
        "IS_RANDOM: whether GAMMA_DELTA sets max/min value for random.",
    )
    parser.add_argument(
        "--random_gamma",
        action="store_true",
        help="If set, use default random gamma adjustment.",
    )
    # Color jittering arguments
    parser.add_argument(
        "--adjust_hsv",
        type=float,
        nargs=4,
        default=(0.0, 0.0, 0.0, 1.0),
        metavar=("HUE_DELTA", "SATURATION_DELTA", "VALUE_DELTA", "IS_RANDOM"),
        help="If set, apply delta to HSV values of the images."
        "HUE_DELTA: maximum hue adjustment in degrees (0-360). "
        "SATURATION_DELTA: maximum saturation scale adjustment (-1.0, 1.0). "
        "VALUE_DELTA: maximum value scale adjustment (-1.0, 1.0). "
        "IS_RANDOM: whether *_DELTA sets max/min value for random.",
    )
    parser.add_argument(
        "--random_hsv",
        action="store_true",
        help="If set, use default random HSV adjustments.",
    )
    # Random augmentations
    parser.add_argument(
        "--random_augments",
        action="store_true",
        help="If set, use default random augmentations (brightness, contrast, gamma, HSV).",
    )

    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    process_images(args)
