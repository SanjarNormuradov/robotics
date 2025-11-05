from PIL import Image, ImageOps
from pathlib import Path


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
        if fname.suffix.lower() not in (
            ".jpg",
            ".jpeg",
            ".png",
            ".bmp",
            ".tif",
            ".tiff",
        ):
            continue
        out_img = Image.open(fname)
        out_fpath = (
            dst / fname.name
            if not args.rename_files
            else dst
            / (
                f"{args.dst_name_prefix}_{start_num + fid:0{num_zeros+len(str(start_num))}d}{fname.suffix}"
            )
        )
        in_ext = fname.suffix.lower()[1:]  # without dot

        # Fix orientation via EXIF if needed
        out_img = ImageOps.exif_transpose(out_img)
        # If source is CMYK, convert to RGB
        if out_img.mode == "CMYK":
            out_img = out_img.convert("RGB")

        if args.change_ext:
            out_ext = args.change_ext.lower()
            assert out_ext in ["jpg", "png", "bmp", "tiff"], "Unsupported target format"
            # Save as JPG (lossy, no alpha)
            out_fpath = out_fpath.with_suffix(f".{out_ext}")

        if args.resize:
            width, height = args.resize
            out_img = out_img.resize((width, height))

        if args.dry_run:
            message = "[Dry Run] Would"
            if args.resize:
                message += f" resize {fname} to {width}x{height};"
            if args.change_ext:
                message += f" convert from {in_ext} to {out_ext};"
            message += f" save to {out_fpath}"
            print(message)
            continue

        if out_ext == "jpg":
            out_img.save(out_fpath, quality=95)
        elif out_ext == "png":
            out_img.save(out_fpath, optimize=True)
        else:
            out_img.save(out_fpath)

        message = f"{fname.stem}: "
        if args.resize:
            message += f" resized to {width}x{height};"
        if args.change_ext:
            message += f" converted from {in_ext} to {out_ext};"
        message += f" saved to {out_fpath}"
        print(message)


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
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    process_images(args)
