#!/usr/bin/env python3
"""
Utility script for renaming image files in batch.
Supports various renaming patterns and operations.
"""

import os
import re
import argparse
from pathlib import Path
from typing import List, Tuple, Optional


class FileRenamer:
    def __init__(self, directory: str, dry_run: bool = True):
        """
        Initialize the file renamer.

        Args:
            directory: Path to the directory containing files to rename
            dry_run: If True, only show what would be renamed without actually doing it
        """
        self.directory = Path(directory)
        self.dry_run = dry_run

    def get_image_files(self, pattern: str = "*.png") -> List[Path]:
        """Get all image files matching the pattern."""
        return list(self.directory.glob(pattern))

    def rename_with_pattern(
        self, old_pattern: str, new_pattern: str, file_extension: str = ".png"
    ) -> List[Tuple[str, str]]:
        """
        Rename files using regex patterns.

        Args:
            old_pattern: Regex pattern to match current filenames
            new_pattern: New naming pattern (can use regex groups like \1, \2)
            file_extension: File extension to work with

        Returns:
            List of (old_name, new_name) tuples
        """
        files = self.get_image_files(f"*{file_extension}")
        renamed_files = []

        for file_path in files:
            filename = file_path.name
            match = re.match(old_pattern, filename)

            if match:
                # Replace the pattern with new pattern
                new_name = re.sub(old_pattern, new_pattern, filename)
                old_path = file_path
                new_path = file_path.parent / new_name

                renamed_files.append((str(old_path), str(new_path)))

                if not self.dry_run:
                    old_path.rename(new_path)
                    print(f"Renamed: {filename} -> {new_name}")
                else:
                    print(f"Would rename: {filename} -> {new_name}")

        return renamed_files

    def rename_sequential(
        self,
        prefix: str = "image",
        num_items: int = 0,
        start_id: int = 0,
        start_num: int = 0,
        padding: int = 4,
        file_extension: str = ".png",
    ) -> List[Tuple[str, str]]:
        """
        Rename files to sequential numbering.

        Args:
            prefix: Prefix for new filenames
            start_num: Starting number for sequence
            padding: Number of digits to pad (e.g., 4 -> 0001)
            file_extension: File extension

        Returns:
            List of (old_name, new_name) tuples
        """
        files = sorted(self.get_image_files(f"*{file_extension}"))
        if num_items > 0:
            files = files[start_id : start_id + num_items]
        renamed_files = []

        for i, file_path in enumerate(files):
            filename = file_path.name
            new_name = f"{prefix}_{str(start_num + i).zfill(padding)}{file_extension}"
            old_path = file_path
            new_path = file_path.parent / new_name

            renamed_files.append((str(old_path), str(new_path)))

            if not self.dry_run:
                old_path.rename(new_path)
                print(f"Renamed: {filename} -> {new_name}")
            else:
                print(f"Would rename: {filename} -> {new_name}")

        return renamed_files

    def rename_replace_text(
        self, old_text: str, new_text: str, file_extension: str = ".png"
    ) -> List[Tuple[str, str]]:
        """
        Simple text replacement in filenames.

        Args:
            old_text: Text to replace
            new_text: Replacement text
            file_extension: File extension

        Returns:
            List of (old_name, new_name) tuples
        """
        files = self.get_image_files(f"*{file_extension}")
        renamed_files = []

        for file_path in files:
            filename = file_path.name

            if old_text in filename:
                new_name = filename.replace(old_text, new_text)
                old_path = file_path
                new_path = file_path.parent / new_name

                renamed_files.append((str(old_path), str(new_path)))

                if not self.dry_run:
                    old_path.rename(new_path)
                    print(f"Renamed: {filename} -> {new_name}")
                else:
                    print(f"Would rename: {filename} -> {new_name}")

        return renamed_files


def main():
    parser = argparse.ArgumentParser(description="Batch rename image files")
    parser.add_argument("dir", help="Directory containing files to rename")
    parser.add_argument(
        "--dry-run",
        action="store_true",
        default=True,
        help="Show what would be renamed without actually doing it",
    )
    parser.add_argument(
        "--execute",
        action="store_true",
        help="Actually perform the renaming (overrides --dry-run)",
    )

    # Renaming methods
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument(
        "--pattern",
        nargs=2,
        metavar=("OLD", "NEW"),
        help="Rename using regex pattern. OLD=regex pattern, NEW=replacement",
    )
    group.add_argument(
        "--sequential",
        nargs="*",
        help="Rename to sequential numbers: [prefix] [num_items] [start_id] [start_num] [padding]",
    )
    group.add_argument(
        "--replace",
        nargs=2,
        metavar=("OLD", "NEW"),
        help="Replace text in filenames. OLD=text to replace, NEW=replacement",
    )

    parser.add_argument(
        "--extension", default=".png", help="File extension to work with"
    )

    args = parser.parse_args()

    # Set dry_run based on arguments
    dry_run = args.dry_run and not args.execute

    renamer = FileRenamer(args.dir, dry_run=dry_run)

    if args.pattern:
        old_pattern, new_pattern = args.pattern
        renamer.rename_with_pattern(old_pattern, new_pattern, args.extension)

    elif args.sequential is not None:
        # Parse sequential arguments
        prefix = args.sequential[0] if len(args.sequential) > 0 else "image"
        num_items = int(args.sequential[1]) if len(args.sequential) > 1 else 0
        start_id = int(args.sequential[2]) if len(args.sequential) > 2 else 0
        start_num = int(args.sequential[3]) if len(args.sequential) > 3 else 0
        padding = int(args.sequential[4]) if len(args.sequential) > 4 else 4
        renamer.rename_sequential(
            prefix, num_items, start_id, start_num, padding, args.extension
        )

    elif args.replace:
        old_text, new_text = args.replace
        renamer.rename_replace_text(old_text, new_text, args.extension)


if __name__ == "__main__":
    main()
