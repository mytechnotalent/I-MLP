#!/usr/bin/env python3

"""
Script to convert Jupyter Notebook (.ipynb) files to Markdown using jupyter nbconvert.
Requires Jupyter (nbconvert) to be installed in the environment.

Usage:
    python convert_ipynb_to_md.py <input_ipynb> [output_md]

If `output_md` is not provided, the Markdown file will be created next to the input
notebook with the same base name and a .md extension. Any accompanying resources
images folder produced by nbconvert (e.g. `notebook_files/`) will be moved/renamed
to match the output basename if an explicit output path is provided.
"""

import subprocess
import sys
import os
import shutil
import re


def ensure_nbconvert_installed():
    """
    Ensure that `nbconvert` / `jupyter` is available in the current Python environment.

    Attempts to import `nbconvert` and, if missing, invokes pip to install
    `nbconvert` and `jupyter` into the active Python interpreter. Returns
    True if nbconvert appears available after this call, False otherwise.
    """
    try:
        import nbconvert  # type: ignore

        return True
    except Exception:
        print("`nbconvert` not found — attempting to install via pip...")
        try:
            # Upgrade pip first to reduce install issues
            subprocess.check_call(
                [sys.executable, "-m", "pip", "install", "--upgrade", "pip"]
            )
            subprocess.check_call(
                [sys.executable, "-m", "pip", "install", "nbconvert", "jupyter"]
            )
        except subprocess.CalledProcessError as e:
            print(f"Automatic pip install failed: {e}")
            return False

        # Re-check import
        try:
            import importlib

            importlib.invalidate_caches()
            import nbconvert  # type: ignore

            return True
        except Exception:
            return False


def convert_ipynb_to_md(input_file, output_file=None):
    """
    Convert an .ipynb file to Markdown using `jupyter nbconvert`.

    Args:
        input_file (str): Path to the input .ipynb file.
        output_file (str, optional): Desired output .md file path. If omitted,
                                     the .md will be written alongside the input
                                     file using the same base name.

    Returns:
        bool: True on success, False on failure.
    """
    if not os.path.exists(input_file):
        print(f"Error: Input file '{input_file}' does not exist.")
        return False

    # Ensure nbconvert/jupyter is available (try to install if missing)
    if not ensure_nbconvert_installed():
        print(
            "Error: 'nbconvert' (jupyter) is not available and automatic install failed."
        )
        print("Install manually with: pip install nbconvert jupyter")
        return False

    input_file = os.path.abspath(input_file)
    input_dir = os.path.dirname(input_file) or "."
    input_base = os.path.splitext(os.path.basename(input_file))[0]

    # Determine target output path and base names
    if output_file:
        output_file = os.path.abspath(output_file)
        out_dir = os.path.dirname(output_file) or "."
        out_base = os.path.splitext(os.path.basename(output_file))[0]
        if os.path.splitext(output_file)[1].lower() != ".md":
            output_file = os.path.join(out_dir, out_base + ".md")
    else:
        output_file = os.path.join(input_dir, input_base + ".md")
        out_dir = input_dir
        out_base = input_base

    # Run nbconvert in the input file's directory so relative resources are created there
    try:
        # Use the active Python interpreter to run nbconvert as a module so
        # we don't rely on `jupyter` being on PATH (some installs put scripts
        # into a bin directory not on PATH).
        result = subprocess.run(
            [
                sys.executable,
                "-m",
                "nbconvert",
                "--to",
                "markdown",
                os.path.basename(input_file),
                "--output",
                input_base,
            ],
            cwd=input_dir,
            capture_output=True,
            text=True,
        )

        if result.returncode != 0:
            print("Error: jupyter nbconvert failed:")
            print(result.stderr)
            return False

        generated_md = os.path.join(input_dir, input_base + ".md")
        generated_resources = os.path.join(input_dir, input_base + "_files")

        # Fix resource links inside the generated markdown so that image
        # references point to the desired output resource folder name
        try:
            if os.path.exists(generated_md):
                with open(generated_md, "r", encoding="utf-8") as fh:
                    md_text = fh.read()

                # Remove any <style>...</style> blocks that nbconvert may embed
                # (commonly produced by pandas DataFrame HTML output). This
                # prevents CSS like the `.dataframe` block from appearing in
                # the final Markdown.
                md_text = re.sub(
                    r"<style.*?>.*?</style>\s*", "", md_text, flags=re.S | re.I
                )

                # Replace occurrences like "inputbase_files/" and "./inputbase_files/"
                src_folder = input_base + "_files"
                dst_folder = out_base + "_files"
                if src_folder != dst_folder:
                    md_text = md_text.replace(src_folder + "/", dst_folder + "/")
                    md_text = md_text.replace("./" + src_folder + "/", dst_folder + "/")
                    with open(generated_md, "w", encoding="utf-8") as fh:
                        fh.write(md_text)
        except Exception as e:
            # Non-fatal: continue and move files if possible
            print(f"Warning: failed to rewrite resource links: {e}")

        # If the desired output path differs from the generated one, move/rename
        if os.path.abspath(generated_md) != os.path.abspath(output_file):
            # Ensure output directory exists
            os.makedirs(os.path.dirname(output_file), exist_ok=True)
            shutil.move(generated_md, output_file)

        # Move/rename resources folder if present
        if os.path.isdir(generated_resources):
            desired_resources = os.path.join(
                os.path.dirname(output_file), out_base + "_files"
            )
            # If resource folder already exists at destination, remove it first
            if os.path.isdir(desired_resources):
                shutil.rmtree(desired_resources)
            shutil.move(generated_resources, desired_resources)

        print(f"Successfully converted '{input_file}' → '{output_file}'")
        return True

    except FileNotFoundError:
        print("Error: 'jupyter' (nbconvert) is not installed or not on PATH.")
        print("Install with: pip install nbconvert jupyter")
        return False
    except Exception as e:
        print(f"An unexpected error occurred: {e}")
        return False


def main():
    if len(sys.argv) < 2:
        print("Usage: python convert_ipynb_to_md.py <input_ipynb> [output_md]")
        sys.exit(1)

    input_file = sys.argv[1]
    output_file = sys.argv[2] if len(sys.argv) > 2 else None

    ok = convert_ipynb_to_md(input_file, output_file)
    sys.exit(0 if ok else 1)


if __name__ == "__main__":
    main()
