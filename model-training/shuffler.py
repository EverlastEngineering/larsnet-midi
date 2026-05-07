import random
import sys
from pathlib import Path

def shuffle_manifest(input_path, output_path=None):
    """
    Reads a tab-separated manifest file, shuffles the lines, 
    and writes to a new file.
    """
    input_file = Path(input_path)
    if not input_file.exists():
        print(f"Error: {input_path} not found.")
        return

    # Read lines and strip empty ones
    with open(input_file, 'r') as f:
        lines = [line.strip() for line in f if line.strip()]

    if not lines:
        print("File is empty.")
        return

    # Shuffle in place
    random.seed(42)  # Seed for reproducibility, remove for true random
    random.shuffle(lines)

    # Determine output path
    if output_path is None:
        output_path = input_file.parent / f"{input_file.stem}_shuffled{input_file.suffix}"

    # Write shuffled lines
    with open(output_path, 'w') as f:
        for line in lines:
            f.write(line + '\n')

    print(f"Successfully shuffled {len(lines)} pairs.")
    print(f"Output saved to: {output_path}")

if __name__ == "__main__":
    # Usage: python shuffle_manifest.py batch1.txt
    file_to_shuffle = sys.argv[1] if len(sys.argv) > 1 else "batch1.txt"
    shuffle_manifest(file_to_shuffle)