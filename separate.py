from separation_utils import process_stems
from pathlib import Path
from typing import Union, Optional
import argparse


def separate(input_dir: Union[str, Path], output_dir: Union[str, Path], wiener_exponent: Optional[float], device: str):
    """
    Separate drums without post-processing EQ.
    
    This is a simple wrapper around process_stems with EQ disabled.
    """
    process_stems(
        input_dir=input_dir,
        output_dir=output_dir,
        wiener_exponent=wiener_exponent,
        device=device,
        apply_eq=False,
        verbose=True
    )


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('-i', '--input_dir', type=str, required=True, help="Path to the root directory where to find the target drum mixtures.")
    parser.add_argument('-o', '--output_dir', type=str, default='separated_stems', help="Path to the directory where to save the separated tracks.")
    parser.add_argument('-w', '--wiener_exponent', type=float, default=None, help="Positive α-Wiener filter exponent (float). Use it only if Wiener filtering is to be applied.")
    parser.add_argument('-d', '--device', type=str, default='cpu', help="Torch device. Default 'cpu'")

    args = vars(parser.parse_args())

    separate(
        input_dir=args['input_dir'],
        output_dir=args['output_dir'],
        wiener_exponent=args['wiener_exponent'],
        device=args['device']
    )
