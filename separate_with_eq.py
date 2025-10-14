from separation_utils import process_stems
from pathlib import Path
from typing import Union, Optional
import argparse


def separate_with_eq(
    input_dir: Union[str, Path],
    output_dir: Union[str, Path],
    wiener_exponent: Optional[float],
    device: str,
    apply_eq: bool = True,
    eq_config_path: str = "eq.yaml"
):
    """
    Separate drums with optional post-processing EQ to reduce bleed.
    
    Args:
        input_dir: Directory containing drum mixes
        output_dir: Directory to save separated stems
        wiener_exponent: Wiener filter exponent (None to disable)
        device: 'cpu' or 'cuda'
        apply_eq: Whether to apply frequency cleanup
        eq_config_path: Path to EQ configuration file
    """
    process_stems(
        input_dir=input_dir,
        output_dir=output_dir,
        wiener_exponent=wiener_exponent,
        device=device,
        apply_eq=apply_eq,
        eq_config_path=eq_config_path,
        verbose=True
    )


if __name__ == '__main__':
    parser = argparse.ArgumentParser(
        description="Separate drums with post-processing EQ to reduce bleed between stems."
    )
    parser.add_argument('-i', '--input_dir', type=str, required=True,
                        help="Path to the root directory where to find the target drum mixtures.")
    parser.add_argument('-o', '--output_dir', type=str, default='separated_stems_eq',
                        help="Path to the directory where to save the separated tracks.")
    parser.add_argument('-w', '--wiener_exponent', type=float, default=1.5,
                        help="Positive α-Wiener filter exponent (float). Recommended: 1.0-2.0. Use 0 to disable.")
    parser.add_argument('-d', '--device', type=str, default='cpu',
                        help="Torch device. Use 'cuda' if available for faster processing.")
    parser.add_argument('--no-eq', action='store_true',
                        help="Disable post-processing EQ cleanup.")
    parser.add_argument('--eq-config', type=str, default='eq.yaml',
                        help="Path to EQ configuration YAML file.")

    args = parser.parse_args()
    
    # Convert wiener_exponent=0 to None
    wiener_exp = None if args.wiener_exponent == 0 else args.wiener_exponent

    separate_with_eq(
        input_dir=args.input_dir,
        output_dir=args.output_dir,
        wiener_exponent=wiener_exp,
        device=args.device,
        apply_eq=not args.no_eq,
        eq_config_path=args.eq_config
    )
