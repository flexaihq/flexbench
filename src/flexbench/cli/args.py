"""CLI argument parser for FlexBench."""

from flexbench.args import create_cli_parser, validate_args


def get_cli_args():
    """Get CLI arguments."""
    parser = create_cli_parser()
    args = parser.parse_args()
    return validate_args(args)
