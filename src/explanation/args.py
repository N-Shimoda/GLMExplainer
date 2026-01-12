import argparse


def check_non_negative_int(value: str) -> int:
    try:
        ivalue = int(value)
    except ValueError:
        raise argparse.ArgumentTypeError(f"`{value}` is not an integer.")
    if ivalue < 0:
        raise argparse.ArgumentTypeError("Value must be non-negative.")
    return ivalue
