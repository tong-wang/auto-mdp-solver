"""Protocol evaluation for the random-action baseline."""

from secretary_benchmark_common import build_arg_parser, evaluate
from secretary_benchmark_random import make_policy


def main() -> None:
    args = build_arg_parser("Evaluate the random secretary baseline.").parse_args()
    evaluate("random", make_policy, args)


if __name__ == "__main__":
    main()
