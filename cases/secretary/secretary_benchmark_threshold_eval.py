"""Protocol evaluation for the literature-optimal threshold policy."""

from secretary_benchmark_common import build_arg_parser, evaluate
from secretary_benchmark_threshold import make_policy, optimal_skip_count


def main() -> None:
    args = build_arg_parser("Evaluate the optimal secretary threshold.").parse_args()
    print(f"optimal skip count: {optimal_skip_count(100)}")
    evaluate("threshold", make_policy, args)


if __name__ == "__main__":
    main()
