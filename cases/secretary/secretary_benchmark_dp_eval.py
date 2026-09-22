"""Protocol evaluation for the backward-induction optimum."""

from secretary_benchmark_common import build_arg_parser, evaluate
from secretary_benchmark_dp import make_policy, solve


def main() -> None:
    args = build_arg_parser("Evaluate the exact secretary DP.").parse_args()
    solution = solve(100)
    print(f"DP skip count: {solution.skip_count}; value={solution.value[1]:.12f}")
    evaluate("dp", make_policy, args)


if __name__ == "__main__":
    main()
