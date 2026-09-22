"""Protocol evaluation for the structural rule fitted from PPO."""

from __future__ import annotations

from secretary_benchmark_common import build_arg_parser, evaluate
from secretary_benchmark_fitted import make_policy_from_file


def main() -> None:
    parser = build_arg_parser("Evaluate the fitted secretary rule.")
    parser.add_argument("--fit-path", required=True)
    args = parser.parse_args()
    evaluate("fitted", make_policy_from_file(args.fit_path), args)


if __name__ == "__main__":
    main()
