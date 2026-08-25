"""`geoloc-agent-eval`: pattern comparison -> reports/agent_eval.md"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

from geoloc_agent.agent.eval import build_scenarios, run_all
from geoloc_agent.agent.patterns import PATTERNS
from geoloc_agent.agent.report import build_agent_report


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="geoloc-agent-eval", description=__doc__)
    parser.add_argument("--out", type=Path, default=Path("reports/agent_eval.md"))
    parser.add_argument("--pattern", action="append", choices=sorted(PATTERNS))
    parser.add_argument(
        "--backend", default="rules", choices=("rules", "claude"),
        help="'claude' requires ANTHROPIC_API_KEY and the anthropic package.",
    )
    args = parser.parse_args(argv)

    if args.backend == "claude":
        from geoloc_agent.agent.backend import ClaudeBackend

        if not ClaudeBackend().available:
            parser.error(
                "claude backend unavailable: set ANTHROPIC_API_KEY and install `anthropic`"
            )

    patterns = args.pattern or list(PATTERNS)
    scenarios = build_scenarios()
    print(f"scenarios: {len(scenarios)}  patterns: {', '.join(patterns)}")
    scores = run_all(patterns, scenarios)
    report = build_agent_report(scores, args.out, backend=args.backend)

    from geoloc_agent.agent.eval import summarise

    for name, summary in summarise(scores).items():
        print(
            f"  {name:18s} task={summary['task_success']:.3f} "
            f"calls/track={summary['tool_calls_per_track']:.2f} "
            f"missed_escalations={summary['missed_escalations']}"
        )
    print(f"\nwrote {report}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
