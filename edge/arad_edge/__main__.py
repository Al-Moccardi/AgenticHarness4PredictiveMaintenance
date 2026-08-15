"""Unified CLI:  python -m arad_edge <command> [args]

Commands:
  detect      frozen Isolation-Forest detection over the test set -> KB-schema CSV
  interpret   SLM natural-language interpretation of detected anomalies
  sample      pick reference test units (stratified) for a pilot/demo run
  collect     collection simulator: replay test_FD002 as a growing CSV
  daemon      online detection daemon (tails the collector, emits anomaly queue)
  telemetry   hardware telemetry sampler (CPU/RAM/GPU/temp/power)
  stats       aggregate all logs into report + figures
  smoke       run the offline guard suite

Run `python -m arad_edge <command> --help` for each command's options.
"""
import sys


def main() -> None:
    if len(sys.argv) < 2 or sys.argv[1] in ("-h", "--help"):
        print(__doc__)
        sys.exit(0)
    cmd = sys.argv.pop(1)
    if cmd == "detect":
        from .detector import main as m
    elif cmd == "interpret":
        from .interpreter import main as m
    elif cmd == "sample":
        from .sampling import main as m
    elif cmd == "collect":
        from .collector import main as m
    elif cmd == "daemon":
        from .daemon import main as m
    elif cmd == "telemetry":
        from .telemetry import main as m
    elif cmd == "stats":
        from .stats import main as m
    elif cmd == "smoke":
        from tests.smoke import main as m  # type: ignore
    else:
        print(f"unknown command: {cmd}\n")
        print(__doc__)
        sys.exit(2)
    m()


if __name__ == "__main__":
    main()
