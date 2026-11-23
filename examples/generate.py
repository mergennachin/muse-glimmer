"""Send a prompt to Glimmer and optionally ask a follow-up.

    muse-glimmer serve examples/generate.py -- "Name a lunar coffee shop." "Give it a tagline."

The launcher reuses a running Glimmer server at the configured address or
starts one for the duration of the script.
"""

import argparse

from muse_glimmer import Glimmer


def main() -> None:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument("prompt", nargs="?", default="Name a coffee shop on the moon.")
    parser.add_argument("follow_up", nargs="?", help="A second prompt about the first reply")
    args = parser.parse_args()

    # Print time to first token (ms) and prefill and decode speeds (tokens/s) to stderr.
    g = Glimmer(timings=True)
    system = "Be concise."
    answer = g.generate(args.prompt, system=system, temperature=0.7, max_tokens=2048)
    print(answer)

    if args.follow_up:
        g2 = Glimmer()
        messages = [
            {"role": "system", "content": system},
            {"role": "user", "content": args.prompt},
            {"role": "assistant", "content": answer},
            {"role": "user", "content": args.follow_up},
        ]
        reply = g2.chat(messages)
        print(reply["choices"][0]["message"]["content"])


if __name__ == "__main__":
    main()
