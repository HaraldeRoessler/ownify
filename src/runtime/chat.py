"""
ownify — Interactive chat using MLX with openclaw adapter.

Includes MolTrust integration for agent identity and escalation trust.
Trust layer built on MolTrust Protocol (https://moltrust.ch/) by MoltyCel.

Usage:
    python src/runtime/chat.py
    python src/runtime/chat.py --adapter-path adapters/openclaw-mlx-v2
    python src/runtime/chat.py --trust    # enable MolTrust integration
"""

import argparse
import re
import sys
import os

from mlx_lm import load, generate
from mlx_lm.sample_utils import make_sampler

# Add project root to path for imports
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "../.."))


def detect_escalation(response: str) -> tuple[bool, str]:
    """Check if the model wants to escalate."""
    match = re.search(r'<escalate\s+reason="([^"]+)"\s*/>', response)
    if match:
        return True, match.group(1)
    return False, ""


def main():
    parser = argparse.ArgumentParser(description="ownify chat")
    parser.add_argument("--model", default="Qwen/Qwen3.5-4B")
    parser.add_argument("--adapter-path", default="adapters/openclaw-mlx-v2")
    parser.add_argument("--max-tokens", type=int, default=500)
    parser.add_argument("--temp", type=float, default=0.7)
    parser.add_argument("--trust", action="store_true", help="Enable MolTrust integration")
    args = parser.parse_args()

    sampler = make_sampler(temp=args.temp, top_p=0.9)

    # Initialize MolTrust if enabled
    trust = None
    if args.trust:
        try:
            from src.trust.moltrust import MolTrustClient
            trust = MolTrustClient()
            status = trust.status()
            if status["registered"]:
                print(f"MolTrust: registered as {status['did']}")
            else:
                print("MolTrust: enabled but not registered. Use /register to set up identity.")
            if not status["api_key_set"]:
                print("MolTrust: no API key set. Export MOLTRUST_API_KEY for full functionality.")
        except ImportError:
            print("MolTrust: requests library not installed. Run: pip install requests")
            trust = None

    print("Loading ownify...")
    model, tokenizer = load(args.model, adapter_path=args.adapter_path)
    print("Ready. Type 'quit' to exit. Type '/help' for commands.\n")

    history = []

    while True:
        try:
            user_input = input("you> ").strip()
        except (EOFError, KeyboardInterrupt):
            print("\nBye.")
            break

        if not user_input:
            continue

        # Commands
        if user_input.lower() in ("quit", "exit", "/bye"):
            print("Bye.")
            break

        if user_input == "/help":
            print("\nCommands:")
            print("  /help       — show this help")
            print("  /status     — show ownify and trust status")
            print("  /register   — register DID with MolTrust")
            print("  /trust <did> — check an agent's trust score")
            print("  /clear      — clear conversation history")
            print("  /bye        — exit\n")
            continue

        if user_input == "/status":
            print(f"\nModel: {args.model}")
            print(f"Adapter: {args.adapter_path}")
            print(f"History: {len(history)} messages")
            if trust:
                s = trust.status()
                print(f"MolTrust: {'registered' if s['registered'] else 'not registered'}")
                if s["did"]:
                    print(f"  DID: {s['did']}")
                print(f"  Cached agents: {s['cached_agents']}")
                print(f"  API key: {'set' if s['api_key_set'] else 'not set'}")
            else:
                print("MolTrust: disabled (use --trust to enable)")
            print()
            continue

        if user_input == "/clear":
            history = []
            print("History cleared.\n")
            continue

        if user_input.startswith("/register"):
            if not trust:
                print("MolTrust not enabled. Restart with --trust\n")
                continue
            if trust.is_registered:
                print(f"Already registered as {trust.did}\n")
                continue
            name = input("Agent name (max 40 chars): ").strip()
            if not name:
                print("Cancelled.\n")
                continue
            try:
                result = trust.register(name)
                print(f"Registered: {trust.did}\n")
            except Exception as e:
                print(f"Registration failed: {e}\n")
            continue

        if user_input.startswith("/trust "):
            if not trust:
                print("MolTrust not enabled. Restart with --trust\n")
                continue
            did = user_input[7:].strip()
            try:
                rep = trust.check_reputation(did)
                print(f"\n  DID: {did}")
                print(f"  Trust score: {rep.get('trust_score', 'unknown')}")
                print(f"  Risk level: {rep.get('risk_level', 'unknown')}")
                print(f"  Total ratings: {rep.get('total_ratings', 0)}\n")
            except Exception as e:
                print(f"Trust check failed: {e}\n")
            continue

        # Regular chat
        history.append({"role": "user", "content": user_input})

        prompt = tokenizer.apply_chat_template(
            history, tokenize=False, add_generation_prompt=True,
            enable_thinking=False,
        )

        response = generate(
            model, tokenizer, prompt=prompt,
            max_tokens=args.max_tokens,
            sampler=sampler,
            verbose=False,
        )

        # Clean up response
        response = response.strip()
        if response.startswith("<think>"):
            think_end = response.find("</think>")
            if think_end != -1:
                response = response[think_end + 8:].strip()

        # Check for escalation
        wants_escalation, reason = detect_escalation(response)
        if wants_escalation and trust and trust.is_registered:
            # Sign the escalation request for provenance
            try:
                ipr = trust.sign_output(
                    f"escalation_request: {reason}",
                    metadata={"type": "escalation", "reason": reason},
                )
                ipr_id = ipr.get("ipr_id", "unknown")
                response += f"\n[MolTrust: escalation signed as {ipr_id}]"
            except Exception:
                pass  # Don't block chat on trust failures

        history.append({"role": "assistant", "content": response})
        print(f"\nownify> {response}\n")


if __name__ == "__main__":
    main()
