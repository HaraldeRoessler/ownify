"""
ownify — Agent runtime with tool execution loop.

The model decides what to do. The runtime just executes tools
and feeds results back. All intelligence is in the weights.

Usage:
    python src/runtime/agent.py
    python src/runtime/agent.py --working-dir /path/to/project
    python src/runtime/agent.py --allow-network
"""

import argparse
import json
import os
import re
import sys

from mlx_lm import load, generate
from mlx_lm.sample_utils import make_sampler

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "../.."))

from src.runtime.tools import ToolExecutor, TOOL_DEFINITIONS


MAX_TOOL_ROUNDS = 10  # Prevent infinite loops


def build_system_context(tools: list[dict], working_dir: str) -> str:
    """Build the tool context that tells the model what tools are available."""
    tool_descriptions = []
    for t in tools:
        f = t["function"]
        params = ", ".join(f["parameters"].get("required", []))
        tool_descriptions.append(f"- {f['name']}({params}): {f['description']}")

    return (
        f"You are ownify, a personal AI agent running locally. "
        f"You have these tools:\n\n"
        + "\n".join(tool_descriptions)
        + f"\n\nWorking directory: {working_dir}\n\n"
        f"To use a tool, output:\n"
        f'<tool_call>\n{{"name": "tool_name", "arguments": {{"key": "value"}}}}\n</tool_call>\n\n'
        f"You can chain multiple tool calls. After getting results, "
        f"continue working or respond to the user.\n"
        f"If a task is beyond your capability, use: <escalate reason=\"...\" />\n"
        f"Be direct. No fluff. Get things done."
    )


def parse_tool_calls(text: str) -> list[dict]:
    """Extract tool calls from model output."""
    calls = []
    pattern = r'<tool_call>\s*(\{.*?\})\s*</tool_call>'
    for match in re.finditer(pattern, text, re.DOTALL):
        try:
            call = json.loads(match.group(1))
            if "name" in call and "arguments" in call:
                calls.append(call)
        except json.JSONDecodeError:
            continue
    return calls


def strip_tool_calls(text: str) -> str:
    """Remove tool call tags from text, leaving only the conversational part."""
    cleaned = re.sub(r'<tool_call>\s*\{.*?\}\s*</tool_call>', '', text, flags=re.DOTALL)
    return cleaned.strip()


def main():
    parser = argparse.ArgumentParser(description="ownify agent")
    parser.add_argument("--model", default="Qwen/Qwen3.5-4B")
    parser.add_argument("--adapter-path", default="adapters/openclaw-mlx-v2")
    parser.add_argument("--working-dir", default=".", help="Working directory for file operations")
    parser.add_argument("--allow-network", action="store_true", help="Allow HTTP tool")
    parser.add_argument("--max-tokens", type=int, default=1000)
    parser.add_argument("--temp", type=float, default=0.3)
    parser.add_argument("--auto-approve", action="store_true", help="Skip confirmation for tool execution")
    args = parser.parse_args()

    working_dir = os.path.abspath(args.working_dir)
    executor = ToolExecutor(working_dir=working_dir, allow_network=args.allow_network)
    sampler = make_sampler(temp=args.temp, top_p=0.9)

    print("Loading ownify agent...")
    model, tokenizer = load(args.model, adapter_path=args.adapter_path)

    system_context = build_system_context(TOOL_DEFINITIONS, working_dir)

    print(f"Ready. Working dir: {working_dir}")
    print(f"Network: {'enabled' if args.allow_network else 'disabled'}")
    print("Type 'quit' to exit.\n")

    conversation = [{"role": "system", "content": system_context}]

    while True:
        try:
            user_input = input("you> ").strip()
        except (EOFError, KeyboardInterrupt):
            print("\nBye.")
            break

        if not user_input:
            continue
        if user_input.lower() in ("quit", "exit", "/bye"):
            print("Bye.")
            break

        conversation.append({"role": "user", "content": user_input})

        # Agent loop — model can call tools multiple times
        for round_num in range(MAX_TOOL_ROUNDS):
            prompt = tokenizer.apply_chat_template(
                conversation, tokenize=False, add_generation_prompt=True,
                enable_thinking=False,
            )

            response = generate(
                model, tokenizer, prompt=prompt,
                max_tokens=args.max_tokens,
                sampler=sampler,
                verbose=False,
            )

            response = response.strip()
            if response.startswith("<think>"):
                think_end = response.find("</think>")
                if think_end != -1:
                    response = response[think_end + 8:].strip()

            # Check for tool calls
            tool_calls = parse_tool_calls(response)

            if not tool_calls:
                # No tools — this is the final response
                conversation.append({"role": "assistant", "content": response})
                print(f"\nownify> {response}\n")
                break

            # Execute tool calls
            conversation.append({"role": "assistant", "content": response})

            for call in tool_calls:
                name = call["name"]
                arguments = call["arguments"]

                # Show what's about to happen
                print(f"\n  [tool] {name}({json.dumps(arguments, ensure_ascii=False)})")

                # Ask for approval unless auto-approve
                if not args.auto_approve:
                    approval = input("  Execute? [Y/n] ").strip().lower()
                    if approval == "n":
                        result = "Tool execution denied by user."
                        print(f"  [result] Denied")
                        conversation.append({"role": "tool", "name": name, "content": result})
                        continue

                # Execute
                result = executor.execute(name, arguments)

                # Truncate display
                display = result if len(result) < 200 else result[:200] + "..."
                print(f"  [result] {display}")

                conversation.append({"role": "tool", "name": name, "content": result})

        else:
            print("\nownify> (reached max tool rounds, stopping)\n")


if __name__ == "__main__":
    main()
