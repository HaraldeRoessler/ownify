"""
ownify — Standard tool executor.

Provides 5 standard tools for the agent. The tools are dumb —
they execute what the model asks. The model decides what to do.

Safety boundaries are enforced here, not in the model.
"""

import json
import os
import subprocess
from pathlib import Path
from typing import Optional

import requests


# Commands that are never allowed
BLOCKED_COMMANDS = [
    "rm -rf /", "rm -rf /*", "mkfs", "dd if=", "> /dev/sd",
    "chmod -R 777 /", ":(){ :|:& };:",
]

TOOL_DEFINITIONS = [
    {
        "type": "function",
        "function": {
            "name": "read_file",
            "description": "Read the contents of a file at the given path",
            "parameters": {
                "type": "object",
                "properties": {
                    "path": {"type": "string", "description": "File path to read"}
                },
                "required": ["path"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "write_file",
            "description": "Write content to a file, creating directories if needed",
            "parameters": {
                "type": "object",
                "properties": {
                    "path": {"type": "string", "description": "File path to write"},
                    "content": {"type": "string", "description": "Content to write"},
                },
                "required": ["path", "content"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "shell",
            "description": "Execute a shell command and return stdout/stderr",
            "parameters": {
                "type": "object",
                "properties": {
                    "command": {"type": "string", "description": "Shell command to execute"}
                },
                "required": ["command"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "http",
            "description": "Make an HTTP request to a URL",
            "parameters": {
                "type": "object",
                "properties": {
                    "method": {"type": "string", "enum": ["GET", "POST", "PUT", "DELETE"]},
                    "url": {"type": "string", "description": "URL to request"},
                    "headers": {"type": "object", "description": "Request headers"},
                    "body": {"type": "string", "description": "Request body"},
                },
                "required": ["method", "url"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "list_dir",
            "description": "List files and directories at the given path",
            "parameters": {
                "type": "object",
                "properties": {
                    "path": {"type": "string", "description": "Directory path to list"}
                },
                "required": ["path"],
            },
        },
    },
]


class ToolExecutor:
    """Executes tool calls from the model. Thin and dumb by design."""

    def __init__(self, working_dir: str = ".", allow_network: bool = False):
        self.working_dir = Path(working_dir).resolve()
        self.allow_network = allow_network

    def execute(self, name: str, arguments: dict) -> str:
        """Execute a tool call and return the result as a string."""
        try:
            if name == "read_file":
                return self._read_file(arguments["path"])
            elif name == "write_file":
                return self._write_file(arguments["path"], arguments["content"])
            elif name == "shell":
                return self._shell(arguments["command"])
            elif name == "http":
                return self._http(
                    arguments["method"],
                    arguments["url"],
                    arguments.get("headers"),
                    arguments.get("body"),
                )
            elif name == "list_dir":
                return self._list_dir(arguments["path"])
            else:
                return f"Error: unknown tool '{name}'"
        except Exception as e:
            return f"Error: {e}"

    def _resolve_path(self, path: str) -> Path:
        """Resolve path relative to working directory. Block escapes."""
        resolved = (self.working_dir / path).resolve()
        if not str(resolved).startswith(str(self.working_dir)):
            raise PermissionError(f"Path {path} escapes working directory")
        return resolved

    def _read_file(self, path: str) -> str:
        resolved = self._resolve_path(path)
        if not resolved.exists():
            return f"Error: file not found: {path}"
        content = resolved.read_text(errors="replace")
        if len(content) > 10000:
            return content[:10000] + f"\n... (truncated, {len(content)} chars total)"
        return content

    def _write_file(self, path: str, content: str) -> str:
        resolved = self._resolve_path(path)
        resolved.parent.mkdir(parents=True, exist_ok=True)
        resolved.write_text(content)
        return f"File written: {path} ({len(content)} chars)"

    def _shell(self, command: str) -> str:
        # Safety check
        for blocked in BLOCKED_COMMANDS:
            if blocked in command:
                return f"Error: command blocked for safety: {blocked}"

        try:
            result = subprocess.run(
                command,
                shell=True,
                capture_output=True,
                text=True,
                timeout=30,
                cwd=str(self.working_dir),
            )
            output = result.stdout
            if result.stderr:
                output += ("\n" if output else "") + result.stderr
            if result.returncode != 0:
                output += f"\n(exit code: {result.returncode})"
            if not output.strip():
                output = "(no output)"
            if len(output) > 5000:
                output = output[:5000] + "\n... (truncated)"
            return output
        except subprocess.TimeoutExpired:
            return "Error: command timed out (30s limit)"

    def _http(self, method: str, url: str, headers: Optional[dict] = None,
              body: Optional[str] = None) -> str:
        if not self.allow_network:
            return "Error: network access disabled. Start with --allow-network to enable."

        try:
            response = requests.request(
                method=method,
                url=url,
                headers=headers or {},
                data=body,
                timeout=15,
            )
            result = f"HTTP {response.status_code}\n"
            content = response.text
            if len(content) > 5000:
                content = content[:5000] + "\n... (truncated)"
            result += content
            return result
        except requests.RequestException as e:
            return f"Error: HTTP request failed: {e}"

    def _list_dir(self, path: str) -> str:
        resolved = self._resolve_path(path)
        if not resolved.exists():
            return f"Error: directory not found: {path}"
        if not resolved.is_dir():
            return f"Error: not a directory: {path}"

        entries = []
        for item in sorted(resolved.iterdir()):
            if item.name.startswith("."):
                continue
            if item.is_dir():
                entries.append(f"{item.name}/")
            else:
                size = item.stat().st_size
                if size > 1024 * 1024:
                    size_str = f"{size / 1024 / 1024:.1f}MB"
                elif size > 1024:
                    size_str = f"{size / 1024:.0f}KB"
                else:
                    size_str = f"{size}B"
                entries.append(f"{item.name} ({size_str})")

        if not entries:
            return "(empty directory)"
        return "\n".join(entries)
