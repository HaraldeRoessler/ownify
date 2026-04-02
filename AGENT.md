# ownify — Agent Architecture

## Vision

ownify is not a chatbot. It's an agent. The difference:

- **Chatbot:** answers questions with text
- **Agent:** takes actions to accomplish goals using tools

The agent behavior — planning, tool selection, error recovery, self-improvement — is trained into the LoRA adapter, not hardcoded in wrapper code. The runtime provides standard tools. The model decides when and how to use them.

## Tool Format

ownify uses Qwen's native function calling format. The model outputs structured tool calls, the runtime executes them and feeds results back.

### Available Tools

The runtime provides these standard tools. Nothing more. The intelligence is in the model.

| Tool | What it does | Example |
|------|-------------|---------|
| `read_file` | Read a file's contents | Read a config file |
| `write_file` | Write/create a file | Write code, configs |
| `shell` | Execute a shell command | Run tests, git, build |
| `http` | Make an HTTP request | Call APIs, check endpoints |
| `list_dir` | List directory contents | Explore project structure |

### Tool Call Format

The model calls tools using Qwen's function calling format:

```
<tool_call>
{"name": "read_file", "arguments": {"path": "src/main.py"}}
</tool_call>
```

The runtime executes and returns:

```
<tool_response>
{"name": "read_file", "content": "import sys\n\ndef main():\n    ..."}
</tool_response>
```

The model can chain multiple tool calls to accomplish complex tasks.

### Tool Definitions

Tools are defined as JSON schemas passed to the model:

```json
[
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
        "required": ["path"]
      }
    }
  },
  {
    "type": "function",
    "function": {
      "name": "write_file",
      "description": "Write content to a file, creating it if it doesn't exist",
      "parameters": {
        "type": "object",
        "properties": {
          "path": {"type": "string", "description": "File path to write"},
          "content": {"type": "string", "description": "Content to write"}
        },
        "required": ["path", "content"]
      }
    }
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
        "required": ["command"]
      }
    }
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
          "body": {"type": "string", "description": "Request body (for POST/PUT)"}
        },
        "required": ["method", "url"]
      }
    }
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
        "required": ["path"]
      }
    }
  }
]
```

## Agent Loop

The runtime implements a simple loop. All intelligence is in the model.

```
User gives task
       │
       ▼
┌─────────────────┐
│  Model thinks   │◄──────────────────┐
│  (what to do    │                   │
│   next)         │                   │
└───────┬─────────┘                   │
        │                             │
        ├── Text response ──► Show to user (done)
        │
        ├── <tool_call> ──► Execute tool
        │                        │
        │                   <tool_response>
        │                        │
        └────────────────────────┘
              (loop back with result)
```

The model decides:
1. **Which tool** to use (or none — just answer)
2. **What arguments** to pass
3. **What to do** with the result
4. **When to stop** and respond to the user
5. **When to escalate** if the task is too complex

## Training Data Format

Agentic training examples show complete tool-use sequences:

### Simple tool use
```json
{"messages": [
  {"role": "user", "content": "What's in the README?"},
  {"role": "assistant", "content": "", "tool_calls": [{"type": "function", "function": {"name": "read_file", "arguments": "{\"path\": \"README.md\"}"}}]},
  {"role": "tool", "name": "read_file", "content": "# My Project\nA simple web app."},
  {"role": "assistant", "content": "It's a simple web app project. The README just has a title and one-line description."}
]}
```

### Multi-step task
```json
{"messages": [
  {"role": "user", "content": "Create a hello world Python script and run it"},
  {"role": "assistant", "content": "", "tool_calls": [{"type": "function", "function": {"name": "write_file", "arguments": "{\"path\": \"hello.py\", \"content\": \"print('Hello, World!')\"}"}}]},
  {"role": "tool", "name": "write_file", "content": "File written: hello.py"},
  {"role": "assistant", "content": "", "tool_calls": [{"type": "function", "function": {"name": "shell", "arguments": "{\"command\": \"python hello.py\"}"}}]},
  {"role": "tool", "name": "shell", "content": "Hello, World!"},
  {"role": "assistant", "content": "Done. Created hello.py and ran it — output: Hello, World!"}
]}
```

### Error recovery
```json
{"messages": [
  {"role": "user", "content": "Run the tests"},
  {"role": "assistant", "content": "", "tool_calls": [{"type": "function", "function": {"name": "shell", "arguments": "{\"command\": \"pytest\"}"}}]},
  {"role": "tool", "name": "shell", "content": "ERROR: ModuleNotFoundError: No module named 'pytest'"},
  {"role": "assistant", "content": "", "tool_calls": [{"type": "function", "function": {"name": "shell", "arguments": "{\"command\": \"pip install pytest && pytest\"}"}}]},
  {"role": "tool", "name": "shell", "content": "collected 5 items\n5 passed in 0.3s"},
  {"role": "assistant", "content": "pytest wasn't installed. Installed it and ran the tests — all 5 passed."}
]}
```

### Escalation during tool use
```json
{"messages": [
  {"role": "user", "content": "Refactor the entire authentication module to use OAuth2"},
  {"role": "assistant", "content": "", "tool_calls": [{"type": "function", "function": {"name": "list_dir", "arguments": "{\"path\": \"src/auth/\"}"}}]},
  {"role": "tool", "name": "list_dir", "content": "middleware.py (450 lines)\nhandlers.py (380 lines)\nmodels.py (200 lines)\ntokens.py (150 lines)\nutils.py (90 lines)"},
  {"role": "assistant", "content": "<escalate reason=\"Auth module is 1270 lines across 5 files — full OAuth2 refactor requires reasoning across all files simultaneously, which exceeds my context capacity\" />\n\nI looked at the auth module — it's 5 files, 1270 lines total. A full OAuth2 refactor at this scale needs a model that can hold and reason across all of it at once. Want me to escalate?"}
]}
```

## What Lives Where

| Component | Where | Trained or coded? |
|-----------|-------|-------------------|
| Tool selection | LoRA weights | Trained |
| Planning | LoRA weights | Trained |
| Error recovery | LoRA weights | Trained |
| Escalation decisions | LoRA weights | Trained |
| Communication style | LoRA weights | Trained |
| Tool execution | Runtime (Python) | Coded |
| Safety guardrails | Runtime (Python) | Coded |
| File I/O | Runtime (Python) | Coded |

The runtime is thin and dumb. The model is smart.

## Safety

The runtime enforces boundaries the model cannot override:

- **No network access** unless explicitly allowed (escalation, configured APIs)
- **Working directory sandbox** — file operations restricted to project scope
- **Command allowlist** — shell tool rejects dangerous commands (rm -rf /, etc.)
- **User confirmation** — destructive actions require approval before execution
- **Escalation logging** — all external calls are logged locally

These are in the runtime, not the model. The model can't bypass them.

## Roadmap

1. **Tool executor runtime** — implement the 5 tools and the agent loop
2. **Agentic training data** — 200+ examples showing tool use, planning, error recovery
3. **Train openclaw v3** — merge agentic data with existing behavior data
4. **Test end-to-end** — can ownify write code, run it, fix errors, commit?
5. **Self-improvement loop** — can ownify generate its own training data and retrain?
