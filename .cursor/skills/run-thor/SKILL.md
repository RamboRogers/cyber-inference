---
name: run-thor
description: Deploy and test cyber-inference on the Thor lab server. Use when the user wants to test on Thor, deploy to Thor, run on Thor, verify the server, or mentions thor.lab.
---

# Deploy & Test on Thor

Thor is the GPU lab server used for integration testing of cyber-inference. It is accessible via SSH and hosts the production-like test environment.

## Connection Details

| Field | Value |
|-------|-------|
| Host | `thor.lab` |
| User | `matt` |
| SSH | `ssh matt@thor.lab` |
| Project path | `/home/matt/Local/cyber-inference` |
| Server URL | `http://thor.lab:8337` |

## Deploy Workflow

Follow these steps in order. Each depends on the previous.

### 1. Commit & Push (local machine)

```bash
git add -A && git commit -m "<message>" && git push
```

### 2. Pull on Thor (remote)

```bash
ssh matt@thor.lab "cd /home/matt/Local/cyber-inference && git pull"
```

### 3. Start the Server (remote)

The server runs via `start.sh` which handles `uv sync` and auto-restart.

```bash
# Interactive (see logs live) - use for debugging
ssh -t matt@thor.lab "cd /home/matt/Local/cyber-inference && ./start.sh"

# Background (detached) - use for long-running tests
ssh matt@thor.lab "cd /home/matt/Local/cyber-inference && nohup ./start.sh > /tmp/cyber-inference.log 2>&1 &"
```

SGLang + CUDA PyTorch wheels are installed automatically when NVIDIA hardware is detected.
To force disable SGLang:
```bash
ssh -t matt@thor.lab "cd /home/matt/Local/cyber-inference && CYBER_INFERENCE_NO_SGLANG=1 ./start.sh"
```

### 4. Verify the Server

```bash
# Health check
curl -s http://thor.lab:8337/health

# List models
curl -s http://thor.lab:8337/v1/models | python3 -m json.tool

# System status
curl -s http://thor.lab:8337/admin/status | python3 -m json.tool

# SGLang status (if enabled)
curl -s http://thor.lab:8337/admin/sglang/status | python3 -m json.tool
```

The web GUI is available at: `http://thor.lab:8337`

### 5. Test Inference

```bash
# Chat completion
curl -s http://thor.lab:8337/v1/chat/completions \
  -H "Content-Type: application/json" \
  -d '{"model": "<model_name>", "messages": [{"role": "user", "content": "Hello"}]}' \
  | python3 -m json.tool

# Embeddings
curl -s http://thor.lab:8337/v1/embeddings \
  -H "Content-Type: application/json" \
  -d '{"model": "<model_name>", "input": "test text"}' \
  | python3 -m json.tool
```

## Quick One-Liner Deploy

Pull latest and restart in one command:

```bash
ssh -t matt@thor.lab "cd /home/matt/Local/cyber-inference && git pull && ./start.sh"
```

## Troubleshooting

- **Server won't start**: Check logs with `ssh matt@thor.lab "tail -50 /tmp/cyber-inference.log"`
- **Port in use**: Kill existing process with `ssh matt@thor.lab "pkill -f 'cyber-inference serve'"`
- **Check running processes**: `ssh matt@thor.lab "ps aux | grep cyber-inference"`
- **GPU/CUDA issues**: `ssh matt@thor.lab "nvidia-smi"`
