#!/usr/bin/env python3
"""
Test script for Cyber-Inference model operations.

Tests:
- Model download
- Model loading/unloading
- Inference (chat completions)
- Streaming responses

Usage:
    python scripts/test_model_inference.py

Requirements:
    - Cyber-Inference server running on localhost:8337
    - Network access to HuggingFace for downloads
"""

import argparse
import asyncio
import json
import sys
import time
from typing import Optional

import httpx

# Configuration
BASE_URL = "http://localhost:8337"
TEST_MODEL_REPO = "bartowski/Qwen3VL-2B-Instruct-GGUF"
TEST_MODEL_FILE = "Qwen3VL-2B-Instruct-Q4_K_M.gguf"
TEST_MODEL_NAME = "Qwen3VL-2B-Instruct-Q4_K_M"

# Colors for output
class Colors:
    HEADER = '\033[95m'
    BLUE = '\033[94m'
    CYAN = '\033[96m'
    GREEN = '\033[92m'
    YELLOW = '\033[93m'
    RED = '\033[91m'
    ENDC = '\033[0m'
    BOLD = '\033[1m'


def print_header(msg: str) -> None:
    print(f"\n{Colors.HEADER}{Colors.BOLD}{'='*60}{Colors.ENDC}")
    print(f"{Colors.HEADER}{Colors.BOLD}{msg}{Colors.ENDC}")
    print(f"{Colors.HEADER}{Colors.BOLD}{'='*60}{Colors.ENDC}\n")


def print_success(msg: str) -> None:
    print(f"{Colors.GREEN}✓ {msg}{Colors.ENDC}")


def print_error(msg: str) -> None:
    print(f"{Colors.RED}✗ {msg}{Colors.ENDC}")


def print_info(msg: str) -> None:
    print(f"{Colors.CYAN}→ {msg}{Colors.ENDC}")


def print_warning(msg: str) -> None:
    print(f"{Colors.YELLOW}⚠ {msg}{Colors.ENDC}")


async def check_server_health() -> bool:
    """Check if the server is running and healthy."""
    print_info("Checking server health...")

    try:
        async with httpx.AsyncClient() as client:
            response = await client.get(f"{BASE_URL}/health", timeout=5.0)

            if response.status_code == 200:
                data = response.json()
                print_success(f"Server is healthy: {data}")
                return True
            else:
                print_error(f"Health check failed: {response.status_code}")
                return False

    except httpx.ConnectError:
        print_error("Cannot connect to server. Is it running?")
        return False
    except Exception as e:
        print_error(f"Health check error: {e}")
        return False


async def list_models() -> list:
    """List all available models."""
    print_info("Listing available models...")

    async with httpx.AsyncClient() as client:
        response = await client.get(f"{BASE_URL}/v1/models", timeout=10.0)

        if response.status_code == 200:
            data = response.json()
            models = data.get("data", [])
            print_success(f"Found {len(models)} models")
            for model in models:
                print(f"    - {model['id']}")
            return models
        else:
            print_error(f"Failed to list models: {response.status_code}")
            return []


async def download_model(repo_id: str, filename: Optional[str] = None) -> bool:
    """Download a model from HuggingFace."""
    print_info(f"Downloading model from: {repo_id}")
    if filename:
        print_info(f"Specific file: {filename}")

    try:
        async with httpx.AsyncClient(timeout=600.0) as client:
            start_time = time.time()

            response = await client.post(
                f"{BASE_URL}/admin/models/download",
                json={
                    "name": filename.replace(".gguf", "") if filename else None,
                    "hf_repo_id": repo_id,
                    "hf_filename": filename,
                },
            )

            elapsed = time.time() - start_time

            if response.status_code == 200:
                data = response.json()
                print_success(f"Download complete in {elapsed:.1f}s")
                print(f"    Name: {data.get('name')}")
                print(f"    Size: {data.get('size_bytes', 0) / (1024**3):.2f} GB")
                return True
            else:
                print_error(f"Download failed: {response.status_code}")
                try:
                    error_data = response.json()
                    print_error(f"Error: {error_data.get('detail', 'Unknown')}")
                except:
                    print_error(f"Response: {response.text[:200]}")
                return False

    except httpx.TimeoutException:
        print_warning("Download is taking a long time - it may still be in progress")
        return False
    except Exception as e:
        print_error(f"Download error: {e}")
        return False


async def load_model(model_name: str) -> bool:
    """Load a model into memory."""
    print_info(f"Loading model: {model_name}")

    try:
        async with httpx.AsyncClient(timeout=300.0) as client:
            start_time = time.time()

            response = await client.post(
                f"{BASE_URL}/admin/models/{model_name}/load",
            )

            elapsed = time.time() - start_time

            if response.status_code == 200:
                data = response.json()
                print_success(f"Model loaded in {elapsed:.1f}s")
                print(f"    Port: {data.get('port')}")
                print(f"    Status: {data.get('status')}")
                return True
            else:
                print_error(f"Load failed: {response.status_code}")
                try:
                    error_data = response.json()
                    print_error(f"Error: {error_data.get('detail', 'Unknown')}")
                except:
                    print_error(f"Response: {response.text[:200]}")
                return False

    except Exception as e:
        print_error(f"Load error: {e}")
        return False


async def unload_model(model_name: str) -> bool:
    """Unload a model from memory."""
    print_info(f"Unloading model: {model_name}")

    try:
        async with httpx.AsyncClient(timeout=30.0) as client:
            response = await client.post(
                f"{BASE_URL}/admin/models/{model_name}/unload",
            )

            if response.status_code == 200:
                print_success("Model unloaded")
                return True
            else:
                print_error(f"Unload failed: {response.status_code}")
                return False

    except Exception as e:
        print_error(f"Unload error: {e}")
        return False


async def test_inference_non_streaming(model_name: str) -> bool:
    """Test non-streaming chat completion."""
    print_info(f"Testing non-streaming inference with: {model_name}")

    try:
        async with httpx.AsyncClient(timeout=120.0) as client:
            start_time = time.time()

            response = await client.post(
                f"{BASE_URL}/v1/chat/completions",
                json={
                    "model": model_name,
                    "messages": [
                        {"role": "user", "content": "Say 'Hello, World!' and nothing else."}
                    ],
                    "max_tokens": 50,
                    "temperature": 0.1,
                    "stream": False,
                },
            )

            elapsed = time.time() - start_time

            if response.status_code == 200:
                data = response.json()
                choices = data.get("choices", [])

                if choices:
                    content = choices[0].get("message", {}).get("content", "")
                    usage = data.get("usage", {})

                    print_success(f"Inference complete in {elapsed:.1f}s")
                    print(f"    Response: {content[:100]}")
                    print(f"    Tokens: {usage.get('total_tokens', 'N/A')}")
                    return True
                else:
                    print_error("No choices in response")
                    return False
            else:
                print_error(f"Inference failed: {response.status_code}")
                try:
                    error_data = response.json()
                    print_error(f"Error: {error_data.get('detail', 'Unknown')}")
                except:
                    print_error(f"Response (first 500 chars): {response.text[:500]}")
                return False

    except Exception as e:
        print_error(f"Inference error: {e}")
        import traceback
        traceback.print_exc()
        return False


async def test_inference_streaming(model_name: str) -> bool:
    """Test streaming chat completion."""
    print_info(f"Testing streaming inference with: {model_name}")

    try:
        async with httpx.AsyncClient(timeout=120.0) as client:
            start_time = time.time()
            collected_content = ""
            chunk_count = 0

            async with client.stream(
                "POST",
                f"{BASE_URL}/v1/chat/completions",
                json={
                    "model": model_name,
                    "messages": [
                        {"role": "user", "content": "Count from 1 to 5."}
                    ],
                    "max_tokens": 50,
                    "temperature": 0.1,
                    "stream": True,
                },
            ) as response:
                if response.status_code != 200:
                    print_error(f"Stream failed: {response.status_code}")
                    return False

                async for line in response.aiter_lines():
                    if line.startswith("data: "):
                        data = line[6:]
                        if data == "[DONE]":
                            break

                        try:
                            chunk = json.loads(data)
                            choices = chunk.get("choices", [])
                            if choices:
                                delta = choices[0].get("delta", {})
                                content = delta.get("content", "")
                                if content:
                                    collected_content += content
                                    chunk_count += 1
                        except json.JSONDecodeError:
                            continue

            elapsed = time.time() - start_time

            if collected_content:
                print_success(f"Streaming complete in {elapsed:.1f}s")
                print(f"    Response: {collected_content[:100]}")
                print(f"    Chunks received: {chunk_count}")
                return True
            else:
                print_error("No content received from stream")
                return False

    except Exception as e:
        print_error(f"Streaming error: {e}")
        import traceback
        traceback.print_exc()
        return False


async def get_status() -> dict:
    """Get system status."""
    print_info("Getting system status...")

    async with httpx.AsyncClient() as client:
        response = await client.get(f"{BASE_URL}/admin/status", timeout=10.0)

        if response.status_code == 200:
            data = response.json()
            print_success("Status retrieved")
            print(f"    Running models: {data.get('running_models', [])}")
            print(f"    CPU: {data.get('cpu_percent', 0):.1f}%")
            print(f"    Memory: {data.get('memory_percent', 0):.1f}%")
            print(f"    GPU: {data.get('gpu_available', False)}")
            return data
        else:
            print_error(f"Status failed: {response.status_code}")
            return {}


async def wait_for_model_ready(model_name: str, max_wait: int = 30) -> bool:
    """Wait for model to be ready to receive requests."""
    print_info(f"Waiting for model to be fully ready (max {max_wait}s)...")

    for i in range(max_wait):
        try:
            async with httpx.AsyncClient(timeout=5.0) as client:
                # Try a simple inference
                response = await client.post(
                    f"{BASE_URL}/v1/chat/completions",
                    json={
                        "model": model_name,
                        "messages": [{"role": "user", "content": "hi"}],
                        "max_tokens": 1,
                        "stream": False,
                    },
                )

                if response.status_code == 200:
                    print_success(f"Model ready after {i+1}s")
                    return True

        except Exception:
            pass

        await asyncio.sleep(1)
        print(f"    Waiting... {i+1}s")

    print_warning("Model may not be fully ready")
    return False


async def run_full_test(
    model_repo: str = TEST_MODEL_REPO,
    model_file: str = TEST_MODEL_FILE,
    model_name: str = TEST_MODEL_NAME,
    skip_download: bool = False,
) -> bool:
    """Run the full test suite."""

    print_header("CYBER-INFERENCE MODEL TEST")

    results = {
        "health": False,
        "download": False,
        "load": False,
        "inference_sync": False,
        "inference_stream": False,
        "unload": False,
    }

    # Step 1: Health check
    print_header("STEP 1: Server Health Check")
    results["health"] = await check_server_health()
    if not results["health"]:
        print_error("Server is not healthy. Aborting tests.")
        return False

    # Step 2: List models
    print_header("STEP 2: Check Existing Models")
    models = await list_models()
    model_exists = any(m["id"] == model_name for m in models)

    # Step 3: Download model (if needed)
    print_header("STEP 3: Model Download")
    if model_exists:
        print_info(f"Model {model_name} already exists, skipping download")
        results["download"] = True
    elif skip_download:
        print_warning("Download skipped by user")
        results["download"] = True
    else:
        results["download"] = await download_model(model_repo, model_file)
        if not results["download"]:
            print_error("Download failed. Checking if model exists anyway...")
            models = await list_models()
            model_exists = any(m["id"] == model_name for m in models)
            if model_exists:
                print_success("Model found despite download error (may have been downloaded previously)")
                results["download"] = True
            else:
                print_error("Model not available. Aborting tests.")
                return False

    # Step 4: Load model
    print_header("STEP 4: Model Loading")
    status = await get_status()
    if model_name in status.get("running_models", []):
        print_info(f"Model {model_name} already loaded")
        results["load"] = True
    else:
        results["load"] = await load_model(model_name)
        if not results["load"]:
            print_error("Failed to load model. Aborting tests.")
            return False

    # Step 4b: Wait for model to be truly ready
    await wait_for_model_ready(model_name)

    # Step 5: Test inference (non-streaming)
    print_header("STEP 5: Non-Streaming Inference")
    results["inference_sync"] = await test_inference_non_streaming(model_name)

    # Step 6: Test inference (streaming)
    print_header("STEP 6: Streaming Inference")
    results["inference_stream"] = await test_inference_streaming(model_name)

    # Step 7: Unload model
    print_header("STEP 7: Model Unloading")
    results["unload"] = await unload_model(model_name)

    # Summary
    print_header("TEST SUMMARY")

    passed = 0
    failed = 0

    for test_name, result in results.items():
        if result:
            print_success(f"{test_name}: PASSED")
            passed += 1
        else:
            print_error(f"{test_name}: FAILED")
            failed += 1

    print(f"\n{Colors.BOLD}Total: {passed} passed, {failed} failed{Colors.ENDC}")

    return failed == 0


def main():
    global BASE_URL

    parser = argparse.ArgumentParser(
        description="Test Cyber-Inference model operations"
    )
    parser.add_argument(
        "--repo", "-r",
        default=TEST_MODEL_REPO,
        help=f"HuggingFace repo ID (default: {TEST_MODEL_REPO})"
    )
    parser.add_argument(
        "--file", "-f",
        default=TEST_MODEL_FILE,
        help=f"Model filename (default: {TEST_MODEL_FILE})"
    )
    parser.add_argument(
        "--name", "-n",
        default=TEST_MODEL_NAME,
        help=f"Model name (default: {TEST_MODEL_NAME})"
    )
    parser.add_argument(
        "--skip-download", "-s",
        action="store_true",
        help="Skip download step (assume model exists)"
    )
    parser.add_argument(
        "--url", "-u",
        default=BASE_URL,
        help=f"Server URL (default: {BASE_URL})"
    )

    args = parser.parse_args()

    # Update base URL if provided
    BASE_URL = args.url

    # Run tests
    success = asyncio.run(run_full_test(
        model_repo=args.repo,
        model_file=args.file,
        model_name=args.name,
        skip_download=args.skip_download,
    ))

    sys.exit(0 if success else 1)


if __name__ == "__main__":
    main()

