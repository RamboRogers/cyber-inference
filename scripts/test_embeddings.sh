#!/bin/bash
#
# Test script for Cyber-Inference embeddings endpoint
#
# Tests the Qwen3-Embedding-0.6B-GGUF model with sample text
#
# Usage:
#     ./scripts/test_embeddings.sh
#     ./scripts/test_embeddings.sh "Your custom text here"
#
# Requirements:
#     - Cyber-Inference server running on localhost:8337
#     - Qwen3-Embedding-0.6B-GGUF model downloaded
#

set -e

BASE_URL="${CYBER_INFERENCE_URL:-http://localhost:8337}"
MODEL_NAME="${CYBER_INFERENCE_EMBEDDING_MODEL:-Qwen3-Embedding-0.6B-Q8_0}"

# Default test text if none provided
TEST_TEXT="${1:-The quick brown fox jumps over the lazy dog}"

echo "═══════════════════════════════════════════════════════════"
echo "  Cyber-Inference Embeddings Test"
echo "═══════════════════════════════════════════════════════════"
echo ""
echo "Server: $BASE_URL"
echo "Model:  $MODEL_NAME"
echo "Text:   $TEST_TEXT"
echo ""

# Check if server is running
echo "Checking server health..."
if ! curl -s -f "$BASE_URL/health" > /dev/null; then
    echo "❌ Error: Server is not running at $BASE_URL"
    echo "   Please start the server with: uv run cyber-inference serve"
    exit 1
fi
echo "✅ Server is running"
echo ""

# Make embeddings request
echo "Requesting embeddings..."
echo "───────────────────────────────────────────────────────────"

RESPONSE=$(curl -s -w "\n%{http_code}" \
    -X POST "$BASE_URL/v1/embeddings" \
    -H "Content-Type: application/json" \
    -d "{
        \"model\": \"$MODEL_NAME\",
        \"input\": \"$TEST_TEXT\",
        \"encoding_format\": \"float\"
    }")

# Split response and status code (macOS-compatible)
HTTP_CODE=$(echo "$RESPONSE" | tail -n 1)
HTTP_BODY=$(echo "$RESPONSE" | sed '$d')

if [ "$HTTP_CODE" -eq 200 ]; then
    echo "✅ Success! (HTTP $HTTP_CODE)"
    echo ""
    echo "Response:"
    echo "$HTTP_BODY" | jq '.' 2>/dev/null || echo "$HTTP_BODY"
    echo ""

    # Extract embedding info
    EMBEDDING_SIZE=$(echo "$HTTP_BODY" | jq -r '.data[0].embedding | length' 2>/dev/null || echo "unknown")
    MODEL_RESPONSE=$(echo "$HTTP_BODY" | jq -r '.model' 2>/dev/null || echo "unknown")
    USAGE=$(echo "$HTTP_BODY" | jq -r '.usage' 2>/dev/null || echo "unknown")

    echo "───────────────────────────────────────────────────────────"
    echo "Embedding Details:"
    echo "  Model:        $MODEL_RESPONSE"
    echo "  Vector Size:  $EMBEDDING_SIZE dimensions"
    echo "  Usage:        $USAGE"
    echo ""

    # Show first few dimensions as sample
    if command -v jq > /dev/null; then
        echo "Sample embedding values (first 5 dimensions):"
        echo "$HTTP_BODY" | jq -r '.data[0].embedding[0:5]' 2>/dev/null || echo "  (jq required for this)"
    fi

    echo ""
    echo "✅ Test completed successfully!"

elif [ "$HTTP_CODE" -eq 503 ]; then
    echo "❌ Error: Model not loaded (HTTP $HTTP_CODE)"
    echo ""
    echo "Response:"
    echo "$HTTP_BODY" | jq '.' 2>/dev/null || echo "$HTTP_BODY"
    echo ""
    echo "The model will be loaded automatically on first request."
    echo "If this persists, check:"
    echo "  1. Model is downloaded: curl $BASE_URL/admin/models"
    echo "  2. Server logs for errors"
    exit 1

elif [ "$HTTP_CODE" -eq 404 ]; then
    echo "❌ Error: Model not found (HTTP $HTTP_CODE)"
    echo ""
    echo "Response:"
    echo "$HTTP_BODY" | jq '.' 2>/dev/null || echo "$HTTP_BODY"
    echo ""
    echo "Please download the model first:"
    echo "  curl -X POST $BASE_URL/admin/models/download \\"
    echo "    -H 'Content-Type: application/json' \\"
    echo "    -d '{\"hf_repo_id\": \"Qwen/Qwen3-Embedding-0.6B-GGUF\"}'"
    exit 1

else
    echo "❌ Error: Request failed (HTTP $HTTP_CODE)"
    echo ""
    echo "Response:"
    echo "$HTTP_BODY" | jq '.' 2>/dev/null || echo "$HTTP_BODY"
    exit 1
fi

