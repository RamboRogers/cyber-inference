"""Focused regression tests for the MTP download controls."""

from __future__ import annotations

import re
import shutil
import subprocess
from pathlib import Path

import pytest

TEMPLATE_PATH = Path("src/cyber_inference/web/templates/models.html")


def test_mtp_download_controls_use_backend_contract() -> None:
    template = TEMPLATE_PATH.read_text()

    assert 'id="mtpFileSelect"' in template
    assert 'name="hf_mtp_filename"' in template
    assert "data.mtp_files || []" in template
    assert "data.suggested_mtp" in template
    assert "requestBody.hf_mtp_filename = mtpFilename" in template
    assert "Do not download projector (text-only)" in template
    assert "combined MTP and vision" in template
    assert 'placeholder="2"' in template
    assert "mtp.spec_draft_n_max ?? 2" in template
    assert "MTP {% if model.mtp_draft_path %}separate{% else %}embedded{% endif %}" in template
    assert "ggml-org/Qwen3.6-27B-GGUF" in template
    assert "ggml-org/Qwen3.6-35B-A3B-GGUF" in template


def test_split_mtp_repo_selects_head_and_keeps_projector_opt_in() -> None:
    if not shutil.which("node"):
        pytest.skip("node is required for the focused downloader JavaScript harness")

    template = TEMPLATE_PATH.read_text()
    script_match = re.search(r"<script>([\s\S]*)</script>", template)
    assert script_match is not None
    script = script_match.group(1)
    harness = f"""
class Element {{
  constructor(id) {{
    this.id = id;
    this.children = [];
    this.listeners = {{}};
    this.value = '';
    this.textContent = '';
    this.disabled = false;
    this.selected = false;
    this.className = '';
    this.style = {{}};
    this.classList = {{
      add: (...names) => {{ this.className += ' ' + names.join(' '); }},
      remove: (...names) => {{
        const remove = new Set(names);
        this.className = this.className.split(/\\s+/).filter(x => !remove.has(x)).join(' ');
      }},
      contains: (name) => this.className.split(/\\s+/).includes(name),
    }};
  }}
  set innerHTML(value) {{
    this._innerHTML = value;
    this.children = [];
    this.value = '';
  }}
  get innerHTML() {{ return this._innerHTML || ''; }}
  appendChild(child) {{
    this.children.push(child);
    if (child.selected) this.value = child.value;
  }}
  prepend(child) {{ this.children.unshift(child); }}
  removeChild(child) {{ this.children = this.children.filter(x => x !== child); }}
  get lastChild() {{ return this.children[this.children.length - 1]; }}
  addEventListener(name, callback) {{ this.listeners[name] = callback; }}
  reset() {{ this.value = ''; }}
}}
const elements = {{}};
globalThis.document = {{
  getElementById: (id) => elements[id] || (elements[id] = new Element(id)),
  createElement: (tag) => new Element(tag),
}};
globalThis.window = {{ location: {{ protocol: 'http:', host: 'test' }} }};
globalThis.location = {{ reload: () => {{}} }};
globalThis.WebSocket = function() {{}};
globalThis.setTimeout = () => {{}};
globalThis.alert = (message) => {{ throw new Error(message); }};
globalThis.console = console;
let capturedRequest = null;
globalThis.adminFetch = async (endpoint, options) => {{
  capturedRequest = {{ endpoint, body: JSON.parse(options.body) }};
  return {{ ok: true }};
}};
{script}
repoData = {{
  is_mtp_candidate: true,
  is_multimodal: true,
  suggested_model: 'Qwen3.6-27B-Q4_K_M.gguf',
  suggested_mtp: 'mtp-Qwen3.6-27B-Q4_0.gguf',
  suggested_mmproj: 'mmproj-Qwen3.6-27B-F16.gguf',
  model_files: [{{
    filename: 'Qwen3.6-27B-Q4_K_M.gguf',
    size_bytes: 19000000000,
    quantization: 'q4_k_m',
    is_split: false,
  }}],
  mtp_files: [{{
    filename: 'mtp-Qwen3.6-27B-Q4_0.gguf',
    size_bytes: 1680000000,
    quantization: 'q4_0',
    artifact_type: 'mtp',
  }}],
  mmproj_files: [{{
    filename: 'mmproj-Qwen3.6-27B-F16.gguf',
    size_bytes: 900000000,
    artifact_type: 'mmproj',
  }}],
}};
populateFileSelections(repoData);
if (document.getElementById('mtpFileSelect').value !== repoData.suggested_mtp) {{
  throw new Error('recommended MTP head was not selected');
}}
if (document.getElementById('mmprojFileSelect').value !== '') {{
  throw new Error('projector must remain opt-in');
}}
if (document.getElementById('mtpDraftSection').className.includes('hidden')) {{
  throw new Error('MTP selector should be visible');
}}
if (!document.getElementById('mtpSizeInfo').textContent.includes('GB')) {{
  throw new Error('MTP file size should be shown');
}}
document.getElementById('repoId').value = 'ggml-org/Qwen3.6-27B-GGUF';
await document.getElementById('downloadForm').listeners.submit({{
  preventDefault: () => {{}},
}});
if (capturedRequest.endpoint !== '/admin/models/download') {{
  throw new Error(capturedRequest.endpoint);
}}
if (capturedRequest.body.hf_mtp_filename !== repoData.suggested_mtp) {{
  throw new Error('MTP filename was not sent');
}}
if ('hf_mmproj_filename' in capturedRequest.body) {{
  throw new Error('projector should not be sent unless explicitly selected');
}}
"""

    result = subprocess.run(
        ["node", "--input-type=module", "-e", harness],
        text=True,
        capture_output=True,
        check=False,
    )
    assert result.returncode == 0, result.stderr
