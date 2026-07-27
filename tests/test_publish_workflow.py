from pathlib import Path

WORKFLOW = Path(__file__).resolve().parents[1] / ".github" / "workflows" / "publish-containers.yml"


def test_publish_workflow_has_weekly_llama_cpp_refresh_window():
    workflow = WORKFLOW.read_text(encoding="utf-8")

    assert "schedule:" in workflow
    assert 'cron: "0 6 * * 1"' in workflow
    assert 'cron: "0 7 * * 1"' in workflow
    assert "America/New_York" in workflow
    assert 'local_hour="$(TZ=America/New_York date +%H)"' in workflow
    assert 'local_weekday="$(TZ=America/New_York date +%u)"' in workflow
    assert "python3 - <<'PY'" in workflow


def test_publish_workflow_skips_scheduled_rebuild_without_new_llama_cpp():
    workflow = WORKFLOW.read_text(encoding="utf-8")

    assert "llama-cpp-refresh:" in workflow
    assert 'org.cyberinference.llama-cpp.tag' in workflow
    assert 'if [[ "$current_llama_tag" == "$LLAMA_CPP_TAG" ]]' in workflow
    assert "should_publish=false" in workflow
    assert "needs.llama-cpp-refresh.outputs.should_publish == 'true'" in workflow


def test_publish_workflow_pins_and_labels_thor_llama_cpp_build():
    workflow = WORKFLOW.read_text(encoding="utf-8")

    assert 'git clone --depth 1 --branch "${LLAMA_CPP_TAG}"' in workflow
    assert "LLAMA_CPP_COMMIT=$(git rev-parse HEAD)" in workflow
    assert "docker/build/llama-bin/llama/BUILD_INFO" in workflow
    assert 'org.cyberinference.llama-cpp.tag=${LLAMA_CPP_TAG}' in workflow
    assert 'org.cyberinference.llama-cpp.revision=${LLAMA_CPP_COMMIT}' in workflow


def test_publish_workflow_smoke_tests_complete_thor_mtp_contract():
    workflow = WORKFLOW.read_text(encoding="utf-8")

    assert "llama_help=$(/app/bin/llama-server --help 2>&1)" in workflow
    for capability in (
        "draft-mtp",
        "--spec-draft-model",
        "--spec-draft-n-max",
        "--parallel",
        "--flash-attn",
        "--chat-template-kwargs",
    ):
        assert capability in workflow
    assert "llama-server is missing required MTP capability" in workflow


def test_publish_workflow_never_deploys_to_production_thor():
    workflow = WORKFLOW.read_text(encoding="utf-8")

    assert "deploy-thor:" not in workflow
    assert "Deploy latest on Thor" not in workflow
    assert "Replace Thor service with latest image" not in workflow
    assert "docker stop cyber-inference" not in workflow


def test_publish_workflow_links_all_images_to_public_repository():
    workflow = WORKFLOW.read_text(encoding="utf-8")

    source_label = (
        '--label "org.opencontainers.image.source=https://github.com/${GITHUB_REPOSITORY}"'
    )
    assert workflow.count(source_label) == 2
