from pathlib import Path

from tools.extract_ntops_test_specs import extract


def test_extracts_test_evidence_and_excludes_dispatch_wrapper(tmp_path: Path) -> None:
    tests = tmp_path / "tests"
    tests.mkdir()
    (tmp_path / "LICENSE").write_text(
        "Apache License\nVersion 2.0, January 2004\n", encoding="utf-8"
    )
    (tests / "utils.py").write_text(
        "def generate_arguments():\n"
        "    return ('shape', [((4,),)])\n",
        encoding="utf-8",
    )
    (tests / "test_mean.py").write_text(
        "import torch\n"
        "from ntops.torch.mean import mean as ntops_mean\n"
        "\n"
        "def test_mean():\n"
        "    input_tensor = torch.randn((4,), device='cuda')\n"
        "    output = ntops_mean(input_tensor)\n"
        "    reference = torch.mean(input_tensor)\n"
        "    assert torch.allclose(output, reference)\n",
        encoding="utf-8",
    )
    (tests / "test_matmul.py").write_text(
        "def test_matmul():\n"
        "    assert True\n",
        encoding="utf-8",
    )

    result = extract(tmp_path)

    assert result["source"]["license"] == {
        "spdx": "Apache-2.0",
        "file": "LICENSE",
    }
    assert result["scope"] == {
        "operator_count": 1,
        "excluded": [
            {
                "operator": "matmul",
                "source": "tests/test_matmul.py",
                "reason": "dispatch wrapper over mm/bmm; no independent kernel",
            }
        ],
    }
    operator = result["operators"][0]
    assert operator["operator"] == "mean"
    assert operator["tests"][0]["candidate_calls"] == ["ntops_mean(input_tensor)"]
    assert operator["tests"][0]["tensor_factories"] == [
        "torch.randn((4,), device='cuda')"
    ]
    assert result["shared_helpers"]["functions"][0]["name"] == "generate_arguments"
