#!/usr/bin/env python3
"""Materialize standalone, requires-only ntops operator specifications."""

from __future__ import annotations

import argparse
import ast
import json
import re
from pathlib import Path
from typing import Any

import yaml


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_NTOPS = ROOT / "build" / "upstream" / "ntops"
DEFAULT_OUTPUT = ROOT / "specs" / "ntops"
EVIDENCE_PATH = ROOT / "specs" / "ntops_test_evidence.json"
CATALOG_PATH = ROOT / "docs" / "ntops_test_specs.md"
NTOPS_COMMIT = "9ae4166ad342e4745f0eed13a5a20d069e994fc0"
OBSOLETE_FILES = {Path("common.yaml")}

TENSOR_ARGUMENTS = {
    "attn_mask",
    "bias",
    "cos_table",
    "exponent",
    "input",
    "key",
    "mat",
    "mat1",
    "mat2",
    "max",
    "min",
    "other",
    "present_key",
    "present_key_slot",
    "present_value",
    "present_value_slot",
    "query",
    "running_mean",
    "running_var",
    "sin_table",
    "tensors",
    "value",
    "vec",
    "weight",
    "weights",
}

CATEGORIES = {
    "elementwise": {
        "abs", "acosh", "atan", "celu", "clamp", "cos", "cosh", "exp",
        "gelu", "isinf", "isnan", "neg", "pow", "relu", "round", "rsqrt",
        "sgn", "sigmoid", "sign", "signbit", "silu", "sin", "tanh", "threshold",
    },
    "binary_and_comparison": {
        "add", "bitwise_and", "bitwise_not", "bitwise_or", "div", "eq", "fmax",
        "ge", "gt", "le", "lt", "maximum", "mul", "ne", "sub",
    },
    "linear_algebra_and_normalization": {
        "addmm", "addmv", "batch_norm", "bmm", "instance_norm", "layer_norm",
        "mm", "rms_norm", "rotary_position_embedding", "scaled_dot_product_attention",
    },
    "reduction_sort_and_shape": {
        "argsort", "diag", "logsumexp", "max", "mean", "median", "msort",
        "quantile", "rot90", "select_copy", "softmax", "sort", "stack",
    },
    "pooling_convolution_and_random": {
        "adaptive_avg_pool2d", "adaptive_max_pool2d", "alpha_dropout", "avg_pool2d",
        "bincount", "conv2d", "dropout", "lp_pool1d", "lp_pool2d", "lp_pool3d",
        "max_pool1d", "max_pool2d", "max_pool3d",
    },
}

FLOAT_DOMAIN = {
    "abs", "add", "addmm", "addmv", "alpha_dropout", "batch_norm", "bitwise_not",
    "bmm", "celu",
    "clamp", "cos", "cosh", "div", "eq", "exp", "fmax", "ge", "gelu", "gt",
    "instance_norm", "isinf", "isnan", "layer_norm", "le", "logsumexp", "lt",
    "max", "maximum", "mean", "mm", "msort", "mul", "ne", "neg", "pow",
    "quantile", "relu", "rms_norm", "rot90", "round", "rsqrt", "select_copy", "sgn",
    "sigmoid", "sign", "signbit", "silu", "sin", "softmax", "sort", "sub", "tanh",
    "threshold", "dropout",
}
INTEGER_DOMAIN = {"bitwise_and", "bitwise_not", "bitwise_or"}
MATMUL_DOMAIN = {"addmm", "addmv", "bmm", "mm"}
FLOAT32_ASSERTED_ONLY = {"cos", "exp", "pow", "quantile", "sigmoid", "sin", "tanh"}

SAME_SHAPE_INPUTS = {
    "add": ["input", "other"],
    "bitwise_and": ["input", "other"],
    "bitwise_or": ["input", "other"],
    "clamp": ["input", "min", "max"],
    "div": ["input", "other"],
    "eq": ["input", "other"],
    "ge": ["input", "other"],
    "gt": ["input", "other"],
    "le": ["input", "other"],
    "lt": ["input", "other"],
    "mul": ["input", "other"],
    "ne": ["input", "other"],
    "pow": ["input", "exponent"],
    "sub": ["input", "other"],
}

SPECIFIC_REQUIRES: dict[str, list[dict[str, Any]]] = {
    "mm": [
        {"op": "rank_equals", "tensor": "input", "value": 2},
        {"op": "rank_equals", "tensor": "mat2", "value": 2},
        {"op": "dim_equal", "left": ["input", 1], "right": ["mat2", 0]},
    ],
    "bmm": [
        {"op": "rank_equals", "tensor": "input", "value": 3},
        {"op": "rank_equals", "tensor": "mat2", "value": 3},
        {"op": "dim_equal", "left": ["input", 0], "right": ["mat2", 0]},
        {"op": "dim_equal", "left": ["input", 2], "right": ["mat2", 1]},
    ],
    "addmm": [
        {"op": "shape_equals", "tensor": "input", "value": ["m", "n"]},
        {"op": "shape_equals", "tensor": "mat1", "value": ["m", "k"]},
        {"op": "shape_equals", "tensor": "mat2", "value": ["k", "n"]},
    ],
    "addmv": [
        {"op": "shape_equals", "tensor": "input", "value": ["m"]},
        {"op": "shape_equals", "tensor": "mat", "value": ["m", "k"]},
        {"op": "shape_equals", "tensor": "vec", "value": ["k"]},
    ],
    "batch_norm": [
        {"op": "rank_between", "tensor": "input", "lower": 2, "upper": 4},
        {"op": "channel_axis", "tensor": "input", "axis": 1},
        {"op": "optional_shape_equals", "tensor": "weight", "shape": ["C"]},
        {"op": "optional_shape_equals", "tensor": "bias", "shape": ["C"]},
        {"op": "equals", "left": "training", "right": True},
    ],
    "instance_norm": [
        {"op": "rank_between", "tensor": "input", "lower": 3, "upper": 4},
        {"op": "channel_axis", "tensor": "input", "axis": 1},
        {"op": "optional_shape_equals", "tensor": "weight", "shape": ["C"]},
        {"op": "optional_shape_equals", "tensor": "bias", "shape": ["C"]},
        {"op": "optional_shape_equals", "tensor": "running_mean", "shape": ["C"]},
        {"op": "optional_shape_equals", "tensor": "running_var", "shape": ["C"]},
    ],
    "layer_norm": [
        {"op": "nonempty_shape_suffix", "shape": "normalized_shape", "of": "input"},
        {"op": "optional_shape_equals", "tensor": "weight", "shape": "normalized_shape"},
        {"op": "optional_shape_equals", "tensor": "bias", "shape": "normalized_shape"},
    ],
    "rms_norm": [
        {"op": "nonempty_shape_suffix", "shape": "normalized_shape", "of": "input"},
        {"op": "optional_shape_equals", "tensor": "weight", "shape": "normalized_shape"},
    ],
    "rotary_position_embedding": [
        {"op": "rank_equals", "tensor": "input", "value": 4},
        {"op": "dim_even", "tensor": "input", "axis": 3},
        {"op": "shape_equals", "tensor": "sin_table", "value": ["L", "D/2"]},
        {"op": "same_shape", "values": ["sin_table", "cos_table"]},
    ],
    "scaled_dot_product_attention": [
        {"op": "rank_equals", "tensor": "query", "value": 4},
        {"op": "rank_equals", "tensor": "key", "value": 4},
        {"op": "rank_equals", "tensor": "value", "value": 4},
        {"op": "dtype_in", "tensor": "query", "values": ["float16", "float32"]},
        {"op": "dtype_in", "tensor": "key", "values": ["float16", "float32"]},
        {"op": "dtype_in", "tensor": "value", "values": ["float16", "float32"]},
        {
            "op": "dtype_in",
            "tensor": "attn_mask",
            "values": ["bool", "float16", "float32"],
        },
        {
            "op": "dtype_in",
            "tensor": "present_key",
            "values": ["float16", "float32"],
        },
        {
            "op": "dtype_in",
            "tensor": "present_value",
            "values": ["float16", "float32"],
        },
        {
            "op": "dtype_in",
            "tensor": "present_key_slot",
            "values": ["float16", "float32"],
        },
        {
            "op": "dtype_in",
            "tensor": "present_value_slot",
            "values": ["float16", "float32"],
        },
        {"op": "dim_equal", "left": ["query", 3], "right": ["key", 3]},
        {"op": "dim_equal", "left": ["query", 3], "right": ["value", 3]},
        {"op": "dim_equal", "left": ["key", 2], "right": ["value", 2]},
        {"op": "dim_equal", "left": ["query", 0], "right": ["key", 0]},
        {"op": "dim_equal", "left": ["query", 0], "right": ["value", 0]},
        {"op": "dim_equal", "left": ["key", 1], "right": ["value", 1]},
        {"op": "same_dtype", "values": ["query", "key", "value"]},
        {"op": "head_count_divisible", "query": "query", "key": "key"},
        {
            "op": "not_both",
            "left": {"op": "is_not_null", "value": "attn_mask"},
            "right": {"op": "is_true", "value": "is_causal"},
        },
        {"op": "equals", "left": "dropout_p", "right": 0},
        {"op": "equals", "left": "enable_gqa", "right": True},
    ],
    "diag": [{"op": "rank_in", "tensor": "input", "values": [1, 2]}],
    "fmax": [{"op": "broadcastable", "values": ["input", "other"]}],
    "maximum": [{"op": "broadcastable", "values": ["input", "other"]}],
    "argsort": [
        {"op": "rank_between", "tensor": "input", "lower": 1, "upper": 3},
        {"op": "nonnegative_valid_axis", "tensor": "input", "axis": "dim"},
    ],
    "median": [
        {"op": "rank_between", "tensor": "input", "lower": 1, "upper": 3},
        {"op": "nonnegative_valid_axis", "tensor": "input", "axis": "dim"},
        {"op": "equals", "left": "keepdim", "right": False},
    ],
    "select_copy": [
        {"op": "nonnegative_valid_axis", "tensor": "input", "axis": "dim"},
        {"op": "index_in_axis", "tensor": "input", "axis": "dim", "index": "index"},
    ],
    "logsumexp": [{"op": "valid_axis", "tensor": "input", "axis": "dim"}],
    "max": [{"op": "optional_valid_axis", "tensor": "input", "axis": "dim"}],
    "mean": [{"op": "optional_valid_axis", "tensor": "input", "axis": "dim"}],
    "softmax": [{"op": "nonnegative_valid_axis", "tensor": "input", "axis": "dim"}],
    "sort": [{"op": "valid_axis", "tensor": "input", "axis": "dim"}],
    "rot90": [
        {"op": "rank_between", "tensor": "input", "lower": 2, "upper": 4},
        {"op": "two_distinct_valid_axes", "tensor": "input", "axes": "dims"},
    ],
    "quantile": [
        {"op": "all_values_in_half_open_interval", "value": "q", "lower": 0, "upper": 1},
        {"op": "optional_nonnegative_valid_axis", "tensor": "input", "axis": "dim"},
    ],
    "stack": [
        {"op": "sequence_length_between", "value": "tensors", "lower": 2, "upper": 5},
        {"op": "rank_equals", "tensor": "tensors[*]", "value": 3},
        {"op": "dtype_in", "tensor": "tensors[*]", "values": ["float32"]},
        {"op": "all_dims_between", "tensor": "tensors[*]", "lower": 16, "upper": 64},
        {"op": "generated_by", "tensor": "tensors[*]", "generator": "torch.randn"},
        {"op": "all_same_shape", "values": "tensors"},
        {"op": "all_same_dtype", "values": "tensors"},
        {"op": "value_in", "value": "dim", "values": [0, 1, 2]},
    ],
    "conv2d": [
        {"op": "shape_equals", "tensor": "input", "value": [2, 3, 112, 112]},
        {"op": "shape_equals", "tensor": "weight", "value": [4, 3, "r", "s"]},
        {"op": "shape_equals", "tensor": "bias", "value": [4]},
        {"op": "equals", "left": "groups", "right": 1},
    ],
    "adaptive_avg_pool2d": [{"op": "rank_equals", "tensor": "input", "value": 4}],
    "adaptive_max_pool2d": [{"op": "rank_equals", "tensor": "input", "value": 4}],
    "avg_pool2d": [{"op": "shape_equals", "tensor": "input", "value": [2, 3, 112, 112]}],
    "max_pool1d": [
        {"op": "rank_equals", "tensor": "input", "value": 3},
        {
            "op": "value_pair_in",
            "values": ["kernel_size", "stride"],
            "pairs": [[2, 2], [3, 2], [3, 1]],
        },
    ],
    "max_pool2d": [{"op": "shape_equals", "tensor": "input", "value": [2, 3, 112, 112]}],
    "max_pool3d": [{"op": "rank_equals", "tensor": "input", "value": 5}],
    "lp_pool1d": [{"op": "rank_equals", "tensor": "input", "value": 3}],
    "lp_pool2d": [{"op": "rank_equals", "tensor": "input", "value": 4}],
    "lp_pool3d": [{"op": "rank_equals", "tensor": "input", "value": 5}],
    "bincount": [
        {"op": "rank_equals", "tensor": "input", "value": 1},
        {"op": "dtype_in", "tensor": "input", "values": ["int32"]},
        {"op": "all_elements_nonnegative", "tensor": "input"},
        {"op": "dtype_in", "tensor": "weights", "values": ["float32"]},
        {"op": "integer_greater_equal", "value": "minlength", "lower": 0},
        {"op": "same_shape", "values": ["input", "weights"], "when": "weights is not null"},
    ],
}

DIRECT_ARGUMENT_REQUIRES: dict[str, dict[str, list[dict[str, Any]]]] = {
    "add": {
        "alpha": [{"op": "generated_by", "value": "alpha", "generator": "random.gauss(0, 1)"}]
    },
    "sub": {
        "alpha": [{"op": "generated_by", "value": "alpha", "generator": "random.gauss(0, 1)"}]
    },
    "addmm": {
        name: [{"op": "generated_by", "value": name, "generator": "random.gauss(0, 1)"}]
        for name in ("alpha", "beta")
    },
    "addmv": {
        name: [{"op": "generated_by", "value": name, "generator": "random.gauss(0, 1)"}]
        for name in ("alpha", "beta")
    },
    "alpha_dropout": {
        "p": [{"op": "value_between", "value": "p", "lower": 0.1, "upper": 0.5}],
        "training": [{"op": "value_in", "value": "training", "values": [False, True]}],
        "inplace": [{"op": "equals", "left": "inplace", "right": False}],
    },
    "avg_pool2d": {
        "count_include_pad": [
            {"op": "equals", "left": "count_include_pad", "right": True}
        ],
        "divisor_override": [
            {"op": "equals", "left": "divisor_override", "right": None}
        ],
    },
    "celu": {"alpha": [{"op": "equals", "left": "alpha", "right": 1.0}]},
    "dropout": {
        "p": [
            {
                "op": "value_between",
                "value": "p",
                "lower": 0.0,
                "upper": 1.0,
                "upper_inclusive": False,
            }
        ],
        "training": [{"op": "equals", "left": "training", "right": True}],
        "inplace": [{"op": "equals", "left": "inplace", "right": False}],
    },
    "instance_norm": {
        "momentum": [{"op": "equals", "left": "momentum", "right": 0.1}]
    },
    "lp_pool1d": {
        "kernel_size": [
            {
                "op": "generated_by",
                "value": "kernel_size",
                "generator": "random.randint(1, min(5, input.shape[2]))",
            }
        ],
        "stride": [
            {
                "op": "generated_by",
                "value": "stride",
                "generator": "None or random.randint(1, kernel_size)",
            }
        ],
    },
    "lp_pool2d": {
        "kernel_size": [
            {
                "op": "generated_by",
                "value": "kernel_size",
                "generator": "scalar or pair; each axis is between 1 and min(5, input spatial size)",
            }
        ],
        "stride": [
            {
                "op": "generated_by",
                "value": "stride",
                "generator": "None or scalar/pair no larger than kernel_size per axis",
            }
        ],
    },
    "lp_pool3d": {
        "kernel_size": [
            {
                "op": "generated_by",
                "value": "kernel_size",
                "generator": "scalar or triple; each axis is between 1 and min(4, input spatial size)",
            }
        ],
        "stride": [
            {
                "op": "generated_by",
                "value": "stride",
                "generator": "None or scalar/triple no larger than kernel_size per axis",
            }
        ],
    },
    "max_pool1d": {
        "kernel_size": [
            {"op": "value_in", "value": "kernel_size", "values": [2, 3]}
        ],
        "stride": [{"op": "value_in", "value": "stride", "values": [1, 2]}],
        "dilation": [{"op": "equals", "left": "dilation", "right": 1}],
    },
    "max_pool2d": {
        "return_indices": [{"op": "equals", "left": "return_indices", "right": False}]
    },
    "max_pool3d": {
        "kernel_size": [
            {
                "op": "generated_by",
                "value": "kernel_size",
                "generator": "2, 3, or a length-3 tuple with each value in {2,3}",
            }
        ],
        "stride": [
            {
                "op": "generated_by",
                "value": "stride",
                "generator": "1, 2, or a length-3 tuple with each value in {1,2}",
            }
        ],
        "dilation": [{"op": "equals", "left": "dilation", "right": 1}],
    },
    "mean": {"dtype": [{"op": "equals", "left": "dtype", "right": None}]},
    "rot90": {
        "k": [
            {
                "op": "generated_by",
                "value": "k",
                "generator": "base in {0,1,2,3} + 4 * random.randint(-100, 100)",
            }
        ]
    },
    "round": {"decimals": [{"op": "equals", "left": "decimals", "right": 0}]},
    "scaled_dot_product_attention": {
        "is_causal": [
            {"op": "value_in", "value": "is_causal", "values": [False, True]}
        ],
        "scale": [
            {
                "op": "generated_by",
                "value": "scale",
                "generator": "None or random.uniform(0.05, 0.5)",
            }
        ],
        "causal_variant": [
            {
                "op": "value_in",
                "value": "causal_variant",
                "values": [None, "CausalVariant.LOWER_RIGHT", "CausalVariant.UPPER_LEFT"],
            }
        ],
    },
    "silu": {"inplace": [{"op": "equals", "left": "inplace", "right": False}]},
    "softmax": {
        "dtype": [
            {
                "op": "value_in",
                "value": "dtype",
                "values": ["float16", "float32", "float64"],
            }
        ]
    },
    "threshold": {
        "threshold": [
            {"op": "value_between", "value": "threshold", "lower": -1.0, "upper": 1.0}
        ],
        "value": [{"op": "value_between", "value": "value", "lower": 0.0, "upper": 1.0}],
        "inplace": [{"op": "equals", "left": "inplace", "right": False}],
    },
}

PREDICATE_OPS = {
    "all_dims_between", "all_elements_nonnegative", "all_same_dtype", "all_same_shape",
    "all_values_in_half_open_interval", "broadcastable", "channel_axis", "device_is",
    "dim_equal", "dim_even", "dtype_in", "equals", "generated_by",
    "head_count_divisible", "index_in_axis", "integer_between", "integer_greater_equal",
    "is_not_null", "is_true", "nonempty_shape_suffix", "not_both", "nonnegative_valid_axis",
    "optional_nonnegative_valid_axis", "optional_shape_equals", "optional_valid_axis",
    "rank_between", "rank_equals", "rank_in", "same_dtype", "same_shape",
    "sequence_length_between", "shape_equals", "two_distinct_valid_axes", "valid_axis",
    "value_between", "value_in", "value_pair_in",
}

_SKIPPED_CASE = object()


def _category_map() -> dict[str, str]:
    result: dict[str, str] = {}
    for category, operators in CATEGORIES.items():
        for operator in operators:
            if operator in result:
                raise ValueError(f"operator appears in multiple categories: {operator}")
            result[operator] = category
    return result


def _catalog() -> dict[str, dict[str, str]]:
    pattern = re.compile(r"^\| `([^`]+)` \| (.*) \| (.*) \|$")
    result = {}
    for line in CATALOG_PATH.read_text(encoding="utf-8").splitlines():
        match = pattern.match(line)
        if match:
            name, input_summary, _ = match.groups()
            result[name] = {"input_domain_zh": input_summary}
    return result


def _default_source(node: ast.expr | None, source: str) -> Any:
    if node is None:
        return None
    segment = ast.get_source_segment(source, node)
    if segment == "None":
        return None
    if segment in {"True", "False"}:
        return segment == "True"
    try:
        return ast.literal_eval(node)
    except (ValueError, SyntaxError):
        return segment


def _signature(ntops_repo: Path, operator: str) -> dict[str, Any]:
    path = ntops_repo / "src" / "ntops" / "torch" / f"{operator}.py"
    source = path.read_text(encoding="utf-8")
    tree = ast.parse(source, filename=str(path))
    function = next(
        node
        for node in tree.body
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef))
        and node.name == operator
    )
    positional = [*function.args.posonlyargs, *function.args.args]
    positional_defaults = [None] * (len(positional) - len(function.args.defaults)) + list(
        function.args.defaults
    )
    arguments = [
        (argument.arg, default)
        for argument, default in zip(positional, positional_defaults, strict=True)
    ]
    arguments.extend(
        (argument.arg, default)
        for argument, default in zip(
            function.args.kwonlyargs, function.args.kw_defaults, strict=True
        )
    )

    inputs = []
    parameters = []
    for name, default_node in arguments:
        default = _default_source(default_node, source)
        if name == "out":
            parameters.append(
                {"name": name, "kind": "optional_output_buffer", "default": default}
            )
        elif name in TENSOR_ARGUMENTS and not (
            operator == "threshold" and name == "value"
        ):
            kind = "tensor_sequence" if name == "tensors" else "tensor"
            entry: dict[str, Any] = {
                "name": name,
                "kind": kind,
                "shape_var": f"S_{name}",
                "dtype_var": f"T_{name}",
            }
            if default_node is not None:
                entry.update({"optional": default is None, "default": default})
            inputs.append(entry)
        else:
            entry = {"name": name, "kind": "scalar_or_shape_parameter"}
            if default_node is not None:
                entry["default"] = default
            else:
                entry["required"] = True
            parameters.append(entry)

    return {"inputs": inputs, "parameters": parameters}


def _tensor_domain(operator: str, tensor: str) -> list[dict[str, Any]]:
    """Return a copied, self-contained base domain for one tensor argument."""
    constraints: list[dict[str, Any]] = [
        {"op": "device_is", "tensor": tensor, "value": "cuda"}
    ]
    if operator in MATMUL_DOMAIN:
        constraints.insert(
            0,
            {
                "op": "dtype_in",
                "tensor": tensor,
                "values": ["float16", "float32"],
            },
        )
        return constraints
    if operator in INTEGER_DOMAIN:
        return [
            {"op": "rank_between", "tensor": tensor, "lower": 1, "upper": 4},
            {
                "op": "dtype_in",
                "tensor": tensor,
                "values": ["bool", "int8", "int16", "int32"],
            },
            {"op": "device_is", "tensor": tensor, "value": "cuda"},
            {"op": "all_dims_between", "tensor": tensor, "lower": 1, "upper": 1024},
            {
                "op": "generated_by",
                "tensor": tensor,
                "generator": "torch.rand > 0.5 for bool; torch.randint(-10, 10) otherwise",
            },
        ]
    if operator in FLOAT_DOMAIN:
        dtypes = (
            ["float32"]
            if operator in FLOAT32_ASSERTED_ONLY
            else ["float16", "float32"]
        )
        return [
            {"op": "rank_between", "tensor": tensor, "lower": 1, "upper": 4},
            {"op": "dtype_in", "tensor": tensor, "values": dtypes},
            {"op": "device_is", "tensor": tensor, "value": "cuda"},
            {"op": "all_dims_between", "tensor": tensor, "lower": 1, "upper": 1024},
            {"op": "generated_by", "tensor": tensor, "generator": "torch.randn"},
        ]
    return constraints


def _standalone_summary(summary: str) -> str:
    return (
        summary.replace("一个 `F` 张量", "一个由测试生成的浮点张量")
        .replace("两个同 shape 的 `F` 张量", "两个同 shape 的浮点张量")
        .replace("`F`/`randn`", "由 `torch.randn` 生成的浮点")
        .replace("`F` shape", "测试生成的浮点 shape")
        .replace("`F` 中", "测试生成的浮点输入中")
        .replace("`F` 的", "测试生成的浮点输入中的")
        .replace("`F`", "测试生成的浮点输入域")
        .replace("`I`", "测试生成的整数输入域")
        .replace("`M`", "测试生成的矩阵维度域")
    )


def _predicate_targets(predicate: dict[str, Any], argument_names: set[str]) -> set[str]:
    targets: set[str] = set()
    for key in (
        "tensor",
        "value",
        "shape",
        "axis",
        "axes",
        "index",
        "left",
        "right",
    ):
        value = predicate.get(key)
        if isinstance(value, str) and value in argument_names:
            targets.add(value)
        elif isinstance(value, str) and value.endswith("[*]"):
            sequence = value.removesuffix("[*]")
            if sequence in argument_names:
                targets.add(sequence)
    return targets


def _decorator_value(node: ast.expr) -> Any:
    if isinstance(node, ast.Constant):
        return node.value
    if isinstance(node, (ast.List, ast.Tuple, ast.Set)):
        return [_decorator_value(item) for item in node.elts]
    if isinstance(node, ast.UnaryOp) and isinstance(node.op, ast.USub):
        return -_decorator_value(node.operand)
    if isinstance(node, ast.Attribute):
        return node.attr
    if isinstance(node, ast.Call) and ast.unparse(node.func).endswith("pytest.param"):
        if any(".skip" in ast.unparse(keyword.value) for keyword in node.keywords):
            return _SKIPPED_CASE
        return [_decorator_value(argument) for argument in node.args]
    raise ValueError("parameter value is generated dynamically")


def _observed_argument_values(module: dict[str, Any]) -> dict[str, list[Any]]:
    observed: dict[str, list[Any]] = {}
    for test in module["tests"]:
        for parametrization in test["parametrization"]:
            try:
                call = ast.parse(parametrization, mode="eval").body
                if not isinstance(call, ast.Call) or len(call.args) < 2:
                    continue
                names = _decorator_value(call.args[0]).split(",")
                names = [name.strip() for name in names]
                cases = _decorator_value(call.args[1])
                if not isinstance(cases, list):
                    cases = [cases]
                rows = cases if len(names) > 1 else [[case] for case in cases]
                for row in rows:
                    if row is _SKIPPED_CASE or (
                        isinstance(row, list) and _SKIPPED_CASE in row
                    ):
                        continue
                    if not isinstance(row, list) or len(row) < len(names):
                        raise ValueError("invalid parametrization row")
                    for name, value in zip(names, row, strict=False):
                        values = observed.setdefault(name, [])
                        if value not in values:
                            values.append(value)
            except (SyntaxError, TypeError, ValueError):
                continue
    return observed


def _requirements(
    operator: str,
    signature: dict[str, Any],
    summary: str,
    module: dict[str, Any],
) -> dict[str, Any]:
    signature_arguments = [*signature["inputs"], *signature["parameters"]]
    argument_names = {argument["name"] for argument in signature_arguments}
    observed_values = _observed_argument_values(module)
    direct_constraints = DIRECT_ARGUMENT_REQUIRES.get(operator, {})
    observed_dtypes = [
        value
        for value in observed_values.get("dtype", [])
        if isinstance(value, str)
        and value.startswith(("bool", "int", "float", "complex"))
    ]
    candidate_calls = [
        call for test in module["tests"] for call in test["candidate_calls"]
    ]
    has_explicit_out = any(re.search(r"\bout\s*=", call) for call in candidate_calls)
    arguments: dict[str, Any] = {}
    for argument in signature_arguments:
        name = argument["name"]
        entry = {"kind": argument["kind"]}
        for field in ("required", "optional", "default"):
            if field in argument:
                entry[field] = argument[field]
        entry["constraints"] = (
            _tensor_domain(operator, name)
            if argument["kind"] in {"tensor", "tensor_sequence"}
            else []
        )
        if (
            observed_dtypes
            and argument["kind"] in {"tensor", "tensor_sequence"}
            and not any(item["op"] == "dtype_in" for item in entry["constraints"])
        ):
            entry["constraints"].append(
                {"op": "dtype_in", "tensor": name, "values": observed_dtypes}
            )
        if argument["kind"] == "optional_output_buffer":
            if has_explicit_out:
                entry["constraints"].append(
                    {
                        "op": "generated_by",
                        "value": name,
                        "generator": "None or a compatible output buffer constructed by the test",
                    }
                )
            else:
                entry["constraints"].append(
                    {"op": "equals", "left": name, "right": None}
                )
        arguments[name] = entry

    for name, values in observed_values.items():
        if name in arguments and name not in direct_constraints:
            arguments[name]["constraints"].append(
                {"op": "value_in", "value": name, "values": values}
            )
    for name, constraints in direct_constraints.items():
        arguments[name]["constraints"].extend(constraints)

    relations: list[dict[str, Any]] = []
    predicates: list[dict[str, Any]] = []
    if operator in MATMUL_DOMAIN:
        predicates.append(
            {"op": "integer_between", "values": ["m", "n", "k"], "lower": 1, "upper": 1024}
        )
    if operator in SAME_SHAPE_INPUTS:
        predicates.extend(
            [
                {"op": "same_shape", "values": SAME_SHAPE_INPUTS[operator]},
                {"op": "same_dtype", "values": SAME_SHAPE_INPUTS[operator]},
            ]
        )
    predicates.extend(SPECIFIC_REQUIRES.get(operator, []))
    for predicate in predicates:
        targets = _predicate_targets(predicate, argument_names)
        if len(targets) == 1:
            constraints = arguments[targets.pop()]["constraints"]
            if predicate not in constraints:
                constraints.append(predicate)
        else:
            relations.append(predicate)
    return {
        "observed_domain_zh": _standalone_summary(summary),
        "arguments": arguments,
        "relations": relations,
    }


def _schema() -> dict[str, Any]:
    predicate = {
        "type": "object",
        "required": ["op"],
        "properties": {"op": {"type": "string", "enum": sorted(PREDICATE_OPS)}},
        "additionalProperties": True,
    }
    return {
        "$schema": "https://json-schema.org/draft/2020-12/schema",
        "$id": "urn:etv:operator-requires:v1",
        "title": "ETV standalone test-derived operator requirements",
        "type": "object",
        "required": [
            "schema", "operator", "requires", "evidence",
        ],
        "properties": {
            "schema": {"const": "operator-requires/v1"},
            "operator": {
                "type": "object",
                "required": ["id", "name", "revision", "category"],
                "properties": {
                    "id": {"type": "string"},
                    "name": {"type": "string"},
                    "revision": {"type": "integer", "minimum": 1},
                    "category": {"type": "string"},
                },
                "additionalProperties": False,
            },
            "requires": {
                "type": "object",
                "required": ["observed_domain_zh", "arguments", "relations"],
                "properties": {
                    "observed_domain_zh": {"type": "string"},
                    "arguments": {
                        "type": "object",
                        "minProperties": 1,
                        "additionalProperties": {
                            "type": "object",
                            "required": ["kind", "constraints"],
                            "properties": {
                                "kind": {"type": "string"},
                                "required": {"type": "boolean"},
                                "optional": {"type": "boolean"},
                                "default": {},
                                "constraints": {"type": "array", "items": predicate},
                            },
                            "additionalProperties": False,
                        },
                    },
                    "relations": {"type": "array", "items": predicate},
                },
                "additionalProperties": False,
            },
            "evidence": {
                "type": "object",
                "required": ["status", "source", "tests", "reviewed"],
                "properties": {
                    "status": {"const": "inferred_from_tests"},
                    "source": {
                        "type": "object",
                        "required": ["repository", "commit", "license"],
                        "properties": {
                            "repository": {"type": "string"},
                            "commit": {"type": "string"},
                            "license": {"type": "string"},
                        },
                        "additionalProperties": False,
                    },
                    "tests": {
                        "type": "array",
                        "minItems": 1,
                        "items": {
                            "type": "object",
                            "required": ["path", "name", "line"],
                            "properties": {
                                "path": {"type": "string"},
                                "name": {"type": "string"},
                                "line": {"type": "integer", "minimum": 1},
                            },
                            "additionalProperties": False,
                        },
                    },
                    "reviewed": {"const": True},
                },
                "additionalProperties": False,
            },
        },
        "additionalProperties": False,
    }


def _validate_spec(spec: dict[str, Any]) -> None:
    required = set(_schema()["required"])
    missing = required - spec.keys()
    if missing:
        raise ValueError(f"{spec.get('operator')}: missing fields {sorted(missing)}")
    predicate_groups = [spec["requires"]["relations"]]
    predicate_groups.extend(
        argument["constraints"]
        for argument in spec["requires"]["arguments"].values()
    )
    for group in predicate_groups:
        if not isinstance(group, list) or any(
            not isinstance(predicate, dict) or not predicate.get("op")
            for predicate in group
        ):
            raise ValueError(f"invalid predicate group in {spec['operator']['id']}")
        for predicate in group:
            _validate_predicate_tree(predicate, spec["operator"]["id"])


def _validate_predicate_tree(value: Any, operator_id: str) -> None:
    if isinstance(value, dict):
        if "op" in value and value["op"] not in PREDICATE_OPS:
            raise ValueError(f"unknown predicate {value['op']!r} in {operator_id}")
        for child in value.values():
            _validate_predicate_tree(child, operator_id)
    elif isinstance(value, list):
        for child in value:
            _validate_predicate_tree(child, operator_id)


def build_specs(ntops_repo: Path = DEFAULT_NTOPS) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    evidence = json.loads(EVIDENCE_PATH.read_text(encoding="utf-8"))
    if evidence["source"]["commit"] != NTOPS_COMMIT:
        raise ValueError("test evidence is not from the expected ntops commit")
    catalog = _catalog()
    categories = _category_map()
    evidence_by_name = {item["operator"]: item for item in evidence["operators"]}
    if set(catalog) != set(categories) or set(catalog) != set(evidence_by_name):
        raise ValueError("catalog, categories, and test evidence do not cover the same operators")

    specs = []
    index_entries = []
    for operator in sorted(catalog):
        module = evidence_by_name[operator]
        tests = module["tests"]
        signature = _signature(ntops_repo, operator)
        spec = {
            "schema": "operator-requires/v1",
            "operator": {
                "id": f"ntops.torch.{operator}",
                "name": operator,
                "revision": 1,
                "category": categories[operator],
            },
            "requires": _requirements(
                operator, signature, catalog[operator]["input_domain_zh"], module
            ),
            "evidence": {
                "status": "inferred_from_tests",
                "source": {
                    "repository": evidence["source"]["repository"],
                    "commit": evidence["source"]["commit"],
                    "license": evidence["source"]["license"]["spdx"],
                },
                "tests": [
                    {
                        "path": module["source"],
                        "name": test["name"],
                        "line": test["line"],
                    }
                    for test in tests
                ],
                "reviewed": True,
            },
        }
        _validate_spec(spec)
        specs.append(spec)
        index_entries.append(
            {
                "id": spec["operator"]["id"],
                "path": f"operators/{operator}.yaml",
                "category": categories[operator],
                "test": module["source"],
            }
        )

    index = {
        "schema": "operator-requires-index/v1",
        "source": {
            "repository": evidence["source"]["repository"],
            "commit": evidence["source"]["commit"],
            "license": "Apache-2.0",
        },
        "operator_schema": "operator_spec.schema.json",
        "operator_count": len(specs),
        "operators": index_entries,
    }
    return index, specs


class _NoAliasDumper(yaml.SafeDumper):
    def ignore_aliases(self, data: Any) -> bool:
        return True


def _yaml(document: dict[str, Any]) -> str:
    return yaml.dump(
        document,
        Dumper=_NoAliasDumper,
        allow_unicode=True,
        sort_keys=False,
        default_flow_style=False,
        width=100,
    )


def expected_files(ntops_repo: Path = DEFAULT_NTOPS) -> dict[Path, str]:
    index, specs = build_specs(ntops_repo)
    result = {
        Path("index.yaml"): _yaml(index),
        Path("operator_spec.schema.json"): json.dumps(
            _schema(), indent=2, ensure_ascii=True
        )
        + "\n",
    }
    for spec in specs:
        result[Path("operators") / f"{spec['operator']['name']}.yaml"] = _yaml(spec)
    return result


def materialize(output: Path, ntops_repo: Path = DEFAULT_NTOPS, check: bool = False) -> None:
    expected = expected_files(ntops_repo)
    if check:
        errors = []
        for relative, content in expected.items():
            path = output / relative
            if not path.is_file():
                errors.append(f"missing: {relative}")
            elif path.read_text(encoding="utf-8") != content:
                errors.append(f"stale: {relative}")
        expected_operator_files = {
            relative for relative in expected if relative.parent == Path("operators")
        }
        actual_operator_files = {
            path.relative_to(output)
            for path in (output / "operators").glob("*.yaml")
        }
        for relative in sorted(actual_operator_files - expected_operator_files):
            errors.append(f"unexpected: {relative}")
        for relative in sorted(OBSOLETE_FILES):
            if (output / relative).exists():
                errors.append(f"unexpected: {relative}")
        if errors:
            raise ValueError("generated ntops specs are inconsistent:\n" + "\n".join(errors))
        return

    for relative, content in expected.items():
        path = output / relative
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(content, encoding="utf-8")
    for relative in OBSOLETE_FILES:
        path = output / relative
        if path.is_file():
            path.unlink()


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--ntops-repo", type=Path, default=DEFAULT_NTOPS)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--check", action="store_true")
    arguments = parser.parse_args()
    materialize(
        arguments.output.resolve(),
        arguments.ntops_repo.resolve(),
        check=arguments.check,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
