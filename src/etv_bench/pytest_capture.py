"""Freeze upstream pytest identities and capture unmodified operator calls."""

import hashlib
import inspect
import json
import math
import os
from pathlib import Path
import subprocess
import sys
import time

import pytest

from .capture import Capture


def encode(value):
    if value is None or type(value) in (str, int, bool, float):
        return value
    if isinstance(value, (tuple, list)):
        return [encode(v) for v in value]
    if isinstance(value, dict):
        return {str(k): encode(v) for k, v in sorted(value.items())}
    if hasattr(value, "shape") and hasattr(value, "dtype"):
        return {"shape": list(value.shape), "dtype": str(value.dtype),
                "stride": list(value.stride()), "device": str(value.device)}
    return str(value)


def write(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n")


def pytest_addoption(parser):
    parser.addoption("--etv-freeze", type=Path)
    parser.addoption("--etv-selection", type=Path)
    parser.addoption("--etv-capture-root", type=Path)


def pytest_collection_modifyitems(config, items):
    destination = config.getoption("--etv-freeze")
    records = [{"nodeid": item.nodeid, "function": item.nodeid.split("[")[0],
                "parameters": encode(getattr(getattr(item, "callspec", None), "params", {}))}
               for item in items]
    if destination:
        root = Path(config.rootpath)
        sha = subprocess.check_output(["git", "-C", str(root), "rev-parse", "HEAD"], text=True).strip()
        selected = {}
        def rank(record):
            params = record["parameters"]
            shape = params.get("shape", [])
            size = math.prod(shape) if isinstance(shape, list) and all(type(v) is int for v in shape) else 0
            return (params.get("dtype", "torch.float32") != "torch.float32",
                    bool(params.get("inplace", False)), size, record["nodeid"])
        for record in sorted(records, key=rank):
            selected.setdefault(record["function"], record["nodeid"])
        files = {str(p.relative_to(root)): hashlib.sha256(p.read_bytes()).hexdigest()
                 for p in sorted((root / "tests").rglob("*.py"))}
        write(destination, {"format": "etv-frozen-suite-v1", "source_commit": sha,
                            "selection_policy": "one per test function; prefer float32, out-of-place, smallest shape, lexical nodeid",
                            "files": files, "all_cases": records,
                            "selected": sorted(selected.values())})
    selection = config.getoption("--etv-selection")
    if selection:
        requested = set(json.loads(selection.read_text())["selected"])
        found = {item.nodeid for item in items} & requested
        if found != requested:
            raise pytest.UsageError("frozen node IDs missing: " + repr(sorted(requested - found)))
        deselected = [item for item in items if item.nodeid not in requested]
        items[:] = [item for item in items if item.nodeid in requested]
        config.hook.pytest_deselected(items=deselected)


def clone_arguments(arguments):
    import torch
    memo = {}
    def clone(value):
        if isinstance(value, torch.Tensor):
            if id(value) not in memo:
                memo[id(value)] = value.detach().clone(memory_format=torch.preserve_format)
            return memo[id(value)]
        if isinstance(value, tuple):
            return tuple(clone(v) for v in value)
        if isinstance(value, list):
            return [clone(v) for v in value]
        if isinstance(value, dict):
            return {k: clone(v) for k, v in value.items()}
        return value
    return clone(arguments)


def tensor_inputs(arguments):
    import torch
    result = {}
    def visit(value, key):
        if isinstance(value, torch.Tensor):
            result[key] = value
        elif isinstance(value, (tuple, list)):
            for i, element in enumerate(value):
                visit(element, key + str(i))
        elif isinstance(value, dict):
            for name, element in value.items():
                visit(element, key + "_" + name)
    for name, value in arguments.items():
        if name != "out":
            visit(value, "Input_" + name)
    return result


def reference(name):
    import torch
    if hasattr(torch, name):
        return getattr(torch, name)
    if hasattr(torch.nn.functional, name):
        return getattr(torch.nn.functional, name)
    raise ValueError("no public torch reference for " + name)


def capture_call(directory, nodeid, function, arguments):
    import torch
    start = time.perf_counter()
    status = {"nodeid": nodeid, "operator": function.__name__, "arguments": encode(arguments)}
    try:
        expected = reference(function.__name__)
        # A copied argument set preserves input values from before the upstream call.
        inputs = clone_arguments(arguments)
        capture = Capture(directory)
        capture.inputs(tensor_inputs(inputs))
        compiled = torch.compile(expected, fullgraph=True)
        compiled(**clone_arguments(inputs))
        torch.cuda.synchronize()
        with capture.side("lhs"):
            lhs = function(**inputs)
        if not isinstance(lhs, torch.Tensor):
            raise ValueError("multiple/non-tensor outputs require separate observations")
        capture.outputs("lhs", {"Output": lhs})
        with capture.side("rhs"):
            rhs = compiled(**inputs)
        if not isinstance(rhs, torch.Tensor):
            raise ValueError("multiple/non-tensor reference outputs")
        capture.outputs("rhs", {"Output": rhs})
        torch.cuda.synchronize()
        capture.write({"repository": "https://github.com/InfiniTensor/ntops",
                       "commit": "9ae4166ad342e4745f0eed13a5a20d069e994fc0", "test_id": nodeid})
        status.update(status="captured", launches=len(capture.launches), diagnostics=capture.diagnostics)
    except Exception as exc:
        status.update(status="capture_failed", error=type(exc).__name__ + ": " + str(exc)[:2000])
    status["seconds"] = time.perf_counter() - start
    write(directory / "case_status.json", status)


@pytest.hookimpl(hookwrapper=True)
def pytest_pyfunc_call(pyfuncitem):
    root = pyfuncitem.config.getoption("--etv-capture-root")
    if root is None:
        yield
        return
    calls = []
    active = set()
    def profile(frame, event, result):
        module = frame.f_globals.get("__name__", "")
        if not module.startswith("ntops.torch.") or module == "ntops.torch.utils":
            return
        function = frame.f_globals.get(frame.f_code.co_name)
        if not inspect.isfunction(function) or function.__code__ is not frame.f_code:
            return
        if frame.f_code.co_name.startswith("_") or frame.f_code.co_name != module.rsplit(".", 1)[-1]:
            return
        if event == "call":
            if not active:
                arguments = {name: frame.f_locals[name] for name in inspect.signature(function).parameters}
                calls.append((function, clone_arguments(arguments)))
            active.add(id(frame))
        elif event == "return":
            active.discard(id(frame))
    previous = sys.getprofile()
    if previous is not None:
        raise RuntimeError("capture requires an unused profiler")
    sys.setprofile(profile)
    try:
        outcome = yield
    finally:
        sys.setprofile(previous)
    case_id = hashlib.sha256(pyfuncitem.nodeid.encode()).hexdigest()[:16]
    directory = root / case_id
    write(directory / "upstream.json", {"nodeid": pyfuncitem.nodeid,
          "status": "failed" if outcome.excinfo else "passed", "operator_calls": len(calls),
          "failure": str(outcome.excinfo[1])[:2000] if outcome.excinfo else ""})
    if len(calls) == 1:
        capture_call(directory, pyfuncitem.nodeid, *calls[0])
    else:
        write(directory / "case_status.json", {"nodeid": pyfuncitem.nodeid, "status": "capture_failed",
              "error": "expected one outer operator call, found " + str(len(calls))})
