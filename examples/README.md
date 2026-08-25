# Examples

The examples machine-check the archive's add frontier proof and its required
negative cases.

| Spec | Verdict | Purpose |
| --- | --- | --- |
| `specs/add_proved.json` | `PROVED` | 2-D stride-aware ntops vs linear Inductor |
| `specs/add_fma_proved.json` | `PROVED` | Z3-admitted `fma == mul+add` under `ABSTRACT_FLOAT` |
| `specs/add_bad_stride.json` | `DISPROVED` | output address differs at the first later row |
| `specs/add_bad_mask.json` | `DISPROVED` | rhs omits logical element 127 |
| `specs/add_bad_compute.json` | `DISPROVED` | exact value model distinguishes add from sub |
| `specs/add_missing_alias.json` | `UNKNOWN` | required no-alias premise is absent |

Run all cases:

```bash
.venv/bin/python -m etv check examples/specs/add_proved.json
.venv/bin/python -m etv check examples/specs/add_fma_proved.json
.venv/bin/python -m etv check examples/specs/add_bad_stride.json
.venv/bin/python -m etv check examples/specs/add_bad_mask.json
.venv/bin/python -m etv check examples/specs/add_bad_compute.json
.venv/bin/python -m etv check examples/specs/add_missing_alias.json
```

The negative commands exit with 1 and UNKNOWN exits with 2, which is intentional
for CI. Add `--out build/<case>` to retain `report.json` and `report.md`.
