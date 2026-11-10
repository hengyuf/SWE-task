#!/usr/bin/env python3
"""teacher-deepswe setup (2026-10-30): every offline check of the SETUP phase in one run; writes
analysis/offline-checks.json and prints a summary. No Slurm job, no sandbox call; hub: one key_budget.py check of
the campaign key NVAPI_KEY_NEW_2 (one 4-token GLM call; the key since 2026-10-31 18:15 per OPERATOR_NOTE 18:15; from
2026-10-30 21:45 until its floor stop it was NVAPI_KEY_NEW_3, before that NVAPI_KEY_NEW_2). Read-only outside this
directory.

    /usr/bin/python3 -I -B setup/offline_checks.py [--skip-budget]

Checks:
  compile        every *.py here (compile() in memory, no bytecode written) + bash -n submit.sh / driver.sbatch
  dry_run        driver.py --dry-run for all 18 campaigns (3 models x pilot + w1-g0..2 + w2-g0..1), with the ATP
                 controller python as driver.sbatch runs it: 113 tasks per model, 79 in wave 1, no duplicates, the same
                 task lists for every model, every task dir has task.toml, instruction.md and tests/
  dataset_row    synth rollouts.dataset_row() for all 113 task dirs: image == task.toml docker_image, task_id, a
                 non-empty instruction, a positive verifier timeout
  labels         the longest value of every OpenSandbox / k8s label ATP builds (driver.sandbox_labels) <= 63
  budget         key_budget.py NVAPI_KEY_NEW_2 (name and numbers; ok also needs >= the $1,700 start floor)
  config         setup/config_check.py all_ok (ATP build_config: limits, sampling, model ids, no prompt override)
  code_identity  brokenreply.classify/_reverse_lines/last_reply/broken_end and degenerate.classify/scan are KR4's
                 code (ast dump equality); probe_worker / driver stock-prompt sha256; driver specs == settings.json
                 (kimi, glm) and == teacher-rollouts-v2 driver.py HUB_MODELS["deepseek"] (ast literal)
  hygiene        no __pycache__ here; no "nvapi-" / "sk-" credential-looking string in any text file here
  inodes         inodes and bytes per slot measured on completed KR4 (Kimi, 256k) and TR2 (GLM / DeepSeek, 1M)
                 campaigns, x 565 slots per model x (1 + re-run share) vs lfs quota free (user, project 10581)
"""
import sys

sys.dont_write_bytecode = True

import argparse  # noqa: E402
import ast  # noqa: E402
import hashlib  # noqa: E402
import json  # noqa: E402
import os  # noqa: E402
from pathlib import Path  # noqa: E402
import re  # noqa: E402
import subprocess  # noqa: E402

TD = Path(__file__).resolve().parents[1]
S = Path("/lustre/fsw/portfolios/nemotron/users/hengyuf/swe-task")
PY = str(S / "atp/runtime/controller/bin/python3")
KR4 = S / "sft/kimi-rollouts-v4-stock256k-20261028"
TR2 = S / "sft/teacher-rollouts-v2-20261025"
TASKS = Path("/lustre/fsw/portfolios/nemotron/users/hengyuf/swe-rl-v2-sc-migration/eval/deepswe-v1.1/cache/tasks")
MODELS = ("kimi", "glm", "deepseek")
DEFAULT_CONCURRENCY = {"kimi": 8, "glm": 8, "deepseek": 5}  # submit.sh DEFAULT_CONCURRENCY (checked below)
CAMPAIGNS = [("pilot", 0, "pilot"), ("deepswe-w1", 0, "w1-g0"), ("deepswe-w1", 1, "w1-g1"), ("deepswe-w1", 2, "w1-g2"),
             ("deepswe-w2", 0, "w2-g0"), ("deepswe-w2", 1, "w2-g1")]


def check_compile() -> dict:
    files = sorted(p for p in TD.rglob("*.py") if "selftest" not in p.parts)
    errors = {}
    for path in files:
        try:
            compile(path.read_text(), str(path), "exec")
        except SyntaxError as error:
            errors[str(path)] = str(error)
    shells = {}
    for name in ("submit.sh", "driver.sbatch"):
        result = subprocess.run(["bash", "-n", str(TD / name)], capture_output=True, text=True)
        shells[name] = result.returncode == 0 and not result.stderr
    return {"python_files": len(files), "errors": errors, "shell_syntax": shells,
            "ok": not errors and all(shells.values())}


def check_dry_run() -> dict:
    out = {"campaigns": {}}
    lists = {}
    for model in MODELS:
        for taskset, group, name in CAMPAIGNS:
            cap = "1" if name == "pilot" else str(DEFAULT_CONCURRENCY[model])  # submit.sh's defaults
            args = [PY, "-B", str(TD / "driver.py"), "--model", model, "--taskset", taskset, "--group", str(group),
                    "--max-concurrency", cap, "--dry-run"]
            result = subprocess.run(args, capture_output=True, text=True, cwd=str(S / "synth"), timeout=600)
            if result.returncode:
                out["campaigns"][f"{model}-{name}"] = {"error": result.stderr[-500:]}
                continue
            data = json.loads(result.stdout)
            lists[(model, name)] = data
            out["campaigns"][f"{model}-{name}"] = {"tasks": len(data["tasks"]), "slots": data["slots"],
                                                   "repeats": data["repeats"], "wave": data["wave"],
                                                   "max_concurrency": data["max_concurrency"],
                                                   "spec": data["spec"], "broken_reply_reruns": data["broken_reply_reruns"]}
    per_model = {}
    for model in MODELS:
        w1 = [t for name in ("w1-g0", "w1-g1", "w1-g2") for t in lists.get((model, name), {}).get("tasks", [])]
        w2 = [t for name in ("w2-g0", "w2-g1") for t in lists.get((model, name), {}).get("tasks", [])]
        every = w1 + w2
        missing_files = [t for t in every if not ((TASKS / t / "task.toml").is_file() and
                                                  (TASKS / t / "instruction.md").is_file() and (TASKS / t / "tests").is_dir())]
        per_model[model] = {"total": len(every), "unique": len(set(every)), "wave1": len(w1), "wave2": len(w2),
                            "wave_overlap": len(set(w1) & set(w2)), "missing_files": missing_files,
                            "slots": sum(lists.get((model, n), {}).get("slots", 0) for _, _, n in CAMPAIGNS if n != "pilot"),
                            "pilot": lists.get((model, "pilot"), {}).get("tasks")}
    same = all(lists.get((model, name), {}).get("tasks") == lists.get(("kimi", name), {}).get("tasks")
               for model in MODELS for _, _, name in CAMPAIGNS)
    submit = (TD / "submit.sh").read_text()
    out["submit_defaults_match"] = "declare -A DEFAULT_CONCURRENCY=([kimi]=8 [glm]=8 [deepseek]=5)" in submit
    out["per_model"] = per_model
    out["same_tasks_for_every_model"] = same
    out["ok"] = (len(lists) == 18 and same and out["submit_defaults_match"] and all(
        v["total"] == 113 and v["unique"] == 113 and v["wave1"] == 79 and v["wave2"] == 34 and not v["wave_overlap"]
        and not v["missing_files"] and v["slots"] == 565 for v in per_model.values()))
    out["labels"] = {}
    for data in lists.values():
        for key, value in data["labels"].items():
            if value["chars"] > out["labels"].get(key, {}).get("chars", -1):
                out["labels"][key] = value
    return out


DATASET_ROW_CODE = r'''
import json, sys, tomllib
from pathlib import Path
sys.path.insert(0, "/lustre/fsw/portfolios/nemotron/users/hengyuf/swe-task/synth")
import rollouts
root = Path(sys.argv[1])
bad, rows = [], 0
for task_dir in sorted(p for p in root.iterdir() if p.is_dir()):
    config = tomllib.loads((task_dir / "task.toml").read_text())
    row = rollouts.dataset_row(task_dir)
    rows += 1
    problems = []
    if row["image"] != config["environment"]["docker_image"]: problems.append("image")
    if row["task_id"] != config["metadata"]["task_id"] or row["task_id"] != task_dir.name: problems.append("task_id")
    if not row["responses_create_params"]["input"][0]["content"].strip(): problems.append("instruction")
    if not row["_atp_verifier_timeout"] > 0: problems.append("verifier_timeout")
    if not row["image"].startswith("public.ecr.aws/"): problems.append("image_form")
    if problems: bad.append({"task": task_dir.name, "problems": problems})
print(json.dumps({"rows": rows, "bad": bad, "image_example": row["image"], "subset": row["subset"]}))
'''


def check_dataset_row() -> dict:
    result = subprocess.run([PY, "-B", "-c", DATASET_ROW_CODE, str(TASKS)], capture_output=True, text=True, timeout=600,
                            cwd=str(S / "synth"))
    if result.returncode:
        return {"ok": False, "error": result.stderr[-800:]}
    data = json.loads(result.stdout)
    data["ok"] = data["rows"] == 113 and not data["bad"]
    data["note"] = ("image = the task's public ECR docker_image (ATP's plain image path; synth probe_worker's snapshot "
                    "patch applies only to 'snapshot:<id>' images); 'subset' is pass-through provenance Gym never reads")
    return data


def check_budget(skip: bool) -> dict:
    if skip:
        return {"skipped": True, "ok": True}
    result = subprocess.run(["/usr/bin/python3", "-I", "-B", str(TD / "key_budget.py"), "--key-name", "NVAPI_KEY_NEW_2",
                             "--tag", "offline check"], capture_output=True, text=True, timeout=300)
    record = json.loads(result.stdout.strip().splitlines()[-1])
    return {"record": record, "start_floor": 1700.0, "guard_floor": 1500.0,
            "ok": record.get("key") == "NVAPI_KEY_NEW_2" and record.get("remaining") is not None
            and record["remaining"] >= 1700.0}


def check_config() -> dict:
    result = subprocess.run([PY, "-B", str(TD / "setup" / "config_check.py")], capture_output=True, text=True,
                            timeout=600, cwd=str(S / "synth"))
    if result.returncode:
        return {"ok": False, "error": result.stderr[-800:]}
    data = json.loads(result.stdout)
    return {"ok": data["all_ok"], "opencode_limit": {a: v["opencode_limit"] for a, v in data["aliases"].items()},
            "sampling": {a: v["sampling_overrides"] for a, v in data["aliases"].items()},
            "model_name": {a: v["policy_model_name"] for a, v in data["aliases"].items()},
            "stream_default": {a: v["ATP_DEEPSWE_UPSTREAM_STREAM_default"] for a, v in data["aliases"].items()},
            "kr4_pilot_recorded_limit": data["kr4_pilot_recorded"]["limit"]}


def functions(path: Path, names) -> dict:
    tree = ast.parse(path.read_text())
    found = {node.name: ast.dump(node) for node in tree.body if isinstance(node, ast.FunctionDef) and node.name in names}
    return found


def check_code_identity() -> dict:
    out = {}
    names = ("_reverse_lines", "last_reply", "broken_end")
    mine, theirs = functions(TD / "brokenreply.py", names), functions(KR4 / "brokenreply.py", names)
    out["brokenreply_functions_identical"] = {n: mine.get(n) == theirs.get(n) and n in mine for n in names}
    names = ("classify", "scan")
    mine, theirs = functions(TD / "degenerate.py", names), functions(KR4 / "degenerate.py", names)
    out["degenerate_functions_identical"] = {n: mine.get(n) == theirs.get(n) and n in mine for n in names}
    out["degenerate_file_identical"] = (TD / "degenerate.py").read_bytes() == (KR4 / "degenerate.py").read_bytes()
    stock = hashlib.sha256((TD / "prompt" / "opencode-1.17.11-default.txt").read_bytes()).hexdigest()
    out["stock_prompt_sha256"] = stock
    out["stock_prompt_matches_kr4"] = stock == hashlib.sha256(
        (KR4 / "prompt" / "opencode-1.17.11-default.txt").read_bytes()).hexdigest()
    driver_text = (TD / "driver.py").read_text()
    tree = ast.parse(driver_text)
    specs = next(ast.literal_eval(node.value) for node in tree.body if isinstance(node, ast.Assign)
                 and getattr(node.targets[0], "id", None) == "BASE_SPECS")
    settings = json.loads((S / "synth" / "settings.json").read_text())["models"]
    tr2_tree = ast.parse((TR2 / "driver.py").read_text())
    tr2 = next(ast.literal_eval(node.value) for node in tr2_tree.body if isinstance(node, ast.Assign)
               and getattr(node.targets[0], "id", None) == "HUB_MODELS")
    out["specs"] = specs
    out["spec_kimi_eq_settings"] = specs["kimi"] == settings.get("kimi")
    out["spec_glm_eq_settings"] = specs["glm"] == settings.get("glm") == tr2.get("glm")
    out["spec_deepseek_eq_tr2"] = specs["deepseek"] == tr2.get("deepseek") and "deepseek" not in settings
    out["ok"] = (all(out["brokenreply_functions_identical"].values()) and all(out["degenerate_functions_identical"].values())
                 and out["stock_prompt_matches_kr4"] and out["spec_kimi_eq_settings"] and out["spec_glm_eq_settings"]
                 and out["spec_deepseek_eq_tr2"])
    return out


def check_hygiene() -> dict:
    caches = [str(p) for p in TD.rglob("__pycache__")]
    pattern = re.compile(rb"nvapi-[A-Za-z0-9_-]{8,}|\bsk-[A-Za-z0-9]{16,}")
    hits = []
    for path in TD.rglob("*"):
        if not path.is_file() or path.is_symlink() or "selftest" in path.parts or path.stat().st_size > 50_000_000:
            continue
        if path.suffix in (".gz",):
            continue
        if pattern.search(path.read_bytes()):
            hits.append(str(path))  # the path only, never the matching text
    return {"pycache_dirs": caches, "credential_like_files": hits, "ok": not caches and not hits}


def count_tree(path: Path) -> tuple:
    inodes = 0
    size = 0
    for root, dirs, files in os.walk(path):
        inodes += len(dirs) + len(files)
        for name in files:
            try:
                size += os.lstat(os.path.join(root, name)).st_size
            except OSError:
                pass
    return inodes, size


def lfs(kind, who) -> dict:
    out = subprocess.run(["lfs", "quota", f"-{kind}", who, "/lustre/fsw"], capture_output=True, text=True, timeout=120)
    fields = [line.split() for line in out.stdout.splitlines() if line.strip().startswith("/lustre/fsw")][0]
    used_kb, limit_kb = int(fields[1].rstrip("*")), int(fields[3])
    files, files_limit = int(fields[5].rstrip("*")), int(fields[7])
    return {"used_tib": round(used_kb / 2 ** 30, 2), "limit_tib": round(limit_kb / 2 ** 30, 2),
            "free_tib": round((limit_kb - used_kb) / 2 ** 30, 2), "files": files, "files_limit": files_limit,
            "free_files": files_limit - files}


def check_inodes() -> dict:
    samples = {"kimi": [KR4 / "probes/kimi" / c for c in ("sel50-g0", "rest47-g2", "baihe-g3", "sel50-g1-brk1")],
               "glm": [TR2 / "probes/glm" / c for c in ("group-0", "group-5")],
               "deepseek": [TR2 / "probes/deepseek" / c for c in ("group-0", "group-5")]}
    measured = {}
    for model, dirs in samples.items():
        rows = []
        for d in dirs:
            inodes, size = count_tree(d)
            slots = len([p for p in (d / "models" / model / "attempts").iterdir() if p.is_dir()])
            rows.append({"campaign": str(d.relative_to(d.parents[2])), "slots": slots, "inodes": inodes, "bytes": size,
                         "inodes_per_slot": round(inodes / slots, 1), "gb_per_slot": round(size / slots / 1e9, 3)})
        measured[model] = rows
    attempt = KR4 / "probes/kimi/sel50-g0/models/kimi/attempts"
    per_attempt = [count_tree(p)[0] for p in sorted(attempt.glob("*/attempt-*"))[:10]]
    rerun_share = {"kimi": 66 / 970, "glm": 0.0, "deepseek": 0.0}
    est = {}
    for model, rows in measured.items():
        per_slot = max(r["inodes_per_slot"] for r in rows)
        gb = max(r["gb_per_slot"] for r in rows)
        slots = 565 * (1 + rerun_share[model]) + 1  # + re-runs (KR4 rate; none seen for glm / deepseek) + the pilot
        est[model] = {"inodes_per_slot_used": per_slot, "slots": round(slots), "inodes": round(slots * per_slot),
                      "gb_per_slot_used": gb, "tb": round(slots * gb / 1000, 2)}
    total_inodes = sum(v["inodes"] for v in est.values()) + 10_000  # + dataset, logs, babysit, analysis
    total_tb = round(sum(v["tb"] for v in est.values()), 2)
    user, project = lfs("u", os.environ.get("USER", "hengyuf")), lfs("p", "10581")
    return {"measured": measured, "files_per_kr4_attempt_dir": per_attempt, "estimate": est,
            "total_inodes": total_inodes, "total_tb": total_tb, "quota_user": user, "quota_project_10581": project,
            "free_inodes_after_user": user["free_files"] - total_inodes,
            "free_inodes_after_project": project["free_files"] - total_inodes,
            "free_tib_after_user": round(user["free_tib"] - total_tb / 1.0995, 2),
            "ok": project["free_files"] - total_inodes >= 4_000_000 and user["free_files"] - total_inodes >= 4_000_000}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--skip-budget", action="store_true")
    args = parser.parse_args()
    report = {"compile": check_compile()}
    dry = check_dry_run()
    labels = dry.pop("labels")
    report["dry_run"] = dry
    report["dataset_row"] = check_dataset_row()
    report["labels"] = {"longest": labels, "limit": 63,
                        "ok": all(v["chars"] <= 63 for k, v in labels.items())}
    report["budget"] = check_budget(args.skip_budget)
    report["config"] = check_config()
    report["code_identity"] = check_code_identity()
    report["hygiene"] = check_hygiene()
    report["inodes"] = check_inodes()
    report["all_ok"] = all(v.get("ok") for v in report.values() if isinstance(v, dict))
    (TD / "analysis" / "offline-checks.json").write_text(json.dumps(report, indent=1, ensure_ascii=False) + "\n")
    print(json.dumps({k: (v.get("ok") if isinstance(v, dict) else v) for k, v in report.items()}, indent=1))


if __name__ == "__main__":
    main()
