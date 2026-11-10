"""Reviewer (round 1): outcomes of the read-only sftv3 patcher and the staged tasksall patcher on COPIES of the three
forms a job's staged deepswe_capture.py can have (Baihe's original, sftv3-patched, the staged fix), plus: the fixed
copy is never written; a tasksall 'patched' copy is byte-identical to an sftv3 'patched' copy and imports.
Run: <atp>/runtime/controller/bin/python -I -B verify_patchers.py <private scratch dir>
"""
import sys

sys.dont_write_bytecode = True

import hashlib  # noqa: E402
import importlib.util  # noqa: E402
import os  # noqa: E402
from pathlib import Path  # noqa: E402
import shutil  # noqa: E402
import tempfile  # noqa: E402

S = Path("/lustre/fsw/portfolios/nemotron/users/hengyuf/swe-task")
FIXDIR = S / "sft/glm-rollouts-tasksall-20261108/atp-fix"
PATCHERS = {"sftv3": S / "sft/glm-rollouts-sftv3-20261107/capture_patch.py",
            "tasksall-installed": S / "sft/glm-rollouts-tasksall-20261108/capture_patch.py",
            "tasksall-staged": FIXDIR / "capture_patch.py"}
FORMS = {"baihe": S / "atp/scripts/deepswe_capture.py", "fix": FIXDIR / "app/scripts/deepswe_capture.py"}
RESULTS = []


def check(ok, message):
    RESULTS.append(bool(ok))
    print(("PASS " if ok else "FAIL ") + message, flush=True)


def load(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def main():
    scratch = Path(sys.argv[1]).resolve()
    if (FIXDIR / "scratch").resolve() not in scratch.parents:
        raise SystemExit("scratch must be under atp-fix/scratch")
    scratch.mkdir(parents=True, exist_ok=True)
    root = Path(tempfile.mkdtemp(prefix="patchers-", dir=scratch))
    before = {name: digest(path) for name, path in {**PATCHERS, **FORMS}.items()}
    patchers = {name: load("p_" + name.replace("-", "_"), path) for name, path in PATCHERS.items()}
    sftv3_form = root / "sftv3-form" / "deepswe_capture.py"
    sftv3_form.parent.mkdir()
    shutil.copyfile(FORMS["baihe"], sftv3_form)
    check(patchers["sftv3"].patch_file(sftv3_form) == "patched", "sftv3 patcher makes the sftv3 form from Baihe's")
    forms = {**FORMS, "sftv3": sftv3_form}
    expected = {("sftv3", "baihe"): "patched", ("sftv3", "sftv3"): "already", ("sftv3", "fix"): "already",
                ("tasksall-installed", "baihe"): "patched", ("tasksall-installed", "sftv3"): "already",
                ("tasksall-installed", "fix"): "already",
                ("tasksall-staged", "baihe"): "patched", ("tasksall-staged", "sftv3"): "already",
                ("tasksall-staged", "fix"): "upstream"}
    outputs = {}
    for (patcher, form), want in expected.items():
        target = root / f"{patcher}--{form}" / "scripts" / "deepswe_capture.py"
        target.parent.mkdir(parents=True)
        shutil.copyfile(forms[form], target)
        (target.parent / "__pycache__").mkdir()
        (target.parent / "__pycache__" / "deepswe_capture.cpython-313.pyc").write_bytes(b"stale")
        original = target.read_bytes()
        mtime = os.stat(target).st_mtime_ns
        try:
            got = patchers[patcher].patch_file(target)
        except Exception as error:  # noqa: BLE001 - reported
            got = f"raised {type(error).__name__}: {error}"
        written = target.read_bytes() != original or os.stat(target).st_mtime_ns != mtime
        pyc = (target.parent / "__pycache__" / "deepswe_capture.cpython-313.pyc").exists()
        outputs[patcher, form] = target.read_bytes()
        check(got == want and written == (want == "patched") and pyc == (want != "patched"),
              f"{patcher} on {form}: {got!r} (want {want!r}); file written: {written}; stale pyc kept: {pyc}")
    check(outputs["tasksall-staged", "baihe"] == outputs["sftv3", "baihe"] == sftv3_form.read_bytes(),
          "tasksall-staged 'patched' output == sftv3 'patched' output (byte-identical)")
    compile(outputs["tasksall-staged", "baihe"].decode(), "patched", "exec")
    check(outputs["tasksall-staged", "fix"] == FORMS["fix"].read_bytes(), "fixed copy untouched by tasksall-staged")
    # a copy staged from the fixed app and then (wrongly) half-edited must not pass as 'upstream'
    broken = root / "broken" / "deepswe_capture.py"
    broken.parent.mkdir()
    text = FORMS["fix"].read_text()
    broken.write_text(text.replace("                await _in_stream_io(persist, chunk)\n",
                                   "                persist(chunk)\n", 1))
    try:
        got = patchers["tasksall-staged"].patch_file(broken)
    except RuntimeError as error:
        got = f"raised: {error}"
    check(got.startswith("raised"), f"tasksall-staged on a half-reverted fix fails closed ({got[:90]})")
    after = {name: digest(path) for name, path in {**PATCHERS, **FORMS}.items()}
    check(before == after, "patchers and source forms unchanged on disk")
    print(f"SUMMARY {sum(RESULTS)}/{len(RESULTS)} PASS; scratch {root}")
    raise SystemExit(0 if all(RESULTS) else 1)


if __name__ == "__main__":
    main()
