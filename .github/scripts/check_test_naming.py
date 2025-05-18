import subprocess
import sys
import re

# Get the list of added Python files in the PR
result = subprocess.run(
    ["git", "diff", "--diff-filter=A", "--name-only", "origin/main...HEAD"],
    capture_output=True, text=True
)
files = result.stdout.strip().split("\n")

# Filter new .py files
py_files = [f for f in files if f.endswith(".py")]

bad_files = []
bad_funcs = []

for f in py_files:
    if not (f.startswith("test_") or f.endswith("_test.py")):
        bad_files.append(f)
        continue

    with open(f, "r", encoding="utf-8") as code:
        lines = code.readlines()
        for i, line in enumerate(lines):
            if line.strip().startswith("def "):
                func_name = re.findall(r"def (\w+)\(", line)
                if func_name and not func_name[0].startswith("test_"):
                    bad_funcs.append(f"{f}:{i+1} -> {func_name[0]}")

# Report violations
if bad_files:
    print("❌ Invalid test file names:")
    for f in bad_files:
        print(f"  - {f} (must start with 'test_' or end with '_test.py')")

if bad_funcs:
    print("❌ Invalid test function names:")
    for f in bad_funcs:
        print(f"  - {f} (must start with 'test_')")

if bad_files or bad_funcs:
    sys.exit(1)  # Fail the job
else:
    print("✅ Test file and function naming passed.")
