import os
import py_compile

base_dir = "/Users/varunvemuri/Downloads/IPPD/Foundation model/ECG-FM_EVAL/label_create/git_ecg_finetuned"
bad_strings = ["ecgfm.py", "eval_finetuned.py"]
bad_filenames = ["preprocess_ecgfm.py", "create_labels_ecgfm.py", "eval_ecgfm.py", "eval_finetuned.py"]

print("--- Checking for old filenames ---")
has_old_files = False
for root, dirs, files in os.walk(base_dir):
    if ".git" in root:
        continue
    for f in files:
        if f in bad_filenames:
            print(f"ERROR: Found old file: {os.path.join(root, f)}")
            has_old_files = True

if not has_old_files:
    print("SUCCESS: No old python scripts found.")

print("\n--- Checking for old string references ---")
has_old_strings = False
for root, dirs, files in os.walk(base_dir):
    if ".git" in root or "__pycache__" in root or "finetuned_model" in root:
        continue
    for f in files:
        if not f.endswith((".py", ".md", ".txt", ".csv", ".sh")):
            continue
        path = os.path.join(root, f)
        try:
            with open(path, "r", encoding="utf-8") as file:
                content = file.read()
                for bad_str in bad_strings:
                    if bad_str in content:
                        print(f"ERROR: Found '{bad_str}' in {path}")
                        has_old_strings = True
        except Exception as e:
            print(f"Could not read {path}: {e}")

if not has_old_strings:
    print("SUCCESS: No old string references found.")

print("\n--- Syntax Checking Python Scripts ---")
has_syntax_errors = False
py_files = []
for root, dirs, files in os.walk(base_dir):
    if ".git" in root or "site-packages" in root: # ignore hidden and venvs
        continue
    for f in files:
        if f.endswith(".py"):
            py_files.append(os.path.join(root, f))

for py_file in py_files:
    try:
        py_compile.compile(py_file, doraise=True)
        print(f"Syntax OK: {os.path.relpath(py_file, base_dir)}")
    except py_compile.PyCompileError as e:
        print(f"SYNTAX ERROR in {py_file}:\n{e}")
        has_syntax_errors = True

if not has_syntax_errors:
    print("SUCCESS: All python scripts have valid syntax.")
