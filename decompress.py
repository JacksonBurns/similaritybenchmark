import tarfile
import shutil

for fname in {"MultiAssay", "SingleAssay", "chembl_20.smi"}:
    with tarfile.open(f"{fname}.tar.xz") as f:
        f.extractall(".")

shutil.move("chembl_20.smi", "benchlib")
