"""Restore the supplied trip files from the public repository's three data archives."""
import argparse
import csv
import hashlib
from pathlib import Path, PurePosixPath
import tempfile
from zipfile import ZipFile


def main():
    p=argparse.ArgumentParser()
    p.add_argument("--archives",type=Path,default=Path("data_archives"))
    p.add_argument("--data",type=Path,default=Path("data"))
    args=p.parse_args()
    with (args.archives/"manifest.csv").open(newline="",encoding="utf-8") as stream:
        entries=list(csv.DictReader(stream))
    expected_by_archive={}
    for row in entries:
        expected_by_archive.setdefault(row["archive"],{})[row["path"]]=row["sha256"]
    total=0
    for name,expected in sorted(expected_by_archive.items()):
        archive=args.archives/name
        if not archive.is_file():
            # The private download already contains the original extracted files.
            for member,digest in expected.items():
                current=args.data/member
                if not current.is_file() or hashlib.sha256(current.read_bytes()).hexdigest()!=digest:
                    raise ValueError(f"Missing archive {name} and no matching local copy of {member}")
            total+=len(expected)
            continue
        with ZipFile(archive) as z:
            if set(z.namelist())!=set(expected) or len(z.namelist())!=len(expected):
                raise ValueError(f"Archive contents do not match the manifest: {name}")
            for member,h in expected.items():
                parts=PurePosixPath(member).parts
                if len(parts)!=3 or parts[0] not in ("train","student_test") or not parts[1].isdigit() or parts[2] not in {parts[1]+"."+ext for ext in ("track","nodes","arcs","route")}:
                    raise ValueError(f"Unexpected trip path: {member}")
                dest=args.data.joinpath(*parts)
                raw=z.read(member)
                if hashlib.sha256(raw).hexdigest()!=h:raise ValueError(f"Checksum mismatch: {member}")
                if dest.exists():
                    if hashlib.sha256(dest.read_bytes()).hexdigest()!=h:
                        raise ValueError(f"An existing file differs: {dest}")
                else:
                    dest.parent.mkdir(parents=True,exist_ok=True)
                    with tempfile.NamedTemporaryFile(dir=dest.parent,delete=False) as temp:
                        temp.write(raw);temporary=Path(temp.name)
                    temporary.replace(dest)
                total+=1
    print(f"Verified {total} original files in {len({(r['path'].split('/')[0],r['path'].split('/')[1]) for r in entries})} trips.")
if __name__=="__main__":main()
