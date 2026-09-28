"""Step E5 on the Mac: put the checker files from the downloaded Kaggle outputs in place and join the shards.

Copies every results/features/<TAG>/checkers*.npz and meta_checkers*.json from the given output zips into
results/features/<TAG>/ (never overwrites; an existing file with other bytes is reported and the run stops), then joins
each run's part files with score_checkers.py --merge auto (every case and every sentence.csv row exactly once; each part
aligned to a sentence.csv with the local file's sha256) and prints the E5 row checks of every run. The GASP reruns in the
zips are not extracted (the local canon runs are the reference).

    python scripts/assemble_checkers.py ~/Downloads/results(15).zip ~/Downloads/results(16).zip ...
"""
import argparse
import json
import re
import sys
import zipfile
from pathlib import Path
from types import SimpleNamespace

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "scripts"))

import score_checkers  # noqa: E402

FILE = re.compile(r"results/features/([^/]+_K5)/((?:meta_)?checkers[^/]*\.(?:npz|json))$")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("zips", nargs="+")
    ap.add_argument("--outroot", default=str(ROOT / "results" / "features"))
    ap.add_argument("--canon_root", default=str(ROOT / "results" / "gasp_repro" / "canon_results"))
    args = ap.parse_args()
    out, canon = Path(args.outroot), Path(args.canon_root)

    copied, same = [], []
    for z in args.zips:
        with zipfile.ZipFile(z) as zf:
            for name in zf.namelist():
                m = FILE.search(name)
                if not m:
                    continue
                tag, fname = m.groups()
                data = zf.read(name)
                dest = out / tag / fname
                if dest.exists():
                    if dest.read_bytes() != data:
                        sys.exit(f"STOP: {dest} exists with other bytes than {z}:{name}")
                    same.append(str(dest))
                    continue
                dest.parent.mkdir(parents=True, exist_ok=True)
                dest.write_bytes(data)
                copied.append(f"{Path(z).name}: {tag}/{fname}")
    print(f"copied {len(copied)} files ({len(same)} already present with the same bytes)")
    for c in copied:
        print("  ", c)

    ok = True
    for ds in ("ragtruth", "tofueval", "ragbench", "techqa", "expertqalong", "triviapluslong"):
        tags = sorted(p.name for p in canon.glob(f"*_{ds}_K5"))
        if not tags:
            continue
        parts = sorted({p.name for t in tags for p in (out / t).glob("checkers.part*of*.npz")})
        if parts:
            print(f"\n{ds}: joining {parts}")
            ok &= score_checkers.merge(SimpleNamespace(canon_dirs=[str(canon / t) for t in tags], outroot=str(out),
                                                       max_cases=0, merge="auto"))
        for t in tags:
            f = out / t / "meta_checkers.json"
            if not f.exists():
                print(f"  MISSING {t}/checkers.npz")
                ok = False
                continue
            m = json.load(open(f))
            print(f"  {t}: rows {m['gasp_rows']}, cases {m['cases']}, NaN {m['nan_minicheck']}/{m['nan_lettuce']}, "
                  f"tokens outside one span {m['tokens_not_in_one_span']}, multi-chunk {m['lettuce_multi_chunk_contexts']}, "
                  f"truncated {m['lettuce_truncated_contexts']}, smaller MiniCheck batches "
                  f"{m.get('minicheck_smaller_batch_cases', 0)}, checks {'PASS' if m['checks_pass'] else 'FAIL'}")
            ok &= m["checks_pass"]
    print("\nALL E5 ROW CHECKS PASS" if ok else "\nSOMETHING FAILED OR IS MISSING (see above)")
    sys.exit(0 if ok else 1)


if __name__ == "__main__":
    main()
