"""Extract the real collectible quality table and item pools from the game files.

Everything in `bag_of_crafting.cpp` except the fixed recipes used to be invented.
The two pieces of data it actually needs -- each collectible's quality (0-4) and
which item pools it belongs to -- ship with the game and are recovered here.

Item *names* are NOT taken from the game: items.xml stores localisation keys
(`#THE_SAD_ONION_NAME`), so `resources/items.json` remains the name table and
this script only cross-checks against it.

Quality lives in `items_metadata.xml`, not `items.xml` -- the latter carries no
quality attribute in Repentance v1.7.8a.

Usage:
    python3 extract_game_data.py [--game-dir DIR] [--extract-dir DIR]
                                 [--skip-extract] [--out FILE]
"""

import argparse
import json
import shutil
import subprocess
import sys
import tempfile
import xml.etree.ElementTree as ET
from pathlib import Path

HERE = Path(__file__).resolve().parent
REPO = HERE.parent
ITEMS_JSON = REPO / "recipe_suggestor" / "resources" / "items.json"
DEFAULT_OUT = REPO / "recipe_suggestor" / "resources" / "collectibles.json"
DEFAULT_POOLS_OUT = REPO / "recipe_suggestor" / "resources" / "itempools.json"

DEFAULT_GAME_DIR = Path.home() / (
    "Programs/The.Binding.of.Isaac.Rebirth.Repentance.Nexusgames.to/"
    "The Binding of Isaac Rebirth Repentance v1.7.8a"
)

# The DLC3 (Repentance) resource set is the authoritative one.
RES = "resources-dlc3"


def run_extractor(game_dir: Path, out_dir: Path) -> None:
    """Unpack the game's .a archives with the bundled native extractor."""
    exe = game_dir / "tools" / "ResourceExtractor" / "Linux" / "resource_extractor"
    if not exe.exists():
        sys.exit(f"Resource extractor not found: {exe}")
    if not (game_dir / "resources" / "packed").is_dir():
        sys.exit(f"No packed resources under {game_dir}")

    out_dir.mkdir(parents=True, exist_ok=True)
    print(f"Extracting {game_dir} -> {out_dir} (this unpacks ~1 GB, takes a minute)")
    proc = subprocess.run(
        [str(exe), str(game_dir), str(out_dir)],
        stdin=subprocess.DEVNULL,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
    )
    if proc.returncode != 0:
        sys.exit(f"Extractor failed ({proc.returncode}):\n{proc.stdout[-2000:]}")
    print("Extraction finished.")


def parse_quality(path: Path):
    """id -> (quality, tags) from items_metadata.xml."""
    out = {}
    for el in ET.parse(path).getroot():
        if el.tag != "item":
            continue
        q = el.get("quality")
        if q is None:
            continue
        out[int(el.get("id"))] = (int(q), (el.get("tags") or "").split())
    return out


def parse_pools(path: Path):
    """id -> sorted list of pool names, from itempools.xml."""
    pools = {}
    for pool in ET.parse(path).getroot():
        name = pool.get("Name")
        if not name:
            continue
        for item in pool.findall("Item"):
            iid = item.get("Id")
            if iid is None:
                continue
            pools.setdefault(int(iid), set()).add(name)
    return {k: sorted(v) for k, v in pools.items()}


def parse_pool_table(path: Path):
    """The pools in XML order, each with its items and their weights.

    Crafting indexes pools by their POSITION in itempools.xml (treasure 0,
    shop 1, boss 2, devil 3, angel 4, secret 5, shell game 7, golden chest 8,
    red chest 9, curse 12, planetarium 26), and it weights each candidate by the
    item's own Weight attribute, so both the order and the weights have to
    survive extraction. A collectible can appear in a pool more than once; the
    weights add up, which is exactly what the game does.
    """
    table = []
    for pool in ET.parse(path).getroot():
        name = pool.get("Name")
        if not name:
            continue
        items = {}
        for item in pool.findall("Item"):
            iid = item.get("Id")
            if iid is None:
                continue
            weight = float(item.get("Weight", 1))
            items[int(iid)] = items.get(int(iid), 0.0) + weight
        table.append({"name": name,
                      "items": [[iid, w] for iid, w in sorted(items.items())]})
    return table


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--game-dir", type=Path, default=DEFAULT_GAME_DIR,
                    help="Isaac install directory (default: %(default)s)")
    ap.add_argument("--extract-dir", type=Path,
                    help="where to unpack; default is a temp dir that is deleted after")
    ap.add_argument("--skip-extract", action="store_true",
                    help="reuse an existing --extract-dir instead of unpacking again")
    ap.add_argument("--out", type=Path, default=DEFAULT_OUT)
    ap.add_argument("--pools-out", type=Path, default=DEFAULT_POOLS_OUT,
                    help="where to write the ordered, weighted pool table")
    args = ap.parse_args(argv)

    tmp = None
    if args.extract_dir:
        extract_dir = args.extract_dir
    elif args.skip_extract:
        sys.exit("--skip-extract requires --extract-dir")
    else:
        tmp = tempfile.mkdtemp(prefix="isaac_extract_")
        extract_dir = Path(tmp)

    try:
        if not args.skip_extract:
            run_extractor(args.game_dir, extract_dir)

        meta_xml = extract_dir / RES / "items_metadata.xml"
        pools_xml = extract_dir / RES / "itempools.xml"
        for p in (meta_xml, pools_xml):
            if not p.exists():
                sys.exit(f"Missing {p} -- extraction incomplete?")

        quality = parse_quality(meta_xml)
        pools = parse_pools(pools_xml)
        pool_table = parse_pool_table(pools_xml)
        names = json.loads(ITEMS_JSON.read_text())

        print(f"items_metadata.xml: {len(quality)} items with a quality")
        print(f"itempools.xml:      {len(pools)} items across "
              f"{len({p for v in pools.values() for p in v})} pools")
        print(f"items.json:         {len(names)} names")

        collectibles = {}
        for key, name in names.items():
            iid = int(key)
            if iid not in quality:
                continue  # blank, cut or unused -- not craftable
            q, tags = quality[iid]
            collectibles[key] = {
                "name": name,
                "quality": q,
                "pools": pools.get(iid, []),
                "tags": tags,
            }

        skipped = sorted(int(k) for k in names if int(k) not in quality)
        if skipped:
            print(f"\nNo metadata for {len(skipped)} ids (excluded as non-craftable):")
            for iid in skipped:
                print(f"  {iid}: {names[str(iid)]!r}")

        orphans = sorted(i for i in quality if str(i) not in names)
        if orphans:
            print(f"\n{len(orphans)} metadata ids absent from items.json "
                  f"(likely trinkets/other id spaces): {orphans[:10]}"
                  f"{' ...' if len(orphans) > 10 else ''}")

        no_pool = [k for k, v in collectibles.items() if not v["pools"]]
        print(f"\n{len(no_pool)} collectibles belong to no pool "
              f"(quest/unlockable items -- craftable set should exclude them)")

        args.out.parent.mkdir(parents=True, exist_ok=True)
        args.out.write_text(json.dumps(collectibles, indent=1, sort_keys=True,
                                       ensure_ascii=False) + "\n")
        print(f"\nWrote {len(collectibles)} collectibles -> {args.out}")

        args.pools_out.write_text(json.dumps(
            {"_source": "itempools.xml, in file order -- crafting indexes pools by position",
             "pools": pool_table}, indent=1, ensure_ascii=False) + "\n")
        print(f"Wrote {len(pool_table)} pools -> {args.pools_out}")
        for idx in (0, 1, 2, 3, 4, 5, 7, 8, 9, 12, 26):
            if idx < len(pool_table):
                p = pool_table[idx]
                print(f"  pool {idx:2d} {p['name']:<16} {len(p['items'])} items")

        dist = {}
        for v in collectibles.values():
            dist[v["quality"]] = dist.get(v["quality"], 0) + 1
        print("quality distribution: " +
              ", ".join(f"q{q}={dist[q]}" for q in sorted(dist)))
    finally:
        if tmp:
            shutil.rmtree(tmp, ignore_errors=True)


if __name__ == "__main__":
    main()
