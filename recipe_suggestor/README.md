# Isaac Assistant

The Isaac crafting assistant. Captures the screen, detects the contents of
Tainted Cain's crafting bag and the pickups lying on the floor with two YOLOv8 ONNX models, and
prints ranked recipe suggestions.

## Prerequisites

- **CMake** 3.16+
- **OpenCV** 4.x
- **X11** development headers (screen capture; without them only the offline tests build)
- **OpenSSL** (`libcrypto`)
- **ONNX Runtime** — vendored under `lib/onnxruntime-linux-x64-1.16.3`, fetched by `./setup_env.sh`
- **The two trained models**, which are not in the repository — see below

## Build

```bash
./setup_env.sh        # once, downloads ONNX Runtime into lib/
cmake -S . -B build
cmake --build build -j
```

`resources/` and `libonnxruntime.so` are staged into `build/` automatically, so every binary runs
from `build/` with paths relative to itself.

Optional: `cmake -S . -B build -DRS_SANITIZE=ON` for an ASan/UBSan build.

## Getting the models

`resources/models/boc_best.onnx` and `resources/models/floor_best.onnx` are **not committed**: they
are ~12 MB of build output, reproducible from the datasets. Train and export them from
`../models_training/`, which writes straight into `resources/models/`:

```bash
cd ../models_training
pip install ultralytics roboflow python-dotenv pyyaml
python3 train_boc.py   --data BoC---TBOI-6/data.yaml
python3 train_floor.py --data pickups---TBOI-2/data.yaml
```

Without a local dataset the scripts pull it from Roboflow using a `.env` you supply
(`ROBOFLOW_API_KEY`, `ROBOFLOW_WORKSPACE`, the project and version names). `.env` is gitignored and
no key belongs in the tree.

Rebuild after dropping a model in — `resources/` is staged into `build/` at build time, so a model
copied into the source tree only reaches the binaries on the next `cmake --build`.

Missing models degrade rather than crash: the pipeline logs and runs BoC-only without the floor
model, and `floor_test` skips itself. Both missing means no detections at all.

## Run

```bash
cd build

# Live, against a running game
./RecipeSuggestor --seed "7W2N L9AK" --start-seed 3735928559

# Offline, over a directory of images -- no X11 and no game needed
./RecipeSuggestor --replay resources/test_images/screens --seed "7W2N L9AK"
```

| Flag | Meaning |
|---|---|
| `--seed "<run seed>"` | the run seed; crafting results depend on it |
| `--replay <dir>` | run over images instead of capturing the screen |
| `--start-seed <n\|file>` | the run's 32-bit crafting seed, which is what names the crafted item; recover it with `find_start_seed`. Also accepts that tool's `--out` file, so a half-narrowed search is usable as is |
| `--help` | the flag list |

Ctrl-C shuts the pipeline down cleanly.

### What it prints

One block per change of game state, built to be read out of the corner of an eye mid-run: the
situation on top, one instruction under it, alternatives below that.

```
----------------------------------------------------------------
BAG   8/8   quality 13
      Penny  Key  Bomb  Penny  Penny  Key  Key  Card   (oldest first)
FLOOR Bomb, Penny x2, Soul Heart

  >> PICK UP  Soul Heart   quality 13 -> 18  (tier 1-2)
     pushes out the oldest: Penny, Key

     or
        Penny + Soul Heart                q 17  tier 1-2   drops Penny, Key, Bomb
        Bomb                              q 14  tier 0-1   drops Penny
        craft now                         q 13  tier 0-1
```

- The `>>` line is the whole point: what to walk over, and the quality it takes the bag to.
- The bag lists **slots in order**, oldest first, because that is the order things get pushed out.
- `tier` is the collectible tier the bag rolls in, plus the forced pool when one applies.
- A `*` marks a guaranteed `fixed.json` recipe, whose item is named exactly. Those rank first;
  everything else ranks by quality.
- Only three alternatives are shown (`constants::max_shown_alternatives`); the list is ranked, so
  more is noise while playing.
- Nothing craftable prints the same header plus one line saying why:
  `nothing craftable yet: the bag needs 2 more, and the floor is empty`.
- Output is colourised on a terminal and plain when piped or redirected (`NO_COLOR` is honoured).

### What it suggests

`suggest()` plans over the floor: it takes the bag as it stands, enumerates every combination of
whole floor pickups, and reports the bags they lead to, ranked. Three things make it useful rather
than a lookup:

- **A full bag is not a dead end.** The bag is a FIFO queue of 8: picking up past full drops the
  oldest entry and appends the new one. The plan says what to walk over *and* what it costs you.
  This was verified in game rather than assumed — a full `penny key bomb penny penny key key card`
  plus one Red Heart became `key bomb penny penny key key card red_heart` — and both captures are
  pinned in `screen_regression_test`. The detector's reading order is the queue order, so the
  oldest entry is simply the leftmost one.
- **Pickups are indivisible.** A Soul Heart is two bag units, a double heart four. A plan takes a
  whole pickup or none of it — never half of one.
- **Quality is real even when the item is not.** Every plan carries the bag's real quality score,
  the tier it rolls in and the pool it is forced into, all from the game's own data. Without
  `--start-seed` the item name is reported as unknown rather than invented; the ranking is
  unaffected, because it is quality that orders the list.

Pickup *order* does not matter for a single craft: for a fixed number of pickups the resulting bag
is the same whatever sequence you walk them in. Only the count matters, because that is what decides
how many old entries fall out.

### The crafting algorithm, and the one number it needs

`BagOfCrafting::craft_item` is a port of the **real** Repentance routine, transcribed from the
reverse-engineered implementation in [External Item Descriptions](https://github.com/wofsauge/External-Item-Descriptions)
(`features/eid_bagofcrafting.lua`, `EID:calculateBagOfCrafting`). It:

1. sorts the 8 ingredients by id — the result depends on the multiset, **not** on bag order;
2. shifts a xorshift RNG once per ingredient, each ingredient selecting its own shift triple,
   starting from the run's 32-bit start seed;
3. bands the bag's total pickup value into a quality range **per pool** (Devil, Angel and Secret
   band 5 points lower);
4. sums every candidate's weight across the pools the bag unlocks (Treasure 1x, Shop 2x, Boss 2x,
   Devil 10x per Black Heart, and so on);
5. rolls that distribution, retrying up to 20 times when it lands on an item the run cannot give.

Its tables are the game's own: `resources/crafting_rng.json` (shift triples and pickup values),
`resources/itempools.json` (pools in file order, with per-item weights — crafting indexes pools by
*position*), `resources/collectibles.json` (quality), `resources/fixed.json` (checked first).

The port is verified against a transliteration of the reference: 300 random (seed, bag) pairs, same
answer every time.

**The start seed is the catch.** The RNG runs on the run's 32-bit start seed. The game shows you a
seed *string* on the pause screen, and the encoding between the two is not public — so the string is
useless here. Recover the number from crafts you have actually made:

```bash
cat > obs.txt <<'EOT'
# collectible id, then the 8 component ids that were in the bag
177  8 8 8 8 8 8 8 8
EOT
cd build && ./find_start_seed obs.txt      # scans all 2^32 seeds
./RecipeSuggestor --seed "9W4T 9ZJ2" --start-seed <the number it prints>
```

The scan takes about 3.5 minutes on 12 cores and reports how many seeds survive each craft. One
craft leaves around 20 million candidates (a ~0.5% hit rate); each further craft divides that by
roughly the number of items its bag could have produced, so **five or six crafts** pin it exactly —
four was not enough in testing (294 left). The seed is per run, so this is a once-per-run cost.

> Build with optimisation or this is unbearable. `cmake -S . -B build` used to leave
> `CMAKE_BUILD_TYPE` empty, which means **no `-O` flag at all** — the same search took over 16
> minutes. The CMakeLists now defaults to Release.

Without `--start-seed`, every plan still carries its real quality, tier and pool — only the item
name is withheld, rather than invented.

**What can still be wrong with a correct seed:** the game rerolls a result when the collectible is
already gone from the pool or not unlocked on your save. We cannot see either, so a craft whose
first roll hits an item your run has already taken will differ. Component ids also have to be right:
what the detector calls a `soul_heart` is two Soul Heart units, and mislabelling one shifts the whole
RNG chain.

### Preprocessing must match the dataset — and the two datasets differ

Each detector fits its crop to the 640x640 model input the way *its own* dataset was built
(`types::Preprocess`, passed by each subclass's constructor). This is not a stylistic choice:
getting it wrong costs every detection.

| model | dataset geometry | mode |
|---|---|---|
| floor | whole 1920x1080 frames, Roboflow "Resize to 640x640 (Stretch)" | `Stretch` |
| BoC | a wide bag strip **black-padded to a square** before Roboflow saw it — content occupies rows 235..405 of every training image, a 640x170 band centred in 640x640 | `LetterboxBlack` |

Letterboxing the floor frames instead costs about 18 points of recall (68.6% vs 86.9% of
ground-truth instances on the floor validation set). Conversely, squashing the live 730x194 bag
strip to 640x640 — which is what the code did until the strip was calibrated — stretches sprites
3.8x vertically against what the BoC model learned, and it detects **nothing at all**.
`boc_geometry_test` is the guard: it rebuilds a full frame from a training crop and checks the
routed path reproduces the detections of the raw crop.

### The bag strip rect

`constants::boc_crop_*_factor` / `crop_start_x_factor` put the strip at (1191, 886) 729x194 on a
1920x1080 screen — flush with the right and bottom edges, aspect 3.758 against the training band's
3.765. Two independent measurements agree on it: template-matching the bag sprite from a BoC
training crop onto a live frame (match 0.96), and the black bag mask baked into every floor
training image. To re-derive it on a different resolution or HUD scale:

```bash
import -window root ~/frame.png      # or any full-screen capture
cd build && ./roi_preview ~/frame.png
```

It prints the rects and, when the models are present, what each detector finds inside them, and
writes `outputs/roi_overlay.png` plus the exact images the two detectors receive.

## Tests

```bash
cd build && ctest --output-on-failure
```

Tests that need a model skip themselves when it is absent, so a fresh clone with no models still
builds and runs the offline suite. `floor_test` in particular skips without
`resources/models/floor_best.onnx`; see **Getting the models** above.

Two of the tests guard the **live** path rather than the dataset one, and they are the ones that
fail if the strip rect or the preprocessing drifts: `boc_geometry_test` (synthetic, no game needed)
and `screen_regression_test`, which runs both detectors over the real 1920x1080 captures in
`resources/test_images/screens/` and asserts the contents read off them by eye — a bag holding a
penny and a key, a floor holding two coins and a bomb, and a paused frame where the honest answer
is nothing.

## Project structure

- `src/`, `include/` — sources and headers; everything except `main.cpp` is compiled into `rs_core`
- `tests/` — one self-contained executable per test, all registered with CTest
- `resources/` — models, `consumables.json` (the crafting quality table plus each consumable's
  item-pool influence), `items.json` (726 collectible names), `collectibles.json` (721 entries of
  real quality / pool data from the game), `fixed.json` (known-good recipes), `class_map.json`
  (detector class → consumable mapping)
- `tools/` — not tests: `roi_preview` (recalibrate the capture regions), `find_start_seed` (recover
  the run's crafting seed), `craft_recorder` (record real crafts from the live game), `bag_log`
  (read the bag out of saved frames)
- `HASHING.md` — notes on the Tainted Cain crafting mechanics
- `HANDOFF.md` — current state, what is verified, and the one open experiment
- `TODO.md` — remaining work

Not in the repository, by design: `build/`, `lib/` (ONNX Runtime, fetched), `resources/models/`
(trained weights), `resources/test_images/` apart from `screens/` (dataset crops).
