# Isaac Assistant

[![CI](https://github.com/SalvatoreLS/recipe-suggestor/actions/workflows/ci.yml/badge.svg)](https://github.com/SalvatoreLS/recipe-suggestor/actions/workflows/ci.yml)
[![License: MIT](https://img.shields.io/badge/License-MIT-blue.svg)](LICENSE)

A real-time crafting assistant for **The Binding of Isaac: Repentance**, for Tainted Cain.

It watches the game screen, reads what is in the crafting bag and what pickups are lying on the
floor with two YOLOv8 detectors, and prints ranked advice: which pickup to walk over, what quality
that takes the bag to, and what it pushes out of the 8-slot FIFO.

```
BAG   8/8   quality 13
      Penny  Key  Bomb  Penny  Penny  Key  Key  Card   (oldest first)
FLOOR Bomb, Penny x2, Soul Heart

  >> PICK UP  Soul Heart   quality 13 -> 18  (tier 1-2)
     pushes out the oldest: Penny, Key
```

## Repository layout

| Path | What it is |
|---|---|
| [`recipe_suggestor/`](recipe_suggestor/) | The application: C++20, OpenCV, ONNX Runtime, X11 capture. **This is the project.** |
| [`models_training/`](models_training/) | Python scripts that train and export the two YOLOv8 detectors, and extract the real game data the crafting algorithm needs |
| `scripts/` | Dev helpers (serving the run log over the LAN with `ttyd`) |

There was once a parallel Java implementation. It is retired: C++ is the only version, and the
crafting algorithm it once prototyped now lives in `recipe_suggestor/src/bag_of_crafting.cpp` as a
port of the real game routine.

## Quick start

```bash
git clone git@github.com:SalvatoreLS/recipe-suggestor.git
cd recipe-suggestor/recipe_suggestor

./setup_env.sh                        # fetch ONNX Runtime into lib/
cmake -S . -B build && cmake --build build -j
```

The trained models are **not in this repository** (see below). Put them in
`recipe_suggestor/resources/models/`, then:

```bash
cd build
./RecipeSuggestor --seed "7W2N L9AK"                      # live, against the running game
./RecipeSuggestor --replay resources/test_images/screens   # offline, no game needed
ctest --output-on-failure                                  # 12 tests
```

Full build, run, calibration and troubleshooting instructions:
**[`recipe_suggestor/README.md`](recipe_suggestor/README.md)**.

### The models

`boc_best.onnx` and `floor_best.onnx` are ~12 MB each and are **deliberately not committed** —
they are build output, reproducible from the datasets. Train them yourself:

```bash
cd models_training
pip install ultralytics roboflow python-dotenv pyyaml
python3 train_boc.py   --data BoC---TBOI-6/data.yaml
python3 train_floor.py --data pickups---TBOI-2/data.yaml
```

Both scripts export straight into `recipe_suggestor/resources/models/`. The datasets are Roboflow
exports; with a `.env` holding your own `ROBOFLOW_API_KEY` and project names the scripts download
them, otherwise pass a `data.yaml` already on disk. See
[`models_training/README.md`](models_training/README.md).

The app degrades rather than crashing when a model is missing: without the floor model it runs
BoC-only, and every test that needs a model skips itself and passes — so a fresh clone builds and
runs the offline suite with nothing extra installed. That is what CI does on each push.

## Project status

**The app runs end to end.** It captures the screen, detects the bag and the floor pickups, plans
which pickups to take, and ranks the plans by the bag's real quality. `ctest` is green, and two of
the tests (`boc_geometry_test`, `screen_regression_test`) guard the live path against real 1080p
captures rather than only the dataset path.

What is solid:

- **Detection geometry.** The bag strip rect and the two different preprocessing modes the two
  datasets require are measured, not guessed, and pinned by tests. Getting either wrong costs every
  detection — squashing the bag strip detects literally nothing.
- **The crafting algorithm.** `BagOfCrafting::craft_item` is a port of the real Repentance routine
  (transcribed from [External Item Descriptions](https://github.com/wofsauge/External-Item-Descriptions)),
  verified against a transliteration of the reference on 300 random (seed, bag) pairs.
- **The game data.** Qualities, item pools and weights are extracted from the installed game, not
  hand-entered, so a plan's quality, tier and forced pool are the game's own numbers.
- **Planning over a full bag.** The bag is a FIFO of 8; picking up past full drops the oldest. The
  plan says what to walk over *and* what it costs. Verified in game, pinned by a regression test.

The two open weaknesses, both documented in detail rather than papered over:

- **Naming the crafted item needs the run's 32-bit start seed**, which the game never displays. The
  seed *string* on the pause screen encodes it by a scheme that is not public.
  `tools/find_start_seed` recovers the number by brute-forcing all 2^32 seeds against crafts you
  have actually made (~3.5 min on 12 cores; five or six crafts pin it). Until it converges on a real
  run, plans still carry their real quality, tier and pool — the item name is reported as unknown
  rather than invented. Current state and the open experiment:
  [`recipe_suggestor/HANDOFF.md`](recipe_suggestor/HANDOFF.md).
- **The floor detector is uneven.** mAP50 0.740, but that figure covers only the 14 of 23 classes
  that have validation instances. `eternal_heart` scores AP 0 and does not work — and it is one of
  the pool-forcing pickups. Read the per-class table in
  [`models_training/README.md`](models_training/README.md), not the headline number.

Remaining work is tracked in [`recipe_suggestor/TODO.md`](recipe_suggestor/TODO.md).

## License

MIT — see [`LICENSE`](LICENSE).

## Requirements

- Linux with X11 (screen capture; without X11 headers only the offline tests build)
- CMake 3.16+, a C++20 compiler, OpenCV 4.x, OpenSSL (`libcrypto`)
- ONNX Runtime 1.16.3 — fetched by `recipe_suggestor/setup_env.sh`
- The game running **fullscreen at 1920x1080**; every capture region is a fraction of the frame
- Python 3.10+ with Ultralytics, only if you are training the models
