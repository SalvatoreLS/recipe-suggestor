# Isaac Assistant (C++) — TODO

Status legend: `[x]` done · `[ ]` open · `[~]` partially done / needs rework

This file is the checklist; each item carries its own rationale inline.

---

## Already done

- [x] `Detector` class (ONNX Runtime inference + NMS + left-to-right sort)
- [x] `Pipeline` class (5 worker threads over bounded queues)
- [x] `FrameCapturer` class (X11 `XGetImage`)
- [x] `SuggestionTrie` class (standalone, tested — deliberately **not** wired into the pipeline, see F6)
- [x] `CircularList`, `BoundedQueue`
- [x] Object/consumable ID files (`resources/items.json`, `consumables.json`, `fixed.json`)
- [x] BGR / RGB order verified when handing the image to the detector
- [x] BoC model trained and exported (`boc_best.onnx`, mAP50 0.847 / mAP50-95 0.687)

---

## A — Floor model (blocking: the app cannot start without it, or without B1)

The floor model has **never been trained**. `runs/detect/train7` is the only pickups run and it
aborted after ~4 s with an empty `weights/`. `pipeline.cpp` and `floor_test.cpp` both load
`resources/models/floor_best.onnx`, which does not exist.

- [x] **A1** Fix the stale-export bug in `train_boc.py:65` / `train_floor.py:65`. Both hardcode
      `YOLO('runs/detect/train/weights/best.pt')`, but Ultralytics auto-increments run dirs
      (`train2`…`train8` exist), so `train_floor.py` would export the **BoC** model into
      `floor_best.onnx`. Use `Path(results.save_dir)/'weights'/'best.pt'`.
- [x] **A2** Assert `trained_model.names` matches the expected class list before `shutil.copy` —
      a tripwire for exactly the failure in A1.
- [x] **A3** Fix the `.env` typo `ROBOFLOW_VERSION_FOOR` → `ROBOFLOW_VERSION_FLOOR`
      (`train_floor.py` reads the correct name and silently falls back to v1 today).
- [x] **A4** Add a `--data <path>` / `DATA_YAML` override so training can run from the on-disk
      `pickups---TBOI-1/` without a Roboflow key.
- [x] **A5** Pass `project='runs/detect', name='floor'` so runs are identifiable.
- [x] **A6** Trained and exported `floor_best.onnx` from the **v2** dataset:
      `cd models_training && python3 train_floor.py --data pickups---TBOI-2/data.yaml`
      (note `python3`; plain `python` is not on PATH). Roboflow v2 is far better than the
      v1 copy that was on disk: 285 train images / 1446 instances against 85 / ~267, and
      **every** class has training instances (v1 had none at all for `golden_bomb` and
      `mega_battery`). v2 also adds two classes -- `black_heart` and `micro_battery` -- so
      the floor section of `class_map.json` is now 23 classes, not 21.
      Two environment traps, both now handled in-tree: `nvrtc: error: failed to open
      libnvrtc-builtins.so.13.0` (export
      `LD_LIBRARY_PATH=~/.local/lib/python3.10/site-packages/nvidia/cu13/lib`), and a CUDA OOM
      around epoch 26 because Ultralytics defaults to batch 16, which does not fit in a 4 GB
      card — `train_common.py` now defaults to `--batch 8` and exposes `--batch` / `--device`.
      Training takes well over an hour; start it detached (`setsid nohup ... &`).

> Result: **mAP50 0.740 / mAP50-95 0.575**, well above the 0.4-0.6 originally expected. But that
> covers only the 14 of 23 classes that have validation instances; the other 9 are simply not
> measured. `eternal_heart` scores **AP 0** (recall 0 on 4 instances) and should be treated as
> not working — it is one of the pool-forcing pickups, so a missed Eternal Heart silently changes
> which pool a craft rolls from.

---

## B — Blocking bugs (the app currently aborts at startup)

- [x] **B1** Graceful degradation: wrap each detector construction in `Pipeline::initialize()` in
      `try/catch(const Ort::Exception&)`, leave the pointer null, log and run BoC-only. Today a
      missing model propagates out of the constructor and aborts `main`.
- [x] **B2** `Pipeline::stop()` is declared (`pipeline.hpp:80`) and **never defined** — there is no
      way to shut the pipeline down. Make it public, keep the worker threads as a member, have
      `run()` spawn and return, add `join()`, and make `stop()` clear `running` and push
      `std::nullopt` into `frame_queue` (which already cascades downstream).
- [x] **B3** `~Pipeline()` deletes the detector nodes while workers are still inside
      `session.Run()` → use-after-free. Must `stop(); join();` first.
- [x] **B4** `main.cpp`: install a SIGINT handler and own the pipeline with `unique_ptr`
      (the `new Pipeline` is currently never deleted).
- [x] **B5** `pipeline_test.cpp:35-38` deletes the `Pipeline` while a *detached* thread is still
      running `run()` — a guaranteed UAF. Replace with `stop(); join(); delete`.
- [x] **B6** Pixel-format mismatch: `frame_capturer.cpp:37` allocates and indexes **4** bytes/px
      (BGRA) while `router.cpp:32,50` and `utils.cpp:18` assume **3**. Not an out-of-bounds read
      (3 < 4) but the image is garbage. Fix the **producer only** — emit packed BGR24 — and do
      *not* add `channels`/`stride` to `ScreenCapture`; the whole downstream already assumes packed
      BGR24 and a second invariant would drift.
- [x] **B7** Same edit: replace the per-pixel `XGetPixel` loop (~2M virtual calls per frame, the
      pipeline's throughput bottleneck) with a `cv::Mat` wrap over `bytes_per_line` + `cvtColor`.
      Keep the old loop as a fallback for exotic visuals.
- [x] **B8** `router.cpp:55-58` uses `boc_crop_width_factor` (0.35) as a *pixel count*;
      `static_cast<u_int16_t>(0.35) == 0`, so `x2 == x1` and the BoC region is never covered.
      Extract one `cv::Rect boc_rect(int w, int h) const` shared by `process_boc` and
      `process_floor` so the crop and the mask cannot diverge again.
- [x] **B9** `process_boc`: clamp `start_X` / `start_Y` and the crop extents to the source bounds.
- [x] **B10** Router hardcodes `640`; use `constants::img_width` / `img_height`.
- [x] **B11** `cover_region_inplace`: `std::max((uint16_t)0, x1)` is a no-op on an unsigned type —
      take `int` parameters and clamp there.
- [x] **B12** `bag_of_crafting_test.cpp:104-117` (`setup_resources()`) **overwrites**
      `resources/fixed.json` with `{}`. Add a `fixed_path` constructor parameter, delete
      `setup_resources()`, and test against the real file.
- [x] **B13** CMake: `floor_test` does not link `src/pipeline/nodes/floor_detector.cpp` →
      undefined symbol.
- [x] **B14** CMake: factor the 7 duplicated source lists into `add_library(rs_core STATIC …)`.
- [x] **B15** CMake: POST_BUILD-copy `resources/` next to every binary and standardise all test
      paths to `"resources/…"` (half currently use `"../resources/…"`, so they only work from one
      working directory).
- [x] **B16** CMake: drop the dead `list(REMOVE_ITEM … detector_torch.cpp)`, add `-Wall -Wextra`,
      guard the two X11-dependent test targets.

---

## C — Detector correctness

- [x] **C1** Dedup `Detector` / `FloorDetector` — `floor_detector.cpp:5-116` is a near-verbatim copy
      of `detector.cpp:145-261`. Extract
      `protected: std::vector<Prediction> Detector::run_inference(const cv::Mat&)`.
      **Do this before C2:** letterbox changes the pre/post-processing contract, and with the
      duplication in place it gets implemented twice and the copies drift — which is the bug class
      that created the duplication in the first place.
- [x] **C2** Fix the signature lie: `detector.hpp:23` declares `CircularList<ItemID>`,
      `detector.cpp:145` defines `CircularList<ConsumableID>`. Both alias `u_int16_t` so it
      compiles. A detector emits *consumables* — pick `ConsumableID`.
- [~] **C3** Letterbox — implemented, but **off by default**; see H4. The reasoning was right in
      general and wrong for these datasets.
- [x] **C3a** Letterbox. `_preprocess_image` does a plain resize; Ultralytics trains *and validates*
      with letterbox, so a 1920×1080 floor frame squashed to 640×640 is a distribution shift the
      model never saw. Add `letterbox()` to `utils.cpp` (gray 114, centred), store
      `scale`/`pad_x`/`pad_y`, and invert them in `_filter_predictions`.
- [x] **C4** Delete `Router::pad_boc_image_to_model_size` (it silently crops the right 32 px of a
      672-wide BoC crop) and `Router::resize_floor_image_to_model_size`. The Detector should own
      all model-space geometry; the Router hands over native-resolution crops.
- [x] **C5** `constants.hpp:10` `nms_threshold = 0.1f` is the IoU **above which boxes are
      suppressed** — at 0.1 any pair overlapping by >10% collapses, catastrophic for the tight
      8-slot BoC row. Set `0.45f` (Ultralytics default).
- [x] **C6** Add `floor_conf_threshold = 0.35f` — the weaker floor model needs a lower gate than
      BoC's 0.5.
- [x] **C7** Read class names from ONNX metadata (`names` in `metadata_props`, present in both
      models) and expose `class_names()`. This removes the need for any `synset.txt`
      (`detector_test.cpp:39` references a missing one) and is the **only** reliable guard against
      the wrong-model hazard: both models are `nc: 21` with different class lists, so a swapped
      file emits an identical `[1,25,8400]` tensor and silently mislabels everything.
- [x] **C8** Assert `output dim[1] == 4 + class_names().size()` at construction.

---

## D — Data layer

- [x] **D1** New `resources/class_map.json`: per model section, `expects_classes` (the C7 tripwire)
      plus a map from class index to a **list** of `(consumable_id, qty)`.
- [x] **D2** The list (not a scalar) is required because the six floor-only labels are **quantity
      multipliers** — `half_heart` → 1× Red Heart, `red_heart` → 2×, `double_heart` → 4×,
      `double_bomb` → 2× Bomb — and `red_soul_heart` is genuinely compound (1 red + 1 soul).
      BoC's `extra` maps to the empty list and is dropped. **The multipliers need confirming.**
- [x] **D3** Reuse `serializeBag` as the one canonical bag key — it already emits the sorted-CSV
      `"1,1,1,1,1,1,1,1"` shape that `fixed.json` uses. Do not introduce a second key format.
- [x] **D4** Add `loadJsonToStringMap` (`items.json` has *string* values, so the existing
      integer-checking `loadJsonToUnorderedMap` rejects it). Keep `loadJsonToUnorderedMap` for
      `fixed.json` unchanged.
- [x] **D5** Add `load_consumables` — `consumables.json` is an **array** of objects, which
      `loadJsonToUnorderedMap` cannot read.
- [x] **D6** Add `load_class_map(path, section, model_names)`, throwing when
      `model_names != expects_classes`.
- [x] **D7** `utils.cpp:11` `get_object_from_id()` returns `{}` — a stub. Prefer deleting it and
      the `Consumable` struct in favour of a single `ConsumableInfo`.
- [x] **D8** Constants: `consumables_path`, `class_map_path`, both model paths, `bag_size = 8`,
      `collectibles_path`, and `item_names_path = "resources/items.json"` — the complete 726-entry
      table. (F3 renamed it: the file that used to be `items_old.json` is the good one, and the
      95-entry fragment that used to hold the `items.json` name is gone.)
- [x] **D9** `constants::max_elements_rank` (15) is declared and never used — make it the
      suggestion-list cap.
- [x] **D10** New `tests/data_test.cpp`: 29 consumables load; `items[45] == "Yum Heart"`; both
      class-map sections have 21 entries; every referenced consumable id is 1..29 and present.

---

## E — RecipeSuggestor (the core missing feature)

`RecipeSuggestor::suggest` currently only prints counts.

- [x] **E1** Rewrite the header around a
      `Suggestion { item_id, item_name, recipe, to_add, quality_score, bool exact }` and
      `suggest(const std::vector<ConsumableID>& bag, const std::map<ConsumableID, Quantity>& floor)`,
      plus `bind_models(boc_names, floor_names)` and `static std::string format(...)`.
- [x] **E2** Take a plain vector, **not** `CircularList&`. The pipeline snapshots under
      `results_mtx` and computes outside it; the current signature would force the whole
      computation to hold the lock and block both detectors.
- [x] **E3** Translate bag and floor through the class map (apply `qty`, drop `extra`, truncate to
      8 so a spurious 9th detection cannot poison the bag).
- [x] **E4** Memo cache keyed on `serializeBag(bag) + "|" + serialize(floor)`. This is the
      "cache that prevents us always computing the same objects" item — inputs only change when the
      game state does, so at 30 Hz it short-circuits ~29 of every 30 calls.
- [x] **E5** Enumerate multisets of size `8 - bag.size()` from the floor supply, respecting
      available quantities, deduped by sorted key, capped at 20 000 with a warning. Real floors have
      ≤6 distinct pickup types, so this stays in the hundreds.
- [x] **E6** Resolve each candidate: a `fixed.json` hit ⇒ `exact = true`; otherwise the heuristic ⇒
      `exact = false`.
- [x] **E7** Rank by `exact` desc, `quality_score` desc, `to_add.size()` asc, `item_id` asc
      (last key for determinism); truncate to `max_elements_rank`; resolve names from
      `items.json` with an `"Item #<id>"` fallback.
- [x] **E8** Console output — the Pipeline prints, the suggestor formats. `*` marks a guaranteed
      `fixed.json` recipe; everything else carries `(approx)`.
- [x] **E9** Pipeline integration: snapshot under `results_mtx`, compute outside it, call
      `bind_models()` in `initialize()`, and raise the suggestor poll from 33 ms to ~200 ms
      (30 suggestion blocks per second is unreadable).
- [x] **E10** Add `CircularList::snapshot()` (~8 lines, copies under its own lock). `begin()`
      (`circular_list.hpp:296`) takes the spin lock and **releases it on return**, so iteration is
      unsynchronised.
- [x] **E11** Add `--replay <dir>` to `main` (~30 lines) feeding `resources/test_images/*.jpg`
      through Router → Detectors → Suggestor, so the demo is reproducible without X11 or the game.
- [x] **E12** Add `--seed "<run seed>"`. `BagOfCrafting` needs a run seed and nothing supplies one;
      there is no seed OCR anywhere in the codebase.
- [x] **E13** New `tests/suggestor_test.cpp`: 8 pennies ⇒ exact hit on `"8,8,8,8,8,8,8,8": 177`;
      7 pennies + floor `{penny:1}` ⇒ same recipe with `to_add == {8}`; `extra` dropped;
      `{double_heart:1}` ⇒ supply `{1: 4}`; identical inputs twice ⇒ memo hit.

### E — BagOfCrafting: the pass-1 honesty ceiling

`src/bag_of_crafting.cpp` is a 1:1 transliteration of the **explicitly-dummy** Python in
`HASHING.md` §3. Everything except the 20 `fixed.json` recipes is fabricated.

- [x] **E14** Replace the fake `qualities.push_back(id % 5)` (`:61`) with the **real** quality field
      from `consumables.json` (it matches `HASHING.md` §2 exactly).
- [~] **E15** `get_quality_range`'s ladder (9/14/18/22/26/30) is still unverified, and we now know
      `fixed.json` **cannot** verify it. With the real quality table in place the ladder agrees
      with only 13 of the 19 fixed recipes — but that is not evidence against it: those recipes
      are hardcoded exceptions that bypass the quality roll entirely, so they say nothing about
      the ladder either way. `bag_of_crafting_test` prints the comparison for the record and
      deliberately does not assert on it. Real validation needs seed+bag test vectors (G6).
- [x] **E16** Delete the fabricated `get_item_quality` (`:99`, buckets item IDs by `<=143/286/…`)
      and `is_craftable_item` (`:107`, always true), and the `find_matching_item` retry loop that
      filters on them. A loop over invented constraints is worse than no loop — it hides that the
      output is arbitrary.
- [x] **E17** Align the LCG with the Java reference (`CraftingComputator:112-125`,
      `1664525 / 1013904223 & 0xFFFFFFFF`) so the two ports at least agree with each other. The C++
      side currently uses `1103515245 / 12345 & 0x7FFFFFFF`. **Neither is the real Repentance RNG.**
- [x] **E18** Document at the top of `bag_of_crafting.cpp` that it is an approximation.
- [x] **E19** Default `heuristic_enabled_ = false`, behind a `--heuristic` flag. The default demo
      shows the 19 exact recipes plus honest quality scores. A hash that emits plausible-looking
      *wrong* item names is worse than fewer results.

---

## F — Tech debt / polish

- [x] **F1** Deleted `libtorch/` (732 MB, unused) and its now-pointless `.gitignore` entry.
- [x] **F2** Delete `resources/models/best.onnx` (a duplicate of `boc_best.onnx`) and repoint
      `detector_test`.
- [x] **F3** Resolve the `items.json` (95-entry fragment) / `items_old.json` (real 726-entry table)
      naming inversion — the *older*-named file is the good one.
- [x] **F4** `CircularList::to_string()` uses `std::to_string(temp->value)`, so it only compiles for
      arithmetic `T`. Guard with `if constexpr (std::is_arithmetic_v<T>)` or an ADL `to_string`.
- [x] **F5** `CircularList::rotate` compares signed `int i` against unsigned `size_` — now flagged
      by `-Wall`.
- [x] **F6** Record the decision **not** to wire `SuggestionTrie` into the pipeline: a bag is a
      sorted multiset, not a sequence, so a trie needs all 8! orderings or a canonical sort — at
      which point the existing hash map is strictly better; `move_along` (`trie.cpp:39-47`) is
      destructive, frees every sibling branch and has no reset, while the bag changes
      non-monotonically; it is not thread-safe; and with 20 recipes prefix-sharing saves nothing.
      Keep it as a tested standalone. It becomes useful only with a large real recipe database plus
      incremental as-you-pick-up querying.
- [x] **F7** Delete unused `Pipeline::frame_count` and `FrameCapturer::width/height` (or use the
      latter to validate the captured size).
- [x] **F8** `enable_testing()` + `add_test` for the hand-rolled test mains so `ctest` is a
      one-command regression gate. No GTest needed.
- [x] **F9** Refresh `recipe_suggestor/README.md`: the ONNX Runtime path
      (`models_training/cpp_compatibility_test/lib`) and the `test1.jpg` test resource are both
      stale, and the documented output filename does not match what the code writes.
- [x] **F10** Refresh `models_training/README.md`: it still documents the deleted `train.py`, the
      old single-model `.env` schema, and "100 epochs" where the scripts use 200.
- [x] **F11** Update the root `README.md` status line once the pipeline runs end-to-end.

---

## G — The real Repentance algorithm (pass 2)

Deliberately sequenced after everything above, behind the interface E1 establishes. E16 deleted
the three fabricated hooks (`get_item_quality`, `is_craftable_item`, `find_matching_item`) rather
than leaving invented constraints in place, so G3/G4 **add** the real-data versions instead of
replacing stubs.

- [x] **G1** `models_training/extract_game_data.py` runs the bundled native Linux
      `tools/ResourceExtractor/Linux/resource_extractor` against the install (default
      `~/Programs/The.Binding.of.Isaac...v1.7.8a`, override with `--game-dir`). It unpacks to a
      temp dir and deletes it afterwards; `--extract-dir`/`--skip-extract` reuse an extraction.
- [x] **G2** Quality is **not** in `items.xml` — that file has no `quality` attribute in
      v1.7.8a. It lives in `resources-dlc3/items_metadata.xml`. That plus `itempools.xml`
      (pool membership) produces `resources/collectibles.json`: **721** entries of
      `{name, quality 0-4, pools, tags}`, across 31 pools. Names still come from
      `items.json`, because `items.xml` stores localisation keys (`#THE_SAD_ONION_NAME`).
      Five ids in `items.json` (43, 61, 235, 587, 718) have no metadata — blank, cut or
      unused — and are excluded, so they can no longer be "crafted".
- [x] **G3** `craft_item` now derives the quality band from the bag's real quality sum and
      picks only from collectibles whose **real** quality falls inside it. `item_quality()` and
      `candidates(lo, hi, pool)` are the lookups. Items belonging to no pool are not craftable
      at all: a craft rolls out of a pool.
- [x] **G4** Pool weighting via `pool_points()` / `forced_pool()`. The point values are data,
      not code: `consumables.json` now carries machine-readable `pool` / `pool_points` fields
      beside the human `pool_influence` label. Covers §4 (Angel/Devil/Secret/Curse) **and** the
      three §2-only entries §4 omits: Gold Heart → goldenChest +10, Cracked Key → redChest +10,
      Poop Nugget → shellGame +10. Rune / Soul Stone's Planetarium influence is conditional
      ("if no Penny/Bomb/Key/Heart"), not a point value, so it is not implemented.
- [x] **G5** Replaced the LCG with the **real** crafting RNG, ported from the reverse-engineered
      implementation in External Item Descriptions (`features/eid_bagofcrafting.lua`,
      `EID:calculateBagOfCrafting`) and its data tables (`EID.BoC.PickupValues`,
      `EID.BoC.ComponentShifts`, now `resources/crafting_rng.json`). The pools come from the game's
      own `itempools.xml`, in file order and with per-item weights (`resources/itempools.json`),
      because crafting indexes pools by position. Verified against a transliteration of the
      reference on 300 random (seed, bag) pairs: identical output. Two corrections fell out of it —
      the quality ladder was wrong (the real bands OVERLAP, and Devil/Angel/Secret band 5 points
      lower), and pool selection is weighted per pool rather than a single "forced pool".
      What blocks exactness now is not the algorithm:
- [x] **G5a** The RNG runs on the run's 32-bit START SEED, which the game never displays; the seed
      string on the pause screen encodes it by a scheme that is not public. `tools/find_start_seed`
      recovers the number instead, by scanning all 2^32 seeds for the ones that reproduce crafts you
      actually made (the candidate distribution is seed-independent, so it is built once per
      observation and each seed costs 8 xorshifts plus a binary search). Three or four crafts pin it.
- [ ] **G5b** Superseded note: the old LCG. This is a research task — the decompiled
      crafting routine — not an implementation task, and it is blocked on an **external**
      reference rather than on anything in this repo: `HASHING.md` §3 states outright that its
      Python is a dummy, and the Java `CraftingComputator` is an independently invented LCG.
      Everything a real implementation needs *around* it (quality bands, pools) now exists.
- [x] **G6a** Validated against real crafts from three runs recorded by hand (`crafts.txt`): every
      run came back **consistent** -- run 1 (2 crafts) 16 391 surviving seeds, run 2 (3 crafts) 65,
      run 3 (2 crafts) 49 770. Wrong pool data, a wrong quality ladder or a wrong RNG would leave
      zero, since each craft is a ~1/250 constraint and they must all hold at once. Every crafted
      item also landed inside the band our port computes for its bag. What this does NOT yet prove
      is a forward prediction; see G6.
- [ ] **G6** Validate end to end against this machine's game: recover a start seed with
      `find_start_seed` from real crafts, then confirm the app predicts the NEXT craft before it
      happens. The port matching the reference proves the transcription, not that our component ids
      and pool data line up with what the game actually does in a live run.
- [x] **G7** `--heuristic` is gone. There is no approximation left to gate: with a start seed the
      item is computed by the game's own algorithm, and without one it is reported as unknown.
      Item availability is the remaining honest caveat (see the README).

---

## H — Found while completing A/G

- [x] **H1** `Pipeline::stop()` closed the queues but never drained them. `pop()` keeps handing
      out whatever is still buffered, so both detectors ran inference over the entire backlog
      (up to `queue_max_size` frames each) before reaching the sentinel: shutdown took ~5.5 s and
      `pipeline_test`'s 2 s budget failed. `stop()` now drains and frees the backlog — 239 ms.
      `BoundedQueue::drain()` already existed for exactly this and was simply not called.
- [x] **H2** `--replay` now runs the **floor** detector too. It previously bound the floor map as
      `{}` and only ever ran BoC, despite the comment promising that a full-resolution frame goes
      through the Router first. Full-screen frames are now routed into the bag and floor regions
      exactly as the live pipeline does; a 640x640 image is still treated as a bag crop, which is
      what the images in `resources/test_images/` are. The floor model stays optional.
- [x] **H3** `resources/test_images/` held only 640x640 BoC dataset crops, so nothing *in the
      repo* exercised the routed full-screen path. Two real 1920x1080 captures now live in
      `resources/test_images/screens/` and `screen_regression_test` asserts their contents
      (bag: penny + key; floor: 2 pennies + bomb; and a paused frame that must yield nothing).
      Original note follows.
      `resources/test_images/` holds only 640x640 BoC dataset crops, so nothing *in the
      repo* exercises the routed full-screen path. It has been verified once out-of-tree with a
      synthetic 1920x1080 frame (a floor validation image stretched back to full resolution):
      Router -> FloorDetector returned exactly the image's ground truth,
      `double_bomb penny penny`. Note `process_floor` masks out the bag region **and** the left
      HUD strip, so anything placed there is invisible to the floor detector. Capturing two or
      three real screenshots would turn this into a committed regression gate.

---

- [x] **H4** Preprocessing geometry did not match training. `_preprocess_image` letterboxed
      (C3) on the grounds that "Ultralytics trains and validates with letterbox" — true in
      general, but both datasets are exported from Roboflow with **"Resize to 640x640
      (Stretch)"**, so Ultralytics only ever receives pre-squashed squares and its letterbox is a
      no-op. The models learned stretched sprites; letterboxing a live 1920x1080 frame fed them a
      distribution they had never seen. Measured on the floor validation set, simulating the live
      path: **letterbox matched 68.6% of ground-truth instances, stretch 86.9%**. Preprocessing
      now squashes to match the data, `constants::letterbox_preprocessing` flips it back for a
      model trained on a dataset exported *without* Roboflow's resize, and `Detector` tracks
      independent `scale_x_`/`scale_y_`. Verified end-to-end: 4/4 exact ground-truth matches on
      full-resolution frames, empty frame included.

---

- [x] **H5** `train_common.py` hardcoded `yolov8n.pt`. It now takes `--weights` (local file or a
      name for Ultralytics to fetch), and the export verifies the ONNX output shape is
      `[1, 4+nc, N]` before shipping — the training-side twin of the C++ C8 assertion, so a model
      family with a different head cannot reach `resources/models/`. yolo11 shares YOLOv8's head
      and needs no C++ change; `yolo11s` is the recommended upgrade.

---

## I — Live-path calibration (found while trying to actually run it)

- [x] **I1** The bag strip rect was wrong: 672x194 at `crop_start_x_factor` 0.97, i.e. 58 px too
      narrow and stopping short of the right screen edge. Calibrated to (1191, 886) 729x194 from
      two independent measurements that agree — template-matching the bag sprite of a BoC training
      crop onto a real frame (match 0.96 at scale 1.14, giving origin (1189, 885) 730x194), and the
      black bag mask baked into every floor training image ((1191, 888) 729x192 after undoing the
      640x640 stretch). Factors are now 0.38 / 0.18 / 1.0.
- [x] **I2** Preprocessing was one global flag for two datasets that disagree. H4 measured the
      floor dataset (stretched) and applied its answer to both, but the BoC dataset is a wide strip
      **black-padded to a square**: content sits in rows 235..405 of every training image, a 640x170
      band, aspect 3.765 — which is exactly the calibrated strip's aspect. Squashing that strip to
      640x640 fed the model 3.8x vertically stretched sprites. Now `types::Preprocess`, pinned per
      subclass: `BoCDetector` = `LetterboxBlack`, `FloorDetector` = `Stretch`. `letterbox()` takes a
      pad colour because the BoC data is padded black, not YOLO's gray 114.
- [x] **I3** The bug was invisible to the suite because `boc_test` feeds the already-padded 640x640
      dataset image straight to the detector, skipping the Router. `boc_geometry_test` closes it:
      it undoes a training crop's padding, pastes the band into a synthetic 1920x1080 frame at the
      strip rect, routes it, and requires the same classes and counts as the raw crop (box drift
      0.69 px). With the old stretch preprocessing restored, the routed frame detects **nothing** —
      so live bag detection had never worked.
- [x] **I4** `tools/roi_preview.cpp` (target `roi_preview`, not a test): draws the strip and HUD
      rects over any full-screen capture, runs both detectors, and writes what each one receives.
      This is how to recalibrate for another resolution or HUD scale.
- [x] **I5** The console stayed completely silent when no recipe was possible, which is the normal
      case — an 8-slot recipe needs more pickups than a room usually holds — and looked exactly like
      a hung pipeline. `RecipeSuggestor::format_state` now prints the detected bag and floor with the
      reason there is no suggestion, deduplicated so an unchanged state prints once.

Verified against the running game (seed `9W4T 9ZJ2`):

```
[RecipeSuggestor] bag: Penny, Key  (2/8, quality 3)   floor: Penny x2, Bomb x1   -- no recipe: needs 6 more, floor supplies 3
```

which is exactly what was on screen.

- [x] **I8** `suggest()` could only ever fill the bag to exactly 8 from empty slots, so a full bag
      produced no plans at all, and any bag without a `fixed.json` match was dropped outright
      (`continue`) when the heuristic was off — which is nearly every bag. Between them the console
      showed nothing during normal play. It now plans over the floor: every combination of whole
      pickups, with overflow pushing the oldest entries out of the bag, ranked by the real quality
      score, and reported with its tier and forced pool even when the item id is unknowable.
      `Suggestion` gained `pickups`, `to_drop`, `band_lo/hi` and `pool`.
- [x] **I9** The overflow model assumes the bag is FIFO. **Verified in game**: with the bag full at
      `penny key bomb penny penny key key card`, walking over one Red Heart gave
      `key bomb penny penny key key card red_heart` — the oldest entry falls off the front, the new
      one lands at the back, and the detector's reading order (left to right, top to bottom) is
      exactly that queue order, so no re-indexing is needed. Both captures are committed as
      `resources/test_images/screens/fifo_{before_full_bag,after_heart_pickup}.png` and
      `screen_regression_test` asserts the relation, not just the contents.
- [ ] **I6** The left HUD strip is masked at inference but is **not** masked in the floor training
      images, where the HUD is visible and unlabelled — so the model was fitted to ignore it and the
      mask is an unnecessary black rectangle it never saw. It costs nothing on the frames tested
      (no false positives either way) but it is a gratuitous mismatch; worth removing and
      re-measuring on more frames.
- [ ] **I7** Only 1920x1080 single-monitor is calibrated. `XGetImage` grabs the whole virtual
      screen, so a second monitor changes the frame width and every factor with it.

---

## J — Build and tooling

- [x] **J1** `CMAKE_BUILD_TYPE` was empty, so every target -- detectors, pipeline, tests, and the
      2^32 seed scan -- compiled with **no optimisation flag at all**. The scan took 16+ minutes;
      with the Release default it takes 3.5. CMakeLists now sets Release when no build type is
      given. Nothing else in the repo hinted at this, and the ONNX inference paths had been
      measured in that state.
- [ ] **J2** The seed scan averages ~2.7 cores busy despite `hardware_concurrency() == 12` on this
      machine (9m19 CPU in 3m27 wall). Worth checking for a cgroup CPU quota before optimising the
      code -- the per-thread survivor vectors sharing cache lines is the other suspect.

---

- [ ] **G8** The seed-string -> start-seed encoding is still unknown. Probed the obvious base-32
      schemes (five alphabets x forward/reversed x four 32-bit extractions) against run 2's 65
      candidate seeds: no hit. Until someone reverse-engineers it, `find_start_seed` is the only
      way in, and it costs 5-6 crafts per run.

---

## Open questions

- [x] The `class_map.json` `qty` multipliers (D2) were confirmed: `half_heart` = 1x Red Heart,
      `red_heart` = 2x, `double_heart` = 4x, `double_bomb` = 2x Bomb, `red_soul_heart` = 1 red +
      1 soul.
- [ ] Reading the run seed off the screen (OCR) — currently a `--seed` CLI argument (E12).
- [x] Merging the six near-identical heart classes was **not** needed, and would have been
      actively harmful: the heart labels *are* the quantity signal (`half_heart` 1x,
      `red_heart` 2x, `double_heart` 4x), so collapsing them loses the information the bag
      needs. Labelling more data was the right call — Roboflow v2 already had it.
- [ ] The floor dataset's **validation** split is now the weak point: 8 of 23 classes have zero
      validation instances, and `mega_battery` / `red_soul_heart` have only 3 training instances
      each. Worth topping up before trusting per-class AP for those.
