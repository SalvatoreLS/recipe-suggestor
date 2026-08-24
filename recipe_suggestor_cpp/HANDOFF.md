# Where this is, and what to do next

Working notes for picking this back up. `TODO.md` is the full checklist; this is the short version
plus the one experiment that is currently open.

---

## One-paragraph status

The app runs end to end against the live game: it reads the crafting bag and the floor pickups off
the screen correctly, plans which pickups to grab, and ranks the plans by the bag's real quality.
All of that is verified. The one thing that does **not** work yet is naming the collectible a craft
will produce, and the reason is narrowed down to a single unknown: the run's 32-bit crafting seed,
which the game never displays. There is a tool that recovers it from crafts you have actually made,
but on the crafts recorded so far no seed reproduces more than four of them, and that is expected by
chance rather than evidence. The open experiment is to re-record crafts with the bags read off the
screen instead of typed, which decides between "the bags were mistyped" and "v1.7.8a behaves
differently from the reference implementation".

---

## Verified, do not re-derive

| Fact | Evidence |
|---|---|
| Bag strip is at (1191, 886) 729x194 on 1920x1080 | Bag sprite template-matched from a training crop (0.96), and the black bag mask baked into every floor training image. Two independent measurements agreeing. |
| BoC model needs **black letterbox** preprocessing, floor model needs **stretch** | The two datasets were built differently. With stretch on the bag, the BoC detector returns *nothing*: `boc_geometry_test` fails outright with the old setting. |
| The bag is a **FIFO queue** of 8 | Recorded in game: full bag `penny key bomb penny penny key key card` + one Red Heart became `key bomb penny penny key key card red_heart`. Oldest out the front, new one at the back. Pinned by `screen_regression_test`. |
| Detector reading order (left-to-right, top-to-bottom) **is** the queue order | Same capture pair. So the leftmost slot is the oldest, and `to_drop` taking from the front is correct. |
| Crafting depends on the **multiset**, not bag order | The reference sorts the 8 components before anything else. Bag order only decides what gets pushed out on overflow. |
| Our port of the crafting algorithm is faithful | 300 random (seed, bag) pairs, identical to a transliteration of the reference Lua. |
| Component ids and the fixed-recipe table are right | 8 pennies produced Portable Slot in game, exactly as predicted. That path uses no RNG. |
| Pools, weights, qualities, fixed recipes all match the reference | Diffed against EID's own export. Only difference: EID has 1-4 newer items in treasure/devil/angel/curse, from a later game version. Ours come from this install, which is what counts. |
| `craftingquality` does not exist in v1.7.8a | EID uses `item.CraftingQuality or item.Quality`; the attribute is absent from this version's `items_metadata.xml`, so plain quality is correct here. |

---

## The open question

`BagOfCrafting::craft_item` is a port of the real algorithm (see `HASHING.md` section 3), and it
needs the run's **start seed**: a 32-bit number the game uses internally and never shows. The seed
*string* on the pause screen encodes it by a scheme that is not public — probing the obvious base-32
variants against a real candidate set found nothing (TODO G8).

So `tools/find_start_seed` recovers the number by brute force: scan all 2^32 seeds for the ones that
reproduce crafts you actually made. It works — planting a known seed, generating six crafts from it
and running the search recovers exactly that seed and nothing else.

**On real crafts it has not converged.** From `crafts.txt`:

```
run LSV2 S8CB, 8 crafts recorded by hand
  crafts 1-2 (Keeper's Box, Magic Scab)   49770 seeds
  + Sharp Straw                              98
  + Tinytoma                                  2
  + The Ladder                                0   <- breaks here
  + the remaining three                       0
```

Beware the statistics: **expected survivors from N crafts is ~4.3e9 / 250^N**, which is ~275 for 3
crafts and ~1 for 4, *whatever the crafts are*. So the 65 survivors on an earlier 3-craft run and
the 2 here are the null result, not confirmation. The first genuine evidence would be a seed
surviving **5 or more** crafts. None does yet.

Two candidates remain:

1. **The typed bags are wrong somewhere.** One mistyped component invalidates that craft's
   constraint and silently eliminates the true seed. This is what the open experiment tests.
2. **v1.7.8a differs from the reference.** EID targets Repentance+ (v1.9.x); the shift triples or
   pickup values could differ between versions.

---

## The experiment to run

`tools/craft_recorder` removes the typing from the loop: it watches the live bag and, when a full
bag of 8 empties (which is what crafting looks like), writes the exact 8 components that were in it
plus the frame captured right after, so the collectible that popped out can be identified.

```bash
cd recipe_suggestor_cpp/build
./craft_recorder --seconds 1200 --out crafts_run4
```

Then play **one** Tainted Cain run, fullscreen 1920x1080, and craft **six times**. Ctrl-C when done.
It writes `crafts_run4/observations.txt` with one line per craft and `????` where the item id goes,
plus `crafts_run4/craft_N_after.png` for each.

Fill in the item ids (names resolve via `resources/items.json`), then:

```bash
./find_start_seed crafts_run4/observations.txt --out crafts_run4/seeds.txt
```

About 3.5 minutes. It reports how many seeds survive each craft.

### How to read the outcome

- **One seed survives all six** — the algorithm is right and the earlier failures were transcription.
  Run the app with `--start-seed <n>` and item names become the game's own answers.
- **Zero survive** — the bags are now certainly correct, so the suspect is the game version. Next
  step there is to check whether EID's `ComponentShifts` / `PickupValues` changed between v1.7.8a
  and Repentance+, or to test against a Repentance+ install.
- **Hundreds survive** — a craft was recorded wrong (a pickup landed between the capture and the
  craft, say). Record more.

A partly-narrowed candidate set is accepted (`--start-seed seeds.txt`) and names an item when every
candidate agrees — but measured, that essentially never happens: even 2 surviving seeds agreed on 0
of 40 test bags. Treat the exact seed as the requirement.

---

## Running the app

```bash
cd recipe_suggestor_cpp/build
./RecipeSuggestorCPP --seed "9W4T 9ZJ2"                      # live
./RecipeSuggestorCPP --seed "9W4T 9ZJ2" --start-seed 123456  # with item names
./RecipeSuggestorCPP --replay resources/test_images --seed "9W4T 9ZJ2"   # no game needed
ctest                                                        # 12 tests, all green
```

Fullscreen 1920x1080 only — every region is a fraction of the captured screen, and a second monitor
would change the frame width and every factor with it (TODO I7).

If detection ever looks wrong:

```bash
import -window root ~/frame.png
./roi_preview ~/frame.png      # draws both regions, runs both detectors, writes outputs/
```

---

## Tools added while investigating

| Tool | What it is for |
|---|---|
| `roi_preview` | Draws the bag strip and HUD mask over any capture, runs both detectors, writes what each one receives. How to recalibrate for another resolution. |
| `find_start_seed` | Recovers the run's crafting seed from observed crafts. `--out` dumps every candidate. |
| `craft_recorder` | Records crafts from the live game into an observations file. |
| `bag_log` | Reads the bag out of saved frames, as component ids. |

New data files: `resources/crafting_rng.json` (shift triples, pickup values),
`resources/itempools.json` (pools in file order with weights, from this install),
`resources/test_images/screens/*.png` (four real captures used as regression gates).

---

## Traps worth remembering

- **`CMAKE_BUILD_TYPE` was empty**, so everything compiled with no `-O` flag. The seed search took
  16 minutes; with Release it takes 3.5. CMakeLists now defaults to Release. Any timing measured
  before that is meaningless.
- A **floor Soul Heart pickup is worth 2 bag units**; a half soul heart is 1. Plans must take whole
  pickups — there is a test for it.
- `boc_test` feeds the detector a pre-padded dataset image directly, bypassing the Router, so it
  cannot catch live-path geometry bugs. `boc_geometry_test` and `screen_regression_test` are the
  ones that can.
