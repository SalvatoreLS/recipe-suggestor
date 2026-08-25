# Tainted Cain Crafting Mechanics (Repentance v1.7.5+)

The crafting system for Tainted Cain is driven by a deterministic algorithm that uses the run seed and a specific set of 8 ingredients to generate an item ID. While most recipes are randomized per seed, the logic remains constant.

---

## 1. The Crafting Workflow

The process follows a strict hierarchy of operations:

1. **Component Collection:** 8 consumables are added to the Bag of Crafting.
2. **Ingredient Sorting:** The 8 internal IDs are sorted in ascending order.
3. **Quality Calculation:** The sum of the quality weights of the 8 items determines the target item quality tier.
4. **Pool Determination:** Specific "special" consumables add weights to specialized item pools (e.g., Eternal Hearts for the Angel Pool).
5. **Hashing:** The sorted IDs and the seed are passed through a custom PRNG.
6. **Iteration:** If the hashed result does not meet the quality or pool requirements, the PRNG is stepped until a match is found.

---

## 2. Consumable Data Reference

| ID | Consumable | Quality | Pool Influence |
| --- | --- | --- | --- |
| 1 | Red Heart | 1 | — |
| 2 | Soul Heart | 4 | — |
| 3 | Black Heart | 5 | Devil Room (+10) |
| 4 | Eternal Heart | 5 | Angel Room (+10) |
| 5 | Gold Heart | 5 | Golden Chest (+10) |
| 6 | Bone Heart | 5 | Secret Room (+5) |
| 7 | Rotten Heart | 1 | Curse Room (+10) |
| 8 | Penny | 1 | — |
| 9 | Nickel | 3 | — |
| 10 | Dime | 5 | — |
| 11 | Lucky Penny | 8 | — |
| 12 | Key | 2 | — |
| 13 | Golden Key | 7 | — |
| 14 | Charged Key | 5 | — |
| 15 | Bomb | 2 | — |
| 16 | Golden Bomb | 7 | — |
| 17 | Giga Bomb | 10 | — |
| 18 | Micro Battery | 2 | — |
| 19 | Lil' Battery | 4 | — |
| 20 | Mega Battery | 8 | — |
| 21 | Card | 2 | — |
| 22 | Pill | 2 | — |
| 23 | Rune / Soul Stone | 4 | Planetarium (if conditions met) |
| 24 | Dice Shard | 4 | — |
| 25 | Cracked Key | 2 | Red Chest (+10) |
| 26 | Golden Penny | 7 | — |
| 27 | Golden Pill | 7 | — |
| 28 | Golden Battery | 7 | — |
| 29 | Poop Nugget | 0 | Shell Game (+10) |

---

## 3. Implementation of Crafting Logic

> **This section used to hold an invented LCG**, presented as a demonstration and marked as a
> dummy. It was transcribed into C++ and Java anyway, and it is the reason the app named the wrong
> collectible for every craft. What follows is the real algorithm; the code in
> `src/bag_of_crafting.cpp` is a port of it, verified against the reference on 300 random
> (seed, bag) pairs.

Source: [External Item Descriptions](https://github.com/wofsauge/External-Item-Descriptions),
`features/eid_bagofcrafting.lua` (`EID:calculateBagOfCrafting`), a reverse-engineered
reimplementation of the game routine, plus its data tables in `features/eid_data.lua`
(`EID.BoC.PickupValues`, `EID.BoC.ComponentShifts`).

```python
def craft(components, start_seed):          # components: 8 component ids
    comps = sorted(components)              # the result depends on the MULTISET, not bag order

    # 1. Walk a xorshift RNG once per ingredient. Each ingredient selects its own
    #    shift triple, so the ingredients themselves drive the state.
    state = start_seed                      # the run's 32-bit START SEED, not the seed string
    counts, total = [0] * 64, 0
    for c in comps:
        counts[c] += 1
        total += PICKUP_VALUES[c]           # the table in section 2
        state = xorshift(state, SHIFTS[c])
    shift = SHIFTS[6]                       # every later roll uses this one triple

    # 2. Pools the bag unlocks, and how heavily each counts.
    pools = [(0, 1), (1, 2), (2, 2),                        # treasure, shop, boss: always
             (3, counts[3] * 10),   # devil        <- Black Hearts
             (4, counts[4] * 10),   # angel        <- Eternal Hearts
             (5, counts[6] * 5),    # secret       <- Bone Hearts
             (7, counts[29] * 10),  # shell game   <- Poop Nuggets
             (8, counts[5] * 10),   # golden chest <- Gold Hearts
             (9, counts[25] * 10),  # red chest    <- Cracked Keys
             (12, counts[7] * 10)]  # curse        <- Rotten Hearts
    if counts[1] + counts[8] + counts[12] + counts[15] == 0:   # no Red Heart/Penny/Key/Bomb
        pools.append((26, counts[23] * 10))                    # planetarium <- Runes

    # 3. Every candidate's weight, banded by quality PER POOL.
    weights = defaultdict(float)
    for idx, pool_weight in pools:
        if pool_weight <= 0:
            continue
        n = total - (5 if 3 <= idx <= 5 else 0)   # Devil/Angel/Secret band 5 points lower
        qmin, qmax = quality_band(n)
        for item_id, item_weight in ITEM_POOLS[idx]:
            if qmin <= QUALITY[item_id] <= qmax:
                weights[item_id] += item_weight * pool_weight

    # 4. Weighted draw, walking candidates in ascending id order. Up to 20 attempts:
    #    the game rerolls when the item is already gone from the pool or not unlocked.
    for _ in range(20):
        target = xorshift_next(state, shift) * 2.3283061589829401e-10 * sum(weights.values())
        for item_id in sorted(weights):
            target -= weights[item_id]
            if target < 0:
                return item_id if available(item_id) else BREAK_AND_REROLL
    return 25
```

with

```python
def xorshift(state, shift):                 # shift is a triple, per ingredient
    state ^= (state >> shift[0]) & 0xFFFFFFFF
    state ^= (state << shift[1]) & 0xFFFFFFFF
    state ^= (state >> shift[2]) & 0xFFFFFFFF
    return state & 0xFFFFFFFF

def quality_band(n):                        # NOTE: the bands OVERLAP
    if n > 34: return 4, 4
    if n > 26: return 3, 4
    if n > 22: return 2, 4
    if n > 18: return 2, 3
    if n > 14: return 1, 2
    if n > 8:  return 0, 2
    return 0, 1
```

Two details the old dummy got wrong and that matter more than the RNG itself: the quality bands
**overlap** (a 24-point bag can roll anything from quality 2 to 4), and pool selection is a
**weighted mix of every unlocked pool**, not a single forced pool.

### The start seed

The RNG runs on the run's 32-bit **start seed**. The game displays a seed *string* on the pause
screen (`7W2N L9AK`); the encoding between the two is not public, so the string cannot be converted.
`tools/find_start_seed` recovers the number instead, by scanning all 2^32 seeds for the ones that
reproduce crafts you have actually made. Five or six crafts pin it exactly.

## 4. Item Pool Weighting

If specific ingredients are present, the algorithm adds "Pool Points." If a pool has enough points, the game forces the item selection to roll from that specific pool first.

* **Angel Pool:** 1 Eternal Heart = 10 points.
* **Devil Pool:** 1 Black Heart = 10 points.
* **Secret Pool:** 1 Bone Heart = 5 points.
* **Curse Pool:** 1 Rotten Heart = 10 points.

The pool with the highest points (above a threshold) dictates the `GetItemFromPool` function call within the iteration logic.