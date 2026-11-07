---
status: draft
version: 0.1.0
---

# toolz parity

Every public name in toolz, and where it stands in pipekit-ops.

**Read:** [pytoolz/toolz](https://github.com/pytoolz/toolz) commit `451af60dec590a6010e2babdbf391ea8f815122f`, committed 2026-09-18 ("Add pysentry-pre-commit (#637)"), tag 1.2.0, read on 2026-11-06.
The lists come from `__all__` in `toolz/itertoolz.py`, `toolz/functoolz.py`, `toolz/dicttoolz.py` and `toolz/recipes.py`, from the aliases in `toolz/__init__.py` and `toolz/curried/__init__.py`, and from `toolz/sandbox/`.
cytoolz is a drop-in with the same names, so the table covers it.

## How to read the status column

- **exists as X**: shipped before this design, at main `903b0c3`.
- **exists as Y (Pn)**: added by this design, in roadmap phase Pn; P1 to P5 have shipped, in pipekit-ops 0.2.1, so every name of this kind exists today.
  Before they shipped, these rows read "proposed as Y (Pn)".
- **out of scope**: with a reason. "Pure helper" means the function is a one-liner over values; call it inside your own operator, or in a `Lambda` while exploring.

## Deliberate divergences from toolz

1. **Operators, not functions.**
   A pipekit op is configured at construction and called on one carrier: `Filter(pred)(items)`, not `filter(pred, items)`.
   The constructor plays the role of `toolz.curry`.
2. **Configs, not closures.**
   Arguments are JSON values or operators, so the pipeline round-trips through `dumps` / `loads` and has a `pipeline_hash`.
   toolz accepts any callable; pipekit accepts a callable only through `Lambda`, which is flagged.
3. **Lists, not lazy iterators.**
   toolz is lazy by default (`take`, `partition_all`, `concat` return iterators).
   A pipekit op returns a list, so its output can be inspected, cached, hashed and passed to a `Graph` node twice.
   `Map.iter` and `Prefetch` are the lazy exceptions, for streams.
4. **Carriers are opaque.**
   An op indexes containers and calls operators; it never reads inside an item.
5. **One input per call.**
   A toolz function that takes two values (`binop(acc, item)`, `merge_with(func, *dicts)`) takes one tuple in pipekit, because `Sequential` passes one carrier.
6. **Strict by default.**
   `Zip` raises on unequal lengths; toolz and the builtin `zip` truncate.

## functoolz

| toolz | pipekit-ops status | Note |
|---|---|---|
| `identity` | exists as `Identity` | |
| `apply` | out of scope | `apply(f, *args)` is `f(*args)` |
| `thread_first`, `thread_last` | out of scope | Clojure threading macros; `pipe` and `\|` cover them |
| `memoize` | exists as `Cache`, alias `Memoize`; `key=` added (P4) | toolz's `key` argument is a function of the call's arguments; pipekit's is an operator on the input |
| `compose` | exists as `compose` | |
| `compose_left` | exists as `compose_left` | |
| `comp` | out of scope | an alias of `compose` in toolz |
| `pipe` | exists as `pipe` | |
| `complement` | exists as `complement` (a function) and as `Not(pred)` (P1) | the existing function returns a closure; `Not` is the operator form, for use inside `Filter` and `AssertThat` |
| `juxt` | exists as `juxt` (a function returning a closure) and `Fanout` (an operator, dict output) | `juxt` is not an `Operator`, so it cannot sit in a `Sequential`; see decision O13 |
| `do` | exists as `Tap`, `Sink`, and `Log` (P5) | |
| `curry` | out of scope | the constructor is the partial application |
| `flip` | out of scope | pure helper |
| `excepts` | exists as `Try`, `Coalesce` | |

## itertoolz

| toolz | pipekit-ops status | Note |
|---|---|---|
| `map` (builtin) | exists as `Map` (P2) | the execution engine |
| `filter` (builtin) | exists as `Filter` (P1) | |
| `remove` | exists as `Remove` (P1) | |
| `reduce` (functools) | exists as `Reduce` (P1) | `op` takes one `(acc, item)` tuple |
| `sorted` (builtin) | exists as `SortBy` (P1) | named for its `key`, as `GroupBy` and `CountBy` are |
| `accumulate` | out of scope | running totals over a list are rare in pipelines; write an operator |
| `groupby` | exists as `GroupBy` (P1) | |
| `reduceby` | out of scope | `GroupBy(key) \| ValMap(Reduce(op))` |
| `countby` (recipes) | exists as `CountBy` (P1) | |
| `frequencies` | out of scope | `CountBy(Identity())` |
| `partitionby` (recipes) | out of scope | runs of equal keys; rare; write an operator |
| `merge_sorted` | out of scope | pure helper |
| `interleave` | out of scope | pure helper |
| `unique` | exists as `Unique` (P1) | |
| `isiterable`, `isdistinct` | out of scope | predicates on values; pure helpers |
| `take` | exists as `Take` (P1) | for dry runs over a large search |
| `drop`, `take_nth`, `tail` | out of scope | pure helpers |
| `first`, `second`, `last`, `nth` | exists as `Get(0)`, `Get(1)`, `Get(-1)`, `Get(n)` (P1) | on lists; `first` on a lazy iterator is out of scope |
| `get` | exists as `Get` (P1) | same split as toolz: a tuple of keys returns a tuple |
| `pluck` | out of scope | `Map(Get(key))` |
| `concat` | exists as `Concat` (P1) | replaces the planned `Flatten` |
| `concatv` | out of scope | `Concat()` on a tuple |
| `mapcat` | out of scope | `Map(op) \| Concat()` |
| `cons`, `interpose` | out of scope | pure helpers |
| `iterate` | exists as `Until` (P3) | bounded: `max_iter` is required |
| `sliding_window` | out of scope | time windows over carriers belong to pipekit-cycle (`WindowedCycle`) and pipekit-array |
| `partition` | out of scope | `PartitionAll` with a length check |
| `partition_all` | exists as `PartitionAll` (P1) | replaces the planned `Chunk` |
| `count` | exists as `Count` (P1) | |
| `join` | out of scope | a relational join; use a table library |
| `diff` | out of scope | pure helper |
| `topk` | out of scope | `SortBy(key, reverse=True) \| Take(k)` |
| `peek`, `peekn` | out of scope | iterator plumbing; ops return lists |
| `random_sample` | out of scope | a seeded sampler needs an RNG in the config; belongs with the data loaders in pipekit-train |
| `zip` (builtin, not in toolz) | exists as `Zip` (P1) | strict by default |

## dicttoolz

| toolz | pipekit-ops status | Note |
|---|---|---|
| `merge` | exists as `Merge` (P1) | later inputs win |
| `merge_with` | out of scope | `GroupBy`-style merges with a reducer are rare; write an operator |
| `valmap` | exists as `ValMap` (P1) | |
| `keymap` | out of scope | `Rename` covers the fixed case; computed keys are rare |
| `itemmap` | out of scope | pure helper |
| `valfilter`, `keyfilter`, `itemfilter` | out of scope | `Pick` and `Dissoc` cover the fixed-key case |
| `assoc` | exists as `Assoc` (P1) | the value is `op(x)`, computed from the dict |
| `dissoc` | exists as `Dissoc` (P1) | |
| `assoc_in` | out of scope | `UpdateIn(keys, Const(value), default=None)` |
| `update_in` | exists as `UpdateIn` (P1) | replaces the planned `OnKey` |
| `get_in` | exists as `GetIn` (P1) | |

## curried and sandbox

| toolz | pipekit-ops status | Note |
|---|---|---|
| `toolz.curried.*` | out of scope | curried copies of the names above |
| `toolz.curried.operator` | out of scope | curried `operator` module |
| `partial` (functools) | out of scope | the constructor is the partial application |
| `sandbox.fold` | exists as `Fold` (P2) | toolz takes `map=` to parallelise; pipekit takes `executor=` and runs `Map` |
| `sandbox.unzip` | out of scope | pure helper; `Zip` on a transposed list |
| `sandbox.EqualityHashKey` | out of scope | hashing helper; `Unique(key=…)` covers the use |

## Where toolz changed the plan

| Planned | After reading toolz | Why |
|---|---|---|
| `Chunk(n)` | `PartitionAll(n)` | toolz `partition_all` has the exact semantics (last group may be short), and `partition` names the strict version; a new name would hide the lineage |
| `Flatten` | `Concat` | toolz `concat` flattens one level; "flatten" suggests full recursion, which is not what is wanted |
| `OnKey(key, op)` | `UpdateIn(keys, op)` plus `Assoc(key, op)` | `update_in` already handles nested paths and missing keys; `assoc` covers new keys |
| `GetItem` | `Get` and `GetIn` | toolz's `get` / `get_in` split, single key vs path, avoids one class with two meanings |
| `Rename` | kept | toolz has only `keymap(func)`; a fixed mapping is what pipelines need and is JSON |
| `Pick` | kept, with `Dissoc` beside it | toolz uses `keyfilter(lambda k: k in keys)`; a named op is clearer and serialisable |
| observer `Count` | iterable `Count` | toolz `count` counts the items of an iterable; an observer that counts calls would reuse the name with another meaning, so call counts go to `Log` |
| `Fold` | kept, after `sandbox.fold` | toolz separates the fold from the `map` that distributes it, which is exactly the `Map` + executor split here |
| `Until(op, pred, max_iter)` | kept | toolz `iterate` is infinite and lazy; a pipeline needs a bound |
| `Progress` | dropped | toolz has none; progress is a property of `Map`, see decision O11 |
