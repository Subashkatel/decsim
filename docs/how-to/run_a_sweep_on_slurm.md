[decsim docs](../README.md) › [How-to guides](README.md)

# How to run a sweep on Slurm

A real sweep is millions of shots. `decsim collect` saves every point's
shots as pieces the moment each ends, so a job that hits its time limit
loses only the pieces it was running, and the same job submitted again
picks up where it stopped.

## 1. Size the pieces

A piece is one point's block of shots that one process runs from start
to finish. The yaml's `collection` section sets its size in QEC rounds,
`piece_rounds`, 20,000 unless it says otherwise (`configs/reference.yaml`),
so a piece takes about as long at any history length. A smaller piece
loses less to a killed job, at the cost of building the window models
more often, since they are cached per piece.

## 2. Pin the code the job will run

A job imports the tree as it stands when it starts. A commit landing
between submitting and starting gives the job code nobody chose. So
submit from a worktree pinned at a commit, and leave the tree you work
in free:

```bash
git worktree add ../decsim-weak_ler <commit>
cd ../decsim-weak_ler
```

The commit and the dirty flag go into every run folder's
`manifest.json` and every piece's `piece.json`, so a result names its
code. The interpreter decides which tree is imported, not the directory
you submit from: an environment installed with `pip install -e` from
another checkout imports that checkout, so a worktree needs
`PYTHONPATH` naming it.

## 3. Submit the job

```bash
sbatch -c 16 -t 16:00:00 --wrap \
  "decsim collect configs/weak_ler.yaml --processes 16 --out $PWD/results/weak_ler"
```

`--out` must be an absolute path, so the job writes where you will look.
`--processes` runs that many pieces at once, one per worker.

## 4. If the job dies

Submit the same line again. Every piece whose folder exists is skipped;
a piece folder appears only once the piece is whole, so nothing is
counted twice, and the pieces that were running are run again from
their first seed.

## 5. Read the numbers

The run folder is `results/weak_ler/combined/<name>-<id8>/`, folded from
every piece. Its `sweep.csv` holds one row per point, its values and its
counts, and `decsim.results.load` reads it beside every setting
([How to compare two runs](compare_two_runs.md)).

## Read next

- [Your first sweep](../tutorials/first_sweep.md): a small sweep end to end, with the
  error bars explained.
- [The run folder](../reference/run_folder.md): what a piece and a run folder hold.
- [The commands](../reference/cli.md): every flag of `collect`.
