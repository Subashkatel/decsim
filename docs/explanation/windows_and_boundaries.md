# Windows and boundaries

A decoder cannot wait for the end of a computation before it decides
anything, because a fault-tolerant computation does not end. It has to
decide as the data arrives. Windowing is how, and boundaries are what it
costs.

## Why a window exists

The syndrome data of a run is a three-dimensional object: the checks of
the code in two dimensions and the rounds in the third. Decoding it
means finding the most likely set of faults consistent with the
detection events in that volume. Doing that over the whole volume at
once is both impossible online and unnecessary, because a fault's
influence does not reach far in time.

Dennis, Kitaev, Landahl and Preskill said so and gave the method: store
the syndrome history for only a finite time, recognise that the older
syndrome is more trustworthy than the newer, and correct only the older
part, keeping the rest for the next step. They called it the
**overlapping recovery method** (arXiv:quant-ph/0110143, the section of
that name, and Fig. 13).

decsim's **window** is one step of that method. Skoric et al. give the
two regions their names: "a window can be divided into two regions: a
commit region consisting of the 'long-lived' defects in the first
`n_com` rounds, and a buffer region containing the last `n_buf` rounds"
(arXiv:2209.08552, Sec. I B, `2209.08552.txt` lines 196-201). Toshio et
al. write the same two as `r_com` and `r_buf`. In decsim they are
`commit_round_count` and `buffer_round_count`, and both default to the
code distance.

- The **commit region** is the part of the window whose correction is
  taken as final. Nothing revises it later, unless the escalation
  re-decodes it.
- The **buffer region** is read but not committed. It exists so that a
  fault chain crossing the end of the commit region is seen, rather than
  guessed at.

## What a boundary is

Committing part of a window and not the rest leaves a mark. A chain of
tentative corrections that crosses out of the commit region ends
somewhere, and where it ends the next window sees a defect that the
physics did not put there. Skoric calls these **artificial defects**:
they are "defects" the previous window's own correction created at the
boundary, and the next window has to be told about them or it will
decode a syndrome that is not the one it is really facing
(`2209.08552.txt` lines 272-284).

That message is the **boundary**. In decsim it arrives at the receiving
window as `boundary_in`, and it rides the `decoder_to_decoder` hop. The
same thing has other names elsewhere: qLDPC calls it `net_error`, cuda-q
QEC calls it `syndrome_mods`.

The message updates the detectors of the receiving window's oldest round
layer, and nothing else. That is a narrow, checkable claim, and
`decsim/windows/boundary_payloads.py` names four independent sources for
it: Tan et al. arXiv:2209.09219, quits' `syn_update` over one check
layer, cuda-q QEC's `syndrome_mods` bounded to the next window's first
round, and Bombin et al. arXiv:2303.04846.

## How the boundary is priced

`BOUNDARY_PAYLOADS` has two rows, and the choice is a real modelling
question rather than an implementation detail.

- `dense_seam_mask`, the default: one bit per detector of the seam
  layer, whatever the noise did. That is `d*d - 1` bits for a rotated
  surface code patch. It is the default because both compiled
  implementations carry a mask, so the cost does not depend on the error
  rate, which is what a parameter sweep wants.
- `sparse_seam_list`: the flipped detectors and their index width. That
  is how Skoric's blocks exchange the artificial defects themselves, and
  it is what Bombin's "small number of check generators" implies. It is
  the row to pick when the question is whether the bandwidth claim holds.

## When the boundary ships

`BOUNDARY_POLICIES` has two rows, and this one is about revision.

- `eager` ships at every commit, provisional or final. A serial chain of
  windows keeps moving.
- `held` ships only once the committing result is final. That is what a
  run that may revise a weak result needs, because a provisional
  boundary that has already shipped is never corrected when the strong
  tier answers for that window later
  (`decsim/windows/boundary_policies.py`, citing Toshio et al.
  arXiv:2510.25222 Sec. III A).

The default is not written in the yaml but declared by the rows
themselves: an escalation policy that never escalates declares `eager`,
and one that may escalate asks its strong window shape, whose absorption
behaviour decides. `decsim/build/plan.py`, `_boundaries_name`, is where
that happens, and its docstring cites gem5's rule that a default lives
on the class that owns the parameter.

## The four windowing schemes

`WINDOWING_SCHEMES` says how the windows of an operation are laid out.
Each row is one file under `decsim/windows/schemes/`.

| Row | The layout | Source |
| --- | --- | --- |
| `sliding` | window i commits `n_com` rounds and reads `n_buf` more; window i+1 begins where window i's commit ended. A serial chain. | Skoric arXiv:2209.08552 Sec. I B |
| `parallel` | the stream is cut into A blocks that decode at the same time and B blocks that reconcile the seams between them, so the dependency graph has depth two rather than the length of the chain | Skoric arXiv:2209.08552 Sec. I C |
| `sandwich` | type-1 cores of width `s + 2b` every `s` rounds, with a one-layer type-2 seam window between each adjacent pair | Tan et al. arXiv:2209.09219, supplement |
| `naive_online` | one window over the whole operation, decoded as one batch once the rounds have arrived | no windowing; the baseline the others are measured against |

The interesting difference between them is not accuracy but dependency
depth. A sliding chain is serial: window i+1 cannot commit until window
i has. Skoric's parallel blocks and Tan's sandwich both break that
chain, at the price of extra windows and extra seams. decsim's job is to
price that trade, and each row declares its own answer to the questions
the rest of the machine asks: whether its commits form one serial chain,
whether it has a trailing tail of context, and whether it supports a
stream whose length is not known in advance
(`decsim/ports.py`, `WindowingScheme`).

## The seam

A **seam** is the layer two windows share. In a sliding chain it is
where one window's commit ends and the next begins. In Tan's sandwich it
is a window of its own, the type-2 block, which exists only to reconcile
two independently decoded cores.

The seam is where every boundary payload is measured, and where a
mispricing hides: a boundary that names the wrong layer costs the wrong
number of bits and nobody notices, because the correction is still
right. `tests/windows/test_boundary_payloads.py` and
`tests/windows/test_window_interactions.py` pin the layer.

## Read next

- `docs/explanation/two_tiers.md`: what happens when a committed window
  has to be decoded again.
- `docs/reference/glossary.md`: `r_com`, `r_buf`, `r_strong` and the
  rest, with their sources.
- `docs/how-to/add_a_table_row.md`: adding a scheme of your own.
