[decsim docs](../README.md) › Explanation

# Explanation

These pages say why the machine is shaped the way it is. They are for
reading away from the keyboard, and every claim in them names its
source, a paper or a reference implementation, so you can check it.
The code keeps the same habit: each component's module docstring
names the paper or the reference implementation it follows, with the
section or the file and line, so the source of any one rule is beside
the rule.

## The machine

- [Architecture](architecture.md): the components in the order a
  readout travels, what each owns and hands on, the diagram the test
  suite checks, and the package order.
- [The data path, hop by hop](data_path.md): the eleven priced hops,
  what crosses each, how many bits, and where the numbers came from.
- [Time](time.md): ticks, the engine, clock domains, priced cards, the
  wall-clock decoder, and where the reaction time is measured.

## Decoding

- [Windows and boundaries](windows_and_boundaries.md): why decoding is
  cut into windows, what a boundary is and costs, the four windowing
  schemes, and the seam.
- [Two tiers](two_tiers.md): weak and strong decoders, the confidence
  signal, the threshold, the strong window's shape, restart, and the
  bill for all of it.

## Why it is shaped this way

- [The design decisions](decisions.md): the thirteen modelling decisions,
  each with what was decided, why, and its source, and the open rows.
- [The principles behind the shape](principles.md): the eleven ideas
  the tree is built on, each quoted from its source, and what each
  looks like in decsim.
