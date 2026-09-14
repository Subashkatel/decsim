[decsim docs](../README.md) › Reference

# Reference

Reference pages state what the pieces are, in a fixed shape, so that
you can look a thing up while you work. Four of them are generated from
the source by `tools/docs_map.py` and checked by `tests/test_docs.py`
on every run of the suite, so they cannot drift from the code; to
change one, change the source it reads and run the tool.

## What goes in and what comes out

- [The yaml surface](yaml.md): how to read `configs/reference.yaml`,
  which is the key reference, the `extends` rule, `decsim show`, and
  every shipped config with what it is for.
- [The run folder](run_folder.md): every file a run writes, every
  column of every csv, and the latency points.
- [The commands](cli.md): every subcommand and every flag, read from
  the argument parsers. Generated.

## The shape of the package

- [The plug-in tables](tables.md): the seventeen tables, the file each
  lives in, the yaml key that names it, and every row. Generated.
- [The ports](ports.md): every port of `decsim/ports.py` with its
  methods and members, in the order a readout travels. Generated.
- [The map of the package](map.md): every package and module in uses
  order, with the first sentence of its docstring. Generated.
- [Glossary](glossary.md): what the code calls a thing, what the papers
  call it, and where in the papers to read it.
