"""The frontend: everything that happens to a program before and while it
runs on the QPU. The frontends turn an input (a QLX schedule, a small
circuit) into decsim operations, streams and rounds; the planner sizes the
decoding windows and their dependencies ahead of time; the execution
runtime decides which operation runs when. Planning is build time, off the
reaction path."""
