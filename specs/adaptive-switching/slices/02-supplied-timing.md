# 02 Supplied timing

## Question

Does the event engine use exactly the costs we supply, independently of the
host on which the scientific decoders execute?

## Deliverable and ownership

An editable, versioned timing profile loaded through existing settings and
decoder timing boundaries. Reuse the event engine, links and resource owners.
Both timing-only and real-decoding runs use this path.

## Work

1. Define table keys: hardware profile, component/decoder, distance, actual
   window geometry and concurrency mode. Add input features only when the
   chosen service model requires them.
2. Declare service time, transfer costs, resource occupancy and any initiation
   interval separately. State whether copies/launch/control costs are included.
3. Require valid units, nonnegative finite costs, provenance and a declared
   policy for terminal-window cells. Refuse missing cells by default.
4. Price union find, gap computation, belief matching, links, frame update and
   return control path. Exclude host wall time from simulated hardware time.
5. Start with fixed supplied service; later permit feature-based costs or
   measured service traces with their residual variation and profile scope.

## Gate and evidence

Use independently calculated one-job, two-job FIFO, bandwidth serialization,
dependency-wait and feedback-return fixtures. Match expected event times at
tick precision after documented clock rounding. Change a table cell and show
the expected affected stages; repeat to establish deterministic timing. On a
fixed nonadaptive memory fixture, decoder answers must remain unchanged.

## Scope boundary

Test costs can be invented and labeled synthetic. Hardware assumptions need
separate sourcing and sensitivity analysis before scientific claims. Do not
reuse RelayBP measurements for belief matching, double-count transfers, or
infer an end-to-end tail distribution by adding independent stage percentiles.
