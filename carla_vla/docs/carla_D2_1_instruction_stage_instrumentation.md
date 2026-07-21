# D2.1 Instruction-Stage Instrumentation

Each of the 13 subscenarios has a frozen `scenario_stage_contracts.json`
defining the required stages (entry condition, completion condition,
failure condition, strict order, timeout).

For every scored frame:

- `scenario_id`
- `original_instruction` (frozen high-level scenario instruction)
- `current_command` (current G1 local command)
- `current_stage`
- `previous_stage`
- `requested_transition` (from command-manager)
- `accepted_transition` (what was logged)
- `transition_reason`
- `required_stage_count`
- `emitted_stage_count`

## Recall + Order

- Recall = |emitted ∩ required| / |required|
- Order = strictly increasing indices of emitted stages within the required order

Both clauses must hold for an `instruction_stage` PASS.

## Omitted/Out-of-Order

- `omitted_stages`: required stages never emitted
- `out_of_order_stages`: True iff any required stage emitted out of order

`command-manager` success is NOT inferred from vehicle movement alone. It is
the product of the frozen scenario stage contract.
