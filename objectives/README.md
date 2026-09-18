# Objectives / reward hierarchy destination

This subsystem defines what experiments are trying to improve.

Engineering objectives may include quality, speed, memory, compute efficiency, robustness, trainability, or task-specific metrics.

Design requirements:

- support multiple objectives without forcing a single weighted scalar;
- support hard constraints, lexicographic priorities, and Pareto fronts;
- preserve historical objective definitions with each experiment;
- distinguish task reward from safety/operational constraints;
- keep benchmark metrics separate from broader mission/governance policy.

Do not encode philosophical or moral purpose as a simplistic numerical reward. This repository's near-term role is technical architecture research; higher-order governance should constrain research use rather than be collapsed into a score the optimizer can game.
