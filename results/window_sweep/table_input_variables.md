| feature | type | available_at_inference | updated_during_rollout | rollout_rule |
| --- | --- | --- | --- | --- |
| gpu_util_frac | Observed exogenous | History only | No | Hold last observed value fixed |
| gpu_mem_util | Observed exogenous | History only | No | Hold last observed value fixed |
| mem_util_frac | Observed exogenous | History only | No | Hold last observed value fixed |
| qps | Observed exogenous | History only | No | Hold last observed value fixed |
| hour_sin | Engineered calendar | Deterministic future | Yes | Advance from forecast timestamp |
| hour_cos | Engineered calendar | Deterministic future | Yes | Advance from forecast timestamp |
| dow_sin | Engineered calendar | Deterministic future | Yes | Advance from forecast timestamp |
| dow_cos | Engineered calendar | Deterministic future | Yes | Advance from forecast timestamp |
| power_total_kw | Observed target history | Observed history | Yes | Replace with predicted power |
| power_total_kw_roc | Engineered from target history | Derived from history/predictions | Yes | Recompute from predicted power history |
| power_total_kw_roll_mean_12 | Engineered from target history | Derived from history/predictions | Yes | Recompute from predicted power history |
| power_total_kw_roll_std_12 | Engineered from target history | Derived from history/predictions | Yes | Recompute from predicted power history |
| power_total_kw_roll_mean_72 | Engineered from target history | Derived from history/predictions | Yes | Recompute from predicted power history |
