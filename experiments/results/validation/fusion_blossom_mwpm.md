# Gate 3: decsim inner MWPM (PyMatching) vs Fusion Blossom on the same window graphs

Problem: rotated_memory_z d=3, 30 rounds, p=0.01, 300 recorded shots through the loop; 2700 window decodes captured; identical graph (decsim graphlike faults, log-odds weights scaled by 1000 to even integers) handed to both solvers.

| check | agree |
|---|---|
| minimum matching weight equal | 2700/2700 |
| Fusion Blossom correction satisfies the syndrome | 2700/2700 |
| predicted logical class equal | 2700/2700 |
| logical class differs AND weight differs (a real defect would show here) | 0/2700 |

Equal-weight ties can select different physical corrections and, when parallel faults carry different observables, different logical classes; both are valid minimum-weight matchings. Only a logical disagreement at unequal weight would indicate a solver or graph defect.
