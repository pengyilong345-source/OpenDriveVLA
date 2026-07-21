# D2 stop/resume baseline
Source: D1.8.3 + D1.8.2 frozen findings.

- D1.8.3: pedestrian stop-resume test
  - Full stop confirmed
  - Hazard cleared (scripted pedestrian)
  - Model continued producing all-zero for 5+ seconds
  - 0/1 successful resume
  - EG0 speed never exceeded 1.0 m/s after clear

Conclusion: model has confirmed restart limitation.
D2 baseline: model cannot resume from full stop.
