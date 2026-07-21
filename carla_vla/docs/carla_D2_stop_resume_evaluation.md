# D2 stop/resume evaluation
Source: D1.8.3 pedestrian test.

For each scenario requiring restart:
- successful stop + failed restart:
  stop_success = true
  resume_success = false
  task_completed = false
  episode_success = false

Stop failure: not observed in D1.8.2's moving-only scenarios (s1_1-s3_4
all had continuous forward motion). Stop occurred in D1.8.3 s2_1.

Resume success rate: 0/1 in D1.8.3 pedestrian test.
