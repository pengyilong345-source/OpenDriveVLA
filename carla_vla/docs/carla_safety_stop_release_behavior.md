# Safety-stop release behavior (D1.8.1)

## Gateway safety-stop mechanism

The gateway applies safety-stop (brake=1, throttle=0, steer=0) when:
1. The model response times out (status="timeout")
2. The model response is stale (status="stale_first")
3. The model output is all-zero (status="all_zero_abnormal")

## No latching

The safety-stop is **per-decision, not per-episode**. Each decision
cycle independently:
1. Sends a request to the server
2. Waits for a response
3. Applies whatever control the response contains

If the model produces a non-zero trajectory on cycle N+1, the
safety-stop from cycle N does NOT persist. The ego immediately
receives the model's new trajectory-derived control.

## Implications for stop/resume

- If the model stops (all-zero) at decision K, the safety-stop is applied.
- If the model produces non-zero at decision K+1, the safety-stop is released.
- The model must independently decide to resume; no external mechanism
  forces the safety-stop to release.

## Test result

In the pedestrian test (decision 8 all-zero, decisions 9-10 non-zero),
the safety-stop was applied at decision 8 and released at decision 9.
This suggests the safety-stop correctly releases when the model
produces non-zero output.
