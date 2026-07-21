"""IPC unit tests — run in the base env.

Covers the frame buffer (publish/read, torn-read, sha256 check), the socket
envelope round-trip, the latency profiler aggregation, and process health.
"""
