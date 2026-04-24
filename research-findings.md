# Research Findings

## Summary

Parallel LLM worker architectures can reduce latency for a single request by splitting work across multiple model calls, but they also multiply backend load.

## Key Observation

If one user request fans out into 7 parallel LLM calls, then the system-level load grows much faster than the user count.

- 1 user -> 7 parallel LLM calls
- 10 users -> 70 parallel LLM calls
- 100 users -> 700 parallel LLM calls
- 1,000 users -> 7,000 parallel LLM calls

This means latency optimization at the per-request level can create a concurrency amplification problem at the system level.

## Why This Matters

Even if each individual user gets a faster answer path, the backend must still absorb the multiplied request volume.

Potential limitations include:

- Per-deployment throughput limits
- Token-per-minute and request-per-minute quotas
- Temporary queueing under burst traffic
- Higher tail latency when many requests hit the same model deployment at once
- Higher operating cost because each user request now creates multiple LLM calls instead of one

## Practical Interpretation

Parallel LLM calls are useful when:

- The work can be decomposed cleanly
- Each worker meaningfully reduces the final synthesis burden
- The backend capacity is sized for fan-out traffic

Parallel LLM calls become risky when:

- Every request fans out aggressively
- Many users arrive at the same time
- All worker calls hit the same model deployment
- The latency savings from parallel reasoning are smaller than the overhead of creating many extra LLM calls

## Example Limitation For This Project

In the orchestrated RAG design, one user query may create planner, worker, reviewer, and synthesizer calls. If the worker stage fans out into 7 parallel calls, then scaling to real-world traffic can become expensive and operationally heavy.

For example:

- A single user may trigger 7 parallel worker calls
- 1,000 simultaneous users may trigger around 7,000 worker calls, plus planner/reviewer/synthesizer calls

So one limitation of this design is that faster synthesis through parallel worker LLM calls can create very high backend concurrency pressure as user traffic grows.

## Recommended Guardrails

- Cap per-request fan-out
- Use smaller and faster models for worker stages
- Reserve larger models for final synthesis only
- Add per-user and global concurrency controls
- Measure stage-level latency and queueing behavior
- Design for graceful degradation under load

## Conclusion

Parallel LLM orchestration can improve single-request latency, but it does not come for free. It shifts the problem from one large model call to many smaller concurrent calls. At higher user volume, that fan-out can become a major system limitation.