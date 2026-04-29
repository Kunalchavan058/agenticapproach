# Research Findings
## Comparison of Four Tested RAG Approaches

### Purpose

The objective of this analysis was to evaluate how different retrieval-augmented generation approaches perform when answering complex questions.

Normal RAG is already implemented and works well for simpler, direct questions. The purpose of this comparison was to identify the most suitable next step for improving performance on more complex questions while also considering execution speed, implementation risk, and long-term flexibility.

The four approaches evaluated were:

1. Normal RAG
2. Function Calling Agent
3. Function Calling Agent using Responses API
4. Orchestrated RAG

### Overall Finding

The main finding is that search time remains relatively similar across all four approaches. The larger differences come from generation time and orchestration overhead.

In simple terms:

- Search is not the main differentiator.
- The real differentiators are answer synthesis, number of LLM calls, and workflow complexity.
- Simpler approaches are faster but less capable on complex questions.
- More advanced approaches handle complex questions better, but with higher latency.

### Sample Comparison

The following numbers are from a sample comparison run:

| Metric | Normal RAG | Function Calling Agent | Orchestrated RAG | Function Calling Agent using Responses API |
| --- | ---: | ---: | ---: | ---: |
| Search Calls | 1 | 6 | 5 | 7 |
| Chunks Retrieved | 15 | 48 | 40 | 35 |
| LLM Calls | 1 | 2 | 4 | 3 |
| Search Time | 2.45s | 2.08s | 1.85s | 1.95s |
| Generation Time | 14.71s | 18.48s | 28.62s | 5.11s |
| Total Time | 17.16s | 20.56s | 30.47s | 7.06s |

### Key Observations

1. Search time is broadly comparable across all approaches because the same search backend is used in each case.
2. The larger performance gap comes from what happens after retrieval: tool loops, planning, review, synthesis, and prompt size.
3. Approaches that retrieve more chunks and make more LLM calls generally take longer, but they also do a better job on complex questions.
4. The best approach therefore depends on whether the priority is speed, completeness, or long-term extensibility.

### Approach Assessment

#### 1. Normal RAG

Normal RAG is the simplest approach.

- One search call retrieves a fixed set of chunks.
- One LLM call generates the answer from that context.

**Strengths**

- Lowest implementation and operational complexity.
- Fast and straightforward for direct questions.
- Easy to reason about and easy to debug.

**Limitations**

- It depends heavily on the first retrieval being good enough.
- It does not adapt when the initial retrieval is incomplete.
- It is more likely to miss information in complex, multi-part, or broad questions.

**Assessment**

Normal RAG is a good baseline and should remain in place for simple use cases. However, it is not sufficient as the primary approach for complex question answering.

#### 2. Function Calling Agent

This approach uses an agent with function calling.

- The model can break a complex question into smaller search targets.
- It can call the search function multiple times.
- It can perform follow-up searches when earlier results are incomplete.

**Strengths**

- Better handling of complex and multi-part questions.
- More likely to gather missing evidence through iterative retrieval.
- Flexible design, since additional tools can be introduced later.

**Limitations**

- Higher latency than Normal RAG.
- More chunks retrieved means more context to synthesize.
- Repeated search rounds can increase response time.

**Assessment**

This is a meaningful step forward from Normal RAG because it improves completeness. The tradeoff is increased latency.

#### 3. Function Calling Agent using Responses API

This approach follows the same function-calling model, but uses the newer Responses API instead of the older chat completions flow.

- The model still decides when to call the search function.
- It still uses iterative retrieval.
- It supports parallel tool calling and a cleaner execution flow.

**Strengths**

- Best balance of speed and capability in the observed runs.
- Retains the benefits of iterative retrieval.
- Appears to reduce some of the orchestration overhead seen in the standard function-calling approach.

**Limitations**

- Still more complex than Normal RAG.
- Still slower than single-shot retrieval for simple questions.

**Assessment**

This is the strongest short-term implementation option. It improves support for complex questions while keeping latency at a practical level.

#### 4. Orchestrated RAG

Orchestrated RAG is the most structured approach.

- A planner creates targeted retrieval tasks.
- Executors run the searches.
- A reviewer checks whether enough evidence has been collected.
- A synthesizer produces the final answer.

**Strengths**

- Strongest architecture for very complex questions.
- Clear separation of responsibilities.
- Easier to extend in the future with more agents, tools, and workflow logic.

**Limitations**

- Highest latency among the tested options.
- Multiple sequential stages add overhead.
- Final performance depends on how well the planner and reviewer guide the workflow.

**Assessment**

This is the best long-term architecture, but not the best immediate implementation choice if the goal is to improve capability quickly with lower delivery risk.

### Why Search Time Is Similar

Search time remains relatively stable because all four approaches use the same search backend. Individual retrieval calls are comparatively cheap. The larger cost comes after retrieval, when the model has to process larger context, make additional decisions, and synthesize the final answer.

### Why Generation Time Differs More

Generation time increases when:

- more chunks are passed into the model,
- more tool rounds are required,
- more LLM calls are made,
- prompts become larger or more complex,
- the workflow includes explicit planning, review, and synthesis stages.

In this analysis, generation time should be interpreted as overall non-search time. It includes not only final answer generation, but also orchestration overhead and intermediate model calls.

### Prompt Quality

Prompt quality has a direct effect on both answer quality and latency.

- Better prompts reduce unnecessary search iterations.
- Better prompts produce more targeted retrieval.
- Better prompts improve the quality of synthesis from large context.
- Poor prompts can increase both latency and hallucination risk.

This means prompt design remains an important optimization lever regardless of which architecture is selected.

### Additional Notes

- Chunk counts are not perfectly comparable across all approaches because retrieval patterns and top-k limits differ.
- Retrieving more chunks is not always better. It can improve completeness, but it can also increase synthesis time and reduce focus.
- More search calls do not automatically mean worse performance if those calls are precise and can run in parallel.

### Recommended Direction

Normal RAG is already in place and should continue to serve as the baseline for simpler queries.

For the next step, the recommended direction is:

1. Keep Normal RAG as the baseline.
2. Implement the Function Calling Agent using Responses API as the short-term enhancement for complex questions.
3. Move toward Orchestrated RAG as the long-term target architecture.

The reason for this path is straightforward:

- It allows complex question support to improve in the short term.
- It introduces less uncertainty than moving directly to a fully orchestrated model.
- It keeps the door open for a stronger long-term architecture once broader extensibility becomes the priority.

### Final Conclusion

Based on the observed results, the best immediate next step is to implement the Function Calling Agent using Responses API.

It offers the best short-term balance between improved capability for complex questions, execution speed, and implementation confidence.

Orchestrated RAG should remain the longer-term target because it provides the strongest architectural foundation for future expansion.
