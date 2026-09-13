# Transcripts


Evidence for each graded task. **These files were regenerated from the local
graded run** using the real SentenceTransformers index, CrewAI, and Autogen:


> Captured transcript generated locally under `MOCK_LLM=true`, with
> `CREW_MODE=crewai` and `EMBEDDING_BACKEND=sentence_transformers`.


Running the generator replaces all of them with real captured output from your
machine, relabelled as captured:


```bash
python -m scripts.build_indexes
python -m scripts.calibrate_threshold
python -m scripts.generate_demonstrations
```


Regenerate one group at a time with `--only`:


```bash
python -m scripts.generate_demonstrations --only cache review
```


| File | Task | `--only` name |
| --- | --- | --- |
| [rag_demonstration.md](rag_demonstration.md) | Task 4 - grounded generation and the calibrated fallback | `rag` |
| [tool_invocation.md](tool_invocation.md) | Task 7 - both tools invoked via `crew.kickoff()` | `tools` |
| [memory_same_session.md](memory_same_session.md) | Task 8 - state carried within one session | `memory` |
| [memory_fresh_session.md](memory_fresh_session.md) | Task 8 - state correctly absent in a fresh session | `memory` |
| [guardrails.md](guardrails.md) | Task 10 - PII masking, injection block, groundedness refusal | `guardrails` |
| [escalation_score.md](escalation_score.md) | Task 6 - designed escalation score and its threshold | `escalation` |
| [autogen_review.md](autogen_review.md) | Task 14 - review stage approving and revising | `review` |
| [least_autonomy.md](least_autonomy.md) | Task 15 - only the Lookup Agent holds the lookup tool | `autonomy` |
| [budget_rejection.md](budget_rejection.md) | Task 15 - oversized request rejected before the crew | `budget` |
| [cache_hit.md](cache_hit.md) | Task 16 - real cache hit with before/after evidence | `cache` |


## Reading the placeholders


Values written as `<measured at runtime>` depend on the embedding model and can
only come from an actual run. Everything else is fixed by deterministic logic -
appointment ids, mask tokens, response types, verdict structure, cache counters
and the budget arithmetic are what a run will actually emit.



