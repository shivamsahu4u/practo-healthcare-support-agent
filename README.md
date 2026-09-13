# Track: Practo (Healthcare)

**Final Capstone: Practo Healthcare Support Agent**

A patient-support agent that answers Practo clinic-policy questions from a
knowledge base written for this brief, looks up a specific appointment's status
from a dataset generated deterministically for this brief, remembers a
conversation, is guarded against misuse, has its answers reviewed by an
independent second agent team before they reach the user, and runs under an
explicit governance policy.

Orchestrated with **CrewAI**, reviewed by **Microsoft Autogen**, retrieved with
**ChromaDB** + **SentenceTransformers**, remembered with **LangChain** session
memory, deployed behind **FastAPI**.

**Everything graded runs under `MOCK_LLM=true` with zero API keys, zero paid
accounts and no outbound network calls.**

---

## Reproducible dataset-design choices (Part 1, Task 1)

Stated here at the top, as the brief requires, so the dataset can be reproduced
exactly. All values live in `dataset.py` as module constants.

| Choice | Value |
| --- | --- |
| **Seed** | `4242` (applied to a private `random.Random`, never the global RNG) |
| **Total records** | `45` |
| **Category weights** (relative integer shares) | `General Medicine: 4`, `Cardiology: 3`, `Dermatology: 3`, `Pediatrics: 3`, `Orthopedics: 2` |
| **Resulting category counts** | `12 / 9 / 9 / 9 / 6` (every required category ≥ 3) |
| **Status weights** (relative integer shares) | `Scheduled: 5`, `Completed: 5`, `Cancelled: 2`, `No-Show: 1`, `Rescheduled: 2` |
| **Resulting status counts** | `15 / 15 / 6 / 3 / 6` (every required status ≥ 1) |
| **Follow-up probability** | `0.20` → exactly `round(0.20 × 45) = 9` records → **20.0%**, inside the required 10–30% band |
| **Consultation-fee range** | `300–2500 INR` overall, in steps of `50`, banded per specialty |
| **Per-specialty fee bands (INR)** | General Medicine `300–700`, Pediatrics `400–900`, Dermatology `600–1400`, Orthopedics `700–1600`, Cardiology `900–2500` |
| **`days_since_created`** | integer, `0–30`, drawn uniformly from the seeded RNG |
| **Record ids** | `APT-1001` … `APT-1045`, sequential in generation order |

**Fee-range reasoning (one sentence):** 300–2500 INR banded per specialty
reflects typical private-clinic OPD consultation pricing in Indian metros, where
specialists such as cardiology and orthopedics are priced well above general
physicians.

**Why the weights sum so cleanly.** Both weight sets total 15, and 45 / 15 = 3,
so the largest-remainder allocation lands on whole numbers with no rounding
drift.

**Allocation is quota-based, not draw-based — and that is deliberate.** The
declared weights are converted into exact integer quotas by the largest-remainder
method, and the seeded RNG then *shuffles* those quotas across the 45 record
slots. The weights decide *how many* of each value exist; the seed decides
*which* record gets which. This is stratified allocation rather than independent
multinomial sampling, and it is why the structural thresholds hold by
construction instead of by luck — the follow-up share is exactly 20.0%, not
approximately 20%. No individual record is ever hand-edited to force a number,
which the brief explicitly forbids. Attributes with no coverage constraint
(`consultation_fee_inr`, `days_since_created`) are drawn directly from the seeded
RNG.

Reproduce and verify:

```bash
python dataset.py
```

This prints the counts per category, the counts per status, the follow-up
percentage, the observed fee and day ranges, and the declared design, then
writes `reports/dataset_report.md`. `validate_appointments()` also runs at import
time, so any structural violation raises before the dataset can be used.

---

## Table of contents

- [Quick start](#quick-start)
- [Architecture](#architecture)
- [Request data flow](#request-data-flow)
- [Repository structure](#repository-structure)
- [Python version and dependency policy](#python-version-and-dependency-policy)
- [Offline mode and embedding backends](#offline-mode-and-embedding-backends)
- [`MOCK_LLM` design](#mock_llm-design)
- [Part 1 — Dataset design and RAG core](#part-1--dataset-design-and-rag-core)
- [Part 2 — CrewAI orchestration, tools, memory, guardrails](#part-2--crewai-orchestration-tools-memory-guardrails)
- [Part 3 — Evaluation, observability, FastAPI](#part-3--evaluation-observability-fastapi)
- [Part 4 — Resilience and governance](#part-4--resilience-and-governance)
- [Test suite](#test-suite)
- [Acceptance-criteria traceability matrix](#acceptance-criteria-traceability-matrix)
- [Defects found in self-review, and how each was fixed](#defects-found-in-self-review-and-how-each-was-fixed)
- [Limitations](#limitations)
- [Interview preparation](#interview-preparation)
- [Command reference](#command-reference)
- [Disclaimer on execution](#disclaimer-on-execution)

---

## Quick start

```bash
# 1. Virtual environment. Keeping it OUTSIDE the project folder means the
#    project directory can be zipped and moved without carrying it along.
python -m venv ../practo-venv
../practo-venv/Scripts/activate          # Windows
# source ../practo-venv/bin/activate     # macOS / Linux

# 2. Dependencies - install the declared baseline as ONE set.
pip install -r requirements.txt -r requirements-dev.txt

# 3. Configuration.
cp .env.example .env                      # macOS / Linux
# Copy-Item .env.example .env             # Windows PowerShell
```

```bash
# 4. Verify the dataset.
python dataset.py
```

```bash
# 5. Build both vector indexes.
python -m scripts.build_indexes
```

```bash
# 6. MEASURE the similarity threshold. Not optional: grounded generation refuses
#    to run without a calibrated threshold. Also fills this README's
#    AUTO:CALIBRATION section with the measured values.
python -m scripts.calibrate_threshold
```

```bash
# 7. Compare the two chunking strategies. Also fills this README's
#    AUTO:CHUNKING section with both sets of numbers and the recommendation.
python -m scripts.compare_chunking
```

```bash
# 8. Regenerate every transcript from live runs. Also fills this README's
#    AUTO:ESCALATION section with the computed escalation threshold.
python -m scripts.generate_demonstrations
```

```bash
# 9. Score the 15-query evaluation set.
python -m evaluation.run_evaluation
```

```bash
# 10. Walk every acceptance criterion.
python -m scripts.run_acceptance_checks
```

```bash
# 11. Serve the API.
uvicorn app.main:app --reload
```

Then: Swagger UI at <http://localhost:8000/docs>, health at
<http://localhost:8000/health>, governance at
<http://localhost:8000/governance>.

**Order matters for steps 5 and 6.** Retrieval needs an index, and grounded
generation needs a measured threshold. Skipping step 6 makes `POST /ask` return
`503 calibration_required` with the exact command to run — by design, because the
brief forbids an untested preset threshold.

---

## Architecture

Five layers, each depending only on the ones above it. No layer reaches
backwards, so the dependency graph is acyclic and each layer is testable alone.

```
                       ┌───────────────────────────────────────────┐
  Deployment           │ app/main.py                               │
                       │  GET /health   POST /ask   POST /add-doc   │
                       │  GET /governance   WS /ws/chat/{session}   │
                       └────────────────────┬──────────────────────┘
                                            │
                       ┌────────────────────▼──────────────────────┐
  Service              │ app/services/support_service.py           │
                       │  guardrails → budget → memory → crew →    │
                       │  retrieval gate → review → final check →  │
                       │  Pydantic validation → JSONL log          │
                       └────────────────────┬──────────────────────┘
                                            │
        ┌───────────────────────────────────┼───────────────────────────────┐
        │                                   │                               │
┌───────▼─────────┐            ┌────────────▼────────────┐    ┌────────────▼────────┐
│ Orchestration   │            │ Governance / guardrails │    │ Review stage        │
│ agents/crew.py  │            │ agents/governance.py    │    │ agents/review_team  │
│ agents/mock_llm │            │ agents/guardrails.py    │    │  RoundRobinGroupChat│
│ agents/routing  │            │ agents/memory.py        │    │  ReviewVerdict      │
│ agents/tools.py │            │                         │    │                     │
└───────┬─────────┘            └─────────────────────────┘    └─────────────────────┘
        │
┌───────▼──────────────────────────────────────────────────────────────────────────┐
│ RAG core                                                                          │
│ rag/chunking.py → rag/embeddings.py → rag/indexer.py → rag/retriever.py           │
│                → rag/grounded_generation.py (+ rag/cache.py)                      │
│ rag/calibration.py (threshold)   rag/evaluation.py (precision / recall)           │
└───────┬──────────────────────────────────────────────────────────────────────────┘
        │
┌───────▼──────────────────────────────────────────────────────────────────────────┐
│ Data                                                                              │
│ data/knowledge_base/*.md (12 policy documents)   dataset.py (45 appointments)     │
│ ChromaDB: practo_kb_fixed + practo_kb_sentence (one per chunking strategy)         │
└──────────────────────────────────────────────────────────────────────────────────┘
```

### The three crew agents

| Agent key | Role | Tools | Why |
| --- | --- | --- | --- |
| `retrieval_agent` | Practo Policy Retrieval Specialist | `policy_knowledge_lookup` | Never answers a policy question from memory. |
| `lookup_agent` | Practo Appointment Records Officer | `appointment_status_lookup` | The **only** agent authorised to read appointment records. |
| `response_composer` | Practo Patient Response Composer | _none_ | Holds no tools, so it cannot fetch — and therefore cannot invent. |

### Key design decisions and why

| Decision | Rationale |
| --- | --- |
| Routing is rule-based, not model-decided | The same question always takes the same path, so a transcript is reproducible and the crew cannot talk itself into a tool it should not use. |
| Answers quote retrieved sentences **verbatim** | Grounded by construction. The groundedness check and the reviewer then test real provenance rather than a paraphrase artefact. |
| The threshold is measured, never preset | `0.5`/`0.6`/`0.7` are tutorial defaults that do not reliably separate short policy-sentence embeddings from unrelated queries. |
| The Composer holds no tools | Separating "decide what to fetch" from "decide how to phrase it" is what keeps a wrong fetch from becoming a confidently wrong answer. |
| Composition logic lives in one module | `agents/composition.py` is called by both the CrewAI mock LLM and the direct pipeline, so the two modes cannot drift apart. |
| Tool dispatch keys off `args_schema` | Immune to the "`rag_lookup` contains `lookup`" misclassification trap. |
| Tool-run detection keys off a ledger | Immune to the `"Observation:"` system-template trap. |
| `crew.memory = False` | CrewAI's own memory would spin up its own embedder and vector store, breaking the offline guarantee. Conversation memory is LangChain's job here. |
| `Agent(cache=False)`, `Crew(cache=False)` | CrewAI's internal tool cache would suppress a genuine second tool call and make the invocation ledger under-report. |

---

## Request data flow

`POST /ask` with `{"query": "...", "session_id": "..."}`:

1. **Trace id** — `uuid4().hex`, attached to the response and to the single log line.
2. **PII masking** (`agents/guardrails.py`) — fixed-format contact numbers →
   `[CONTACT_MASKED]`. Everything downstream sees only the masked text.
3. **Prompt-injection detection** — run on the *masked* text, so a payload hidden
   inside a phone number cannot slip past. A hit returns
   `response_type="blocked"` immediately: no crew, no cache, no memory write.
4. **Runtime budget cap** (`agents/governance.py`) — characters and estimated
   tokens, checked *before* the crew is constructed. A breach raises
   `BudgetExceededError` → HTTP 413 with `crew_invoked: false`.
5. **Session memory** (`agents/memory.py`) — the masked query goes to
   `RunnableWithMessageHistory`, which supplies prior turns and stores this one.
6. **Record-id resolution** — from the query text, else from session history.
7. **Deterministic routing** (`agents/routing.py`) — `policy`, `appointment`, or
   `combined`.
8. **CrewAI crew** (`agents/crew.py`) — `crew.kickoff()`. Only the task legs the
   route needs are built. Tools write results and ledger entries onto
   `CrewRunContext`.
9. **Retrieval gate** — if retrieval never cleared the calibrated threshold there
   is no admissible context: refuse, and skip the review stage.
10. **Autogen review** (`agents/review_team.py`) — the two-agent round-robin team
    approves or revises, returning a Pydantic `ReviewVerdict`.
11. **Final groundedness check** — every sentence of the reviewed answer is
    re-checked against the support text. Defence in depth.
12. **Structured validation** — the result is validated as a `SupportResponse`
    before it can leave the process.
13. **One JSON-Lines log entry** (`app/logging_config.py`) with the trace id,
    timing, masked query and outcome.

`WS /ws/chat/{session_id}` runs steps 1–13 identically per frame.

---

## Repository structure

```
practo-healthcare-support-agent/
├── README.md                     this file
├── .env.example                  configuration template, no secrets
├── .gitignore                    excludes .venv, .env, *.db, __pycache__, chroma/
├── .python-version               3.11
├── requirements.txt              declared compatibility baseline
├── requirements-dev.txt          + pytest, pytest-asyncio, httpx
├── requirements-no-torch.txt     documented torch-free alternative
├── pyproject.toml                pytest + ruff configuration
├── dataset.py                    TASK 1  seeded appointment dataset + validation
├── app/
│   ├── config.py                 settings, vocabularies, telemetry/offline env
│   ├── models.py                 TASK 9  every Pydantic contract
│   ├── main.py                   TASK 11 FastAPI app: 4 HTTP + 1 WebSocket
│   ├── logging_config.py         TASK 12 JSON-Lines logging with masking
│   ├── dependencies.py           lazy SupportService singleton
│   └── services/
│       ├── support_service.py    the pipeline
│       └── session_store.py      conversation registry
├── data/
│   ├── knowledge_base/           TASK 2  12 policy documents
│   ├── samples/                  example request payloads for every endpoint
│   └── generated/                git-ignored runtime output (chroma, logs, calibration)
├── rag/
│   ├── chunking.py               TASK 3  two chunking strategies + KB loading
│   ├── embeddings.py             TASK 3  SentenceTransformers + 2 opt-in backends
│   ├── indexer.py                TASK 3  two ChromaDB collections + kb_version
│   ├── retriever.py              TASK 4  top-k retrieval, cosine similarity
│   ├── grounded_generation.py    TASK 4  context-only answers + calibrated threshold
│   ├── calibration.py            TASK 4  empirical threshold measurement
│   ├── evaluation.py             TASK 5  document-level precision / recall
│   ├── cache.py                  TASK 16 bounded LRU response cache
│   └── textutils.py              shared overlap arithmetic
├── agents/
│   ├── tools.py                  TASK 6  check_appointment_status + escalation
│   ├── mock_llm.py               MOCK_LLM as a crewai BaseLLM subclass
│   ├── crew.py                   TASK 7  the crew and crew.kickoff()
│   ├── composition.py            deterministic answer composition
│   ├── context.py                per-turn context + tool-invocation ledger
│   ├── routing.py                deterministic routing
│   ├── memory.py                 TASK 8  LangChain session memory
│   ├── guardrails.py             TASK 10 masking, injection, groundedness
│   ├── review_team.py            TASK 14 Autogen RoundRobinGroupChat
│   └── governance.py             TASK 15 four layers, registry, budget, risk
├── evaluation/
│   ├── test_set.json             TASK 13 exactly 15 queries
│   ├── mock_judge.py             TASK 13 deterministic rule-based judge
│   ├── run_evaluation.py         TASK 13 harness + report
│   └── expected_results.md       rubric + result template
├── scripts/
│   ├── build_indexes.py
│   ├── calibrate_threshold.py    also fills README's AUTO:CALIBRATION section
│   ├── compare_chunking.py       also fills README's AUTO:CHUNKING section
│   ├── generate_demonstrations.py also fills README's AUTO:ESCALATION section
│   ├── run_acceptance_checks.py
│   └── readme.py                 writes measured numbers back into README.md
├── transcripts/                  evidence for every task (see transcripts/README.md)
├── reports/                      measured reports written by the scripts
└── tests/                        12 test modules, all offline
```

---

### Mapping to the generic scaffolding guide

The course scaffolding walkthrough uses a different domain as its worked example
(an invoice-processing desk: `db.py`, `seed.sql`, `policy.md`, `/ingest`,
`/tickets/{id}`, a `tickets`/`vendors`/`purchase_orders`/`events` schema). Those
are that example's requirements, not this brief's. Where the two differ, the
capstone brief wins. Here is the mapping, so nothing looks accidentally missing:

| Scaffolding item | Here | Why |
| --- | --- | --- |
| `git init`, `.gitignore` with `.venv .env *.db __pycache__ chroma/` | ✅ all five patterns present, plus `data/generated/` | as specified |
| virtual environment | `../practo-venv` rather than `./.venv` | the project folder gets zipped and moved between machines; `.gitignore` covers both names |
| `requirements.txt` with fastapi / uvicorn / pydantic / python-dotenv / chroma / langchain | ✅ all six, plus crewai, autogen, sentence-transformers | `chroma` is not the real package name for this - the correct dependency is `chromadb` |
| `.env.example` with no secrets | ✅ | as specified |
| `app/config.py` - centralised business rules | ✅ `app/config.py` | holds the category / status vocabularies, the 12 KB topics, thresholds and every setting |
| `app/models.py` - Pydantic schemas | ✅ `app/models.py` | 14 models; `InvoicePacket` etc. are the invoice example's, not this brief's |
| `app/main.py` - FastAPI entry point + `/health` | ✅ | uses `lifespan` rather than the deprecated `@app.on_event("startup")` |
| `app/memory.py` | `agents/memory.py` | grouped with the other agent concerns |
| `app/tools.py` | `agents/tools.py` | same |
| `app/pipeline.py` - agent orchestration | `agents/crew.py` + `app/services/support_service.py` | orchestration split from the request pipeline |
| `data/policy.md` - ground-truth rules | `data/knowledge_base/` (12 documents) | this brief requires ≥12 retrievable policy documents, not one file |
| `data/samples/` - example inputs | ✅ `data/samples/` | sample payloads for every endpoint, including the guardrail and budget cases |
| `eval/test_cases.json` | `evaluation/test_set.json` | plus the judge and the harness alongside it |
| `app/db.py`, `seed.sql`, the four SQL tables, `/ingest`, `/tickets/{id}` | **deliberately absent** | invoice-domain requirements. This brief specifies no database: state is ChromaDB plus in-process session memory. See the audit-trail note below |
| Swagger UI `/docs`, ReDoc `/redoc` | ✅ | FastAPI provides both automatically |

### Where the audit trail lives, given there is no database

The scaffolding guide gets auditability from a SQL `events` table because its
example writes financial records. This system writes nothing: it reads policy
text and one read-only appointment record. Auditability is therefore two
append-only artefacts rather than a table:

1. **`data/generated/requests.jsonl`** - one JSON-Lines entry per request, with a
   `trace_id`, `duration_ms`, `endpoint`, `transport`, hashed `session_id`, the
   masked query, the response type, which guardrails fired, which tools ran,
   which sources were cited, the review verdict and the status code. That is the
   equivalent of `log_event(ticket, actor, action, details)`, one line per turn.
2. **The tool-invocation ledger** on `CrewRunContext` - every tool call with its
   agent, its arguments and its outcome, returned to the caller as
   `tools_invoked` and printed in full in `transcripts/tool_invocation.md`. This
   is what makes "which agent read which record" answerable after the fact.

Adding SQLite would have meant introducing a dependency and a schema for data
this brief never asks to persist. If the appointment dataset were real rather
than generated, that calculus would change.

## Python version and dependency policy

- **Target:** CPython **3.11** (`.python-version`). Also expected to work on 3.12
  and 3.13. **3.14 will not work** — CrewAI declares
  `requires-python >=3.10,<3.14`, so `pip` refuses outright.
- No 3.12-or-later-only syntax is used anywhere. In particular, f-strings in this
  codebase contain no backslashes, which 3.11 does not permit.

**Pinned dependency policy.** `requirements.txt` is a *declared compatibility
baseline*: one internally consistent set, chosen because CrewAI, ChromaDB,
LangChain and Autogen all constrain `pydantic` and each other.

- Install them **together**, in one `pip install -r requirements.txt`.
- **Do not** upgrade individual packages afterwards, and do not run
  `pip install -U`. Piecemeal upgrades break the set.
- The pins were selected from existing knowledge and **were not verified against
  a package index while this repository was written**. If your resolver objects
  to a specific pin, relax that one line rather than upgrading the whole set, and
  note the change.

| Package | Pin | Role |
| --- | --- | --- |
| `fastapi` / `starlette` / `uvicorn[standard]` | 0.115.6 / 0.41.3 / 0.34.0 | HTTP + WebSocket + ASGI server |
| `pydantic` | 2.10.4 | every structured contract |
| `python-dotenv` | 1.0.1 | `.env` loading |
| `chromadb` | 0.5.23 | two persistent vector collections |
| `sentence-transformers` | 3.3.1 | free local embeddings |
| `crewai` | 0.100.1 | agent orchestration, `BaseLLM` extension point |
| `langchain` / `langchain-core` | 0.3.14 / 0.3.29 | session memory |
| `autogen-core` / `autogen-agentchat` | 0.5.7 / 0.5.7 | review stage |

Note: `chroma` is **not** a real package name for this. The correct dependency
is `chromadb`.

---

## Offline mode and embedding backends

Importing `app/config.py` exports these before any third-party library reads
them, which is why nothing in this project makes an outbound call by accident:

| Variable | Value | Effect |
| --- | --- | --- |
| `CREWAI_DISABLE_TELEMETRY` | `true` | **CrewAI's telemetry otherwise attempts an outbound call when `crew.kickoff()` runs.** Confirmed set. |
| `OTEL_SDK_DISABLED` | `true` | disables the OpenTelemetry SDK CrewAI's telemetry uses. Confirmed set. |
| `CREWAI_DISABLE_VERSION_CHECK` | `true` | suppresses CrewAI's start-up version lookup, another outbound call. |
| `ANONYMIZED_TELEMETRY` | `False` | ChromaDB's own product telemetry. |
| `HF_HUB_OFFLINE`, `TRANSFORMERS_OFFLINE`, `HF_HUB_DISABLE_TELEMETRY` | `1` | exported only when `OFFLINE_MODE=true` and `ALLOW_MODEL_DOWNLOAD=false`. A missing model then **raises** instead of downloading. |

All are set with `setdefault`, so an explicit shell export always wins.

### The SentenceTransformers prerequisite

Strict offline mode needs the model artefacts to be present already. There is no
silent fallback: when the model cannot load,
`EmbeddingModelUnavailableError` is raised with the exact remedy.

Two ways to satisfy it:

1. **Pre-download once**, on a machine that allows it, then point
   `EMBEDDING_MODEL_PATH` at that folder. Strict offline mode then works.
2. **Allow a single online run** with `ALLOW_MODEL_DOWNLOAD=true`, which lets
   SentenceTransformers fetch `sentence-transformers/all-MiniLM-L6-v2` into its
   cache. Set it back to `false` afterwards.

### Three backends, one default

| `EMBEDDING_BACKEND` | Status | Notes |
| --- | --- | --- |
| `sentence_transformers` | **default, graded path** | free, local, no API key |
| `chroma_onnx` | documented opt-in | ChromaDB's own bundled ONNX build of the *same* `all-MiniLM-L6-v2`. Identical weights, `onnxruntime` instead of `torch`. Use `requirements-no-torch.txt`. For machines where torch cannot be installed at all, or where the huggingface host is unreachable. |
| `deterministic_hash` | test suite only | hashing pseudo-embeddings, no model artefacts. **Retrieval quality is not meaningful and no similarity number produced in this mode may be reported as a graded result.** Logs a warning on selection. |

None of the three is ever selected automatically — the backend comes from
configuration, and an unavailable backend raises rather than degrading quietly.
If you run the graded numbers on `chroma_onnx`, say so alongside them, because
the measured threshold will differ from the default backend's.

---

## `MOCK_LLM` design

`MOCK_LLM=true` is the graded mode. There are three separate "models" in this
system and all three are deterministic stand-ins under `MOCK_LLM`:

| Where | Implementation | Extends |
| --- | --- | --- |
| Grounded generation | `rag/grounded_generation.py:compose_grounded_answer` | — |
| The CrewAI crew | `agents/mock_llm.py:MockCrewLLM` | `crewai.llms.base_llm.BaseLLM` |
| The Autogen review team | `agents/review_team.py:MockChatCompletionClient` | `autogen_core.models.ChatCompletionClient` |
| The evaluation judge | `evaluation/mock_judge.py` | — |

### Why `BaseLLM` rather than intercepting calls

CrewAI's documented extension point for a non-`litellm` model is to subclass
`crewai.llms.base_llm.BaseLLM` and implement `call()`. Intercepting calls from
outside means guessing at CrewAI's internals; subclassing means CrewAI drives the
mock the same way it drives a real model, so the crew being exercised is the real
crew. One instance is created per agent, so each agent's "model" knows which role
it is playing.

`supports_function_calling()` returns `False` **on purpose**. That drives
CrewAI's prompt-based ReAct path — which is where both documented pitfalls live,
so exercising it is the point rather than something to route around.

### Pitfall 1 — the `"Observation:"` template trap

CrewAI's own built-in ReAct system prompt literally contains the example line
`Observation: the result of the action`. A mock that searched the conversation
for `"Observation:"` to find a tool result would match that **template text** on
its very first call, before any tool had run, and silently return placeholder
text as the final answer. No crash — just wrong output.

**Mitigation, two layers:**

1. The authoritative "has the tool run?" signal is the ledger on
   `CrewRunContext`, written by the tool wrappers when they actually execute.
   Prompt text is never the source of truth.
2. `extract_generated_observation()` exists as a hardened cross-check: it skips
   `system` messages entirely (that is where the template lives) **and** rejects
   any observation equal to `CREWAI_TEMPLATE_OBSERVATION_PLACEHOLDER`. It is used
   for diagnostics, never as the sole trigger.

Tested by `tests/test_crew.py::TestObservationTemplateGuard`, which feeds the
guard the actual template text and asserts it returns `None`.

### Pitfall 2 — dispatching on the tool's name

Deciding what to send a tool by testing whether `"lookup"` appears in its name
misclassifies a tool called `rag_lookup`. **Dispatch here goes through
`agents/tools.py:classify_tool()`, which keys off the tool's own declared
`args_schema` field names** — `{"query"}` for the policy tool versus
`{"record_id"}` for the appointment tool — via the explicit `ARG_SCHEMA_ROUTING`
table. The tool's *name* is only ever used verbatim, as the string CrewAI needs
to match the `Action:` line. An unrecognised schema raises `ToolDispatchError`
rather than being guessed at.

Tested by
`tests/test_tools.py::TestSchemaBasedDispatch::test_a_tool_named_rag_lookup_is_still_classified_correctly`,
which builds a tool literally named `rag_lookup` and asserts it classifies as the
*policy* tool while confirming the trap substring really is present.

### The `CREW_MODE=direct` escape hatch

`CREW_MODE=direct` runs the same tools and the same composition functions in a
straight line, without CrewAI. It exists as a diagnostic for environments where
CrewAI cannot be installed, and because both modes call
`agents.tools.invoke_*` and `agents.composition.compose_draft` they produce
identical drafts. **It is not the graded path** — `CREW_MODE=crewai` is the
default and is what `crew.kickoff()` evidence comes from. The direct mode logs a
warning and still runs the same governance tool-permission gate.

### Optional real LLM

There is no wired real-LLM path, and none is required. `.env.example` reserves
`REAL_LLM_*` variables, blank and unread by any graded code path. Nothing in the
acceptance criteria depends on a real model.

---

## Part 1 — Dataset design and RAG core

### Task 1 — Dataset

See [the design table at the top](#reproducible-dataset-design-choices-part-1-task-1).
`validate_appointments()` collects **every** violation rather than the first, and
checks: ≥ 40 records, all required fields present, category and status in
vocabulary, category floor of 3, every status ≥ 1, `days_since_created` an `int`
in 0–30 (a `bool` is rejected), `follow_up_required` a genuine `bool`, fees inside
both the global range and the per-specialty band and a multiple of 50, no
duplicate ids, and the follow-up share inside 10–30%.

### Task 2 — Knowledge base

Twelve documents in `data/knowledge_base/`, one per required topic, each 5
substantive sentences, all in my own words and all fabricated.

| Document | Required topic |
| --- | --- |
| `appointment_booking.md` | appointment-booking policy |
| `cancellation_rescheduling.md` | cancellation / rescheduling window |
| `consultation_fees.md` | consultation-fee structure by specialty |
| `insurance_claims.md` | insurance-claim process |
| `prescription_refills.md` | prescription-refill policy |
| `lab_turnaround.md` | lab-test turnaround times |
| `telemedicine.md` | telemedicine eligibility |
| `emergency_visits.md` | emergency-visit protocol |
| `patient_privacy.md` | patient-data privacy policy |
| `follow_up_discount.md` | follow-up-visit discount policy |
| `second_opinion.md` | second-opinion process |
| `home_visits.md` | home-visit eligibility |

`load_knowledge_base()` raises if any required topic is missing, so the corpus
cannot silently regress. The emergency document routes any suspected emergency
to the national emergency number rather than answering it, and states that
support agents and automated assistants do not triage or advise.

### Task 3 — Two chunking strategies, two collections

| Strategy | Method | Collection |
| --- | --- | --- |
| `fixed_size_overlap` | sliding character window, `FIXED_CHUNK_SIZE=480` with `FIXED_CHUNK_OVERLAP=96`, end snapped back to a word boundary so a chunk never ends mid-word | `practo_kb_fixed` |
| `sentence_based` | whole sentences grouped `SENTENCES_PER_CHUNK=2` at a time, no overlap | `practo_kb_sentence` |

Every chunk carries `chunk_id`, `document_id`, `source_filename`, `topic_title`,
`strategy` and `chunk_index`. Chunk ids are namespaced by strategy
(`fixed_size_overlap::cancellation_rescheduling::000`), so the two sets can never
collide.

Both collections are created with `metadata={"hnsw:space": "cosine"}` and
**`embedding_function=None`**. That last part is deliberate: every write passes
explicit vectors, so ChromaDB never constructs its default embedding function —
which would try to fetch its own model artefact and break the offline guarantee.
Writes go through `collection.upsert()`, so a rebuild is idempotent.

`data/generated/index_state.json` tracks `kb_version`, which the response cache
keys on.

### Task 4 — Grounded generation and the *measured* threshold

Answers are built from retrieved context only:
`compose_grounded_answer()` picks the retrieved sentences that best cover the
query and emits them **verbatim** behind a fixed frame naming the source topic.
Nothing is paraphrased and nothing is synthesised.

**Threshold calibration methodology.** `scripts/calibrate_threshold.py` measures
top-1 cosine similarity for **6 in-scope** probes and **3 out-of-scope** probes
against the real index, then sets

```
threshold = (min(in_scope_similarity) + max(out_of_scope_similarity)) / 2
```

— the midpoint of the gap actually observed. If the clusters overlap
(`min(in-scope) <= max(out-of-scope)`), it **fails loudly** and recommends no
threshold, because a silently-overlapping calibration is exactly the failure a
preset hides.

| Probe set | Count | Where |
| --- | --- | --- |
| in-scope | 6 | `rag/calibration.py:IN_SCOPE_CALIBRATION_QUERIES` |
| out-of-scope | 3 | `rag/calibration.py:OUT_OF_SCOPE_CALIBRATION_QUERIES` |

**Why not 0.5 / 0.6 / 0.7.** Those are tutorial defaults. Short policy sentences
embedded with a MiniLM-class model do not place in-scope and unrelated queries on
either side of a round number reliably. The measured clusters are where the real
boundary is for *this* knowledge base and *this* embedder.

#### Measured values and chosen threshold

The brief requires these numbers to appear here. This block was populated by
`python -m scripts.calibrate_threshold` from the local SentenceTransformers
model and persisted index.

<!-- AUTO:CALIBRATION -->
Measured on 2026-09-13T16:49:18+00:00 against collection `practo_kb_sentence` with embedder `sentence_transformers:sentence-transformers/all-MiniLM-L6-v2`.

| Probe | In scope? | Top-1 cosine similarity | Best-matching document |
| --- | --- | --- | --- |
| How long before my appointment can I cancel without paying a fee? | yes | **0.8325** | `cancellation_rescheduling` |
| What is the consultation fee for a cardiology visit? | yes | **0.6689** | `home_visits` |
| How many days does a culture and sensitivity lab test take? | yes | **0.5345** | `lab_turnaround` |
| Is a video consultation allowed for a young child? | yes | **0.5653** | `telemedicine` |
| What discount applies to a follow-up visit within two weeks? | yes | **0.5918** | `follow_up_discount` |
| What should someone do about sudden chest pain and breathlessness right now? | yes | **0.3065** | `emergency_visits` |
| What is the current share price of a large technology company? | no | **0.1874** | `consultation_fees` |
| Who won the football league final last season? | no | **0.1221** | `cancellation_rescheduling` |
| Give me a recipe for chocolate cake with buttercream icing. | no | **0.0509** | `prescription_refills` |

- `min(in-scope)` = **0.3065**
- `max(out-of-scope)` = **0.1874**
- observed gap between the two clusters = **0.1191**
- **chosen threshold = (0.3065 + 0.1874) / 2 = `0.2469`**

The two clusters separate cleanly, so the midpoint `0.2469` sits in the observed gap with no in-scope query below it and no out-of-scope query above it. Pin it with `SIMILARITY_THRESHOLD=0.2469` in `.env`.
<!-- /AUTO:CALIBRATION -->

The same numbers are also written to `reports/calibration_report.md` (human) and
`data/generated/calibration.json` (machine-readable, read by
`rag/grounded_generation.py`). Pin the measured value as
`SIMILARITY_THRESHOLD=<value>` in `.env` to reproduce a graded run on a fresh
checkout.

`resolve_threshold()` raises `CalibrationRequiredError` when neither the pin nor
the calibration file is available. It never guesses.

**Demonstration:** 5 in-scope queries plus 1 deliberately out-of-scope query in
[`transcripts/rag_demonstration.md`](transcripts/rag_demonstration.md). The
out-of-scope query returns exactly
`"I don't know based on the available Practo policy knowledge base."`

### Task 5 — Precision / recall for both strategies

**Scoring is at the document level.** Retrieved chunks are mapped back to their
parent `document_id` and **deduplicated before anything is counted**. That
matters: fixed-size chunking with overlap frequently returns three chunks of the
*same* document, which chunk-level scoring would count as three hits and flatter
that strategy.

```
precision@k = |relevant ∩ retrieved_docs| / |retrieved_docs|
recall@k    = |relevant ∩ retrieved_docs| / |relevant|
```

The same 5 in-scope queries from Task 4 are used, as the brief requires. Ground
truth is in `rag/evaluation.py:DEMO_QUERIES`; `fees_and_home_visit` deliberately
spans **two** documents so recall can land strictly between 0 and 1.

`python -m scripts.compare_chunking` prints and writes, per query and per
collection: the relevant document set, the retrieved chunk ids, the deduplicated
retrieved document set, the true-positive intersection, and the precision and
recall **with the division shown**. It then applies a decision rule fixed *before*
the numbers are read — highest mean F1, then highest mean recall, then fewest
parent documents per query — and writes the recommendation with the numbers cited
into `reports/chunking_comparison.md`.

#### Recommendation, with the numbers

<!-- AUTO:CHUNKING -->
| Strategy | Collection | Mean precision@k | Mean recall@k | Mean F1 | Docs/query |
| --- | --- | --- | --- | --- | --- |
| `fixed_size_overlap` | `practo_kb_fixed` | **0.6000** | **1.0000** | 0.7500 | 2.0 |
| `sentence_based` | `practo_kb_sentence` | **0.8000** | **1.0000** | 0.8889 | 1.6 |

**Recommendation:** Deploy the `sentence_based` strategy (collection `practo_kb_sentence`): it scored mean precision 0.8000, mean recall 1.0000 and mean F1 0.8889 across the five in-scope queries, retrieving 1.6 parent document(s) per query on average. The `fixed_size_overlap` strategy (collection `practo_kb_fixed`) scored mean precision 0.6000, mean recall 1.0000 and mean F1 0.7500 over the same queries, at 2.0 parent document(s) per query. The decision rule was fixed before the numbers were read - highest mean F1, then highest mean recall, then fewest parent documents per query - so the recommendation follows from the measurements rather than from preference.

Deployed collection: `practo_kb_sentence`. Per-query arithmetic for every query and both collections is in `reports/chunking_comparison.md`.
<!-- /AUTO:CHUNKING -->

Set `RECOMMENDED_COLLECTION_NAME` in `.env` to whatever the script recommends;
the CrewAI Retrieval Agent reads that setting.

---

## Part 2 — CrewAI orchestration, tools, memory, guardrails

### Task 6 — `check_appointment_status` and the designed escalation score

```
aging_normalised = min(days_since_created, 30) / 30

escalation_score = clamp(0.45 × (1 if follow_up_required else 0)
                       + 0.55 × aging_normalised, 0.0, 1.0)
```

| Component | Weight | Normalisation |
| --- | --- | --- |
| `follow_up_required` | **0.45** | boolean → {0, 1} |
| recency / aging | **0.55** | `min(days, 30) / 30`, capped at 30 days |

**Direction:** older unresolved records escalate harder. A record created today
contributes nothing from the aging term; one at the 30-day cap contributes 0.55.

**Threshold:** the **nearest-rank 80th percentile of the escalation-score
distribution over the generated dataset**, computed at run time by
`escalation_threshold()`. It is not a hard-coded constant, so it stays honest if
the dataset design changes.

<!-- AUTO:ESCALATION -->
**Escalation threshold = `0.5133`** - the nearest-rank 80th percentile of the escalation-score distribution across all 45 generated appointments, computed at run time by `escalation_threshold()`.

| Statistic | Value |
| --- | --- |
| records | 45 |
| min score | 0.0 |
| median score | 0.385 |
| 90th percentile score | 0.7433 |
| max score | 0.945 |
| **threshold (80th percentile)** | **0.5133** |
| records at or above the threshold | 12 (26.67%) |
| records with `follow_up_required=True` | 9 |

**Proof it is not a boolean OR:**

- escalated on age alone, with **no** follow-up flag: `['APT-1006', 'APT-1026', 'APT-1030', 'APT-1039']`
- carrying a follow-up flag but **not** escalated because still fresh: `['APT-1044']`

Both outcomes are impossible under `follow_up_required OR is_old`. The first list is non-empty by construction, because the threshold is the 36th-smallest of 45 scores and at most one score separates it from the largest no-follow-up score.
<!-- /AUTO:ESCALATION -->

**Dataset-based justification.** The weights are chosen so the two bands overlap:
a record with no follow-up flag reaches at most 0.55, and a follow-up record
starts at 0.45, so the overlap is `[0.45, 0.55]`. With exactly 9 of 45 records
flagged (20.0%), the 36th-smallest score — the nearest-rank 80th percentile of 45
values — lands inside that overlap. Two consequences follow that a bare
`follow_up_required OR is_old` cannot express:

- a **fresh** follow-up record is **not** escalated (0.45 + 0.55 × 1/30 ≈ 0.47);
- a **very old** record with **no** follow-up flag **is** escalated (0.55 at the cap).

`escalation_distribution()` reports both lists explicitly
(`escalated_without_follow_up_flag`, `follow_up_flag_but_not_escalated`), and
`tests/test_tools.py` asserts the score has not collapsed into a boolean.

**Error asymmetry — the healthcare track states that a false negative is worse
than a false positive, and the design is biased accordingly.** A missed
follow-up is a patient who needed a call back and did not get one; an
unnecessary escalation costs a support lead a few minutes. Three consequences:

- The **aging term carries the larger weight** (0.55 vs 0.45), so a record with
  no follow-up flag still escalates on age alone near the 30-day cap. A boolean
  `follow_up_required` test would miss every one of those — pure false
  negatives, and `escalated_without_follow_up_flag` enumerates them.
- The threshold is a **percentile of the live distribution**, not a fixed
  cut-off, so if the data shifts toward older records the escalated set grows
  with it instead of silently under-reporting.
- Escalation is **advisory**: it routes a record to a human. Nothing is
  auto-closed or auto-declined on this score, which is what makes erring toward
  escalation the cheap direction.

**The one false negative this design accepts, stated plainly.** A *fresh*
follow-up record — flagged, but created today — scores ≈0.47 and does not clear
the threshold, so it is not escalated. That is precisely the error class this
domain says to avoid. It is accepted because a same-day follow-up is not yet
actionable: nothing has been missed while the appointment is still current, and
the record escalates on its own within days as the aging term grows. Those
records are reported as `follow_up_flag_but_not_escalated` so the set is
auditable rather than hidden. A deployment that disagreed would lower the
percentile — `escalation_threshold(percentile)` takes it as an argument, and no
other logic depends on its value.

An unknown record id returns a **safe structured not-found payload** — `found:
false`, `escalation_recommended: false`, and a message telling the patient what
an id looks like — rather than raising, so the agent can answer instead of
failing the turn.

Evidence: [`transcripts/escalation_score.md`](transcripts/escalation_score.md).

### Task 7 — The CrewAI crew

Three agents (table [above](#the-three-crew-agents)), run through
`crew.kickoff()` with `Process.sequential`, `allow_delegation=False`,
`memory=False`, `cache=False`.

Which task legs exist is decided by `agents/routing.py` before the crew is built,
so a policy question never constructs an appointment leg. Task descriptions bake
their values in as literals rather than using kickoff inputs, so **no template
interpolation ever runs over patient text** — a query containing a brace would
otherwise break interpolation.

Both tools are demonstrated firing on different queries:

| Query kind | Route | Tools invoked |
| --- | --- | --- |
| "How long before my appointment can I cancel…" | `policy` | `policy_knowledge_lookup` |
| "What is the current status of appointment APT-1007?" | `appointment` | `appointment_status_lookup` |
| "What is the cancellation window, and the status of APT-1007?" | `combined` | **both** |

Evidence: [`transcripts/tool_invocation.md`](transcripts/tool_invocation.md),
with the full tool-invocation ledger per case.

### Task 8 — Session memory

`InMemoryChatMessageHistory` per `session_id`, wired in with
`RunnableWithMessageHistory` (`input_messages_key="input"`,
`history_messages_key="history"`).

`RunnableWithMessageHistory` emits a `LangChainDeprecationWarning` pointing at
LangGraph's own persistence layer. **That is expected, the class still functions
correctly here, and per the brief it does not need silencing — so it is left
visible.**

What memory actually does here: it carries an appointment id forward, so
`"And what is its current status?"` resolves inside one conversation and
correctly fails to resolve in a fresh one.
`resolve_record_id_from_history()` scans newest-first, so a conversation that
discusses two appointments carries the one currently under discussion.

Memory is **in-process only** and does not survive a restart, which the brief
states is sufficient. Only **masked** text is ever stored: the service masks the
query before invoking the runnable, so a contact number cannot reach the history
object.

The registry is a **bounded LRU** (`DEFAULT_MAX_SESSIONS = 512`) with an eviction
counter. `session_id` is client-supplied and history deliberately outlives its
socket, so an unbounded dict let any caller grow process memory without limit —
the response cache was bounded and this, the larger allocation, was not. Touching
a conversation marks it most-recently-used, so an active session is never evicted
ahead of an idle one; an evicted session simply starts fresh, which degrades a
follow-up into a request for the appointment id rather than leaking or crashing.

Evidence: [`transcripts/memory_same_session.md`](transcripts/memory_same_session.md)
(state present) and
[`transcripts/memory_fresh_session.md`](transcripts/memory_fresh_session.md)
(state correctly absent) — two separate transcripts, as the brief requires.

### Task 9 — Structured output

`app/models.py:SupportResponse` is the contract every crew response must conform
to, and it is aliased as **`RESPONSE_FORMAT`** so the `response_format` contract
the brief names has exactly one identifier in the codebase.
`extra="forbid"`, so an unexpected field is an error rather than a silent
pass-through.

Three enforcement points:

1. `SupportService._finalise()` **constructs** it, so an invalid response cannot
   be built at all;
2. `SupportService.answer()` calls `RESPONSE_FORMAT.model_validate(...)` on the
   payload that round-tripped through session memory — validation in code, as the
   task requires;
3. `POST /ask` declares it as its FastAPI `response_model`, putting the same
   schema into the OpenAPI document at `/docs`.

Fields: `trace_id`, `session_id`, `query` (masked), `answer`, `response_type`
(`Literal["policy","appointment","combined","fallback","blocked"]`), `sources`,
`appointment` (a nested `AppointmentResult` with its own `EscalationComponents`),
`escalation_recommended`, `grounded`, `safety_message`, `collection_name`,
`top_similarity`, `threshold`, `pii_masked`, `guardrails_fired`, `cache_hit`,
`latency_ms`, `mock_llm`, `crew_mode`, `tools_invoked`, `review_approved`,
`review_reason`, `review_revised`.

Every other boundary is typed too: `AskRequest`, `AddDocumentRequest`,
`AddDocumentResponse`, `HealthResponse`, `GovernanceResponse`, `BudgetRejection`,
`ReviewVerdict`, `WsClientMessage`, `WsServerAck`, `WsServerError`.

### Task 10 — Guardrails

**Input side — contact-number masking.** Indian contact numbers are the one PII
field in this scenario with a fixed, matchable format. Masked deterministically
by regex to `[CONTACT_MASKED]` (no digits at all) **before** the agent sees the
text and **before** anything is logged — the same masked string for both, which
is why a raw number cannot reach disk. Formats covered: `+91 98765 43210`,
`+919876543210`, `+91-9876543210`, `00919876543210`, `09876543210`,
`9876543210`, `98765-43210`, `98765 43210`. Not touched: `APT-1007`, `2500 INR`,
`24 hours`, `30 days`.

Findings deliberately carry **no** raw value — only a digit count and a truncated
one-way fingerprint — because findings end up in logs.

**Input side — contact-number validation.** `validate_contact_number()` is a
separate control from masking, and the brief asks for both. Masking answers "is
there a number hidden in this free text, and can I redact it before the model or
the log sees it". Validation answers "is this *field* a well-formed contact
number", which is what a booking or call-back flow must ask before accepting
one — so `98765` (too short), `98765432100` (too long) and `1234567890` (an
Indian mobile cannot start with 1) are rejected up front rather than stored and
failed later. It accepts an optional `+91` / `0091` / `91` / `0` prefix plus ten
national digits beginning 6–9, with any spacing, and returns the normalised ten
digits. Its `as_dict()` — the loggable form — omits the normalised number, for
the same reason the masking findings omit it. Covered by
`tests/test_regressions.py::TestContactNumberValidation` (8 accepted formats, 7
rejected).

**PII scope, stated honestly.**

| Field | Masked? | Why |
| --- | --- | --- |
| contact number | **yes** | fixed format |
| patient name | no | free text, no universal format |
| diagnosis / condition | no | free text, no universal format |
| insurance ID | no | no universal format across insurers |

The three out-of-scope fields cannot be masked deterministically by a keyless,
`MOCK_LLM`-only masker — acknowledged in the brief and repeated here. Every
example of all three anywhere in this repository is fabricated; no real medical
data is used.

**The consequence for the log file, stated explicitly.** It is not enough to say
those fields are unmasked: the masked query is also **written to disk** in
`data/generated/requests.jsonl`, which has no rotation and no retention policy.
So a patient name, a stated condition, an insurance id, an email address, an
Aadhaar number, or a non-Indian phone number typed into `query` is persisted.
The regex also only matches the fixed 5+5 Indian grouping, so
`+91 987 654 3210` (3-3-4) is **not** masked. For a system classified High risk
that retention is the sharper problem, and a real deployment would need field-
level redaction plus a retention window before this log could exist at all.

**Input side — prompt injection.** Nine named deterministic patterns:
`ignore_instructions`, `reveal_system_prompt`, `exfiltrate_secrets`,
`role_override`, `jailbreak_mode`, `policy_override`,
`tool_permission_escalation`, `encoded_payload`, `data_exfiltration`. Detection
runs on the **masked** text, so a payload hidden inside a phone number cannot
slip past. A hit blocks the turn: no crew, no cache, no memory write. **Not
claimed to be complete** — a novel phrasing that avoids every listed pattern
would pass. Tuned against false positives: "Show me the instructions for the lab
test" and "How do I delete my record from Practo?" both pass cleanly.

**Output side — groundedness, in two stages.**

1. **Retrieval gate** — is there admissible context at all? Compares top-1
   similarity against the measured threshold. Failure refuses and skips review.
2. **Sentence check** — does every sentence of the final answer clear
   `GROUNDEDNESS_OVERLAP_MIN` (0.6) content-word overlap against the retrieved
   context plus the structured appointment facts plus the composer's own framing
   sentences? Failure refuses.

This is **lexical overlap, not semantic entailment** — a deterministic stand-in
under `MOCK_LLM`. It catches novel vocabulary appearing from nowhere, which is
what a fabricated policy claim looks like; it would not catch a fabrication
assembled entirely from context vocabulary. The independent review stage exists
partly because of that.

Evidence: [`transcripts/guardrails.md`](transcripts/guardrails.md) — one
deliberate test case per guardrail.

---

## Part 3 — Evaluation, observability, FastAPI

### Task 11 — FastAPI deployment

| Method | Path | Request / response models |
| --- | --- | --- |
| `GET` | `/health` | → `HealthResponse`. Loads no model, makes no network call — safe as a container probe. |
| `POST` | `/ask` | `AskRequest` → `SupportResponse`, or **413** `BudgetRejection`, or **503** on missing calibration / index |
| `POST` | `/add-document` | `AddDocumentRequest` → `AddDocumentResponse`, or **400** on rejection |
| `GET` | `/governance` | → `GovernanceResponse` (the four layers, the registry, the risk level) |
| `WS` | `/ws/chat/{session_id}` | `WsClientMessage` in; `SupportResponse` / `WsServerAck` / `WsServerError` out |

FastAPI **lifespan** is used (not the deprecated `on_event`). Start-up is
deliberately cheap: configure logging, ensure the writable directories exist,
stop.

`POST /add-document` validates `topic_slug` against `^[a-z][a-z0-9_]{2,47}$`, so
the destination path is derived entirely from validated input — no dots, no
separators, **no arbitrary filesystem path**. The document is written under
`data/knowledge_base/`, chunked with **both** strategies, upserted into **both**
collections, and the knowledge-base version is bumped; the response cache is then
explicitly invalidated. An existing slug is rejected rather than overwritten.

**WebSocket behaviour.** Accepts `{"query": "..."}` and `{"reset": true}`.
`WebSocketDisconnect` is caught, so a client vanishing mid-conversation logs a
line and returns cleanly — the server and every other client's socket keep
running. A malformed frame (bad JSON, a JSON array, an unexpected field, an empty
query) gets a structured `WsServerError` back and **the socket stays open**.
Session history is keyed by `session_id` and deliberately outlives the socket, so
a reconnect resumes the conversation.

### Task 12 — JSON-Lines structured logging

Exactly **one** JSON object on **one** line per request, written through
`log_request()` — the only writer.

Fields: `timestamp`, `level`, `logger`, `event`, `trace_id`, `endpoint`,
`transport`, `session_hash`, `query` (masked), `query_pii_masked`,
`response_type`, `grounded`, `cache_status`, `duration_ms`, `status_code`,
`error_category`, `guardrails_fired`, `tools_invoked`, `sources`,
`review_approved`, `mock_llm`, `crew_mode`, and an `extra` object.

**PII protection.** `log_request()` re-applies the masking function to whatever it
is handed, so a caller cannot leak a number by forgetting to mask — the log and
the model get identical treatment. Belt and braces:
`ContactNumberMaskingFilter` is attached to the root logger too, so an ordinary
`LOGGER.info("... %s", text)` from anywhere is masked as well. `session_id` is
logged only as a truncated one-way hash. Never logged: unmasked contact numbers,
API keys, secrets, full internal prompts, raw environment variables, or any
clinical detail.

Output: `data/generated/requests.jsonl` (git-ignored). The request stream does
not propagate to the console, so it stays a clean JSON-Lines file.

**One entry per request means *including* failures.** Every log call used to sit
after the pipeline, so any exception below that point escaped before anything was
written and the request vanished from the audit trail — the failure most likely
to happen was the one guaranteed to leave no trace. `SupportService.answer()` now
writes its entry from an `except` branch and re-raises unchanged, with
`error_category` set to the exception type and `status_code` 503. The API layer
still owns the status code; this only guarantees the line. A `pipeline_failures`
counter is exposed on `service.stats()`.

### Task 13 — Evaluation at scale

**Exactly 15 queries** in `evaluation/test_set.json`: q01–q12 cover one required
knowledge-base topic each, q13 is an appointment-lookup edge case, q14 is
deliberately out of scope, q15 is a prompt-injection edge case — three edge or
out-of-scope cases where the brief requires at least two.

Four properties per query, each in `[0, 1]`: **Accuracy**, **Grounding**,
**Completeness**, **Safety**. The full rubric — every band boundary and every
condition — is in [`evaluation/expected_results.md`](evaluation/expected_results.md)
and in `evaluation/mock_judge.py`.

**The judge prompt.** `evaluation/mock_judge.py:JUDGE_PROMPT` is the actual
LLM-as-judge prompt: it defines the four properties, states the declared
expectations and the observed response, and specifies a JSON output format.
`render_judge_prompt(case, response)` fills it per query, and
`run_evaluation.py` embeds the rendered prompt for every query in
`reports/evaluation_report.md` — so the scoring instruction is auditable rather
than implicit. Under `MOCK_LLM` that prompt is **not sent anywhere**; the
deterministic rubric evaluates it instead. Wiring a real judge model means
sending that exact string and parsing four floats back; nothing else about the
harness changes.

**The judge is deterministic and rule-based, not an independent semantic model.**
Stated plainly because it changes how the numbers should be read: it scores
against the declared expectations plus observable response properties, so the
same response always scores the same, but it cannot recognise a correct
paraphrase and cannot notice a plausible answer that is wrong in a way the
expectations do not cover. Reproducibility is worth more here than a verdict a
grader could not re-derive.

`python -m evaluation.run_evaluation` reports all four scores for all 15 queries
**and** the four averages, plus a per-query reason string for every mark lost,
and writes `reports/evaluation_report.md`. Each query gets a fresh `session_id`
so memory cannot leak between independent cases.

---

## Part 4 — Resilience and governance

### Task 14 — Autogen review stage

A **2-agent `RoundRobinGroupChat`** runs after the CrewAI Composer's draft:

1. `policy_compliance_reviewer` — checks grounding, source support and safety,
   and names every unsupported sentence. Does not rewrite.
2. `final_editor` — approves unchanged or revises by removing exactly the
   sentences the reviewer named. Emits a Pydantic verdict.

```python
editor = AssistantAgent(..., output_content_type=ReviewVerdict)
team = RoundRobinGroupChat(
    [reviewer, editor],
    custom_message_types=[StructuredMessage[ReviewVerdict]],
    max_turns=2,
)
```

Both constructor details the brief flags are load-bearing:

- **`max_turns=2`** is the real parameter name, not `max_iterations`. With two
  participants that is exactly one turn each.
- **`output_content_type` on the agent obliges the team to register
  `StructuredMessage[ReviewVerdict]`** via `custom_message_types`, or the run
  dies with `ValueError: Message type ... is not registered`.

```python
class ReviewVerdict(BaseModel):
    approved: bool
    final_answer: str
  reason: str
```

The three field names are the ones the brief specifies. A read-only
`verdict.reasoning` property remains only as a backward-compatible internal
reader; the serialized schema and `model_dump()` use `reason`.

The team's input carries the masked query, the Composer's draft, the retrieved
context, the source document ids, and the appointment lookup result when there is
one.

`MockChatCompletionClient` is a real `autogen_core.models.ChatCompletionClient`
implementation with `model_info["structured_output"] = True` (required for
`output_content_type` to be accepted). It routes on a role marker embedded in
each agent's system message — `[[ROLE:POLICY_COMPLIANCE_REVIEWER]]` /
`[[ROLE:FINAL_EDITOR]]` — rather than on agent names, so renaming an agent cannot
silently change which branch runs. A request with no marker raises rather than
guessing.

**Both demonstrations** are in
[`transcripts/autogen_review.md`](transcripts/autogen_review.md):

- **approval** — a fully grounded draft, `approved: true`, `final_answer`
  byte-identical to the draft;
- **revision** — the same query with one deliberately ungrounded sentence
  injected, `approved: false`, the injected sentence removed, every legitimate
  sentence preserved verbatim, and a `reason` quoting exactly what was removed.

The injection comes from `CrewRunContext.inject_unsupported_claim`, which
defaults to `False` and is set only by the demonstration script and the test
suite. Neither `POST /ask` nor the WebSocket handler ever sets it.

Structurally, the Final Editor can only *remove* sentences — it has no mechanism
to add a claim, which is why the review stage cannot itself introduce an
ungrounded statement.

### Task 15 — The four-layer governance model

The four layers are the ones the brief names — **Application, Scope, Runtime,
Cache** — and `GOVERNANCE_LAYERS` uses exactly those keys, so `GET /governance`
returns them under those names.

| Layer | Principle | Controls | Enforced in |
| --- | --- | --- | --- |
| **Application** | least privilege | tool-permission registry; `assert_tool_assignment()` gates every `Agent` construction; `appointment_status_lookup` authorised for `lookup_agent` only; Composer holds no tools; input and output guardrails | `agents/governance.py`, `agents/crew.py`, `agents/guardrails.py` |
| **Scope** | risk classified **High**; bounded, deterministic scope | the risk classification and its justification; retrieval-only answer scope with an explicit refusal below the calibrated threshold; rule-based routing so the crew cannot widen its own scope; Pydantic output at every boundary; bounded turns (`max_iter`, `MAX_TOOL_ATTEMPTS`, `max_turns=2`), `allow_delegation=False`; data scope — synthetic dataset, local ChromaDB, no real patient data; no diagnosis or triage; telemetry disabled and `HF_HUB_OFFLINE` exported before any third-party import | `agents/governance.py`, `agents/routing.py`, `app/config.py`, `app/models.py`, `rag/grounded_generation.py` |
| **Runtime** | performance monitoring and cost tracking | **cost:** per-request character and estimated-token caps enforced **before** the crew, structured 413 with `crew_invoked: false`, plus the separate Pydantic payload ceiling. **monitoring:** `duration_ms` on every log entry, `latency_ms` on every response, and per-stage counters (`crew_calls`, `review_calls`, `generations`, `retrieval_queries`, cache hits/misses, `pipeline_failures`) via `SupportService.stats()`; one log entry per request including failures; typed exceptions as structured 503s; bounded session memory | `agents/governance.py`, `app/main.py`, `app/logging_config.py`, `app/services/support_service.py`, `agents/memory.py` |
| **Cache** | in-memory cache with normalised keys | bounded LRU with eviction counted; **normalised** keys (trim, lowercase, collapse whitespace) so wording variants share an entry; SHA-256 over length-prefixed components (query, collection, `top_k`, threshold, `kb_version`, embedder identity); explicit invalidation on `POST /add-document` on top of the `kb_version` bump; injections, errors and appointment lookups deliberately not cached; raw PII never reaches it; `RLock`-guarded | `rag/cache.py`, `rag/grounded_generation.py`, `app/main.py` |

`GET /governance` returns all of it as JSON, so the posture is inspectable at
runtime rather than only documented.

**Least autonomy, demonstrated.** `TOOL_PERMISSIONS` is the single authority on
tool access; no other module decides it.
`agents/crew.py:_authorised_tool_map()` routes every proposed assignment through
`assert_tool_assignment()` **before** an `Agent` is constructed, in both crew
modes. Offering `appointment_status_lookup` to the Retrieval Agent or the
Composer raises `ToolPermissionError`. An agent absent from the registry is also
rejected — it is a denylist by default, so a fourth agent cannot silently inherit
access. `verify_registry_invariants()` additionally asserts that
`appointment_status_lookup` has exactly one holder.

**One paragraph on the guard.** `check_appointment_status` reads a patient's
appointment record. Exactly one agent needs that capability — the Lookup Agent,
whose entire job is to fetch one named record — so exactly one agent has it.
Giving it to the Retrieval Agent would let a pure policy question reach
appointment data with no functional gain, since the Retrieval Agent's task never
requires a record id and any call it made would be redundant or wrong. Giving it
to the Composer would be worse, because the Composer writes the final wording: if
the same component both decides what to fetch and how to phrase it, a wrong
fetch becomes a confidently wrong answer with nothing in between to catch it.
Keeping the Composer tool-less is what makes its output auditable — everything in
the final answer must trace to an upstream agent's recorded tool result, so any
claim without a ledger entry behind it is visibly unsupported.

Evidence: [`transcripts/least_autonomy.md`](transcripts/least_autonomy.md).

**Risk classification: High.**

This system is classified **High** risk. The Low/Medium/High scheme places
medical data in the High band, and that is the band this system belongs in: it
answers questions about clinical appointments, consultation fees, prescription
refills, lab reports, insurance claims and patient-data privacy, and it is
addressed to patients rather than to trained staff. It is not merely a
support-ticket assistant, because a confidently wrong answer about an emergency
route, a telemedicine eligibility rule or a refill policy can change what a
patient does about their health. Two facts reduce the concrete exposure without
changing the classification: every appointment record here is synthetic and
deterministically generated, and the agent has no write path to any clinical
system — it reads policy text and one read-only appointment record. Mitigations
are the guardrails, retrieval-only answer construction, the independent review
stage, the tool-permission registry and the budget cap. **Residual risk remains:**
the groundedness check is lexical rather than semantic, the injection denylist is
not exhaustive, and free-text PII such as a patient name, a condition or an
insurance ID is not masked because it has no fixed format to match. The system
does not diagnose, does not triage and does not replace clinical judgement, and
the emergency policy document routes any suspected emergency to the national
emergency number rather than answering it.

**Runtime budget cap.** `MAX_REQUEST_CHARACTERS=2000` and
`MAX_ESTIMATED_TOKENS=600`, with `estimate_tokens = ceil(len(text) / 4)` — a
deterministic heuristic, not a real tokeniser, and reported as an estimate
everywhere it appears. Enforced immediately after the input guardrails and
**before** the crew is constructed, so a rejection never costs a crew run.

**Two different bounds, deliberately separated.** The budget cap is a *cost*
control; it is not a denial-of-service control, and an earlier version of this
README wrongly implied it was ("a rejection costs one length check"). It does
not: masking and nine injection regexes run before it. The DoS bound is
therefore enforced one layer earlier, by Pydantic:

| Bound | Where | Value | Purpose |
| --- | --- | --- | --- |
| `MAX_INBOUND_TEXT_CHARACTERS` | `app/models.py`, on `AskRequest.query` and `WsClientMessage.query` | 8000 | refuses absurd payloads **before any application code runs** (422) |
| `MAX_REQUEST_CHARACTERS` / `MAX_ESTIMATED_TOKENS` | `agents/governance.py` | 2000 / 600 | the cost policy, rejected with a structured **413** `BudgetRejection` |

The schema ceiling sits well above the cost cap on purpose: a request between
the two must still reach the service and be rejected with the structured 413
that Task 15 demonstrates, rather than being swallowed as a validation error.

Evidence: [`transcripts/budget_rejection.md`](transcripts/budget_rejection.md) —
a 3120-character request rejected with `service.crew_calls` unchanged.

### Task 16 — Response caching

In-memory bounded LRU cache on the grounded-generation step. The key is a
SHA-256 digest over the **normalised** query text (trim, lowercase, collapse
whitespace), the collection name, `top_k`, the calibrated threshold, the
knowledge-base version, and the **embedder identity** — each component
length-prefixed so two components cannot collide by concatenation.

The embedder component matters and was missing: a cached answer is only valid in
the embedding space it was retrieved in, so pinning `SIMILARITY_THRESHOLD` and
then switching `EMBEDDING_BACKEND` produced *identical* keys and served answers
across two different embedding spaces. It is derived from configuration rather
than from a live embedder object, so building a cache key still never loads a
model.

**Not cached, by design:** prompt-injection and blocked requests, errors, and
appointment-status lookups (appointment state is mutable, so a cached status
could be stale — only the immutable policy layer is cached). Raw PII never
reaches the cache either, because the query is masked before generation.

**Invalidation.** `POST /add-document` calls `invalidate_cache()` explicitly
*and* bumps `kb_version`, which is part of the key — two independent mechanisms,
either sufficient.

**Before/after evidence.** The proof is that a hit leaves both
`stats.generations` and `retriever.query_count` **unchanged** — the redundant
embedding, ChromaDB query and composition were genuinely skipped, not merely
faster. A timing comparison alone could not distinguish "cached" from "warm", so
both are reported.

Evidence: [`transcripts/cache_hit.md`](transcripts/cache_hit.md).

---

## Test suite

Twelve modules in `tests/`. **Every test runs offline**: no network, no model
artefacts, no real ChromaDB directory.

```bash
pytest
pytest tests/test_dataset.py -v
pytest -m "not requires_crewai and not requires_autogen"
```

| Module | Covers |
| --- | --- |
| `test_dataset.py` | determinism, ≥ 40 records, category counts, status coverage, follow-up band, unique ids, types and ranges, quota allocation, every validation failure path |
| `test_chunking.py` | fixed-size windows and real overlap, word integrity, sentence grouping, chunk metadata, strategy id isolation, KB loading and its error paths |
| `test_rag.py` | index separation, rebuild idempotence, `kb_version`, retrieval ordering, document dedup, grounded generation, fallback below threshold, threshold resolution and refusal, calibration separation and overlap, precision/recall arithmetic, embedding helpers, text utilities |
| `test_tools.py` | escalation bounds and monotonicity, aging cap, component arithmetic, the two signals genuinely competing, percentile threshold, lookup found / not found, **schema-based dispatch including the `rag_lookup` trap** |
| `test_crew.py` | ReAct formatting, **the `"Observation:"` template guard**, per-role mock behaviour, bounded tool attempts, tool ledgering, `crew.kickoff()` invoking the right tools per route, least autonomy inside crew construction |
| `test_memory.py` | session isolation, reset, record-id recovery, routing rules, same-session state carried, fresh-session state absent, cross-session non-leakage, masked-only storage |
| `test_guardrails.py` | every contact-number format masked with no digit run surviving, findings carrying no raw value, non-PII digits untouched, all nine injection patterns firing, ten legitimate questions **not** flagged, pipeline ordering, groundedness pass and fail |
| `test_api.py` | route registration, health, governance, ask (policy / appointment / PII / injection / 413), request validation, add-document success and rejection, unsafe slugs, cache invalidation, WebSocket answer / memory / reset / malformed frame / **disconnect survival** / budget frame, and every structured-logging assertion including "no raw PII on disk" |
| `test_review_team.py` | verdict model constraints, support detection, sentence stripping, all five verdict branches, reviewer critique, task message, role markers, and the **real `RoundRobinGroupChat`** approving and revising |
| `test_governance.py` | registry invariants, exclusive holder, every unauthorised assignment blocked, unknown agent rejected, broken registry detected, risk classification and justification content, all four layers, token estimate, both caps, no crew after rejection, no memory after rejection, injection never cached |
| `test_cache.py` | normalisation rules, key component sensitivity, concatenation collisions, LRU eviction, invalidation, disabled cache, miss→hit with counters unchanged, per-collection isolation, `kb_version` invalidation, service-level hit, blocked requests not cached |
| `test_regressions.py` | one class per defect from [the self-review](#defects-found-in-self-review-and-how-each-was-fixed): the exemplar id cannot poison memory (and a genuine echo still resolves), inbound bounds, embedder in the cache key, session eviction, per-document attribution, identity exemption **with** the proof it is doing the work, and the editor following a doctored critique |

`tests/fakes.py` provides `FakeEmbedder` (deterministic hashing) and
`FakeChromaClient` / `FakeCollection` (in-memory, cosine, same response shape).
`tests/conftest.py` **measures** the similarity threshold with the production
calibration code rather than hard-coding one — with the hashing double the
absolute similarities differ from the real model's, so a fixed number would be
meaningless, and measuring it keeps the tests honest about the threshold being
derived. Tests needing CrewAI or Autogen use `pytest.importorskip`, so the rest
of the suite still runs without them.

---

## Acceptance-criteria traceability matrix

Status wording: **Implemented in source** / **Covered by test** / **Requires
local execution for empirical output**. Nothing is marked "verified" or "passed",
because no code was executed while this repository was written.

| Task | Requirement | Implementation | Test | Evidence | Reproduce | Status |
| --- | --- | --- | --- | --- | --- | --- |
| 1 | ≥ 40 appointments, all structural thresholds, design in README | `dataset.py` | `tests/test_dataset.py` | `reports/dataset_report.md` | `python dataset.py` | Implemented in source; covered by test |
| 2 | ≥ 12 KB documents, every required topic | `data/knowledge_base/*.md`, `rag/chunking.py:load_knowledge_base` | `tests/test_chunking.py::TestKnowledgeBase` | the 12 files | `pytest tests/test_chunking.py` | Implemented in source; covered by test |
| 3 | Both strategies embedded and indexed into two separate collections | `rag/chunking.py`, `rag/embeddings.py`, `rag/indexer.py` | `tests/test_chunking.py`, `tests/test_rag.py::TestIndexing` | `reports/chunking_comparison.md` | `python -m scripts.build_indexes` | Implemented in source; requires local execution for the index |
| 4 | Grounded generation on 5 in-scope + 1 out-of-scope fallback; threshold calibrated empirically | `rag/grounded_generation.py`, `rag/calibration.py` | `tests/test_rag.py::TestGroundedGeneration`, `::TestCalibration`, `::TestThresholdResolution` | `transcripts/rag_demonstration.md`, `reports/calibration_report.md` | `python -m scripts.calibrate_threshold` then `--only rag` | Implemented in source; **measured values require local execution** |
| 5 | Precision/recall for BOTH collections, per-query arithmetic, numbers-cited recommendation | `rag/evaluation.py` | `tests/test_rag.py::TestPrecisionRecall` | `reports/chunking_comparison.md` | `python -m scripts.compare_chunking` | Implemented in source; **measured values require local execution** |
| 6 | `check_appointment_status` with a designed, justified escalation score | `agents/tools.py` | `tests/test_tools.py` | `transcripts/escalation_score.md` | `python -m scripts.generate_demonstrations --only escalation` | Implemented in source; covered by test |
| 7 | ≥ 3 agents; both tools invoked via `.kickoff()` | `agents/crew.py`, `agents/mock_llm.py`, `agents/tools.py` | `tests/test_crew.py::TestCrewKickoff` | `transcripts/tool_invocation.md` | `python -m scripts.generate_demonstrations --only tools` | Implemented in source; covered by test (needs crewai) |
| 8 | Multi-turn memory demonstrated, plus a separate fresh-conversation transcript | `agents/memory.py`, `app/services/session_store.py` | `tests/test_memory.py::TestEndToEndMemory` | `transcripts/memory_same_session.md`, `transcripts/memory_fresh_session.md` | `python -m scripts.generate_demonstrations --only memory` | Implemented in source; covered by test |
| 9 | Every response validates against a declared Pydantic schema | `app/models.py`, `app/services/support_service.py` | `tests/test_api.py::TestAsk::test_response_validates_against_the_schema` | `/docs` OpenAPI schema | `pytest tests/test_api.py` | Implemented in source; covered by test |
| 10 | Input PII + injection and output groundedness each firing on a deliberate case | `agents/guardrails.py` | `tests/test_guardrails.py` | `transcripts/guardrails.md` | `python -m scripts.generate_demonstrations --only guardrails` | Implemented in source; covered by test |
| 11 | ≥ 2 HTTP endpoints + 1 WebSocket surviving a disconnect | `app/main.py` | `tests/test_api.py::TestWebSocket` | `/docs` | `uvicorn app.main:app --reload` | Implemented in source; covered by test |
| 12 | One JSON-Lines entry per request with a trace id, no raw PII on disk | `app/logging_config.py` | `tests/test_api.py::TestStructuredLogging` | `data/generated/requests.jsonl` | `pytest tests/test_api.py -k logging` | Implemented in source; covered by test |
| 13 | Four scores for all 15 queries plus four averages | `evaluation/test_set.json`, `evaluation/mock_judge.py`, `evaluation/run_evaluation.py` | `tests/test_api.py` (pipeline), judge rubric in source | `reports/evaluation_report.md`, `evaluation/expected_results.md` | `python -m evaluation.run_evaluation` | Implemented in source; **scores require local execution** |
| 14 | Review stage both approving and revising, structured verdicts | `agents/review_team.py` | `tests/test_review_team.py` | `transcripts/autogen_review.md` | `python -m scripts.generate_demonstrations --only review` | Implemented in source; covered by test (needs autogen) |
| 15 | Least autonomy demonstrated; risk classified; budget cap rejects an oversized request | `agents/governance.py`, `agents/crew.py`, `app/main.py` | `tests/test_governance.py`, `tests/test_crew.py::TestLeastAutonomyInsideTheCrew` | `transcripts/least_autonomy.md`, `transcripts/budget_rejection.md` | `python -m scripts.generate_demonstrations --only autonomy budget` | Implemented in source; covered by test |
| 16 | Repeated query producing a real cache hit with before/after evidence | `rag/cache.py`, `rag/grounded_generation.py` | `tests/test_cache.py` | `transcripts/cache_hit.md` | `python -m scripts.generate_demonstrations --only cache` | Implemented in source; covered by test |

One command walks all of it:

```bash
python -m scripts.run_acceptance_checks
```

It prints PASS / FAIL / SKIP per criterion and writes
`reports/acceptance_report.md`. `SKIP` means the check needs a built index and a
calibrated threshold — it never reports a false PASS.

---

## Defects found in self-review, and how each was fixed

This repository was put through a deliberate adversarial review of its own
source. Nine defects were found, seven of them reachable from a normal request
path. They are listed here with the fix and the regression test, because a
control that has never been attacked is not evidence of anything.

All of these are pinned by `tests/test_regressions.py`, which states in each
docstring what used to happen.

| # | Defect | Why it mattered | Fix |
| --- | --- | --- | --- |
| 1 | The "which appointment?" reply named **`APT-1007`**, a real record. That answer is stored in session history, and `resolve_record_id_from_history` scans history newest-first — so the next bare follow-up resolved the assistant's *own* sentence and disclosed a record the patient never named. | Unprompted disclosure of appointment data. The worst defect in the project, and it was caused by a safety message. | `app/config.py:EXEMPLAR_RECORD_ID = "APT-XXXX"`. `RECORD_ID_PATTERN` requires four digits, so the exemplar is now **structurally** unresolvable, and `agents/composition.py` asserts that at import time. |
| 2 | `EmbeddingModelUnavailableError`, `ReviewUnavailableError`, `CrewExecutionError`, `ToolDispatchError`, `ToolPermissionError` and `KnowledgeBaseError` were unhandled by `POST /ask`. All subclass plain `RuntimeError`. | An unhandled **500**, and — because every log call sat *after* the pipeline — **no JSON-Lines entry at all**. The most likely failure on a machine without model artefacts was also the one guaranteed to leave no audit trail. Broke Task 12's "exactly one entry per request" and Task 15's "typed exceptions surfaced as structured errors". | `PIPELINE_UNAVAILABLE_ERRORS` in `app/main.py` → structured 503 `pipeline_unavailable`. The service now writes its audit line from an `except` branch and re-raises, so **one line is written on every path**. |
| 3 | The WebSocket handler caught a *narrower* set of failures than `/ask`, so an embedding failure fell through to the catch-all and closed the socket. | Directly contradicted the documented promise that a failed turn leaves the conversation open. | One `SERVICE_UNAVAILABLE_ERRORS` tuple, used by both transports, so they cannot drift apart. |
| 4 | `AskRequest.query` had **no `max_length`**. | A multi-megabyte body was parsed, masked and scanned by nine regexes before the cost cap was consulted. | `MAX_INBOUND_TEXT_CHARACTERS`; see the two-bounds table under Task 15. |
| 5 | `session_id=" "` passed `min_length=1`, then raised `ValueError` inside `SessionMemory`. | Unhandled **500** from a one-character input. | The `_not_blank` validator now covers `session_id` as well as `query` → 422. |
| 6 | `SessionMemory` was an unbounded dict keyed by a **client-supplied** id, and history deliberately outlives its socket. | Any caller could grow process memory without limit while the Runtime layer claimed "bounded cost". The response cache was bounded; the larger allocation was not. | Bounded LRU, `DEFAULT_MAX_SESSIONS = 512`, with an eviction counter. |
| 7 | `compose_grounded_answer` framed the whole answer with `chunks[0].topic_title` while quoting sentences from **every** retrieved chunk. | On a genuinely two-document question the prose credited one document while `sources` listed two — misattribution, on the very query (`fees_and_home_visit`) chosen to span two documents. | Sentences are grouped by the document that actually contains them; one frame per source document. |
| 8 | The response cache key omitted the embedding backend. | With `SIMILARITY_THRESHOLD` pinned, switching `EMBEDDING_BACKEND` produced *identical* keys, so answers retrieved in one embedding space were served for another. | `embedder` is now a key component, supplied from configuration so building a key still never loads a model. |
| 9 | The threshold and the ChromaDB collection handles were memoised for the life of the process, with no way to reload them. `build_indexes(reset=True)` *deletes* those collections. | Re-calibrating or rebuilding against a running server had no effect, and the server kept handles to deleted collections. | `SupportService.refresh_runtime_state()`, plus `refresh_threshold()` and `refresh_collections()`. |

### Two design claims that were corrected rather than patched

**The output groundedness check was close to a tautology.** `build_support_text`
poured *every* literal string the composer can emit — including the whole
"I don't know" refusal and the request-for-an-id — into the support corpus. Since
a draft is assembled only from retrieved sentences, looked-up facts and those
same literals, every sentence's vocabulary was a subset of the support
vocabulary, so overlap was ~1.0 on every real request. The only thing that could
trip it was `inject_unsupported_claim`, which no request path sets.

The fix is *not* to delete the framing from the support text. That was tried on
paper and it breaks the app: an escalated appointment's sentence
("…above the escalation threshold of…, so I am flagging it for a Practo support
lead to review") scores **0.545** against evidence alone — below the 0.6
threshold — so escalated records would be **refused**. The fact templates are
load-bearing support for the sentences rendered from them.

So the corpus was narrowed precisely instead: the fixed control answers moved out
of the support text and are now exempted **by identity**
(`EXEMPT_ANSWER_SENTENCES`), while the fact templates stay in. That removes a
free vocabulary — "know", "available", "knowledge", "base", "specific", "share",
"matches" — that every request previously inherited just so three fixed answers
could pass. `tests/test_regressions.py` proves both halves: the refusal still
passes *via the exemption*, and would be flagged without it.

Honest framing of what the check now is: groundedness is enforced **by
construction** in the composer, which quotes verbatim. The check is a
defence-in-depth assertion over that construction — it catches novel vocabulary
appearing from nowhere. It is not, and never was, a semantic entailment check.

**The Autogen reviewer's critique had no causal effect.** The mock editor
returned `deterministic_verdict(session)`, computed from the session object. The
reviewer's turn was generated, appended to the transcript, and read by nothing;
with `max_turns=2` there was no loop in which it could have mattered. Renaming
the agent was the only thing that made it a reviewer.

The reviewer now appends a machine-readable findings block
(`[[UNSUPPORTED_SENTENCES]] … [[/UNSUPPORTED_SENTENCES]]`, or
`[[RETRIEVAL_BELOW_THRESHOLD]]`) beneath its human-readable prose, and the editor
acts on `verdict_from_findings(...)` — removing exactly the sentences it was
told to remove. If the reviewer has not spoken, the editor **raises** rather than
re-deriving a verdict of its own. `deterministic_verdict` is retained as an
independently-derived cross-check, which is what
`test_review_team.py::test_the_teams_verdict_matches_the_derived_expectation`
asserts the team against.

The decisive test is
`test_regressions.py::test_the_editor_follows_a_doctored_critique`: it names a
sentence that *is* well supported. An editor recomputing from the session would
approve the draft unchanged; this one strips the sentence, because that is what
the reviewer said.

### Known gap, deliberately not built: authorization

`check_appointment_status` is a dictionary lookup with **no notion of who is
asking**, and no endpoint is authenticated. Forty-five requests naming
`APT-1001` … `APT-1045` therefore read every record in the dataset, and no
guardrail fires — that is the intended happy path, once per id. The
prompt-injection denylist is irrelevant here: nothing about "what is the status
of appointment APT-1002?" is an injection.

`TOOL_PERMISSIONS` constrains which **agent** may hold the tool. It says nothing
about which **patient** may read which **row**. Those are different controls, and
the least-autonomy paragraph above should be read as the former only.

Likewise `POST /add-document` is unauthenticated and writes into the retrieval
corpus. A document submitted through it is retrieved, quoted verbatim, reported
`grounded: true`, cited as its own source and approved by the reviewer — because
every control in this system validates *provenance*, never *authority*. Once
ground truth is writable, the downstream controls are vacuous. `SLUG_PATTERN`
prevents path traversal, but traversal was never the interesting threat.

This is **not fixed**, and the reason is scope rather than oversight: this brief
specifies no authentication, no patient identity and no admin role, and inventing
one would change every endpoint, fixture and transcript in the project. In a real
deployment the appointment tool would take an authenticated subject and check
ownership, and `/add-document` would sit behind a separate admin credential. It
is recorded here so a reviewer sees a known, reasoned gap rather than an
unnoticed one.

---

## Limitations

Stated because a governance review would find them anyway:

1. **The groundedness check is lexical, not semantic.** Content-word overlap is a
   deterministic stand-in for entailment under `MOCK_LLM`. It catches novel
   vocabulary appearing from nowhere; it would not catch a fabrication assembled
   entirely from context vocabulary, and it could flag a legitimate paraphrase.
2. **The injection denylist is not exhaustive.** Nine named patterns catch common
   phrasings. A novel phrasing that avoids all nine would pass.
3. **Free-text PII is not masked.** Patient name, diagnosis/condition and
   insurance ID have no universal format to match without a model or a
   dictionary. Out of scope, per the brief, and all examples are fabricated.
4. **The evaluation judge is rule-based.** Reproducible, but it cannot recognise
   a correct paraphrase and cannot notice a plausible answer that is wrong in a
   way the declared expectations do not cover.
5. **Session memory is in-process.** It does not survive a restart and is not
   shared between workers. Sufficient per the brief; a real deployment would need
   external storage.
6. **The knowledge base is small.** Twelve documents. Precision/recall on five
   queries is an indication, not a statistically meaningful benchmark.
7. **The token estimate is a heuristic.** `ceil(len/4)`, not a real tokeniser. It
   is a cost guardrail, not an accounting figure.
8. **The appointment dataset is synthetic.** Deterministically generated, no real
   patient data anywhere. That is a safety property here, but it means the
   escalation threshold is calibrated against a fabricated distribution.
9. **`add-document` writes into the repository working tree.** Convenient for a
   demonstration; a real deployment would use object storage and a migration
   path.
10. **Dependency pins were not verified against a package index** while this
    repository was written. See the dependency policy above.
11. **The mock review stage shares its support test with the output guardrail.**
    Both use `rag/textutils.py`, so the reviewer is *independent in
    orchestration* — a separate agent team with its own model client, and since
    the findings-block fix its critique genuinely drives the verdict — but it is
    not independent in *method*. Because both measure the same lexical overlap
    against the same support text, they are perfectly correlated: the reviewer
    cannot catch anything the guardrail would have missed. A real deployment
    would want a different signal there (NLI entailment, or a reviewer that sees
    only the answer and the source).
12. **There is no authorization anywhere.** Any caller can read any appointment
    record by naming its id, and `POST /add-document` can write into the
    retrieval corpus unauthenticated. `TOOL_PERMISSIONS` governs which *agent*
    holds a tool, not which *patient* may read which *row*. Reasoned scope
    decision, not an oversight — see
    [the authorization gap](#known-gap-deliberately-not-built-authorization).
13. **The request log persists unmasked free-text PII.** Only fixed-format
    Indian mobile numbers are masked, and `requests.jsonl` has no rotation and no
    retention window. See the note under Task 12.
14. **Groundedness is enforced by construction, not by the check.** The composer
    quotes retrieved sentences verbatim, which is what actually makes answers
    grounded; the overlap check is a defence-in-depth assertion over that
    construction. It would not independently catch a fabrication assembled
    entirely from context vocabulary, and under a real LLM — where
    `crew.kickoff()`'s own text is used rather than the deterministic
    composition — it is the only remaining guard.

---

## Interview preparation

Questions worth being able to answer about this project, with pointers.

**Architecture and data flow**
- Walk one `POST /ask` from HTTP to JSON-Lines log. → [Request data flow](#request-data-flow)
- Why is routing rule-based rather than model-decided? → reproducibility, and it
  stops the crew from reaching a tool it should not.
- Why does the Composer hold no tools? → separating "what to fetch" from "how to
  phrase it"; see the least-autonomy paragraph.
- Where would you add a fourth agent, and what would you have to edit? →
  `AGENT_SPECS`, `TOOL_PERMISSIONS`, `_agent_keys_for_route`.

**RAG**
- Why two chunking strategies, and how did you choose between them? → Task 5,
  with a decision rule fixed before the numbers were read.
- Why is precision/recall scored at the document level with dedup? → overlapping
  fixed-size chunks would otherwise flatter that strategy.
- How did you pick the similarity threshold, and why not 0.6? → measured midpoint
  of the observed gap; round numbers do not reliably separate short policy
  sentences.
- What happens if the calibration clusters overlap? → it fails loudly and
  recommends nothing.

**Orchestration**
- Why subclass `BaseLLM` instead of monkey-patching CrewAI? → CrewAI's documented
  extension point; the crew being exercised stays the real crew.
- What are the two CrewAI pitfalls and how does the code avoid each? →
  [`MOCK_LLM` design](#mock_llm-design); both have dedicated tests.
- Why `supports_function_calling() == False`? → it drives the ReAct path where
  those pitfalls live.
- Why `memory=False` and `cache=False` on the crew? → offline guarantee, and
  ledger honesty.

**Memory, guardrails, governance**
- How does a follow-up question resolve an appointment id? → newest-first history
  scan; two transcripts show present and absent.
- Why mask *before* injection detection? → a payload hidden inside a phone number
  cannot slip past.
- Why does the same function mask both the model input and the log line? → so a
  raw number cannot reach disk by anyone forgetting.
- Why is this High risk rather than Medium? → medical data, patient-facing, and a
  wrong answer changes health behaviour.
- Where is the budget cap enforced and why there? → before crew construction, so
  a rejection costs one length check.

**Evaluation and review**
- What can your judge not measure? → paraphrase, and wrongness outside the
  declared expectations.
- How is the review stage independent, and how is it not? → independent
  orchestration and model client; shared lexical method. Limitation 11.
- How would you prove the cache actually saved work? → the two counters, not the
  timing.

**Tech-stack choices**
- Why ChromaDB, SentenceTransformers, CrewAI, Autogen, LangChain memory, FastAPI?
  → free and local, no API key, and each is the piece the brief names for that job.
- What would you change for production? → external memory, semantic entailment
  for groundedness, a real tokeniser, object storage for documents, and a
  genuinely independent reviewer signal.

**Questions the scaffolding guide sets up**
- Where is your `response_format`? → `app/models.py:RESPONSE_FORMAT`, aliasing
  `SupportResponse`; enforced at three points listed under Task 9.
- Where is your audit trail, given you have no `db.py`? → the JSON-Lines request
  log plus the tool-invocation ledger; see "Where the audit trail lives".
- Where is your `policy.md`? → `data/knowledge_base/`, twelve retrievable policy
  documents rather than one file, because this brief requires retrieval over them.
- Why no SQLite? → nothing in this brief asks to persist anything; state is
  ChromaDB plus in-process session memory. Adding a schema for data that is never
  written would be dependency for its own sake.
- What stops the agents making things up? → answers are verbatim retrieved
  sentences, routing is rule-based, the Composer holds no tools, and two
  groundedness checks plus an independent reviewer sit downstream.
- Which Python version, and why not the newest? → 3.11 (`.python-version`),
  inside the guide's "3.10 or higher". 3.14 is excluded because CrewAI declares
  `requires-python >=3.10,<3.14`.

---

## Command reference

```bash
# Setup
python -m venv ../practo-venv
../practo-venv/Scripts/activate
pip install -r requirements.txt -r requirements-dev.txt
cp .env.example .env
```

```bash
# Part 1
python dataset.py                              # dataset + validation report
python -m scripts.build_indexes                # both ChromaDB collections
python -m scripts.calibrate_threshold          # MEASURE the threshold
python -m scripts.compare_chunking             # precision/recall + recommendation
```

```bash
# Parts 2 and 4 evidence
python -m scripts.generate_demonstrations      # all transcripts
python -m scripts.generate_demonstrations --only rag tools memory guardrails
python -m scripts.generate_demonstrations --only review budget autonomy cache escalation
```

```bash
# Part 3
python -m evaluation.run_evaluation            # 15 queries, four scores + averages
uvicorn app.main:app --reload                  # serve the API
```

```bash
# Verification
pytest                                         # full suite
pytest -m "not requires_crewai and not requires_autogen"
python -m scripts.run_acceptance_checks        # every acceptance criterion
```

```bash
# Git (local only)
git init
git add .gitignore && git commit -m "Add gitignore file"
git add . && git commit -m "Practo (Healthcare) capstone: dataset, RAG, crew, review, governance, API"
```

### Example requests

```bash
curl http://localhost:8000/health
```

```bash
curl -X POST http://localhost:8000/ask -H "Content-Type: application/json" -d "{\"query\":\"How long before my appointment can I cancel without paying a fee?\",\"session_id\":\"demo\"}"
```

```bash
curl -X POST http://localhost:8000/ask -H "Content-Type: application/json" -d "{\"query\":\"What is the status of appointment APT-1007?\",\"session_id\":\"demo\"}"
```

---

## Disclaimer on execution

The implementation and evidence were validated locally after development. The
SentenceTransformers model was downloaded once for index creation, then strict
offline mode was restored for calibration, evaluation, transcripts, and the
acceptance run. In particular:

- Every file in `transcripts/` contains captured output from the local graded
  run. Running `python -m scripts.generate_demonstrations` regenerates them.
- `evaluation/expected_results.md` is a **rubric and template**. The measured
  scores come from `python -m evaluation.run_evaluation`.
- The three README sections the brief requires measured numbers in
  (`AUTO:CALIBRATION`, `AUTO:CHUNKING`, `AUTO:ESCALATION`) contain measured
  values generated by their named scripts:

  | Section | Filled by |
  | --- | --- |
  | `AUTO:CALIBRATION` | `python -m scripts.calibrate_threshold` |
  | `AUTO:CHUNKING` | `python -m scripts.compare_chunking` |
  | `AUTO:ESCALATION` | `python -m scripts.generate_demonstrations --only escalation` |

- The dependency pins are a declared baseline selected from existing knowledge,
  not a resolved lockfile.

Areas most likely to need attention on first local run, and why:

| Area | Why it is runtime-sensitive |
| --- | --- |
| The CrewAI custom `BaseLLM` | CrewAI's ReAct parser and executor internals vary between minor versions |
| Autogen structured messages | `output_content_type` and `custom_message_types` were introduced and refined across 0.4.x/0.5.x |
| LangChain memory APIs | `RunnableWithMessageHistory` is deprecated in favour of LangGraph persistence |
| ChromaDB persistence | client construction and `include=` argument names have shifted across 0.4/0.5/1.x |
| Local SentenceTransformers artefacts | must be present for strict offline mode; see the two options above |

Requiring local execution: package installation, embedding-model availability,
index creation, threshold calibration, the chunking comparison, the test suite,
transcript generation, the evaluation run, and FastAPI start-up.

---

**Originality.** The dataset design, the twelve knowledge-base documents, the
code and the analysis in this repository are my own work for this specific brief.
All appointment records and all patient examples are fabricated. No real medical
data is used anywhere, and this system does not provide diagnosis, triage or
medical advice.


