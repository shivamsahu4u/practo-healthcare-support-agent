# Sample request payloads


The scaffolding guide asks for a `data/samples/` directory of example inputs for
testing. In the Paydisk invoice example those were sample invoices; for the
Practo track the equivalent inputs are patient queries and knowledge-base
documents.


| File | Endpoint | What it exercises |
| --- | --- | --- |
| [ask_policy.json](ask_policy.json) | `POST /ask` | policy retrieval only - the RAG tool fires, the lookup tool does not |
| [ask_appointment.json](ask_appointment.json) | `POST /ask` | appointment lookup only - the lookup tool fires |
| [ask_combined.json](ask_combined.json) | `POST /ask` | both tools fire in one turn |
| [ask_pii_masking.json](ask_pii_masking.json) | `POST /ask` | the contact-number guardrail fires (number is fabricated) |
| [ask_prompt_injection.json](ask_prompt_injection.json) | `POST /ask` | the injection guardrail blocks before the crew runs |
| [ask_out_of_scope.json](ask_out_of_scope.json) | `POST /ask` | the calibrated "I don't know" fallback fires |
| [ask_oversized.json](ask_oversized.json) | `POST /ask` | the runtime budget cap rejects with HTTP 413 |
| [add_document.json](add_document.json) | `POST /add-document` | a new policy document indexed into both collections |
| [ws_chat_session.json](ws_chat_session.json) | `WS /ws/chat/{session_id}` | a multi-turn frame sequence including `reset` |


Every appointment id, contact number and patient detail in these files is
**fabricated**. No real medical data appears anywhere in this repository.


## Running them


Start the server first:


```bash
uvicorn app.main:app --reload
```


Then send one:


```bash
curl -X POST http://localhost:8000/ask -H "Content-Type: application/json" -d @data/samples/ask_policy.json
```


```bash
curl -X POST http://localhost:8000/add-document -H "Content-Type: application/json" -d @data/samples/add_document.json
```


`ask_oversized.json` is expected to return **413**, not 200 - that is the point
of it. `ask_prompt_injection.json` returns **200** with
`"response_type": "blocked"`, because a blocked request is a successful refusal
rather than an error.


The WebSocket sequence is a list of frames rather than a single body; send them
in order over `ws://localhost:8000/ws/chat/sample-session`. Frame 3 is
`{"reset": true}`, so frame 4 asks the same follow-up as frame 2 and correctly
fails to resolve the appointment - the memory-reset behaviour from
`transcripts/memory_fresh_session.md`.



