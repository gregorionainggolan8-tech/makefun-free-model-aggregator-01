# Free AI model aggregator

One server that serves the OpenAI Chat Completions API (`POST /v1/chat/completions`)
and the Anthropic Messages API (`POST /v1/messages`), each in JSON and in server-sent
events, over ten simulated free-tier providers on the OpenAI, Anthropic and Gemini
wires. It also answers `GET /v1/models` in the shape of the auth header it got.

## How it picks models

It reads the trial brief (`$MAKEFUN_PROMPT_FILE`): the provider directory, the
sealed quality table and the workload manifest. For each request it ranks every
free model that meets the request's requirements (tools, vision, JSON mode,
context length) by the quality table's `rating_milli` and tries the best first.
A provider that refuses (402, 429, 5xx) is counted and the request is retried on
the next-best model; a provider that fails three calls in a row is treated as
spent or out for the rest of the trial. A JSON-mode request from an OpenAI client
only goes to a wire that can carry `response_format` (OpenAI or Gemini).

Every provider answer is read in full, parsed into text, tool calls, usage and
stop reason, and rendered back in the wire the client used, with the provider's
text whole, its usage counts as that wire reports them, and its stop reason
mapped (`end_turn`→`stop`, `max_tokens`→`length`, `tool_use`→`tool_calls`,
`refusal`→`content_filter`). Requests pass through untouched: system prompt,
user text (images included), tools, stop sequences, `max_tokens`, JSON mode.

## How to run

```sh
# print the catalog the server would register, and exit 0
node main.ts --list

# serve a trial (the sandbox sets MAKEFUN_ENV_URL and MAKEFUN_PROMPT_FILE)
node main.ts
```

Node 22 standard library only; nothing is installed. All routing choices derive
from the brief, the requests and the providers' answers — no wall clock, no
unseeded randomness — so a replayed trial leaves the same state export.
