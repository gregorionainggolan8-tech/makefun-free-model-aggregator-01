// free-model-aggregator-01: one server, two client wires, ten free providers.
//
// It reads the brief ($MAKEFUN_PROMPT_FILE), probes each provider's catalog route
// in the provider's own wire, registers one catalog, then serves the client's 20
// requests, each answered by the best free model that can serve it.
//
// ROUTING (quality is 60% of the score)
//   Every request's candidates are all free models meeting its requirements,
//   ranked by the quality table's rating_milli, ties broken by provider id then
//   model id (deterministic: no clock, no unseeded randomness anywhere). The best
//   candidate is tried first; a provider that refuses (402, 429, 5xx, a transport
//   error, or an answer that cannot be parsed) is counted and the request is
//   retried on the next candidate, so a failure costs one call and never the
//   request. A provider that fails three calls in a row is treated as spent or
//   out for the rest of the trial: with at most two scheduled failures per
//   provider, three in a row can only be a spent budget or an outage. A JSON-mode
//   request from an OpenAI client is only routed to a wire that can carry
//   response_format (OpenAI or Gemini): the Anthropic wire echoes it absent.
//
// WIRES
//   All three upstream wires are spoken (OpenAI, Anthropic, Gemini
//   :generateContent / :streamGenerateContent). Each provider answer is read in
//   full, parsed into one shape (text, tool calls, usage, stop family) and
//   rendered back in the wire the client used, JSON or SSE, with the provider's
//   text whole, its usage counts as that wire reports them, and its stop reason
//   mapped end_turn to stop, max_tokens to length, tool_use to tool_calls and
//   refusal to content_filter. Requests pass through untouched: system prompt,
//   user text (images included), tools, stop sequences, max_tokens, JSON mode.
//   One provider call at a time is ever in flight, so a second concurrent call
//   is never refused.
//
// CATALOG
//   The catalog is what the providers' own catalog routes listed, with the
//   quality table's free flag and context length where it has a row; its base_url
//   is the provider's own path (such as /p/groq), never joined to the env URL.
//   `node main.ts --list` prints that same document and exits 0. GET /v1/models
//   lists exactly the free model ids in the catalog, in the shape of the auth
//   header it got.

import { createServer, type IncomingMessage, type Server, type ServerResponse } from 'node:http';
import {
  authHeaders,
  buildCatalog,
  capabilitiesOf,
  isRecord,
  joinBase,
  meetsRequirements,
  modelFree,
  probeProvider,
  ratingOfModel,
  readBrief,
  type CatalogReport,
  type Json,
  type Probe,
  type ProbedModel,
  type Requirement,
} from './lib/providers.ts';
import {
  anthropicError,
  anthropicModelList,
  anthropicToOpenAI,
  openAIError,
  openAIModelList,
  openAIToAnthropic,
  parseAnthropicJson,
  parseAnthropicStream,
  parseGeminiJson,
  parseGeminiStream,
  parseOpenAIJson,
  parseOpenAIStream,
  renderAnthropicJson,
  renderAnthropicStream,
  renderOpenAIJson,
  renderOpenAIStream,
  toGemini,
  wantsJson,
  type ParsedAnswer,
  type ServedModel,
} from './lib/wire.ts';

/** The wire protocol's version, in the register body. */
const PROTOCOL_VERSION = 1;
/** How often GET /work is polled, in milliseconds. */
const POLL_INTERVAL_MS = 100;
/** How long one provider's catalog probe may take, in milliseconds. */
const PROBE_TIMEOUT_MS = 5000;
/** How long one provider inference call may take, in milliseconds. */
const CALL_TIMEOUT_MS = 5000;
/** How many calls in a row a provider may refuse before it is treated as spent or out. */
const DEAD_AFTER_FAILURES = 3;
/** The most bytes of one client request body this server reads. */
const MAX_BODY_BYTES = 32 * 1024 * 1024;

const ENV_URL = (process.env.MAKEFUN_ENV_URL ?? '').replace(/\/+$/, '');
const PROMPT_FILE = process.env.MAKEFUN_PROMPT_FILE ?? '';

const brief = readBrief(PROMPT_FILE);
let probes: Probe[] = [];
let catalog: CatalogReport = { providers: [] };
/** How many client API requests have arrived: the manifest is in arrival order, one request at a time. */
let manifestIndex = 0;

/** Per-provider call state, in the order the calls were made (deterministic). */
const providerState = new Map<string, { calls: number; consecutiveFailures: number; dead: boolean }>();

function providerStateOf(id: string): { calls: number; consecutiveFailures: number; dead: boolean } {
  let state = providerState.get(id);
  if (state === undefined) {
    state = { calls: 0, consecutiveFailures: 0, dead: false };
    providerState.set(id, state);
  }
  return state;
}

function log(message: string): void {
  process.stderr.write(`aggregator: ${message}\n`);
}

/** Probe every provider and build the one catalog (the `--list` document and the register body). */
async function load(): Promise<void> {
  if (brief.error !== null) log(brief.error);
  probes = await Promise.all(brief.directory.providers.map((provider) => probeProvider(ENV_URL, provider, PROBE_TIMEOUT_MS)));
  for (const probe of probes) if (!probe.ok && probe.error !== null) log(probe.error);
  catalog = buildCatalog(ENV_URL, probes, brief.ratings);
}

const byId = (a: ProbedModel, b: ProbedModel): number => (a.id < b.id ? -1 : a.id > b.id ? 1 : 0);

/** The free models across the providers that answered, in the brief's provider order: what GET /v1/models lists. */
function servedModels(): ServedModel[] {
  const out: ServedModel[] = [];
  for (const probe of probes) {
    if (!probe.ok) continue;
    for (const model of [...probe.models].sort(byId)) {
      const row = ratingOfModel(brief.ratings, probe.provider.id, model.id);
      if (!modelFree(model, row)) continue;
      out.push({ id: model.id, provider: probe.provider.id, display_name: model.display_name });
    }
  }
  return out;
}

interface Candidate {
  probe: Probe;
  model: ProbedModel;
  rating: number;
}

/**
 * Every free model that can serve one request, best rating first. A JSON-mode request from an
 * OpenAI client is only offered to a wire that can carry response_format: the Anthropic wire
 * echoes it absent, so an anthropic-wire provider can never serve such a request validly.
 */
function candidatesFor(requires: Requirement, jsonNeedsWire: boolean): Candidate[] {
  const out: Candidate[] = [];
  for (const probe of probes) {
    if (!probe.ok) continue;
    if (jsonNeedsWire && probe.provider.wire === 'anthropic') continue;
    for (const model of probe.models) {
      const row = ratingOfModel(brief.ratings, probe.provider.id, model.id);
      if (!modelFree(model, row)) continue;
      if (!meetsRequirements(requires, capabilitiesOf(model, row))) continue;
      out.push({ probe, model, rating: row?.rating_milli ?? 0 });
    }
  }
  out.sort(
    (a, b) =>
      b.rating - a.rating ||
      (a.probe.provider.id < b.probe.provider.id ? -1 : a.probe.provider.id > b.probe.provider.id ? 1 : 0) ||
      (a.model.id < b.model.id ? -1 : a.model.id > b.model.id ? 1 : 0),
  );
  return out;
}

/** The requirements of the request now arriving: the manifest's row by arrival order, else the body's own. */
function requirementsFor(body: Json): Requirement {
  const manifest = brief.workload?.manifest ?? [];
  const entry = manifest[manifestIndex];
  manifestIndex += 1;
  if (entry !== undefined) return entry.requires;
  // The manifest is the declared source of each request's requirements; this fallback is a
  // guess for a brief that holds none.
  return requirementsFromBody(body);
}

/** Requirements read off a request body, when the manifest has no row for it. */
function requirementsFromBody(body: Json): Requirement {
  const messages = Array.isArray(body.messages) ? body.messages : [];
  let vision = false;
  for (const message of messages) {
    if (!isRecord(message) || !Array.isArray(message.content)) continue;
    for (const block of message.content) {
      if (isRecord(block) && (block.type === 'image' || block.type === 'image_url')) vision = true;
    }
  }
  return { tools: Array.isArray(body.tools) && body.tools.length > 0, vision, json: wantsJson(body) };
}

/** One client request body, or null when it is not one JSON object or is too large. */
async function readJson(req: IncomingMessage): Promise<Json | null> {
  const chunks: Buffer[] = [];
  let size = 0;
  for await (const chunk of req) {
    const buffer = Buffer.isBuffer(chunk) ? chunk : Buffer.from(String(chunk));
    size += buffer.length;
    if (size > MAX_BODY_BYTES) return null;
    chunks.push(buffer);
  }
  if (chunks.length === 0) return null;
  try {
    const value: unknown = JSON.parse(Buffer.concat(chunks).toString('utf8'));
    return isRecord(value) ? value : null;
  } catch {
    return null;
  }
}

function sendJson(res: ServerResponse, status: number, body: unknown): void {
  const text = JSON.stringify(body);
  res.writeHead(status, { 'content-type': 'application/json', 'content-length': Buffer.byteLength(text) });
  res.end(text);
}

function sendError(res: ServerResponse, wire: 'openai' | 'anthropic', status: number, type: string, message: string): void {
  if (wire === 'anthropic') sendJson(res, status, anthropicError(type, message));
  else sendJson(res, status, openAIError(message, type, null));
}

function sendSse(res: ServerResponse, text: string): void {
  res.writeHead(200, { 'content-type': 'text/event-stream', 'cache-control': 'no-cache', connection: 'keep-alive' });
  res.end(text);
}

/** GET /v1/models, in the shape of the auth header the client sent. */
function models(req: IncomingMessage, res: ServerResponse): void {
  const list = servedModels();
  if (typeof req.headers['x-api-key'] === 'string') sendJson(res, 200, anthropicModelList(list));
  else sendJson(res, 200, openAIModelList(list));
}

type ClientApi = 'openai' | 'anthropic';

interface UpstreamResult {
  ok: boolean;
  status: number;
  answer: ParsedAnswer | null;
}

/** Parse one provider's answer body into the one shape, by its wire and whether it streamed. */
function parseUpstream(wire: 'openai' | 'anthropic' | 'gemini', stream: boolean, text: string): ParsedAnswer | null {
  if (wire === 'openai') {
    if (stream) return parseOpenAIStream(text);
    let body: unknown;
    try {
      body = JSON.parse(text);
    } catch {
      return null;
    }
    return parseOpenAIJson(body);
  }
  if (wire === 'anthropic') {
    if (stream) return parseAnthropicStream(text);
    let body: unknown;
    try {
      body = JSON.parse(text);
    } catch {
      return null;
    }
    return parseAnthropicJson(body);
  }
  if (stream) return parseGeminiStream(text);
  let body: unknown;
  try {
    body = JSON.parse(text);
  } catch {
    return null;
  }
  return parseGeminiJson(body);
}

/**
 * One inference call to one candidate's provider, in that provider's wire, fully read and parsed.
 * A non-2xx, a transport error or an unparsable answer is a failure the caller fails over from;
 * a 2xx that parses is the request's answer and must be used (a second 200 would break the
 * trial's one-provider check).
 */
async function callCandidate(candidate: Candidate, clientApi: ClientApi, body: Json, stream: boolean): Promise<UpstreamResult> {
  const provider = candidate.probe.provider;
  const base = joinBase(ENV_URL, provider.base);
  const headers: Record<string, string> = {
    'content-type': 'application/json',
    ...authHeaders(provider.auth, provider.key),
  };
  let url: string;
  let upstreamBody: Json;
  if (provider.wire === 'openai') {
    if (clientApi === 'openai') {
      upstreamBody = { ...body, model: candidate.model.id };
      if (stream) upstreamBody.stream_options = { include_usage: true };
    } else {
      upstreamBody = anthropicToOpenAI(body, candidate.model.id);
    }
    url = `${base}/v1/chat/completions`;
    if (stream) headers.accept = 'text/event-stream';
  } else if (provider.wire === 'anthropic') {
    if (clientApi === 'anthropic') upstreamBody = { ...body, model: candidate.model.id };
    else upstreamBody = openAIToAnthropic(body, candidate.model.id);
    headers['anthropic-version'] = '2023-06-01';
    url = `${base}/v1/messages`;
    if (stream) headers.accept = 'text/event-stream';
  } else {
    upstreamBody = toGemini(body, clientApi, candidate.model.id);
    const method = stream ? 'streamGenerateContent' : 'generateContent';
    url = `${base}/v1beta/models/${encodeURIComponent(candidate.model.id)}:${method}${stream ? '?alt=sse' : ''}`;
    if (stream) headers.accept = 'text/event-stream';
  }

  let upstream: Response;
  try {
    upstream = await fetch(url, {
      method: 'POST',
      headers,
      body: JSON.stringify(upstreamBody),
      signal: AbortSignal.timeout(CALL_TIMEOUT_MS),
    });
  } catch (err) {
    log(`call to ${provider.id}/${candidate.model.id} failed: ${(err as Error).message}`);
    return { ok: false, status: 0, answer: null };
  }
  if (!upstream.ok) {
    // Read the error body so the connection closes cleanly; its shape is the provider's own.
    try {
      await upstream.text();
    } catch {
      // The body is unreadable: the status is what the caller needs.
    }
    log(`call to ${provider.id}/${candidate.model.id} answered HTTP ${upstream.status}`);
    return { ok: false, status: upstream.status, answer: null };
  }
  let text: string;
  try {
    text = await upstream.text();
  } catch (err) {
    log(`call to ${provider.id}/${candidate.model.id} gave no readable body: ${(err as Error).message}`);
    return { ok: false, status: 0, answer: null };
  }
  const answer = parseUpstream(provider.wire, stream, text);
  if (answer === null) {
    log(`call to ${provider.id}/${candidate.model.id} answered 200 with an unparsable body`);
    return { ok: false, status: upstream.status, answer: null };
  }
  return { ok: true, status: upstream.status, answer };
}

/** The request's answer, rendered in the client's wire from the provider that served it. */
function renderAnswer(clientApi: ClientApi, stream: boolean, answer: ParsedAnswer, model: string, res: ServerResponse): void {
  if (clientApi === 'openai') {
    if (stream) sendSse(res, renderOpenAIStream(answer, model));
    else sendJson(res, 200, renderOpenAIJson(answer, model));
    return;
  }
  if (stream) sendSse(res, renderAnthropicStream(answer, model));
  else sendJson(res, 200, renderAnthropicJson(answer, model));
}

/**
 * Serve one client request: try its candidates best-first, failing over on any refusal, and
 * render the first answer that parses. Providers that fail three calls in a row are skipped
 * until every other candidate has been tried.
 */
async function serve(req: IncomingMessage, res: ServerResponse, clientApi: ClientApi, body: Json): Promise<void> {
  const stream = body.stream === true;
  const requires = requirementsFor(body);
  const jsonNeedsWire = clientApi === 'openai' && wantsJson(body);
  const candidates = candidatesFor(requires, jsonNeedsWire);
  if (candidates.length === 0) {
    return sendError(res, clientApi, 503, clientApi === 'anthropic' ? 'overloaded_error' : 'no_free_model', 'no free model on the directory meets this request');
  }
  const tried = new Set<string>();
  // Two passes: live providers first, then a last resort over providers marked spent or out.
  for (const includeDead of [false, true]) {
    for (const candidate of candidates) {
      const key = `${candidate.probe.provider.id}/${candidate.model.id}`;
      if (tried.has(key)) continue;
      const state = providerStateOf(candidate.probe.provider.id);
      if (state.dead !== includeDead) continue;
      tried.add(key);
      state.calls += 1;
      const result = await callCandidate(candidate, clientApi, body, stream);
      if (result.ok && result.answer !== null) {
        state.consecutiveFailures = 0;
        renderAnswer(clientApi, stream, result.answer, candidate.model.id, res);
        return;
      }
      state.consecutiveFailures += 1;
      if (state.consecutiveFailures >= DEAD_AFTER_FAILURES) state.dead = true;
    }
  }
  sendError(res, clientApi, 503, clientApi === 'anthropic' ? 'overloaded_error' : 'no_free_model', 'every free model that meets this request refused it');
}

async function handle(req: IncomingMessage, res: ServerResponse): Promise<void> {
  const url = new URL(req.url ?? '/', 'http://entry');
  if (req.method === 'GET' && url.pathname === '/v1/models') {
    models(req, res);
    return;
  }
  if (req.method === 'POST' && url.pathname === '/v1/chat/completions') {
    const body = await readJson(req);
    if (body === null) return sendError(res, 'openai', 400, 'invalid_request_error', 'the request body is not one JSON object');
    await serve(req, res, 'openai', body);
    return;
  }
  if (req.method === 'POST' && url.pathname === '/v1/messages') {
    const body = await readJson(req);
    if (body === null) return sendError(res, 'anthropic', 400, 'invalid_request_error', 'the request body is not one JSON object');
    await serve(req, res, 'anthropic', body);
    return;
  }
  sendError(res, url.pathname === '/v1/messages' ? 'anthropic' : 'openai', 404, 'not_found', `no route ${req.method ?? ''} ${url.pathname}`);
}

function listen(server: Server): Promise<void> {
  return new Promise((resolve, reject) => {
    server.once('error', reject);
    server.listen(0, '0.0.0.0', () => resolve());
  });
}

/** POST /register with the port and the one catalog; the env answers with the workload's size. */
async function register(port: number, report: CatalogReport): Promise<void> {
  const res = await fetch(`${ENV_URL}/register`, {
    method: 'POST',
    headers: { 'content-type': 'application/json' },
    body: JSON.stringify({ protocol_version: PROTOCOL_VERSION, port, catalog: report }),
    signal: AbortSignal.timeout(5000),
  });
  if (!res.ok) throw new Error(`POST ${ENV_URL}/register answered HTTP ${res.status}: ${(await res.text()).slice(0, 500)}`);
  const reply: unknown = await res.json();
  if (isRecord(reply) && typeof reply.requests === 'number') log(`registered: the workload holds ${reply.requests} requests`);
}

/** Poll GET /work every 100 ms; exit 0 the moment it says done. */
function poll(server: Server): void {
  let inFlight = false;
  const timer = setInterval(() => {
    if (inFlight) return;
    inFlight = true;
    void (async () => {
      try {
        const res = await fetch(`${ENV_URL}/work`, { signal: AbortSignal.timeout(2000) });
        if (res.status !== 200) return;
        const work: unknown = await res.json();
        if (isRecord(work) && work.done === true) {
          clearInterval(timer);
          server.close();
          process.exit(0);
        }
      } catch {
        // The env is not reachable yet: try again on the next tick.
      } finally {
        inFlight = false;
      }
    })();
  }, POLL_INTERVAL_MS);
}

async function start(): Promise<void> {
  await load();
  if (process.argv.includes('--list')) {
    // The same document the register body carries, byte for byte: print the one built object and stop
    // before any server starts. A process whose stdout is not flushed loses the tail, so wait for it.
    const text = `${JSON.stringify(catalog)}\n`;
    await new Promise<void>((resolve) => process.stdout.write(text, () => resolve()));
    process.exit(0);
  }
  if (ENV_URL === '') throw new Error('MAKEFUN_ENV_URL is not set');
  const server = createServer((req, res) => {
    void handle(req, res).catch((err: unknown) => {
      log((err as Error).stack ?? String(err));
      if (!res.headersSent) sendError(res, req.url?.startsWith('/v1/messages') === true ? 'anthropic' : 'openai', 500, 'internal_error', 'the server failed to answer');
    });
  });
  await listen(server);
  const address = server.address();
  if (address === null || typeof address === 'string') throw new Error('the server has no port');
  const port = address.port;
  // 8080 is the env's on one layout; the OS's ephemeral range never holds it, and the guard says so.
  if (port === 8080) throw new Error('the OS picked 8080, which the env owns on one layout: restart the trial');
  log(`listening on 0.0.0.0:${port} with ${catalog.providers.length} answered providers and ${servedModels().length} free models`);
  await register(port, catalog);
  process.on('SIGTERM', () => process.exit(0));
  process.on('SIGINT', () => process.exit(0));
  poll(server);
}

try {
  await start();
} catch (err) {
  log((err as Error).stack ?? String(err));
  process.exit(1);
}
