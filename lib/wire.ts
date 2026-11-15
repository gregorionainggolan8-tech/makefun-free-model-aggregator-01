// The three upstream wires (OpenAI, Anthropic, Gemini) and the two client wires.
//
// The entry buffers each provider's answer fully, parses it into one ParsedAnswer
// (text, tool calls, usage, stop family), then renders that answer in the wire the
// client used, JSON or SSE. Buffering is safe here: the trial's client reads the
// whole body with a 30 s timeout and scores the bytes, not the timing, and it lets
// the entry put the provider's usage in the exact place each client wire expects it
// (OpenAI's trailing usage chunk, Anthropic's message_start / message_delta).
//
// Everything here is a pure function of its arguments: no clock, no randomness, so
// the same body and the same provider answer give the same bytes on a replay. The
// two model listings use fixed timestamps for the same reason.

import { isRecord, type Json } from './providers.ts';

/** One model the entry serves, as GET /v1/models lists it. */
export interface ServedModel {
  id: string;
  provider: string;
  display_name: string;
}

/** A fixed epoch second for every listed model's `created`: nothing in the reply reads the clock. */
const FIXED_OPENAI_CREATED = 1767225600;
/** A fixed instant for every listed model's `created_at`: nothing in the reply reads the clock. */
const FIXED_ANTHROPIC_CREATED_AT = '2026-01-01T00:00:00Z';
/** Fixed ids for the answers this entry renders: nothing in the reply reads the clock. */
const FIXED_CHAT_ID = 'chatcmpl-aggregator';
const FIXED_MESSAGE_ID = 'msg_aggregator';

/** GET /v1/models on an OpenAI-shaped request: the models this entry serves, in OpenAI's list shape. */
export function openAIModelList(models: readonly ServedModel[]): Json {
  return {
    object: 'list',
    data: models.map((m) => ({ id: m.id, object: 'model', created: FIXED_OPENAI_CREATED, owned_by: m.provider })),
  };
}

/** GET /v1/models on an Anthropic-shaped request (an x-api-key header): Anthropic's model list shape. */
export function anthropicModelList(models: readonly ServedModel[]): Json {
  return {
    data: models.map((m) => ({ type: 'model', id: m.id, display_name: m.display_name, created_at: FIXED_ANTHROPIC_CREATED_AT })),
    has_more: false,
  };
}

/** An error in Anthropic's shape (the Messages API's error envelope). */
export function anthropicError(type: string, message: string): Json {
  return { type: 'error', error: { type, message } };
}

/** An error in OpenAI's shape (Chat Completions). */
export function openAIError(message: string, type: string, code: string | null): Json {
  return { error: { message, type, code } };
}

/** The text of a string or of a block array's text blocks; '' when there is none. */
export function textOf(content: unknown): string {
  if (typeof content === 'string') return content;
  if (!Array.isArray(content)) return '';
  return content.flatMap((b) => (isRecord(b) && b.type === 'text' && typeof b.text === 'string' ? [b.text] : [])).join('\n');
}

/** A data: URL split into its media type and base64 payload; null when it is not one. */
function dataUrlParts(url: string): { media: string; data: string } | null {
  const m = /^data:([^;,]*)(?:;base64)?,(.*)$/s.exec(url);
  if (m === null) return null;
  return { media: m[1] === '' ? 'image/png' : m[1], data: m[2] };
}

/** One Anthropic content block as an OpenAI content part: text or an inline image; null for the rest. */
function partOf(block: Json): Json | null {
  if (block.type === 'text' && typeof block.text === 'string') return { type: 'text', text: block.text };
  if (block.type === 'image' && isRecord(block.source)) {
    const source = block.source;
    if (source.type === 'base64' && typeof source.data === 'string') {
      const media = typeof source.media_type === 'string' ? source.media_type : 'image/png';
      return { type: 'image_url', image_url: { url: `data:${media};base64,${source.data}` } };
    }
    if (source.type === 'url' && typeof source.url === 'string') return { type: 'image_url', image_url: { url: source.url } };
  }
  return null;
}

/** A tool_result block's content as text: its text blocks, or its JSON when it holds something else. */
function toolResultText(content: unknown): string {
  if (typeof content === 'string') return content;
  if (Array.isArray(content)) {
    const text = textOf(content);
    return text !== '' ? text : JSON.stringify(content);
  }
  return content === undefined ? '' : JSON.stringify(content);
}

/** A tool call's arguments as an object: the provider's JSON string, or {} when it is not one. */
function argumentsOf(value: unknown): Json {
  if (isRecord(value)) return value;
  if (typeof value === 'string' && value.trim() !== '') {
    try {
      const parsed: unknown = JSON.parse(value);
      if (isRecord(parsed)) return parsed;
    } catch {
      // Not JSON: an empty input.
    }
  }
  return {};
}

/** A stop reason's family: 0 stop, 1 length, 2 tool use, 3 refusal. Unknown reasons read as stop. */
export function stopFamilyOf(reason: unknown): number {
  if (reason === 'length' || reason === 'max_tokens' || reason === 'MAX_TOKENS') return 1;
  if (reason === 'tool_calls' || reason === 'function_call' || reason === 'tool_use') return 2;
  if (reason === 'content_filter' || reason === 'refusal' || reason === 'SAFETY') return 3;
  return 0;
}

/** One stop family as an OpenAI finish_reason. */
export function openAIStopReason(family: number): string {
  if (family === 1) return 'length';
  if (family === 2) return 'tool_calls';
  if (family === 3) return 'content_filter';
  return 'stop';
}

/** One stop family as an Anthropic stop_reason. */
export function anthropicStopReason(family: number): string {
  if (family === 1) return 'max_tokens';
  if (family === 2) return 'tool_use';
  if (family === 3) return 'refusal';
  return 'end_turn';
}

// ---------------------------------------------------------------------------
// Client -> upstream request builders
// ---------------------------------------------------------------------------

/**
 * An Anthropic Messages request as an OpenAI Chat Completions request that names `model`: the system
 * prompt becomes a system message, image blocks become image_url data URLs, tool_use blocks become
 * tool_calls, tool_result blocks become tool messages, tools and tool_choice are rewritten, and the
 * output cap, temperature, top_p, stop sequences and streaming pass through.
 */
export function anthropicToOpenAI(body: Json, model: string): Json {
  const messages: Json[] = [];
  const system = textOf(body.system);
  if (system !== '') messages.push({ role: 'system', content: system });
  const source = Array.isArray(body.messages) ? body.messages : [];
  for (const raw of source) {
    if (!isRecord(raw)) continue;
    const assistant = raw.role === 'assistant';
    const content = raw.content;
    if (typeof content === 'string') {
      messages.push({ role: assistant ? 'assistant' : 'user', content });
      continue;
    }
    if (!Array.isArray(content)) continue;
    if (assistant) {
      const toolCalls: Json[] = [];
      for (const block of content) {
        if (!isRecord(block) || block.type !== 'tool_use') continue;
        toolCalls.push({
          id: typeof block.id === 'string' ? block.id : '',
          type: 'function',
          function: {
            name: typeof block.name === 'string' ? block.name : '',
            arguments: JSON.stringify(isRecord(block.input) ? block.input : {}),
          },
        });
      }
      const text = textOf(content);
      messages.push({
        role: 'assistant',
        content: text !== '' ? text : null,
        ...(toolCalls.length > 0 ? { tool_calls: toolCalls } : {}),
      });
      continue;
    }
    const parts: Json[] = [];
    for (const block of content) {
      if (!isRecord(block)) continue;
      if (block.type === 'tool_result') {
        messages.push({
          role: 'tool',
          tool_call_id: typeof block.tool_use_id === 'string' ? block.tool_use_id : '',
          content: toolResultText(block.content),
        });
        continue;
      }
      const part = partOf(block);
      if (part !== null) parts.push(part);
    }
    if (parts.length > 0) messages.push({ role: 'user', content: parts });
  }

  const out: Json = { model, messages };
  if (typeof body.max_tokens === 'number') out.max_tokens = body.max_tokens;
  if (typeof body.temperature === 'number') out.temperature = body.temperature;
  if (typeof body.top_p === 'number') out.top_p = body.top_p;
  if (Array.isArray(body.stop_sequences) && body.stop_sequences.length > 0) out.stop = body.stop_sequences;
  if (Array.isArray(body.tools)) {
    const tools = body.tools.flatMap((raw) => {
      if (!isRecord(raw) || typeof raw.name !== 'string') return [];
      return [
        {
          type: 'function',
          function: {
            name: raw.name,
            description: typeof raw.description === 'string' ? raw.description : '',
            parameters: isRecord(raw.input_schema) ? raw.input_schema : { type: 'object', properties: {} },
          },
        },
      ];
    });
    if (tools.length > 0) out.tools = tools;
  }
  const choice = body.tool_choice;
  if (isRecord(choice)) {
    if (choice.type === 'auto') out.tool_choice = 'auto';
    else if (choice.type === 'any') out.tool_choice = 'required';
    else if (choice.type === 'tool' && typeof choice.name === 'string') out.tool_choice = { type: 'function', function: { name: choice.name } };
  }
  if (body.stream === true) {
    out.stream = true;
    out.stream_options = { include_usage: true };
  }
  return out;
}

/** One OpenAI content part as an Anthropic content block: text or an inline image; null for the rest. */
function anthropicBlockOf(part: Json): Json | null {
  if (part.type === 'text' && typeof part.text === 'string') return { type: 'text', text: part.text };
  if (part.type === 'image_url' && isRecord(part.image_url) && typeof part.image_url.url === 'string') {
    const url = part.image_url.url;
    const data = dataUrlParts(url);
    if (data !== null) return { type: 'image', source: { type: 'base64', media_type: data.media, data: data.data } };
    return { type: 'image', source: { type: 'url', url } };
  }
  return null;
}

/**
 * An OpenAI Chat Completions request as an Anthropic Messages request that names `model`: system
 * messages become the system string, image_url parts become image blocks, tool_calls become tool_use
 * blocks, tool messages become tool_result blocks, tools and tool_choice are rewritten, and the output
 * cap, temperature, top_p, stop sequences and streaming pass through.
 */
export function openAIToAnthropic(body: Json, model: string): Json {
  const systemParts: string[] = [];
  const messages: Json[] = [];
  const source = Array.isArray(body.messages) ? body.messages : [];
  for (const raw of source) {
    if (!isRecord(raw)) continue;
    const role = typeof raw.role === 'string' ? raw.role : 'user';
    const content = raw.content;
    if (role === 'system') {
      const text = textOf(content);
      if (text !== '') systemParts.push(text);
      continue;
    }
    if (role === 'assistant') {
      const blocks: Json[] = [];
      const text = textOf(content);
      if (text !== '') blocks.push({ type: 'text', text });
      const calls = Array.isArray(raw.tool_calls) ? raw.tool_calls : [];
      for (const call of calls) {
        if (!isRecord(call) || !isRecord(call.function)) continue;
        blocks.push({
          type: 'tool_use',
          id: typeof call.id === 'string' ? call.id : '',
          name: typeof call.function.name === 'string' ? call.function.name : '',
          input: argumentsOf(call.function.arguments),
        });
      }
      if (blocks.length > 0) messages.push({ role: 'assistant', content: blocks });
      continue;
    }
    if (role === 'tool') {
      messages.push({
        role: 'user',
        content: [{ type: 'tool_result', tool_use_id: typeof raw.tool_call_id === 'string' ? raw.tool_call_id : '', content: typeof content === 'string' ? content : textOf(content) }],
      });
      continue;
    }
    if (typeof content === 'string') {
      messages.push({ role: 'user', content });
      continue;
    }
    if (!Array.isArray(content)) continue;
    const blocks: Json[] = [];
    for (const part of content) {
      if (!isRecord(part)) continue;
      const block = anthropicBlockOf(part);
      if (block !== null) blocks.push(block);
    }
    if (blocks.length > 0) messages.push({ role: 'user', content: blocks });
  }

  const out: Json = { model, messages };
  if (systemParts.length > 0) out.system = systemParts.join('\n');
  if (typeof body.max_tokens === 'number') out.max_tokens = body.max_tokens;
  else if (typeof body.max_completion_tokens === 'number') out.max_tokens = body.max_completion_tokens;
  if (typeof body.temperature === 'number') out.temperature = body.temperature;
  if (typeof body.top_p === 'number') out.top_p = body.top_p;
  const stop = body.stop;
  if (typeof stop === 'string') out.stop_sequences = [stop];
  else if (Array.isArray(stop) && stop.length > 0) out.stop_sequences = stop.filter((s) => typeof s === 'string');
  if (Array.isArray(body.tools)) {
    const tools = body.tools.flatMap((raw) => {
      if (!isRecord(raw) || !isRecord(raw.function) || typeof raw.function.name !== 'string') return [];
      return [
        {
          name: raw.function.name,
          description: typeof raw.function.description === 'string' ? raw.function.description : '',
          input_schema: isRecord(raw.function.parameters) ? raw.function.parameters : { type: 'object', properties: {} },
        },
      ];
    });
    if (tools.length > 0) out.tools = tools;
  }
  const choice = body.tool_choice;
  if (choice === 'auto') out.tool_choice = { type: 'auto' };
  else if (choice === 'required' || choice === 'any') out.tool_choice = { type: 'any' };
  else if (choice === 'none') out.tool_choice = { type: 'none' };
  else if (isRecord(choice) && choice.type === 'function' && isRecord(choice.function) && typeof choice.function.name === 'string') {
    out.tool_choice = { type: 'tool', name: choice.function.name };
  }
  if (body.stream === true) out.stream = true;
  return out;
}

/** Whether an OpenAI body asks for JSON mode (a response_format the Gemini wire can also carry). */
export function wantsJson(body: Json): boolean {
  if (!isRecord(body.response_format)) return false;
  return body.response_format.type === 'json_object' || body.response_format.type === 'json_schema';
}

/** One Gemini function declaration from an OpenAI function tool or an Anthropic tool. */
function geminiDeclaration(name: string, description: string, parameters: unknown): Json {
  return { name, description, parameters: isRecord(parameters) ? parameters : { type: 'object', properties: {} } };
}

/**
 * Either client wire as a Gemini :generateContent / :streamGenerateContent body that names `model`:
 * the system prompt becomes systemInstruction, messages become contents (user / model / function
 * roles), images become inlineData or fileData, tools become functionDeclarations, and the output cap,
 * temperature, top_p, stop sequences and JSON mode land in generationConfig, which is what the Gemini
 * wire echoes back.
 */
export function toGemini(body: Json, clientApi: 'openai' | 'anthropic', model: string): Json {
  const contents: Json[] = [];
  const declarations: Json[] = [];
  let systemText = '';
  let maxTokens: number | undefined;
  let temperature: number | undefined;
  let topP: number | undefined;
  let stops: string[] = [];
  let jsonMode = false;

  if (clientApi === 'openai') {
    const source = Array.isArray(body.messages) ? body.messages : [];
    for (const raw of source) {
      if (!isRecord(raw)) continue;
      const role = typeof raw.role === 'string' ? raw.role : 'user';
      const content = raw.content;
      if (role === 'system') {
        const text = textOf(content);
        if (text !== '') systemText = systemText === '' ? text : `${systemText}\n${text}`;
        continue;
      }
      if (role === 'assistant') {
        const parts: Json[] = [];
        const text = textOf(content);
        if (text !== '') parts.push({ text });
        const calls = Array.isArray(raw.tool_calls) ? raw.tool_calls : [];
        for (const call of calls) {
          if (!isRecord(call) || !isRecord(call.function)) continue;
          parts.push({ functionCall: { name: typeof call.function.name === 'string' ? call.function.name : '', args: argumentsOf(call.function.arguments) } });
        }
        if (parts.length > 0) contents.push({ role: 'model', parts });
        continue;
      }
      if (role === 'tool') {
        contents.push({
          role: 'function',
          parts: [{ functionResponse: { name: '', response: { content: typeof content === 'string' ? content : textOf(content) } } }],
        });
        continue;
      }
      const parts: Json[] = [];
      if (typeof content === 'string') {
        if (content !== '') parts.push({ text: content });
      } else if (Array.isArray(content)) {
        for (const part of content) {
          if (!isRecord(part)) continue;
          if (part.type === 'text' && typeof part.text === 'string') parts.push({ text: part.text });
          else if (part.type === 'image_url' && isRecord(part.image_url) && typeof part.image_url.url === 'string') {
            const data = dataUrlParts(part.image_url.url);
            if (data !== null) parts.push({ inlineData: { mimeType: data.media, data: data.data } });
            else parts.push({ fileData: { fileUri: part.image_url.url } });
          }
        }
      }
      if (parts.length > 0) contents.push({ role: 'user', parts });
    }
    if (typeof body.max_tokens === 'number') maxTokens = body.max_tokens;
    else if (typeof body.max_completion_tokens === 'number') maxTokens = body.max_completion_tokens;
    if (typeof body.temperature === 'number') temperature = body.temperature;
    if (typeof body.top_p === 'number') topP = body.top_p;
    if (typeof body.stop === 'string') stops = [body.stop];
    else if (Array.isArray(body.stop)) stops = body.stop.filter((s): s is string => typeof s === 'string');
    if (Array.isArray(body.tools)) {
      for (const raw of body.tools) {
        if (!isRecord(raw) || !isRecord(raw.function) || typeof raw.function.name !== 'string') continue;
        declarations.push(geminiDeclaration(raw.function.name, typeof raw.function.description === 'string' ? raw.function.description : '', raw.function.parameters));
      }
    }
    jsonMode = wantsJson(body);
  } else {
    systemText = textOf(body.system);
    const source = Array.isArray(body.messages) ? body.messages : [];
    for (const raw of source) {
      if (!isRecord(raw)) continue;
      const assistant = raw.role === 'assistant';
      const content = raw.content;
      if (typeof content === 'string') {
        if (content !== '') contents.push({ role: assistant ? 'model' : 'user', parts: [{ text: content }] });
        continue;
      }
      if (!Array.isArray(content)) continue;
      if (assistant) {
        const parts: Json[] = [];
        for (const block of content) {
          if (!isRecord(block)) continue;
          if (block.type === 'text' && typeof block.text === 'string') parts.push({ text: block.text });
          else if (block.type === 'tool_use') {
            parts.push({ functionCall: { name: typeof block.name === 'string' ? block.name : '', args: isRecord(block.input) ? block.input : {} } });
          }
        }
        if (parts.length > 0) contents.push({ role: 'model', parts });
        continue;
      }
      const parts: Json[] = [];
      let isToolResult = false;
      for (const block of content) {
        if (!isRecord(block)) continue;
        if (block.type === 'text' && typeof block.text === 'string') parts.push({ text: block.text });
        else if (block.type === 'image' && isRecord(block.source)) {
          if (block.source.type === 'base64' && typeof block.source.data === 'string') {
            parts.push({ inlineData: { mimeType: typeof block.source.media_type === 'string' ? block.source.media_type : 'image/png', data: block.source.data } });
          } else if (block.source.type === 'url' && typeof block.source.url === 'string') {
            parts.push({ fileData: { fileUri: block.source.url } });
          }
        } else if (block.type === 'tool_result') {
          isToolResult = true;
          parts.push({ functionResponse: { name: '', response: { content: toolResultText(block.content) } } });
        }
      }
      if (parts.length > 0) contents.push({ role: isToolResult ? 'function' : 'user', parts });
    }
    if (typeof body.max_tokens === 'number') maxTokens = body.max_tokens;
    if (typeof body.temperature === 'number') temperature = body.temperature;
    if (typeof body.top_p === 'number') topP = body.top_p;
    if (Array.isArray(body.stop_sequences)) stops = body.stop_sequences.filter((s): s is string => typeof s === 'string');
    if (Array.isArray(body.tools)) {
      for (const raw of body.tools) {
        if (!isRecord(raw) || typeof raw.name !== 'string') continue;
        declarations.push(geminiDeclaration(raw.name, typeof raw.description === 'string' ? raw.description : '', raw.input_schema));
      }
    }
    jsonMode = false;
  }

  const out: Json = { contents };
  if (systemText !== '') out.systemInstruction = { parts: [{ text: systemText }] };
  if (declarations.length > 0) out.tools = [{ functionDeclarations: declarations }];
  const config: Json = {};
  if (maxTokens !== undefined) config.maxOutputTokens = maxTokens;
  if (temperature !== undefined) config.temperature = temperature;
  if (topP !== undefined) config.topP = topP;
  if (stops.length > 0) config.stopSequences = stops;
  if (jsonMode) config.responseMimeType = 'application/json';
  if (Object.keys(config).length > 0) out.generationConfig = config;
  void model;
  return out;
}

// ---------------------------------------------------------------------------
// Upstream answers, parsed into one shape
// ---------------------------------------------------------------------------

/** One tool call a provider returned, arguments kept as a JSON string. */
export interface ParsedToolCall {
  id: string;
  name: string;
  arguments: string;
}

/** A provider's answer, whichever wire carried it. */
export interface ParsedAnswer {
  text: string;
  toolCalls: ParsedToolCall[];
  inputTokens: number | null;
  outputTokens: number | null;
  stopFamily: number;
}

function numberOrNull(value: unknown): number | null {
  return typeof value === 'number' && Number.isFinite(value) ? value : null;
}

/** The text of a Chat Completions message: a string, or its text parts joined. */
function openAIText(content: unknown): string {
  if (typeof content === 'string') return content;
  if (!Array.isArray(content)) return '';
  return content.flatMap((b) => (isRecord(b) && b.type === 'text' && typeof b.text === 'string' ? [b.text] : [])).join('');
}

/** One Chat Completions answer (JSON) as a ParsedAnswer; null when it is not one. */
export function parseOpenAIJson(body: unknown): ParsedAnswer | null {
  if (!isRecord(body)) return null;
  const choices = Array.isArray(body.choices) ? body.choices : [];
  const choice = isRecord(choices[0]) ? choices[0] : null;
  if (choice === null) return null;
  const message = isRecord(choice.message) ? choice.message : {};
  const toolCalls: ParsedToolCall[] = [];
  const calls = Array.isArray(message.tool_calls) ? message.tool_calls : [];
  for (const call of calls) {
    if (!isRecord(call) || !isRecord(call.function)) continue;
    toolCalls.push({
      id: typeof call.id === 'string' ? call.id : '',
      name: typeof call.function.name === 'string' ? call.function.name : '',
      arguments: typeof call.function.arguments === 'string' ? call.function.arguments : JSON.stringify(argumentsOf(call.function.arguments)),
    });
  }
  const usage = isRecord(body.usage) ? body.usage : {};
  return {
    text: openAIText(message.content),
    toolCalls,
    inputTokens: numberOrNull(usage.prompt_tokens),
    outputTokens: numberOrNull(usage.completion_tokens),
    stopFamily: stopFamilyOf(choice.finish_reason),
  };
}

/** The SSE frames of a text body: [{event, data}] with data: lines joined. */
export function parseSseFrames(text: string): { event: string | null; data: string }[] {
  const frames: { event: string | null; data: string }[] = [];
  const normalized = text.replace(/\r\n/g, '\n');
  for (const frame of normalized.split('\n\n')) {
    if (frame.trim() === '') continue;
    let event: string | null = null;
    const data: string[] = [];
    for (const line of frame.split('\n')) {
      if (line.startsWith('event:')) event = line.slice('event:'.length).trim();
      else if (line.startsWith('data:')) data.push(line.slice('data:'.length).replace(/^ /, ''));
    }
    if (data.length > 0) frames.push({ event, data: data.join('\n') });
  }
  return frames;
}

/** One Chat Completions event stream as a ParsedAnswer; null when it carries no finish chunk. */
export function parseOpenAIStream(text: string): ParsedAnswer | null {
  let out = '';
  let inputTokens: number | null = null;
  let outputTokens: number | null = null;
  let stopFamily: number | null = null;
  const callsByIndex = new Map<number, ParsedToolCall>();
  for (const frame of parseSseFrames(text)) {
    const payload = frame.data.trim();
    if (payload === '[DONE]') continue;
    let chunk: unknown;
    try {
      chunk = JSON.parse(payload);
    } catch {
      continue;
    }
    if (!isRecord(chunk)) continue;
    if (isRecord(chunk.usage)) {
      inputTokens = numberOrNull(chunk.usage.prompt_tokens) ?? inputTokens;
      outputTokens = numberOrNull(chunk.usage.completion_tokens) ?? outputTokens;
    }
    const choices = Array.isArray(chunk.choices) ? chunk.choices : [];
    const choice = isRecord(choices[0]) ? choices[0] : null;
    if (choice === null) continue;
    const delta = isRecord(choice.delta) ? choice.delta : {};
    if (typeof delta.content === 'string') out += delta.content;
    const calls = Array.isArray(delta.tool_calls) ? delta.tool_calls : [];
    for (const call of calls) {
      if (!isRecord(call)) continue;
      const index = typeof call.index === 'number' ? call.index : callsByIndex.size;
      let entry = callsByIndex.get(index);
      if (entry === undefined) {
        const fn = isRecord(call.function) ? call.function : {};
        entry = {
          id: typeof call.id === 'string' && call.id !== '' ? call.id : `call_${index}`,
          name: typeof fn.name === 'string' ? fn.name : '',
          arguments: '',
        };
        callsByIndex.set(index, entry);
      }
      const fn = isRecord(call.function) ? call.function : {};
      if (typeof fn.arguments === 'string') entry.arguments += fn.arguments;
    }
    if (typeof choice.finish_reason === 'string') stopFamily = stopFamilyOf(choice.finish_reason);
  }
  if (stopFamily === null) return null;
  return { text: out, toolCalls: [...callsByIndex.values()], inputTokens, outputTokens, stopFamily };
}

/** One Anthropic Messages answer (JSON) as a ParsedAnswer; null when it is not one. */
export function parseAnthropicJson(body: unknown): ParsedAnswer | null {
  if (!isRecord(body) || body.type !== 'message' || !Array.isArray(body.content)) return null;
  let text = '';
  const toolCalls: ParsedToolCall[] = [];
  for (const block of body.content) {
    if (!isRecord(block)) continue;
    if (block.type === 'text' && typeof block.text === 'string') text += block.text;
    else if (block.type === 'tool_use') {
      toolCalls.push({
        id: typeof block.id === 'string' ? block.id : '',
        name: typeof block.name === 'string' ? block.name : '',
        arguments: JSON.stringify(isRecord(block.input) ? block.input : {}),
      });
    }
  }
  const usage = isRecord(body.usage) ? body.usage : {};
  return {
    text,
    toolCalls,
    inputTokens: numberOrNull(usage.input_tokens),
    outputTokens: numberOrNull(usage.output_tokens),
    stopFamily: stopFamilyOf(body.stop_reason),
  };
}

/** One Anthropic event stream as a ParsedAnswer; null when it never reached message_stop. */
export function parseAnthropicStream(text: string): ParsedAnswer | null {
  let out = '';
  let inputTokens: number | null = null;
  let outputTokens: number | null = null;
  let stopFamily: number | null = null;
  let stopped = false;
  const callsByIndex = new Map<number, ParsedToolCall>();
  for (const frame of parseSseFrames(text)) {
    if (frame.event === null) continue;
    let data: unknown;
    try {
      data = JSON.parse(frame.data);
    } catch {
      continue;
    }
    if (!isRecord(data)) continue;
    switch (frame.event) {
      case 'message_start': {
        if (isRecord(data.message) && isRecord(data.message.usage)) inputTokens = numberOrNull(data.message.usage.input_tokens) ?? inputTokens;
        break;
      }
      case 'content_block_start': {
        const index = numberOrNull(data.index);
        if (index === null) break;
        const block = isRecord(data.content_block) ? data.content_block : {};
        if (block.type === 'tool_use') {
          callsByIndex.set(index, {
            id: typeof block.id === 'string' ? block.id : '',
            name: typeof block.name === 'string' ? block.name : '',
            arguments: '',
          });
        }
        break;
      }
      case 'content_block_delta': {
        const index = numberOrNull(data.index);
        if (index === null) break;
        const delta = isRecord(data.delta) ? data.delta : {};
        if (delta.type === 'text_delta' && typeof delta.text === 'string') out += delta.text;
        else if (delta.type === 'input_json_delta' && typeof delta.partial_json === 'string') {
          const entry = callsByIndex.get(index);
          if (entry !== undefined) entry.arguments += delta.partial_json;
        }
        break;
      }
      case 'message_delta': {
        const delta = isRecord(data.delta) ? data.delta : {};
        if (typeof delta.stop_reason === 'string') stopFamily = stopFamilyOf(delta.stop_reason);
        if (isRecord(data.usage)) outputTokens = numberOrNull(data.usage.output_tokens) ?? outputTokens;
        break;
      }
      case 'message_stop': {
        stopped = true;
        break;
      }
      default:
        break;
    }
  }
  if (!stopped) return null;
  return { text: out, toolCalls: [...callsByIndex.values()], inputTokens, outputTokens, stopFamily: stopFamily ?? 0 };
}

/** One Gemini :generateContent answer (JSON) as a ParsedAnswer; null when it is not one. */
export function parseGeminiJson(body: unknown): ParsedAnswer | null {
  if (!isRecord(body) || !Array.isArray(body.candidates) || body.candidates.length === 0) return null;
  const candidate = isRecord(body.candidates[0]) ? body.candidates[0] : null;
  if (candidate === null) return null;
  const content = isRecord(candidate.content) ? candidate.content : {};
  const parts = Array.isArray(content.parts) ? content.parts : [];
  let text = '';
  const toolCalls: ParsedToolCall[] = [];
  for (const part of parts) {
    if (!isRecord(part)) continue;
    if (typeof part.text === 'string') text += part.text;
    else if (isRecord(part.functionCall)) {
      toolCalls.push({
        id: `call_${toolCalls.length}`,
        name: typeof part.functionCall.name === 'string' ? part.functionCall.name : '',
        arguments: JSON.stringify(isRecord(part.functionCall.args) ? part.functionCall.args : {}),
      });
    }
  }
  const usage = isRecord(body.usageMetadata) ? body.usageMetadata : {};
  return {
    text,
    toolCalls,
    inputTokens: numberOrNull(usage.promptTokenCount),
    outputTokens: numberOrNull(usage.candidatesTokenCount),
    stopFamily: stopFamilyOf(candidate.finishReason),
  };
}

/** One Gemini :streamGenerateContent event stream as a ParsedAnswer; null when it never finished. */
export function parseGeminiStream(text: string): ParsedAnswer | null {
  let out = '';
  let inputTokens: number | null = null;
  let outputTokens: number | null = null;
  let stopFamily: number | null = null;
  const toolCalls: ParsedToolCall[] = [];
  for (const frame of parseSseFrames(text)) {
    let chunk: unknown;
    try {
      chunk = JSON.parse(frame.data);
    } catch {
      continue;
    }
    const parsed = parseGeminiJson(chunk);
    if (parsed === null) continue;
    out += parsed.text;
    toolCalls.push(...parsed.toolCalls);
    if (parsed.inputTokens !== null) inputTokens = parsed.inputTokens;
    if (parsed.outputTokens !== null) outputTokens = parsed.outputTokens;
    if (isRecord(chunk) && Array.isArray(chunk.candidates) && isRecord(chunk.candidates[0]) && typeof chunk.candidates[0].finishReason === 'string') {
      stopFamily = stopFamilyOf(chunk.candidates[0].finishReason);
    }
  }
  if (stopFamily === null) return null;
  return { text: out, toolCalls, inputTokens, outputTokens, stopFamily };
}

// ---------------------------------------------------------------------------
// Parsed answers, rendered in the client's wire
// ---------------------------------------------------------------------------

/** One server-sent event: `event: <name>` when named, then the JSON data line, then the blank line. */
export function sseEvent(event: string | null, data: unknown): string {
  return `${event === null ? '' : `event: ${event}\n`}data: ${JSON.stringify(data)}\n\n`;
}

/** Split a text into the short deltas a stream carries; '' stays one empty piece. */
function splitText(text: string): string[] {
  const pieces: string[] = [];
  for (let at = 0; at < text.length; at += 13) pieces.push(text.slice(at, at + 13));
  return pieces.length === 0 ? [''] : pieces;
}

function numberOrZero(value: number | null): number {
  return value === null ? 0 : value;
}

/** A ParsedAnswer as a Chat Completions answer (JSON), naming the model that served it. */
export function renderOpenAIJson(answer: ParsedAnswer, model: string): Json {
  const message: Json = { role: 'assistant', content: answer.text };
  if (answer.toolCalls.length > 0) {
    message.tool_calls = answer.toolCalls.map((c) => ({ id: c.id, type: 'function', function: { name: c.name, arguments: c.arguments } }));
  }
  return {
    id: FIXED_CHAT_ID,
    object: 'chat.completion',
    created: FIXED_OPENAI_CREATED,
    model,
    choices: [{ index: 0, message, finish_reason: openAIStopReason(answer.toolCalls.length > 0 ? 2 : answer.stopFamily), logprobs: null }],
    usage: {
      prompt_tokens: numberOrZero(answer.inputTokens),
      completion_tokens: numberOrZero(answer.outputTokens),
      total_tokens: numberOrZero(answer.inputTokens) + numberOrZero(answer.outputTokens),
    },
  };
}

/** A ParsedAnswer as a Chat Completions event stream, naming the model that served it. */
export function renderOpenAIStream(answer: ParsedAnswer, model: string): string {
  const finish = openAIStopReason(answer.toolCalls.length > 0 ? 2 : answer.stopFamily);
  const chunk = (delta: Json, finishReason: string | null): Json => ({
    id: FIXED_CHAT_ID,
    object: 'chat.completion.chunk',
    created: FIXED_OPENAI_CREATED,
    model,
    choices: [{ index: 0, delta, finish_reason: finishReason }],
  });
  let out = sseEvent(null, chunk({ role: 'assistant', content: '' }, null));
  for (const piece of splitText(answer.text)) {
    if (piece === '') continue;
    out += sseEvent(null, chunk({ content: piece }, null));
  }
  for (const call of answer.toolCalls) {
    out += sseEvent(
      null,
      chunk({ tool_calls: [{ index: 0, id: call.id, type: 'function', function: { name: call.name, arguments: call.arguments } }] }, null),
    );
  }
  out += sseEvent(null, chunk({}, finish));
  out += sseEvent(null, {
    id: FIXED_CHAT_ID,
    object: 'chat.completion.chunk',
    created: FIXED_OPENAI_CREATED,
    model,
    choices: [],
    usage: {
      prompt_tokens: numberOrZero(answer.inputTokens),
      completion_tokens: numberOrZero(answer.outputTokens),
      total_tokens: numberOrZero(answer.inputTokens) + numberOrZero(answer.outputTokens),
    },
  });
  out += 'data: [DONE]\n\n';
  return out;
}

/** A ParsedAnswer as an Anthropic Messages object, naming the model that served it. */
export function renderAnthropicJson(answer: ParsedAnswer, model: string): Json {
  const content: Json[] = [];
  if (answer.text !== '' || answer.toolCalls.length === 0) content.push({ type: 'text', text: answer.text });
  for (const call of answer.toolCalls) {
    content.push({ type: 'tool_use', id: call.id, name: call.name, input: argumentsOf(call.arguments) });
  }
  return {
    id: FIXED_MESSAGE_ID,
    type: 'message',
    role: 'assistant',
    model,
    content,
    stop_reason: anthropicStopReason(answer.toolCalls.length > 0 ? 2 : answer.stopFamily),
    stop_sequence: null,
    usage: { input_tokens: numberOrZero(answer.inputTokens), output_tokens: numberOrZero(answer.outputTokens) },
  };
}

/** A ParsedAnswer as an Anthropic event stream, naming the model that served it. */
export function renderAnthropicStream(answer: ParsedAnswer, model: string): string {
  const stopReason = anthropicStopReason(answer.toolCalls.length > 0 ? 2 : answer.stopFamily);
  let out = sseEvent('message_start', {
    type: 'message_start',
    message: {
      id: FIXED_MESSAGE_ID,
      type: 'message',
      role: 'assistant',
      model,
      content: [],
      stop_reason: null,
      stop_sequence: null,
      usage: { input_tokens: numberOrZero(answer.inputTokens), output_tokens: 0 },
    },
  });
  let index = 0;
  if (answer.text !== '' || answer.toolCalls.length === 0) {
    out += sseEvent('content_block_start', { type: 'content_block_start', index, content_block: { type: 'text', text: '' } });
    for (const piece of splitText(answer.text)) {
      if (piece === '') continue;
      out += sseEvent('content_block_delta', { type: 'content_block_delta', index, delta: { type: 'text_delta', text: piece } });
    }
    out += sseEvent('content_block_stop', { type: 'content_block_stop', index });
    index += 1;
  }
  for (const call of answer.toolCalls) {
    const at = index;
    index += 1;
    out += sseEvent('content_block_start', { type: 'content_block_start', index: at, content_block: { type: 'tool_use', id: call.id, name: call.name, input: {} } });
    if (call.arguments !== '') {
      out += sseEvent('content_block_delta', { type: 'content_block_delta', index: at, delta: { type: 'input_json_delta', partial_json: call.arguments } });
    }
    out += sseEvent('content_block_stop', { type: 'content_block_stop', index: at });
  }
  out += sseEvent('message_delta', { type: 'message_delta', delta: { stop_reason: stopReason, stop_sequence: null }, usage: { output_tokens: numberOrZero(answer.outputTokens) } });
  out += sseEvent('message_stop', { type: 'message_stop' });
  return out;
}
