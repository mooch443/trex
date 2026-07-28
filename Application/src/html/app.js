"use strict";

(() => {
  const MAGIC = 0x58525442;
  const VERSION = 1;
  const MSG = { SNAPSHOT: 1, DELTA: 2, IMAGE: 3, CLIENTS: 4, INPUT: 5, RESYNC: 6, HEARTBEAT: 7, ERROR: 8 };
  const INPUT = { MOVE: 1, BUTTON: 2, WHEEL: 3, KEY: 4, TEXT: 5, BLUR: 6 };
  const PRIMITIVE = { RECT: 1, CIRCLE: 2, TEXT: 3, POLYGON: 4, VERTICES: 5, IMAGE: 6 };
  const STYLE = { BOLD: 1, ITALIC: 2, UNDERLINE: 4, STRIKE: 8, MONO: 16, SYMBOLS: 32 };

  const viewport = document.querySelector("#viewport");
  const canvas = document.querySelector("#gui");
  const inputSink = document.querySelector("#input-sink");
  const status = document.querySelector("#status");
  const clients = document.querySelector("#clients");
  const context = canvas.getContext("2d", { alpha: false, desynchronized: true });
  const decoder = new TextDecoder();
  const encoder = new TextEncoder();

  const state = {
    sequence: 0n,
    width: 1280,
    height: 720,
    primitives: new Map(),
    order: [],
    resources: new Map(),
    tintCache: new Map(),
    dirty: null,
    fullRedraw: true,
    animation: 0,
    socket: null,
    retry: 250,
    heartbeat: 0
  };

  class Reader {
    constructor(buffer) {
      this.view = new DataView(buffer);
      this.offset = 0;
    }
    ensure(size) {
      if (this.offset + size > this.view.byteLength) throw new Error("Truncated protocol message");
    }
    u8() { this.ensure(1); return this.view.getUint8(this.offset++); }
    u16() { this.ensure(2); const value = this.view.getUint16(this.offset, true); this.offset += 2; return value; }
    u32() { this.ensure(4); const value = this.view.getUint32(this.offset, true); this.offset += 4; return value; }
    u64() { this.ensure(8); const value = this.view.getBigUint64(this.offset, true); this.offset += 8; return value; }
    f32() { this.ensure(4); const value = this.view.getFloat32(this.offset, true); this.offset += 4; return value; }
    bytes(size) { this.ensure(size); const value = new Uint8Array(this.view.buffer, this.offset, size); this.offset += size; return value; }
    string(max = 16 * 1024 * 1024) { const size = this.u32(); if (size > max) throw new Error("Protocol string is too large"); return decoder.decode(this.bytes(size)); }
    remaining() { return this.view.byteLength - this.offset; }
  }

  class Writer {
    constructor(type, payloadSize) {
      this.buffer = new ArrayBuffer(16 + payloadSize);
      this.view = new DataView(this.buffer);
      this.offset = 0;
      this.u32(MAGIC); this.u16(VERSION); this.u16(type); this.u64(0n);
    }
    u8(value) { this.view.setUint8(this.offset++, value); }
    u16(value) { this.view.setUint16(this.offset, value, true); this.offset += 2; }
    u32(value) { this.view.setUint32(this.offset, value, true); this.offset += 4; }
    u64(value) { this.view.setBigUint64(this.offset, value, true); this.offset += 8; }
    f32(value) { this.view.setFloat32(this.offset, value, true); this.offset += 4; }
    bytes(value) { new Uint8Array(this.buffer, this.offset, value.length).set(value); this.offset += value.length; }
  }

  const rgba = (color) => `rgba(${color[0]},${color[1]},${color[2]},${color[3] / 255})`;
  const readColor = (reader) => [reader.u8(), reader.u8(), reader.u8(), reader.u8()];
  const transparent = (color) => color[3] === 0;
  const intersects = (a, b) => a.x < b.x + b.width && a.x + a.width > b.x && a.y < b.y + b.height && a.y + a.height > b.y;

  function transformedBounds(matrix, local) {
    const [a, b, c, d, e, f] = matrix;
    const points = [
      [local.x, local.y], [local.x + local.width, local.y],
      [local.x, local.y + local.height], [local.x + local.width, local.y + local.height]
    ].map(([x, y]) => [a * x + c * y + e, b * x + d * y + f]);
    const xs = points.map(point => point[0]);
    const ys = points.map(point => point[1]);
    return { x: Math.min(...xs) - 2, y: Math.min(...ys) - 2, width: Math.max(...xs) - Math.min(...xs) + 4, height: Math.max(...ys) - Math.min(...ys) + 4 };
  }

  function primitiveBounds(primitive) {
    let local;
    switch (primitive.kind) {
      case PRIMITIVE.RECT: local = { x: 0, y: 0, width: primitive.width, height: primitive.height }; break;
      case PRIMITIVE.CIRCLE: local = { x: 0, y: 0, width: primitive.radius * 2, height: primitive.radius * 2 }; break;
      case PRIMITIVE.TEXT: local = { x: 0, y: 0, width: Math.max(1, primitive.text.length * primitive.fontSize * 18), height: primitive.fontSize * 40 }; break;
      case PRIMITIVE.POLYGON: {
        if (!primitive.points.length) return { x: 0, y: 0, width: 0, height: 0 };
        const xs = primitive.points.map(point => point[0]);
        const ys = primitive.points.map(point => point[1]);
        local = { x: Math.min(...xs), y: Math.min(...ys), width: Math.max(...xs) - Math.min(...xs), height: Math.max(...ys) - Math.min(...ys) };
        break;
      }
      case PRIMITIVE.VERTICES: {
        if (!primitive.points.length) return { x: 0, y: 0, width: 0, height: 0 };
        const xs = primitive.points.map(point => point.x);
        const ys = primitive.points.map(point => point.y);
        const pad = primitive.thickness + 2;
        local = { x: Math.min(...xs) - pad, y: Math.min(...ys) - pad, width: Math.max(...xs) - Math.min(...xs) + pad * 2, height: Math.max(...ys) - Math.min(...ys) + pad * 2 };
        break;
      }
      case PRIMITIVE.IMAGE: local = { x: 0, y: 0, width: primitive.width, height: primitive.height }; break;
      default: local = { x: 0, y: 0, width: 0, height: 0 };
    }
    const bounds = transformedBounds(primitive.matrix, local);
    if (!primitive.clip) return bounds;
    const x = Math.max(bounds.x, primitive.clip.x);
    const y = Math.max(bounds.y, primitive.clip.y);
    return { x, y, width: Math.max(0, Math.min(bounds.x + bounds.width, primitive.clip.x + primitive.clip.width) - x), height: Math.max(0, Math.min(bounds.y + bounds.height, primitive.clip.y + primitive.clip.height) - y) };
  }

  function parsePrimitive(reader) {
    const primitive = { kind: reader.u8(), matrix: Array.from({ length: 6 }, () => reader.f32()) };
    if (reader.u8()) primitive.clip = { x: reader.f32(), y: reader.f32(), width: reader.f32(), height: reader.f32() };
    switch (primitive.kind) {
      case PRIMITIVE.RECT:
        primitive.width = reader.f32(); primitive.height = reader.f32();
        primitive.fill = readColor(reader); primitive.line = readColor(reader);
        primitive.corners = reader.u8(); primitive.radius = reader.f32();
        break;
      case PRIMITIVE.CIRCLE:
        primitive.radius = reader.f32(); primitive.fill = readColor(reader); primitive.line = readColor(reader);
        break;
      case PRIMITIVE.TEXT:
        primitive.text = reader.string(); primitive.color = readColor(reader); primitive.fontSize = reader.f32();
        primitive.style = reader.u32(); primitive.align = reader.u8(); primitive.shadow = reader.f32();
        break;
      case PRIMITIVE.POLYGON: {
        const count = reader.u32(); primitive.points = Array.from({ length: count }, () => [reader.f32(), reader.f32()]);
        primitive.fill = readColor(reader); primitive.line = readColor(reader); primitive.showPoints = Boolean(reader.u8());
        break;
      }
      case PRIMITIVE.VERTICES: {
        primitive.mode = reader.u8(); primitive.thickness = reader.f32(); const count = reader.u32();
        primitive.points = Array.from({ length: count }, () => ({ x: reader.f32(), y: reader.f32(), color: readColor(reader) }));
        break;
      }
      case PRIMITIVE.IMAGE:
        primitive.resource = reader.u32(); primitive.revision = reader.u64();
        primitive.width = reader.f32(); primitive.height = reader.f32();
        primitive.sourceWidth = reader.f32(); primitive.sourceHeight = reader.f32();
        primitive.tint = readColor(reader);
        break;
      default: throw new Error(`Unknown primitive kind ${primitive.kind}`);
    }
    primitive.bounds = primitiveBounds(primitive);
    return primitive;
  }

  function readEntries(reader, dirty) {
    const count = reader.u32();
    for (let index = 0; index < count; ++index) {
      const id = reader.u32();
      const size = reader.u32();
      const end = reader.offset + size;
      const previous = state.primitives.get(id);
      if (previous) dirty.push(previous.bounds);
      const primitive = parsePrimitive(reader);
      if (reader.offset !== end) throw new Error("Primitive payload length mismatch");
      state.primitives.set(id, primitive);
      dirty.push(primitive.bounds);
    }
  }

  function readOrder(reader) {
    const count = reader.u32();
    return Array.from({ length: count }, () => reader.u32());
  }

  function resizeCanvas() {
    const dpr = window.devicePixelRatio || 1;
    const width = Math.max(1, Math.ceil(state.width * dpr));
    const height = Math.max(1, Math.ceil(state.height * dpr));
    if (canvas.width !== width || canvas.height !== height) {
      canvas.width = width; canvas.height = height;
      state.fullRedraw = true;
    }
    const fit = Math.min(
      1,
      viewport.clientWidth / state.width,
      viewport.clientHeight / state.height
    );
    canvas.style.aspectRatio = `${state.width} / ${state.height}`;
    canvas.style.width = `${state.width * fit}px`;
    canvas.style.height = `${state.height * fit}px`;
  }

  function dirtyRegion(regions) {
    for (const region of regions) {
      if (!region || region.width <= 0 || region.height <= 0) continue;
      if (!state.dirty) state.dirty = { ...region };
      else {
        const right = Math.max(state.dirty.x + state.dirty.width, region.x + region.width);
        const bottom = Math.max(state.dirty.y + state.dirty.height, region.y + region.height);
        state.dirty.x = Math.min(state.dirty.x, region.x); state.dirty.y = Math.min(state.dirty.y, region.y);
        state.dirty.width = right - state.dirty.x; state.dirty.height = bottom - state.dirty.y;
      }
    }
    if (state.dirty && state.dirty.width * state.dirty.height > state.width * state.height * .45) state.fullRedraw = true;
    if (!state.animation) state.animation = requestAnimationFrame(render);
  }

  function roundedRectPath(ctx, width, height, radius, mask) {
    const r = Math.max(0, Math.min(radius, width / 2, height / 2));
    const tl = mask & 1 ? r : 0, tr = mask & 2 ? r : 0, br = mask & 4 ? r : 0, bl = mask & 8 ? r : 0;
    ctx.beginPath(); ctx.moveTo(tl, 0); ctx.lineTo(width - tr, 0); ctx.quadraticCurveTo(width, 0, width, tr);
    ctx.lineTo(width, height - br); ctx.quadraticCurveTo(width, height, width - br, height);
    ctx.lineTo(bl, height); ctx.quadraticCurveTo(0, height, 0, height - bl);
    ctx.lineTo(0, tl); ctx.quadraticCurveTo(0, 0, tl, 0); ctx.closePath();
  }

  function vertexGradient(ctx, points) {
    if (points.length < 2) return rgba(points[0]?.color || [255, 255, 255, 255]);
    const first = points[0], last = points[points.length - 1];
    const gradient = ctx.createLinearGradient(first.x, first.y, last.x === first.x && last.y === first.y ? last.x + 1 : last.x, last.y);
    points.forEach((point, index) => gradient.addColorStop(index / Math.max(1, points.length - 1), rgba(point.color)));
    return gradient;
  }

  function drawVertices(ctx, primitive) {
    const points = primitive.points;
    if (!points.length) return;
    ctx.lineWidth = primitive.thickness;
    ctx.lineCap = "round";
    if (primitive.mode === 0) {
      for (const point of points) { ctx.beginPath(); ctx.fillStyle = rgba(point.color); ctx.arc(point.x, point.y, Math.max(1, primitive.thickness / 2), 0, Math.PI * 2); ctx.fill(); }
    } else if (primitive.mode === 1) {
      for (let index = 0; index + 1 < points.length; index += 2) {
        const pair = [points[index], points[index + 1]]; ctx.strokeStyle = vertexGradient(ctx, pair); ctx.beginPath(); ctx.moveTo(pair[0].x, pair[0].y); ctx.lineTo(pair[1].x, pair[1].y); ctx.stroke();
      }
    } else if (primitive.mode === 2) {
      ctx.strokeStyle = vertexGradient(ctx, points); ctx.beginPath(); ctx.moveTo(points[0].x, points[0].y);
      for (const point of points.slice(1)) ctx.lineTo(point.x, point.y); ctx.stroke();
    } else {
      const triangles = [];
      if (primitive.mode === 3) for (let index = 0; index + 2 < points.length; index += 3) triangles.push(points.slice(index, index + 3));
      else for (let index = 0; index + 2 < points.length; ++index) triangles.push([points[index], points[index + 1], points[index + 2]]);
      for (const triangle of triangles) { ctx.fillStyle = vertexGradient(ctx, triangle); ctx.beginPath(); ctx.moveTo(triangle[0].x, triangle[0].y); ctx.lineTo(triangle[1].x, triangle[1].y); ctx.lineTo(triangle[2].x, triangle[2].y); ctx.closePath(); ctx.fill(); }
    }
  }

  function imageFor(primitive) {
    const resource = state.resources.get(primitive.resource);
    // Keep the previous revision visible while the replacement is encoded and
    // transferred. Resource IDs are stable for a primitive and are never reused.
    if (!resource || resource.revision > primitive.revision) return null;
    if (primitive.tint[0] === 255 && primitive.tint[1] === 255 && primitive.tint[2] === 255 && primitive.tint[3] === 255) return resource.bitmap;
    const key = `${primitive.resource}:${primitive.revision}:${primitive.tint.join(",")}`;
    if (state.tintCache.has(key)) return state.tintCache.get(key);
    const tinted = document.createElement("canvas"); tinted.width = resource.width; tinted.height = resource.height;
    const ctx = tinted.getContext("2d"); ctx.drawImage(resource.bitmap, 0, 0);
    ctx.globalCompositeOperation = "multiply"; ctx.fillStyle = `rgb(${primitive.tint[0]},${primitive.tint[1]},${primitive.tint[2]})`; ctx.fillRect(0, 0, tinted.width, tinted.height);
    ctx.globalCompositeOperation = "destination-in"; ctx.globalAlpha = primitive.tint[3] / 255; ctx.drawImage(resource.bitmap, 0, 0);
    state.tintCache.set(key, tinted); return tinted;
  }

  function drawPrimitive(primitive, region) {
    const dpr = window.devicePixelRatio || 1;
    context.save();
    context.setTransform(dpr, 0, 0, dpr, 0, 0);
    context.beginPath(); context.rect(region.x, region.y, region.width, region.height); context.clip();
    if (primitive.clip) { context.beginPath(); context.rect(primitive.clip.x, primitive.clip.y, primitive.clip.width, primitive.clip.height); context.clip(); }
    const [a, b, c, d, e, f] = primitive.matrix;
    context.setTransform(a * dpr, b * dpr, c * dpr, d * dpr, e * dpr, f * dpr);
    switch (primitive.kind) {
      case PRIMITIVE.RECT:
        roundedRectPath(context, primitive.width, primitive.height, primitive.radius, primitive.corners);
        if (!transparent(primitive.fill)) { context.fillStyle = rgba(primitive.fill); context.fill(); }
        if (!transparent(primitive.line)) { context.strokeStyle = rgba(primitive.line); context.lineWidth = 1; context.stroke(); }
        break;
      case PRIMITIVE.CIRCLE:
        context.beginPath(); context.arc(primitive.radius, primitive.radius, primitive.radius, 0, Math.PI * 2);
        if (!transparent(primitive.fill)) { context.fillStyle = rgba(primitive.fill); context.fill(); }
        if (!transparent(primitive.line)) { context.strokeStyle = rgba(primitive.line); context.lineWidth = 1; context.stroke(); }
        break;
      case PRIMITIVE.TEXT: {
        const symbols = primitive.style & STYLE.SYMBOLS;
        const mono = !symbols && primitive.style & STYLE.MONO;
        const italic = !symbols && !mono && primitive.style & STYLE.ITALIC ? "italic " : "";
        const weight = !symbols && primitive.style & STYLE.BOLD ? "700 " : "400 ";
        const family = symbols ? "TRex Symbols" : mono ? "TRex Mono" : "TRex Quicksand";
        const faceScale = mono ? 0.85 : 1;
        const offsetX = mono ? -primitive.fontSize : 0;
        const offsetY = symbols ? 3 * primitive.fontSize : mono ? 1.8 * primitive.fontSize : 0;
        context.font = `${italic}${weight}${32 * faceScale * primitive.fontSize}px "${family}"`;
        context.textBaseline = "top"; context.fillStyle = rgba(primitive.color);
        if (primitive.shadow > 0) { context.shadowColor = `rgba(20,20,20,${primitive.shadow})`; context.shadowOffsetX = 1.5; context.shadowOffsetY = 1.5; }
        context.fillText(primitive.text, offsetX, offsetY);
        context.shadowColor = "transparent";
        if (primitive.style & (STYLE.UNDERLINE | STYLE.STRIKE)) {
          const width = context.measureText(primitive.text).width;
          context.fillStyle = rgba(primitive.color);
          if (primitive.style & STYLE.UNDERLINE) context.fillRect(offsetX, offsetY + 31 * faceScale * primitive.fontSize, width, Math.max(1, primitive.fontSize));
          if (primitive.style & STYLE.STRIKE) context.fillRect(offsetX, offsetY + 17 * faceScale * primitive.fontSize, width, Math.max(1, primitive.fontSize));
        }
        break;
      }
      case PRIMITIVE.POLYGON:
        if (primitive.points.length) { context.beginPath(); context.moveTo(...primitive.points[0]); for (const point of primitive.points.slice(1)) context.lineTo(...point); context.closePath();
          if (!transparent(primitive.fill)) { context.fillStyle = rgba(primitive.fill); context.fill("evenodd"); }
          if (!transparent(primitive.line)) { context.strokeStyle = rgba(primitive.line); context.stroke(); }
          if (primitive.showPoints) for (const point of primitive.points) { context.beginPath(); context.arc(point[0], point[1], 1.5, 0, Math.PI * 2); context.stroke(); }
        }
        break;
      case PRIMITIVE.VERTICES: drawVertices(context, primitive); break;
      case PRIMITIVE.IMAGE: {
        const image = imageFor(primitive);
        if (image) {
          const sourceWidth = Math.min(primitive.sourceWidth || image.width, image.width);
          const sourceHeight = Math.min(primitive.sourceHeight || image.height, image.height);
          context.drawImage(image, 0, 0, sourceWidth, sourceHeight, 0, 0, primitive.width, primitive.height);
        }
        break;
      }
    }
    context.restore();
  }

  function render() {
    state.animation = 0; resizeCanvas();
    const region = state.fullRedraw || !state.dirty ? { x: 0, y: 0, width: state.width, height: state.height } : state.dirty;
    const dpr = window.devicePixelRatio || 1;
    context.save(); context.setTransform(1, 0, 0, 1, 0, 0); context.fillStyle = "#171a20";
    context.fillRect(region.x * dpr, region.y * dpr, region.width * dpr, region.height * dpr); context.restore();
    for (const id of state.order) { const primitive = state.primitives.get(id); if (primitive && intersects(primitive.bounds, region)) drawPrimitive(primitive, region); }
    state.dirty = null; state.fullRedraw = false;
  }

  function sceneMessage(reader, type) {
    const dirty = [];
    const width = reader.f32(), height = reader.f32();
    if (width !== state.width || height !== state.height) { state.width = width; state.height = height; state.fullRedraw = true; }
    if (type === MSG.SNAPSHOT) {
      state.primitives.clear(); readEntries(reader, dirty); state.order = readOrder(reader); state.fullRedraw = true;
      const activeResources = new Set(Array.from(state.primitives.values()).filter(primitive => primitive.kind === PRIMITIVE.IMAGE).map(primitive => primitive.resource));
      for (const [id, resource] of state.resources) {
        if (!activeResources.has(id)) { resource.bitmap.close(); state.resources.delete(id); }
      }
      state.tintCache.clear();
    } else {
      readEntries(reader, dirty);
      const removed = reader.u32();
      for (let index = 0; index < removed; ++index) {
        const id = reader.u32();
        const primitive = state.primitives.get(id);
        if (primitive) dirty.push(primitive.bounds);
        const resource = state.resources.get(id);
        if (resource) resource.bitmap.close();
        state.primitives.delete(id);
        state.resources.delete(id);
      }
      if (removed) state.tintCache.clear();
      if (reader.u8()) { state.order = readOrder(reader); state.fullRedraw = true; }
    }
    dirtyRegion(dirty);
  }

  async function resourceMessage(reader) {
    const id = reader.u32(), revision = reader.u64(), width = reader.u32(), height = reader.u32();
    const mime = reader.u8() ? "image/png" : "image/jpeg"; const size = reader.u32(); const bytes = reader.bytes(size).slice();
    const bitmap = await createImageBitmap(new Blob([bytes], { type: mime }));
    const primitive = state.primitives.get(id);
    if (!primitive || primitive.kind !== PRIMITIVE.IMAGE || primitive.revision < revision) { bitmap.close(); return; }
    const current = state.resources.get(id);
    if (current && current.revision > revision) { bitmap.close(); return; }
    if (current) current.bitmap.close();
    state.resources.set(id, { revision, width, height, bitmap }); state.tintCache.clear();
    const dirty = []; for (const primitive of state.primitives.values()) if (primitive.kind === PRIMITIVE.IMAGE && primitive.resource === id) dirty.push(primitive.bounds);
    dirtyRegion(dirty);
  }

  async function receive(buffer) {
    const reader = new Reader(buffer); const magic = reader.u32(), version = reader.u16(), type = reader.u16(), sequence = reader.u64();
    if (magic !== MAGIC || version !== VERSION) throw new Error("Incompatible browser protocol");
    if ((type === MSG.SNAPSHOT || type === MSG.DELTA) && sequence < state.sequence) return;
    if (type === MSG.SNAPSHOT || type === MSG.DELTA) { state.sequence = sequence; sceneMessage(reader, type); }
    else if (type === MSG.IMAGE) await resourceMessage(reader);
    else if (type === MSG.CLIENTS) {
      const count = reader.u32();
      clients.textContent = count > 1
        ? `Shared control: ${count} browsers connected; all clients can send input.`
        : `${count} browser client${count === 1 ? "" : "s"} connected`;
      clients.classList.toggle("shared", count > 1);
    }
    else if (type === MSG.ERROR) { status.textContent = reader.string(4096); status.className = "error"; }
    if (reader.remaining()) throw new Error("Unexpected protocol payload");
  }

  function modifiers(event) { return (event.shiftKey ? 1 : 0) | (event.ctrlKey ? 2 : 0) | (event.altKey ? 4 : 0) | (event.metaKey ? 8 : 0); }
  function send(writer) { if (state.socket?.readyState === WebSocket.OPEN) state.socket.send(writer.buffer); }
  function control(type) { send(new Writer(type, 0)); }
  function sendMove(event) { const point = logicalPoint(event); const writer = new Writer(MSG.INPUT, 10); writer.u8(INPUT.MOVE); writer.u8(modifiers(event)); writer.f32(point.x); writer.f32(point.y); send(writer); }
  function sendButton(event, pressed) { if (event.button !== 0 && event.button !== 2) return; const point = logicalPoint(event); const writer = new Writer(MSG.INPUT, 12); writer.u8(INPUT.BUTTON); writer.u8(modifiers(event)); writer.u8(event.button === 2 ? 1 : 0); writer.u8(pressed ? 1 : 0); writer.f32(point.x); writer.f32(point.y); send(writer); }
  function sendWheel(event) { const writer = new Writer(MSG.INPUT, 10); writer.u8(INPUT.WHEEL); writer.u8(modifiers(event)); writer.f32(-event.deltaX); writer.f32(-event.deltaY); send(writer); }
  function sendKey(event, pressed) { const code = encoder.encode(event.code); const writer = new Writer(MSG.INPUT, 8 + code.length); writer.u8(INPUT.KEY); writer.u8(modifiers(event)); writer.u8(pressed ? 1 : 0); writer.u8(event.repeat ? 1 : 0); writer.u32(code.length); writer.bytes(code); send(writer); }
  function sendText(text) { for (const character of text) { const writer = new Writer(MSG.INPUT, 6); writer.u8(INPUT.TEXT); writer.u8(0); writer.u32(character.codePointAt(0)); send(writer); } }
  function sendBlur() { const writer = new Writer(MSG.INPUT, 2); writer.u8(INPUT.BLUR); writer.u8(0); send(writer); }
  function logicalPoint(event) { const rect = canvas.getBoundingClientRect(); return { x: (event.clientX - rect.left) * state.width / rect.width, y: (event.clientY - rect.top) * state.height / rect.height }; }

  canvas.addEventListener("pointermove", sendMove);
  canvas.addEventListener("pointerdown", event => { canvas.setPointerCapture(event.pointerId); inputSink.focus({ preventScroll: true }); sendButton(event, true); event.preventDefault(); });
  canvas.addEventListener("pointerup", event => { sendButton(event, false); event.preventDefault(); });
  canvas.addEventListener("pointercancel", sendBlur);
  canvas.addEventListener("contextmenu", event => event.preventDefault());
  canvas.addEventListener("wheel", event => { sendWheel(event); event.preventDefault(); }, { passive: false });
  inputSink.addEventListener("keydown", event => {
    sendKey(event, true);
    if (event.code === "Tab" || event.ctrlKey || event.metaKey || event.altKey) event.preventDefault();
  });
  inputSink.addEventListener("keyup", event => { sendKey(event, false); event.preventDefault(); });
  inputSink.addEventListener("beforeinput", event => { if (event.data) sendText(event.data); inputSink.value = ""; event.preventDefault(); });
  window.addEventListener("blur", sendBlur);
  document.addEventListener("visibilitychange", () => { if (document.hidden) sendBlur(); });
  window.addEventListener("resize", () => { state.fullRedraw = true; dirtyRegion([]); });

  function connect() {
    const scheme = location.protocol === "https:" ? "wss" : "ws";
    const socket = new WebSocket(`${scheme}://${location.host}/ws`); socket.binaryType = "arraybuffer"; state.socket = socket;
    status.textContent = "Connecting…"; status.className = "";
    socket.addEventListener("open", () => {
      state.retry = 250;
      state.sequence = 0n;
      status.textContent = "Connected";
      status.className = "connected";
      control(MSG.RESYNC);
      clearInterval(state.heartbeat);
      state.heartbeat = setInterval(() => control(MSG.HEARTBEAT), 15000);
    });
    socket.addEventListener("message", event => receive(event.data).catch(error => { status.textContent = error.message; status.className = "error"; control(MSG.RESYNC); }));
    socket.addEventListener("close", () => { clearInterval(state.heartbeat); status.textContent = "Disconnected; reconnecting…"; status.className = "error"; setTimeout(connect, state.retry); state.retry = Math.min(5000, state.retry * 2); });
    socket.addEventListener("error", () => socket.close());
  }

  async function loadFonts() {
    const required = [
      ['400 32px "TRex Quicksand"', "TRex"],
      ['700 32px "TRex Quicksand"', "TRex"],
      ['italic 400 32px "TRex Quicksand"', "TRex"],
      ['italic 700 32px "TRex Quicksand"', "TRex"],
      ['400 32px "TRex Mono"', "TRex"],
      ['700 32px "TRex Mono"', "TRex"],
      ['400 32px "TRex Symbols"', "▷🗁"]
    ];
    for (const [font, sample] of required) {
      const faces = await document.fonts.load(font, sample);
      if (!faces.length || !document.fonts.check(font, sample)) throw new Error(`Required browser font failed to load: ${font}`);
    }
    await document.fonts.ready;
  }

  async function start() {
    try {
      status.textContent = "Loading fonts…";
      await loadFonts();
      dirtyRegion([]);
      connect();
    } catch (error) {
      status.textContent = error instanceof Error ? error.message : "Required browser fonts failed to load.";
      status.className = "error";
    }
  }

  start();
})();
