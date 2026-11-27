'use strict';

const { fieldsForObject } = require('./fields');
const tokenTypes = ['function', 'variable', 'parameter', 'number', 'keyword', 'string', 'operator', 'commonsEscape', 'type', 'commonsField'];
// Built-in LayoutType names from commons/common/gui/DynamicGUI.h; custom modules are scene-specific.
const elementTypes = ['each', 'condition', 'vlayout', 'hlayout', 'floatinglayout', 'gridlayout', 'collection',
    'line', 'button', 'circle', 'text', 'stext', 'rect', 'textfield', 'checkbox', 'settings', 'combobox',
    'image', 'list', 'taglist'];
const escapes = { '"': '"', '\\': '\\', '/': '/', b: '\b', f: '\f', n: '\n', r: '\r', t: '\t' };
const actionKeys = new Set(['action', 'on_text_changed', 'hover', 'unhover', 'drag', 'click', 'deselect',
    'add_action', 'remove_action']);

function skipTrivia(source, offset) {
    while (offset < source.length) {
        if (/\s/.test(source[offset])) ++offset;
        else if (source.startsWith('//', offset)) {
            const end = source.indexOf('\n', offset + 2);
            offset = end < 0 ? source.length : end + 1;
        } else if (source.startsWith('/*', offset)) {
            const end = source.indexOf('*/', offset + 2);
            offset = end < 0 ? source.length : end + 2;
        } else break;
    }
    return offset;
}

function readString(source, offset) {
    let text = '';
    const starts = [], ends = [];
    while (offset < source.length && source[offset] !== '"' && !/[\r\n]/.test(source[offset])) {
        const start = offset;
        let character = source[offset++];
        if (character === '\\' && offset < source.length && !/[\r\n]/.test(source[offset])) {
            const escape = source[offset++];
            if (escape === 'u' && /^[\da-fA-F]{4}$/.test(source.slice(offset, offset + 4))) {
                character = String.fromCharCode(parseInt(source.slice(offset, offset + 4), 16));
                offset += 4;
            } else character = escapes[escape] ?? escape;
        }
        text += character;
        starts.push(start);
        ends.push(offset);
    }
    if (source[offset] === '"') ++offset;
    return { text, starts, ends, offset };
}

function expressionTokens(string, tokens, action, scopes) {
    const { text, starts, ends } = string;
    const emit = (start, end, type, bracketDepth) => {
        if (end > start) tokens.push({ start: starts[start], end: ends[end - 1], type,
            ...(bracketDepth === undefined ? {} : { bracketDepth }) });
    };
    const brackets = [];
    const bracket = (start, end) => {
        const character = text[start];
        if (character === '{' || character === '[') {
            brackets.push({ character, index: tokens.length });
            emit(start, end, 'operator', brackets.length - 1);
        } else if (brackets.at(-1)?.character === (character === '}' ? '{' : '[')) {
            const opening = tokens[brackets.pop().index];
            emit(start, end, 'operator', brackets.length);
            opening.matchingBracket = starts[start];
            tokens.at(-1).matchingBracket = opening.start;
            scopes.push({ start: opening.start, end: ends[end - 1],
                contentStart: opening.end, contentEnd: starts[start] });
        } else emit(start, end, 'operator', -1);
    };
    let depth = 0;
    let name = false;
    let actionName = action;
    for (let i = 0; i < text.length;) {
        const start = i;
        const character = text[i];
        if (character === '\\') {
            i = Math.min(i + 2, text.length);
            emit(start, i, 'string');
        } else if (character === "'") {
            ++i;
            while (i < text.length) {
                if (text[i] === '\\') i = Math.min(i + 2, text.length);
                else if (text[i++] === "'") break;
            }
            if (depth || action) emit(start, i, 'string');
        } else if (character === '{') {
            ++depth;
            name = true;
            bracket(i, ++i);
        } else if (character === '}') {
            if (depth) --depth;
            name = false;
            bracket(i, ++i);
        } else if (character === '[' || character === ']') {
            name = false;
            if (depth || action || brackets.length) bracket(i, ++i);
            else ++i;
        } else if ((!depth && !action) || /\s/.test(character)) {
            ++i;
        } else if (name || (actionName && !depth && !brackets.length && character !== ':')) {
            if (depth && (character === '#' || character === '.')) {
                emit(i, ++i, 'keyword');
                continue;
            }
            while (i < text.length && !/[{}:[\]'\\\s]/.test(text[i])) ++i;
            if (i === start) { name = false; continue; }
            const type = (!depth && actionName) || text[i] === ':' ? 'function' : 'variable';
            for (const part of text.slice(start, i).matchAll(/[^.]+|\./g)) {
                emit(start + part.index, start + part.index + part[0].length,
                    part[0] === '.' ? 'operator' : /^\d+$/.test(part[0]) ? 'number' : type);
            }
            name = false;
        } else if (':,'.includes(character)) {
            if (!depth && !brackets.length && character === ':') actionName = false;
            emit(i, ++i, 'operator');
        } else {
            while (i < text.length && !/[{}:[\],'\\\s]/.test(text[i])) ++i;
            const word = text.slice(start, i);
            const type = /^(?:true|false|null)$/.test(word) ? 'keyword'
                : /^-?\d+(?:\.\d+)?(?:[eE][+-]?\d+)?$/.test(word) ? 'number' : 'parameter';
            emit(start, i, type);
        }
    }
}

function* stringValues(source, includeKeys = false) {
    const parents = [];
    const strings = [];
    let key;
    for (let i = 0; i < source.length;) {
        i = skipTrivia(source, i);
        const parent = parents.at(-1);
        if (source[i] !== '"') {
            if (source[i] === '{' || source[i] === '[') {
                parents.push({ kind: source[i] === '{' ? 'object' : 'array', parent, key,
                    fields: new Map(), expectKey: source[i] === '{',
                    inLocals: !!parent?.inLocals || (key === 'locals' && parents.length === 1) });
            } else if (source[i] === '}' || source[i] === ']') parents.pop();
            else if (source[i] === ',' && parent?.kind === 'object') parent.expectKey = true;
            key = undefined;
            ++i;
            continue;
        }
        const start = i + 1;
        const string = readString(source, i + 1);
        i = string.offset;
        const next = skipTrivia(source, i);
        const end = string.ends.at(-1) ?? start;
        const isKey = source[next] === ':' || (parent?.kind === 'object' && parent.expectKey);
        const entry = { ...string, key, start, end, closed: source[end] === '"',
            inLocals: !!parent?.inLocals, isKey, hasColon: source[next] === ':', object: parent };
        strings.push(entry);
        if (isKey) {
            key = string.text;
            if (parent?.kind === 'object') {
                parent.fields.set(key, entry);
                parent.expectKey = false;
                if (key === 'type') parent.type = undefined;
            }
            if (source[next] === ':') i = next + 1;
        } else {
            if (key === 'type' && parent?.kind === 'object') parent.type = string.text.toLowerCase();
            key = undefined;
        }
    }
    // Object types are known before keys are consumed, regardless of property order.
    yield* strings.filter(string => includeKeys || !string.isKey);
}

function tokenize(source, scopes = []) {
    const tokens = [];
    for (const string of stringValues(source, true)) {
        if (string.isKey) {
            if (string.text && Object.hasOwn(fieldsForObject(string.object), string.text)) {
                tokens.push({ start: string.start, end: string.end, type: 'commonsField' });
            }
            continue;
        }
        const action = actionKeys.has(string.key);
        // JSON escapes are decoded before template syntax; ranges still address the original document.
        if (string.key === 'type' && !string.inLocals && elementTypes.includes(string.text.toLowerCase())) {
            tokens.push({ start: string.start, end: string.end, type: 'type' });
        } else {
            expressionTokens(string, tokens, action, scopes);
            if (action && string.text.length) scopes.push({ start: string.starts[0], end: string.ends.at(-1) });
        }
    }
    return tokens;
}

module.exports = { tokenize, tokenTypes, stringValues, elementTypes, fieldsForObject };
