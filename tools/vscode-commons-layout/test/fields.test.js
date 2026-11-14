'use strict';

const assert = require('node:assert/strict');
const test = require('node:test');
const { tokenize } = require('../tokens');

function fields(source) {
    return tokenize(source).filter(token => token.type === 'commonsField')
        .map(token => source.slice(token.start, token.end));
}

test('known fields depend on the owning control even when type comes last', () => {
    assert.deepEqual(fields('{"text":"Run","action":"go","radius":2,"on_text_changed":"go","type":"button"}'),
        ['text', 'action', 'type']);
    assert.deepEqual(fields('{"type":"textfield","text":"","on_text_changed":"go","placeholder":"hint"}'),
        ['type', 'text', 'on_text_changed']);
    assert.deepEqual(fields('{"type":"circle","radius":2,"text":"no","pos":[0,0]}'),
        ['type', 'radius', 'pos']);
    assert.deepEqual(fields('{"type":"custom_widget","text":"custom","radius":2}'), []);
});

test('nested style fields and list item fields stay in their own contexts', () => {
    const source = JSON.stringify({ objects: [
        { type: 'button', font: { size: 1, style: 'bold', align: 'center', text: 'ignored' },
            data: { action: 'ignored', size: 1 } },
        { type: 'list', action: 'ignored', template: { text: '{i}', detail: 'detail', disabled: false, action: 'go' },
            items: [{ text: 'One', action: 'go', radius: 2 }], item: { pad: [2, 3], font: { size: 1 }, action: 'ignored' } },
        { type: 'taglist', input: { list_size: [100, 200], placeholder: 'Add', font: { style: 'italic' } } }
    ] });
    assert.deepEqual(fields(source), ['objects', 'type', 'font', 'size', 'style', 'align',
        'type', 'template', 'text', 'detail', 'disabled', 'action', 'items', 'text', 'action',
        'item', 'pad', 'font', 'size', 'type', 'input', 'list_size', 'placeholder', 'font', 'style']);
});

test('field recognition handles JSONC, escaped keys and incomplete controls without leaking into locals', () => {
    const source = String.raw`{
      "locals": {"config": {"type":"button", "text":"data"}},
      "objects": [
        // "type":"button", "action":"comment"
        {"te\u0078t":"Hi", "type" /* comment */ : "button"},
        {"children":[{"type":"circle","radius":3}],"type":"vlayout"},
        {"type":"button", "text":"unfinished`;
    assert.deepEqual(fields(source), ['locals', 'objects', 'te\\u0078t', 'type',
        'children', 'type', 'radius', 'type', 'type', 'text']);
});
