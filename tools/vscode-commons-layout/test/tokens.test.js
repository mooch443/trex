'use strict';

const assert = require('node:assert/strict');
const test = require('node:test');
const { tokenize } = require('../tokens');

function highlighted(source) {
    const tokens = tokenize(source);
    let end = 0;
    for (const token of tokens) {
        assert.ok(token.start >= end && token.end > token.start && token.end <= source.length);
        assert.doesNotMatch(source.slice(token.start, token.end), /[\r\n]/);
        end = token.end;
    }
    return tokens.map(token => [source.slice(token.start, token.end), token.type]);
}

const value = text => JSON.stringify({ text });

test('element type values use type colours without recolouring ordinary text or local setting types', () => {
    assert.deepEqual(highlighted('{"objects":[{"type":"button","text":"button"},{"type":"TAGLIST"}]}'), [
        ['objects', 'commonsField'], ['type', 'commonsField'], ['button', 'type'], ['text', 'commonsField'],
        ['type', 'commonsField'], ['TAGLIST', 'type']
    ]);
    assert.deepEqual(highlighted(String.raw`{"t\u0079pe" /* comment */ : "\u0062utton"}`), [
        [String.raw`t\u0079pe`, 'commonsField'], [String.raw`\u0062utton`, 'type']
    ]);
    assert.deepEqual(highlighted('{"locals":{"mode":{"type":"button"}},"objects":[{"type":"rect"}]}'), [
        ['locals', 'commonsField'], ['objects', 'commonsField'], ['type', 'commonsField'], ['rect', 'type']
    ]);
    assert.deepEqual(highlighted('// "type":"button"\n{"type":"custom_widget","text":"button"}'), []);
    assert.deepEqual(highlighted('{"type":"{kind}"}'), [
        ['{', 'operator'], ['kind', 'variable'], ['}', 'operator']
    ]);
});

test('element type suggestions match the built-in DynamicGUI registry', () => {
    const fs = require('node:fs');
    const path = require('node:path');
    const { elementTypes } = require('../tokens');
    const header = fs.readFileSync(path.join(__dirname, '../../../Application/src/commons/common/gui/DynamicGUI.h'), 'utf8');
    const types = header.match(/LayoutType,([\s\S]*?)unknown\);/)[1].split(',').map(type => type.trim()).filter(Boolean);
    assert.deepEqual([...elementTypes].sort(), types.sort());
});

test('nested expressions are code, surrounding text remains JSON string content', () => {
    assert.deepEqual(highlighted(value('Hello {if:{ready}:{+:1:{object.array.0}}:no}!')), [
        ['{', 'operator'], ['if', 'function'], [':', 'operator'],
        ['{', 'operator'], ['ready', 'variable'], ['}', 'operator'], [':', 'operator'],
        ['{', 'operator'], ['+', 'function'], [':', 'operator'], ['1', 'number'], [':', 'operator'],
        ['{', 'operator'], ['object', 'variable'], ['.', 'operator'], ['array', 'variable'],
        ['.', 'operator'], ['0', 'number'], ['}', 'operator'], ['}', 'operator'],
        [':', 'operator'], ['no', 'parameter'], ['}', 'operator']
    ]);
});

test('single quotes suppress expressions at the root and inside arrays and calls', () => {
    assert.deepEqual(highlighted(value("'{hidden}' ['{also_hidden}']")), []);
    const tokens = highlighted(value("{if:true:'{hidden}':{visible}}"));
    assert.ok(tokens.some(([text, type]) => text === "'{hidden}'" && type === 'string'));
    assert.deepEqual(tokens.filter(([, type]) => type === 'variable'), [['visible', 'variable']]);
    assert.deepEqual(highlighted(value(String.raw`'can\'t {hidden}' {visible}`)).filter(([, type]) => type === 'variable'),
        [['visible', 'variable']]);
});

test('JSON and template escaping are separate layers', () => {
    assert.deepEqual(highlighted(value(String.raw`'{hidden}' \{visible}`)).filter(([, type]) => type === 'variable'),
        [['visible', 'variable']]);
    const source = value("{if:true:'Name: '{name}:no}");
    assert.ok(highlighted(source).some(([text, type]) => text === "'Name: '" && type === 'string'));
    assert.deepEqual(highlighted(value(String.raw`\q \n`)), []);
    assert.deepEqual(highlighted(value('Say "{name}"')).filter(([, type]) => type === 'variable'), [['name', 'variable']]);
});

test('Unicode escapes retain source ranges and obey decoded brace and quote syntax', () => {
    assert.deepEqual(highlighted(String.raw`{"text":"\u007bna\u006de\u007d"}`), [
        [String.raw`\u007b`, 'operator'], [String.raw`na\u006de`, 'variable'], [String.raw`\u007d`, 'operator']
    ]);
    assert.deepEqual(highlighted(String.raw`{"text":"\u0027{hidden}\u0027"}`), []);
    assert.deepEqual(highlighted(value('🐟 {name}')).filter(([, type]) => type === 'variable'), [['name', 'variable']]);
});

test('JSON keys and JSONC comments never become expressions', () => {
    const source = `{
      // "{comment}": "{comment}"
      "{key}" /* {comment} */: "{value}",
      "other": "plain" /* "{comment}" */
    }`;
    assert.deepEqual(highlighted(source), [['{', 'operator'], ['value', 'variable'], ['}', 'operator']]);
});

test('unfinished expressions highlight immediately and do not leak into the next JSON string', () => {
    assert.deepEqual(highlighted('{"text":"{'), [['{', 'operator']]);
    assert.deepEqual(highlighted('{"text":"{if:{rea'), [
        ['{', 'operator'], ['if', 'function'], [':', 'operator'], ['{', 'operator'], ['rea', 'variable']
    ]);
    assert.deepEqual(highlighted('{"text":"{name", "other":"plain"}'), [
        ['{', 'operator'], ['name', 'variable']
    ]);
});

test('optional lookups, rich text, collections, numbers and boolean arguments get code tokens', () => {
    const tokens = highlighted(value('{if:{.ready}:{#item.name}:{at:0:[-1.5,false]}}'));
    for (const expected of [['.', 'keyword'], ['#', 'keyword'], ['item', 'variable'],
        ['at', 'function'], ['[', 'operator'], ['-1.5', 'number'], ['false', 'keyword'], [']', 'operator']]) {
        assert.ok(tokens.some(token => token[0] === expected[0] && token[1] === expected[1]), expected.join(':'));
    }
});

test('expression and collection bracket pairs share their nesting depth', () => {
    const source = value('{if:{ready}:[{name},[1,2]]:[]}');
    const brackets = tokenize(source).filter(token => token.bracketDepth !== undefined);
    assert.deepEqual(brackets.map(token => [source.slice(token.start, token.end), token.bracketDepth]), [
        ['{', 0], ['{', 1], ['}', 1], ['[', 1], ['{', 2], ['}', 2],
        ['[', 2], [']', 2], [']', 1], ['[', 1], [']', 1], ['}', 0]
    ]);
});

test('quoted and escaped brackets do not change the color depth', () => {
    const source = value(String.raw`{if:true:'[{hidden}]':'\{literal\}' {name}}`);
    assert.deepEqual(tokenize(source).filter(token => token.bracketDepth !== undefined)
        .map(token => [source.slice(token.start, token.end), token.bracketDepth]),
        [['{', 0], ['{', 1], ['}', 1], ['}', 0]]);
    const root = value(String.raw`'[literal]' '\{escaped\}' [1,{name}]`);
    assert.deepEqual(tokenize(root).filter(token => token.bracketDepth !== undefined)
        .map(token => [root.slice(token.start, token.end), token.bracketDepth]),
        [['{', 0], ['}', 0]]);
});

test('unbraced action values highlight commands, arguments and nested expressions', () => {
    assert.deepEqual(highlighted(JSON.stringify({ action: 'set:gui_run:true' })), [
        ['set', 'function'], [':', 'operator'], ['gui_run', 'parameter'], [':', 'operator'], ['true', 'keyword']
    ]);
    assert.deepEqual(highlighted(JSON.stringify({ action: 'reset_settings' })), [['reset_settings', 'function']]);
    assert.deepEqual(highlighted(JSON.stringify({ action: 'set:value:{if:{ready}:[1,2]:false}' })), [
        ['set', 'function'], [':', 'operator'], ['value', 'parameter'], [':', 'operator'],
        ['{', 'operator'], ['if', 'function'], [':', 'operator'],
        ['{', 'operator'], ['ready', 'variable'], ['}', 'operator'], [':', 'operator'],
        ['[', 'operator'], ['1', 'number'], [',', 'operator'], ['2', 'number'], [']', 'operator'],
        [':', 'operator'], ['false', 'keyword'], ['}', 'operator']
    ]);
    assert.deepEqual(highlighted(JSON.stringify({ action: 'do_{which}_now:{text}' })), [
        ['do_', 'function'], ['{', 'operator'], ['which', 'variable'], ['}', 'operator'], ['_now', 'function'],
        [':', 'operator'], ['{', 'operator'], ['text', 'variable'], ['}', 'operator']
    ]);
    const generated = "{if:{ready}:'set:gui_run:true':'reset_settings'}";
    assert.deepEqual(highlighted(JSON.stringify({ action: generated })), highlighted(value(generated)));
});

test('action fields are detected through JSONC trivia and do not affect other values', () => {
    const source = String.raw`{"act\u0069on" /* comment */ : "set:count:2", "text":"set:count:2", "nested":{"text":"plain"}}`;
    assert.deepEqual(highlighted(source), [
        ['set', 'function'], [':', 'operator'], ['count', 'parameter'], [':', 'operator'], ['2', 'number']
    ]);
    for (const key of ['on_text_changed', 'hover', 'unhover', 'drag', 'click', 'deselect', 'add_action', 'remove_action']) {
        assert.deepEqual(highlighted(JSON.stringify({ [key]: 'reset_settings' })), [['reset_settings', 'function']]);
    }
    assert.deepEqual(highlighted('{"action":null,"text":"plain","other":["plain"]}'), []);
    assert.deepEqual(highlighted('{"action":["plain"]}'), []);
    assert.deepEqual(highlighted('{"action":"set:count:'), [
        ['set', 'function'], [':', 'operator'], ['count', 'parameter'], [':', 'operator']
    ]);
});

test('plain-text brackets and escaped characters do not receive code colouring', () => {
    assert.deepEqual(highlighted(value('Text [with brackets]')), []);
    assert.deepEqual(highlighted(value(String.raw`\[literal\]`)), []);
    const source = JSON.stringify({ action: String.raw`show:'[quoted]':'[literal]':[{name},2]` });
    assert.deepEqual(highlighted(source), [
        ['show', 'function'], [':', 'operator'], ["'[quoted]'", 'string'], [':', 'operator'],
        ["'[literal]'", 'string'], [':', 'operator'],
        ['[', 'operator'], ['{', 'operator'], ['name', 'variable'], ['}', 'operator'], [',', 'operator'],
        ['2', 'number'], [']', 'operator']
    ]);
});

test('incomplete strings keep their own bracket depth and mismatched closers are marked', () => {
    const source = '{"first":"{if:{name", "second":"{value}", "third":"{a:[1}]"}';
    assert.deepEqual(tokenize(source).filter(token => token.bracketDepth !== undefined)
        .map(token => token.bracketDepth), [0, 1, 0, 0, 0, 1, -1, 1]);
    const unicode = String.raw`{"text":"\u007bname\u007d"}`;
    assert.deepEqual(tokenize(unicode).filter(token => token.bracketDepth !== undefined)
        .map(token => [unicode.slice(token.start, token.end), token.bracketDepth]),
        [[String.raw`\u007b`, 0], [String.raw`\u007d`, 0]]);
});
