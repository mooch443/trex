'use strict';

const assert = require('node:assert/strict');
const fs = require('node:fs');
const path = require('node:path');
const vm = require('node:vm');
const test = require('node:test');
const { tokenize, tokenTypes, stringValues, elementTypes, fieldsForObject } = require('../tokens');
const manifest = require('../package.json');

test('the provider selects layout files, maps document positions and responds to configuration changes', () => {
    let patterns = manifest.contributes.configuration.properties['commonsLayout.filePatterns'].default;
    let registration, onConfiguration, onVisibleEditors, onDocumentChange, onSelectionChange, disposed = 0;
    let completionRegistration, completionsDisposed = 0;
    let hoverRegistration, hoversDisposed = 0;
    const vscode = {
        ThemeColor: class { constructor(id) { this.id = id; } },
        DecorationRangeBehavior: { ClosedClosed: 1 },
        SemanticTokensLegend: class { constructor(types) { this.tokenTypes = types; } },
        SemanticTokensBuilder: class {
            constructor(legend) { this.legend = legend; this.tokens = []; }
            push(range, type) {
                assert.ok(this.legend.tokenTypes.includes(type));
                assert.equal(range.start.line, range.end.line);
                this.tokens.push({ range, type });
            }
            build() { return this.tokens; }
        },
        Range: class { constructor(start, end) { this.start = start; this.end = end; } },
        CompletionItemKind: { Class: 6, Property: 9, EnumMember: 20 },
        CompletionItem: class { constructor(label, kind) { this.label = label; this.kind = kind; } },
        SnippetString: class { constructor(value) { this.value = value; } },
        MarkdownString: class {
            constructor() { this.value = ''; }
            appendText(value) { this.value += value; return this; }
            appendMarkdown(value) { this.value += value; return this; }
            appendCodeblock(value) { this.value += '\n' + value; return this; }
        },
        Hover: class { constructor(contents, range) { this.contents = contents; this.range = range; } },
        window: {
            visibleTextEditors: [],
            createTextEditorDecorationType: options => ({ options, dispose() {} }),
            onDidChangeVisibleTextEditors: callback => { onVisibleEditors = callback; return { dispose() {} }; },
            onDidChangeTextEditorSelection: callback => { onSelectionChange = callback; return { dispose() {} }; }
        },
        workspace: {
            getConfiguration: () => ({ get: () => patterns }),
            onDidChangeConfiguration: callback => { onConfiguration = callback; return { dispose() {} }; },
            onDidChangeTextDocument: callback => { onDocumentChange = callback; return { dispose() {} }; }
        },
        languages: {
            match: (selector, document) => selector.some(filter => filter.language === document.languageId
                && filter.pattern === '**/*layout*.json' && document.fileName.endsWith('_layout.json')) ? 10 : 0,
            registerDocumentSemanticTokensProvider(selector, provider, legend) {
                registration = { selector, provider, legend };
                return { dispose() { ++disposed; } };
            },
            registerCompletionItemProvider(selector, provider, ...triggers) {
                completionRegistration = { selector, provider, triggers };
                return { dispose() { ++completionsDisposed; } };
            },
            registerHoverProvider(selector, provider) {
                hoverRegistration = { selector, provider };
                return { dispose() { ++hoversDisposed; } };
            }
        }
    };
    const sandbox = { module: { exports: {} }, require: name => {
        if (name === 'vscode') return vscode;
        if (name === './tokens') return { tokenize, tokenTypes, stringValues, elementTypes, fieldsForObject };
        throw new Error(name);
    } };
    vm.runInNewContext(fs.readFileSync(path.join(__dirname, '../extension.js'), 'utf8'), sandbox);
    const context = { subscriptions: [] };
    sandbox.module.exports.activate(context);
    assert.equal(registration.selector.length, 4);
    assert.equal(registration.selector[0].language, 'json');
    assert.equal(registration.selector[0].pattern, '**/*layout*.json');
    assert.equal(registration.selector[1].language, 'jsonc');

    let source = '{\n  "text": "🐟 {name}"\n}';
    const document = {
        fileName: '/project/tracking_layout.json',
        languageId: 'json',
        getText: () => source,
        positionAt(offset) {
            const lines = source.slice(0, offset).split('\n');
            return { line: lines.length - 1, character: lines.at(-1).length };
        },
        offsetAt(position) {
            return source.split('\n').slice(0, position.line).reduce((offset, line) => offset + line.length + 1, 0)
                + position.character;
        }
    };
    const tokens = registration.provider.provideDocumentSemanticTokens(document, { isCancellationRequested: false });
    assert.equal(tokens.length, 3);
    assert.equal(tokens[1].type, 'variable');
    assert.equal(source.split('\n')[1].slice(tokens[1].range.start.character, tokens[1].range.end.character), 'name');
    assert.equal(registration.provider.provideDocumentSemanticTokens(document, { isCancellationRequested: true }), undefined);

    const colored = new Map();
    const editor = {
        document,
        selections: [{ active: document.positionAt(source.indexOf('{name}')), isEmpty: true }],
        setDecorations: (decoration, ranges) => colored.set(
            (decoration.options.color ?? decoration.options.backgroundColor).id, ranges)
    };
    vscode.window.visibleTextEditors = [editor];
    onVisibleEditors([editor]);
    assert.equal(colored.get('editorBracketHighlight.foreground1').length, 2);
    assert.equal(colored.get('editorBracketHighlight.foreground2').length, 0);
    const matched = () => Array.from(colored.get('editorBracketMatch.background') ?? [], range => [
        document.offsetAt(range.start), document.offsetAt(range.end)
    ]).sort((a, b) => a[0] - b[0]);
    const select = (...offsets) => {
        editor.selections = offsets.map(offset => ({ active: document.positionAt(offset), isEmpty: true }));
        onSelectionChange({ textEditor: editor, selections: editor.selections });
    };
    const opening = source.indexOf('{name}');
    const closing = source.indexOf('}', opening);
    assert.deepEqual(matched(), [[opening + 1, closing]]);
    select(closing + 1);
    assert.deepEqual(matched(), [[opening + 1, closing]]);
    select(opening + 3);
    assert.deepEqual(matched(), [[opening + 1, closing]]);
    select(0);
    assert.deepEqual(matched(), []);
    source = JSON.stringify({ text: '{a:{b:{c:{name}}}}' });
    onDocumentChange({ document, contentChanges: [{}] });
    assert.equal(colored.get('editorBracketHighlight.foreground1').length, 4);
    assert.equal(colored.get('editorBracketHighlight.foreground2').length, 2);
    assert.equal(colored.get('editorBracketHighlight.foreground3').length, 2);
    assert.equal(colored.get('editorBracketHighlight.unexpectedBracket.foreground').length, 0);
    const inner = source.indexOf('{name}');
    const innerClose = source.indexOf('}', inner);
    select(innerClose);
    assert.deepEqual(matched(), [[inner + 1, innerClose]]);
    select(inner + 3);
    assert.deepEqual(matched(), [[inner + 1, innerClose]]);
    select(source.indexOf('c:') + 1);
    assert.deepEqual(matched(), [[source.indexOf('{c:') + 1, innerClose + 1]]);
    select(innerClose + 1);
    assert.deepEqual(matched(), [[source.indexOf('{c:') + 1, innerClose + 1]]);
    select(source.indexOf('{a:'), inner);
    assert.equal(matched().length, 2);
    editor.selections[0].isEmpty = false;
    editor.selections[1].isEmpty = false;
    onSelectionChange({ textEditor: editor, selections: editor.selections });
    assert.deepEqual(matched(), []);
    source = JSON.stringify({ text: '{a:[1}]' });
    onDocumentChange({ document, contentChanges: [{}] });
    assert.equal(colored.get('editorBracketHighlight.unexpectedBracket.foreground').length, 1);
    select(source.indexOf('}'));
    assert.deepEqual(matched(), [[source.indexOf('[') + 1, source.indexOf(']')]]);
    select(source.indexOf('['));
    assert.equal(matched().length, 1);

    source = String.raw`{"text":"\u007bname\u007d"}`;
    onDocumentChange({ document, contentChanges: [{}] });
    select(source.indexOf('007b'));
    assert.deepEqual(matched(), [[15, 19]]);
    source = '{"text":"{name"}';
    onDocumentChange({ document, contentChanges: [{}] });
    assert.deepEqual(matched(), []);
    source = '{"first":"{name", "second":"value}"}';
    onDocumentChange({ document, contentChanges: [{}] });
    select(source.indexOf('{name'));
    assert.deepEqual(matched(), []);
    source = JSON.stringify({ text: String.raw`'{hidden}' \{literal\} {visible}` });
    onDocumentChange({ document, contentChanges: [{}] });
    select(source.indexOf('{hidden}'));
    assert.deepEqual(matched(), []);
    select(source.indexOf('{literal'));
    assert.deepEqual(matched(), []);
    select(source.indexOf('{visible}'));
    assert.equal(matched().length, 1);

    source = JSON.stringify({ text: 'Some [plain text] and {if:true:[1,{name}]:no}' });
    onDocumentChange({ document, contentChanges: [{}] });
    select(source.indexOf('plain'));
    assert.deepEqual(matched(), []);
    select(source.indexOf('1,'));
    assert.deepEqual(matched(), [[source.indexOf('[1') + 1, source.indexOf(']:no')]]);
    select(source.indexOf('name') + 2);
    assert.deepEqual(matched(), [[source.indexOf('{name}') + 1, source.indexOf('}]:no')]]);
    select(source.indexOf(':no') + 1);
    assert.deepEqual(matched(), [[source.indexOf('{if') + 1, source.indexOf(':no}') + 3]]);

    source = JSON.stringify({ action: 'set:value:{if:{ready}:[1,2]:false}', text: 'plain' });
    onDocumentChange({ document, contentChanges: [{}] });
    select(source.indexOf('value') + 2);
    assert.deepEqual(matched(), [[source.indexOf('set:'), source.indexOf('","text"')]]);
    select(source.indexOf('ready') + 2);
    assert.deepEqual(matched(), [[source.indexOf('{ready}') + 1, source.indexOf('{ready}') + 6]]);
    select(source.indexOf('plain') + 2);
    assert.deepEqual(matched(), []);

    source = JSON.stringify({ action: '{if:{ready}:go:stop}' });
    onDocumentChange({ document, contentChanges: [{}] });
    select(source.indexOf('{if'));
    assert.deepEqual(matched(), [[source.indexOf('{if') + 1, source.indexOf(':stop}') + 5]]);
    source = JSON.stringify({ text: '{if:true:[]:{}}' });
    onDocumentChange({ document, contentChanges: [{}] });
    select(source.indexOf('[]'));
    assert.deepEqual(matched(), []);
    select(source.indexOf('{}') + 1);
    assert.deepEqual(matched(), []);

    source = JSON.stringify({ text: '{name}' });
    onDocumentChange({ document, contentChanges: [{}] });
    select(source.indexOf('{name}'));
    assert.equal(matched().length, 1);

    assert.ok(completionRegistration, 'element type completion provider is registered');
    assert.equal(completionRegistration.selector[0].pattern, '**/*layout*.json');
    assert.ok(completionRegistration.triggers.includes('"'));
    const complete = offset => completionRegistration.provider.provideCompletionItems(
        document, document.positionAt(offset), { isCancellationRequested: false });
    source = '{"objects":[{"type":"bu"}]}';
    const completions = complete(source.indexOf('bu') + 2);
    assert.equal(completions.length, 20);
    const button = completions.find(item => item.label === 'button');
    assert.equal(button.kind, vscode.CompletionItemKind.Class);
    assert.equal(button.insertText, 'button');
    const replaced = source.slice(0, document.offsetAt(button.range.start)) + button.insertText
        + source.slice(document.offsetAt(button.range.end));
    assert.equal(JSON.parse(replaced).objects[0].type, 'button');
    assert.equal(complete(source.indexOf('type') + 2), undefined);
    assert.equal(completionRegistration.provider.provideCompletionItems(document,
        document.positionAt(source.indexOf('bu') + 2), { isCancellationRequested: true }), undefined);
    source = '{"type":"button"}';
    assert.equal(complete(source.indexOf('button') + 2).find(item => item.label === 'button').range.end.character,
        source.indexOf('button') + 'button'.length);
    source = '{"type":""}';
    assert.equal(complete(source.indexOf('""') + 1).length, 20);
    source = '{"type":"bu';
    assert.equal(complete(source.length).find(item => item.label === 'button').insertText, 'button"');
    source = '{\n "objects": [{"type" /* comment */:\n "te"}]\n}';
    assert.ok(complete(source.indexOf('"te"') + 3).some(item => item.label === 'textfield'));
    source = '{"locals":{"mode":{"type":"bu"}},"objects":[{"type":"bu"}]}';
    assert.equal(complete(source.indexOf('bu') + 2), undefined);
    assert.equal(complete(source.lastIndexOf('bu') + 2).length, 20);
    source = '// "type": "bu"\n{"text":"bu", "type":"{kind}"}';
    assert.equal(complete(source.indexOf('bu') + 2), undefined);
    assert.equal(complete(source.indexOf('"text":"bu"') + 10), undefined);
    assert.equal(complete(source.indexOf('kind') + 2), undefined);
    source = '{"type":"bu"}';
    document.fileName = '/project/ordinary.json';
    assert.equal(complete(source.indexOf('bu') + 2), undefined);
    document.fileName = '/project/tracking_layout.json';

    source = '{"type":"button","text":"Run","ac"}';
    const properties = complete(source.indexOf('"ac"') + 3);
    assert.ok(properties?.some(item => item.label === 'action'), 'known button properties are suggested');
    assert.ok(!properties.some(item => ['text', 'type', 'radius', 'on_text_changed'].includes(item.label)));
    const action = properties.find(item => item.label === 'action');
    assert.equal(action.kind, vscode.CompletionItemKind.Property);
    const inserted = action.insertText.value.replace(/\$1/g, '');
    assert.deepEqual(JSON.parse(source.slice(0, document.offsetAt(action.range.start)) + inserted
        + source.slice(document.offsetAt(action.range.end))), { type: 'button', text: 'Run', action: '' });
    source = '{"act":"run", "type":"button"}';
    const renamed = complete(source.indexOf('act') + 2).find(item => item.label === 'action');
    assert.equal(renamed.insertText, 'action');
    assert.equal(JSON.parse(source.slice(0, document.offsetAt(renamed.range.start)) + renamed.insertText
        + source.slice(document.offsetAt(renamed.range.end))).action, 'run');
    source = '{"type":"button","font":{"si"}}';
    const fontProperties = complete(source.indexOf('si') + 2);
    assert.deepEqual(Array.from(fontProperties, item => item.label).sort(), ['align', 'size', 'style']);
    source = '{"type":"vlayout","align":""}';
    assert.deepEqual(Array.from(complete(source.indexOf('""') + 1), item => item.label), ['left', 'center', 'right']);
    source = '{"type":"hlayout","align":""}';
    assert.deepEqual(Array.from(complete(source.indexOf('""') + 1), item => item.label), ['top', 'center', 'bottom']);
    source = '{"type":"button","font":{"style":""}}';
    assert.deepEqual(Array.from(complete(source.indexOf('""') + 1), item => item.label), ['regular', 'bold', 'italic', 'mono']);
    source = '{"type":"button","data":{"si"}}';
    assert.equal(complete(source.indexOf('si') + 2), undefined);
    source = '{"type":"custom_widget","ac"}';
    assert.equal(complete(source.indexOf('ac') + 2), undefined);
    source = '{"type":"button","action":"run","radius":3}';
    assert.ok(hoverRegistration);
    const hover = offset => hoverRegistration.provider.provideHover(document, document.positionAt(offset),
        { isCancellationRequested: false });
    assert.match(hover(source.indexOf('action') + 2).contents.value, /Action command/);
    assert.equal(hover(source.indexOf('radius') + 2), undefined);
    assert.equal(hoverRegistration.provider.provideHover(document, document.positionAt(source.indexOf('action') + 2),
        { isCancellationRequested: true }), undefined);

    source = JSON.stringify({ text: '{name}' });
    onDocumentChange({ document, contentChanges: [{}] });
    select(source.indexOf('{name}'));
    assert.equal(matched().length, 1);

    patterns = ['**/my_gui.json'];
    onConfiguration({ affectsConfiguration: setting => setting === 'commonsLayout.filePatterns' });
    assert.equal(disposed, 1);
    assert.equal(completionsDisposed, 1);
    assert.equal(hoversDisposed, 1);
    assert.equal(completionRegistration.selector[0].pattern, '**/my_gui.json');
    assert.equal(complete(source.indexOf('name') + 2), undefined);
    assert.equal(registration.selector[0].pattern, '**/my_gui.json');
    assert.ok([...colored.values()].every(ranges => ranges.length === 0));
    select(source.indexOf('{name}'));
    assert.deepEqual(matched(), []);
    patterns = [];
    onConfiguration({ affectsConfiguration: () => true });
    assert.equal(disposed, 2);
    assert.equal(completionsDisposed, 2);
    assert.equal(hoversDisposed, 2);
    context.subscriptions.forEach(subscription => subscription.dispose());
});
