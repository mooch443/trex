'use strict';

const vscode = require('vscode');
const { tokenize, tokenTypes, stringValues, elementTypes, fieldsForObject } = require('./tokens');

function activate(context) {
    const legend = new vscode.SemanticTokensLegend(tokenTypes);
    const bracketColors = [1, 2, 3].map(level => `editorBracketHighlight.foreground${level}`);
    bracketColors.push('editorBracketHighlight.unexpectedBracket.foreground');
    const decorations = bracketColors.map(color => vscode.window.createTextEditorDecorationType({
        color: new vscode.ThemeColor(color),
        rangeBehavior: vscode.DecorationRangeBehavior.ClosedClosed
    }));
    const matchingDecoration = vscode.window.createTextEditorDecorationType({
        backgroundColor: new vscode.ThemeColor('editorBracketMatch.background'),
        rangeBehavior: vscode.DecorationRangeBehavior.ClosedClosed
    });
    const documentBrackets = new WeakMap();
    let selector = [];
    const decorateMatching = editor => {
        const ranges = [];
        if (vscode.languages.match(selector, editor.document)) {
            const brackets = documentBrackets.get(editor.document) ?? [];
            const selected = new Set();
            for (const selection of editor.selections) {
                if (!selection.isEmpty) continue;
                const offset = editor.document.offsetAt(selection.active);
                // The smallest enclosing span is the current expression or action layer.
                let token;
                for (const bracket of brackets) {
                    if (bracket.start <= offset && offset < bracket.end
                        && (!token || bracket.end - bracket.start < token.end - token.start)) token = bracket;
                }
                token ??= brackets.find(token => token.end === offset);
                if (token) selected.add(token);
            }
            for (const token of selected) {
                const start = token.contentStart ?? token.start;
                const end = token.contentEnd ?? token.end;
                if (start >= end) continue;
                ranges.push(new vscode.Range(
                    editor.document.positionAt(start), editor.document.positionAt(end)));
            }
        }
        editor.setDecorations(matchingDecoration, ranges);
    };
    const decorate = editor => {
        const ranges = decorations.map(() => []);
        const brackets = [];
        if (vscode.languages.match(selector, editor.document)) {
            for (const token of tokenize(editor.document.getText(), brackets)) {
                if (token.bracketDepth === undefined) continue;
                const color = token.bracketDepth < 0 ? 3 : token.bracketDepth % 3;
                ranges[color].push(new vscode.Range(
                    editor.document.positionAt(token.start), editor.document.positionAt(token.end)));
            }
        }
        documentBrackets.set(editor.document, brackets);
        decorations.forEach((decoration, i) => editor.setDecorations(decoration, ranges[i]));
        decorateMatching(editor);
    };
    const decorateVisible = () => vscode.window.visibleTextEditors.forEach(decorate);
    const provider = {
        provideDocumentSemanticTokens(document, cancellation) {
            const builder = new vscode.SemanticTokensBuilder(legend);
            for (const token of tokenize(document.getText())) {
                if (cancellation.isCancellationRequested) return;
                const start = document.positionAt(token.start);
                const end = document.positionAt(token.end);
                builder.push(new vscode.Range(start, end), token.type);
            }
            return builder.build();
        }
    };
    const completionProvider = {
        provideCompletionItems(document, position, cancellation) {
            if (cancellation.isCancellationRequested || !vscode.languages.match(selector, document)) return;
            const offset = document.offsetAt(position);
            for (const string of stringValues(document.getText(), true)) {
                if (cancellation.isCancellationRequested) return;
                if (string.inLocals || offset < string.start || offset > string.end
                    || !/^[\w-]*$/.test(string.text)) continue;
                const fields = fieldsForObject(string.object);
                if (string.isKey) {
                    const names = Object.keys(fields).filter(name => name === string.text || !string.object.fields.has(name));
                    if (!names.length) return;
                    return names.map(name => {
                        const definition = fields[name];
                        const item = new vscode.CompletionItem(name, vscode.CompletionItemKind.Property);
                        item.detail = 'DynamicGUI field';
                        item.documentation = definition.description;
                        item.range = new vscode.Range(document.positionAt(string.start),
                            document.positionAt(string.end + (!string.hasColon && string.closed ? 1 : 0)));
                        if (string.hasColon) item.insertText = name;
                        else {
                            const value = definition.value === '""' ? '"$1"'
                                : definition.value === '{}' ? '{$1}' : definition.value === '[]' ? '[$1]'
                                : '${1:' + definition.value.replace(/[\\$}]/g, '\\$&') + '}';
                            item.insertText = new vscode.SnippetString(name + '": ' + value);
                        }
                        return item;
                    });
                }
                const definition = Object.hasOwn(fields, string.key) ? fields[string.key] : undefined;
                const values = string.key === 'type' ? elementTypes : definition?.values;
                if (!values) return;
                return values.map(type => {
                    const item = new vscode.CompletionItem(type, string.key === 'type'
                        ? vscode.CompletionItemKind.Class : vscode.CompletionItemKind.EnumMember);
                    item.detail = string.key === 'type' ? 'DynamicGUI element type' : definition.description;
                    item.range = new vscode.Range(document.positionAt(string.start), document.positionAt(string.end));
                    item.insertText = type + (string.closed ? '' : '"');
                    return item;
                });
            }
        }
    };
    const hoverProvider = {
        provideHover(document, position, cancellation) {
            if (cancellation.isCancellationRequested || !vscode.languages.match(selector, document)) return;
            const offset = document.offsetAt(position);
            for (const string of stringValues(document.getText(), true)) {
                if (cancellation.isCancellationRequested) return;
                if (!string.isKey || offset < string.start || offset >= string.end) continue;
                const fields = fieldsForObject(string.object);
                if (!Object.hasOwn(fields, string.text)) return;
                const definition = fields[string.text];
                const contents = new vscode.MarkdownString();
                contents.appendText(definition.description);
                contents.appendCodeblock(JSON.stringify(string.text) + ': ' + definition.value, 'json');
                return new vscode.Hover(contents,
                    new vscode.Range(document.positionAt(string.start), document.positionAt(string.end)));
            }
        }
    };

    let registration;
    let completionRegistration;
    let hoverRegistration;
    const register = () => {
        registration?.dispose();
        completionRegistration?.dispose();
        hoverRegistration?.dispose();
        const patterns = vscode.workspace.getConfiguration('commonsLayout').get('filePatterns', []);
        selector = patterns.flatMap(pattern => [
            { language: 'json', pattern },
            { language: 'jsonc', pattern }
        ]);
        registration = selector.length
            ? vscode.languages.registerDocumentSemanticTokensProvider(selector, provider, legend)
            : undefined;
        completionRegistration = selector.length
            ? vscode.languages.registerCompletionItemProvider(selector, completionProvider, '"')
            : undefined;
        hoverRegistration = selector.length ? vscode.languages.registerHoverProvider(selector, hoverProvider) : undefined;
        decorateVisible();
    };

    register();
    context.subscriptions.push(
        ...decorations,
        matchingDecoration,
        vscode.window.onDidChangeVisibleTextEditors(decorateVisible),
        vscode.window.onDidChangeTextEditorSelection(event => decorateMatching(event.textEditor)),
        vscode.workspace.onDidChangeTextDocument(event => {
            if (!event.contentChanges.length) return;
            vscode.window.visibleTextEditors.forEach(editor => {
                if (editor.document === event.document) decorate(editor);
            });
        }),
        vscode.workspace.onDidChangeConfiguration(event => {
            if (event.affectsConfiguration('commonsLayout.filePatterns')) register();
        }),
        { dispose: () => { registration?.dispose(); completionRegistration?.dispose(); hoverRegistration?.dispose(); } }
    );
}

module.exports = { activate };
