# Commons Layout Expressions

Highlights commons expressions inside JSON layout string values. Function names,
variables, numbers, boolean values, separators and quoted literals use your VS
Code theme's code colours. Built-in DynamicGUI element types also get type
colours and completion suggestions. Known fields are highlighted for their owning
control, with field completion, value suggestions and hover help.

```json
{
  "text": "Hello {name}",
  "size": "{+:10:{*:2:{width}}}",
  "tooltip": "{if:{ready}:Hello {name}:'Waiting {name}'}"
}
```

In the last example, `if`, `ready` and the unquoted `name` are code. The
single-quoted `'Waiting {name}'` remains literal text. Highlighting begins as
soon as you type `{`, including before the expression or JSON string is closed.

## Install

In VS Code, open **Extensions**, select **… → Install from VSIX…**, and choose
`commons-layout.vsix`. To install or replace the local extension:

```sh
code --install-extension tools/vscode-commons-layout/commons-layout.vsix --force
```

Open a layout file in **JSON** or **JSON with Comments** mode. The extension
adds expression colours to that mode, preserving JSON formatting and validation.

## File selection

The default patterns are `**/*layout*.json` and `**/*layout*.jsonc`, covering
files such as `tracking_layout.json` and `settings_layout.json`. To include
differently named layouts, set this in your VS Code settings:

```json
"commonsLayout.filePatterns": [
  "**/*layout*.json",
  "**/*layout*.jsonc",
  "**/my_gui.json"
]
```

Changes take effect immediately. Use `[]` to disable highlighting and completion,
or `"**/*.json"` to include every JSON file. Unrecognised JSON property names
and comments keep their normal highlighting.

## Element types

Inside a `"type": "..."` value, built-in DynamicGUI names such as `button`,
`textfield` and `vlayout` use your theme's type colour. Opening the value's
quote triggers completion suggestions; use **Ctrl+Space** to request them while
editing an existing value. Accepting a suggestion replaces the value and keeps
its JSON quotes intact.

The built-in names are `each`, `condition`, `vlayout`, `hlayout`,
`floatinglayout`, `gridlayout`, `collection`, `line`, `button`, `circle`,
`text`, `stext`, `rect`, `textfield`, `checkbox`, `settings`, `combobox`,
`image`, `list` and `taglist`.

These suggestions apply to the configured layout files, excluding comments,
other string fields and top-level `locals` declarations. Custom module names
remain editable but are not included in the built-in suggestions.

## Control fields

Known field names use your theme's keyword colour. Recognition follows the
owning control: `action` is known on a button, `on_text_changed` on a textfield,
and `radius` on a circle. Property order does not matter. Nested `font`, list
templates, row styles, tag-list input styles and layout defaults have their own
field sets. Unknown fields and custom control properties retain normal JSON
colouring; this is editing assistance, not runtime validation.

Open a quoted property name or press **Ctrl+Space** inside it to suggest fields
for that object. Fields already present are omitted. New fields include a value
placeholder; completing an existing field name preserves its value. Hover a
known field for a short explanation and an example value.

Value completion also covers font styles, font alignment, layout alignment,
floating-layout policies and pointer-event policies. For example, `vlayout`
offers `left`, `center`, `right`, while `hlayout` offers `top`, `center`, `bottom`.

To customise known-field colouring, set the `commonsField` rule in
`editor.semanticTokenColorCustomizations`.

## Actions

Action values use `name:arguments` without outer braces. For example,
`"action": "set:gui_run:true"` highlights the command, separators and arguments,
and `"action": "reset_settings"` highlights the command without parameters.
Nested expressions such as `"action": "set:value:{text}"` retain their own
highlighting. Fully braced generated commands use normal expression syntax.

The same rules apply to `on_text_changed`, `hover`, `unhover`, `drag`, `click`,
`deselect`, `add_action` and `remove_action` string values.

## Quoting and escaping

- Single quotes suppress expression highlighting, including inside arrays.
- JSON's outer double quotes do not suppress expressions. Double quotes within
  the decoded template also do not create template literals.
- Escaping is interpreted in two stages: JSON first, then the template. For
  example, the JSON value `"\\{name\\}"` contains escaped, literal braces,
  while `"{name}"` contains an expression.
- Unicode JSON escapes follow the same rules as their decoded characters.
- Template escapes use normal string colouring, and escaped brackets do not
  contribute to nesting.
- The extension only highlights text; it does not evaluate expressions.

## Colours

The extension uses VS Code's standard semantic token colours for types,
functions, variables, parameters, numbers, keywords, strings and operators. It enables
semantic highlighting by default for JSON and JSON with Comments. An explicit
user setting takes precedence; if you have disabled semantic highlighting,
enable `editor.semanticHighlighting.enabled` for those language modes.

Expression braces `{…}` and collection brackets `[…]` inside expressions or
actions are also coloured by nesting depth. Square brackets in ordinary text
remain plain text. Each matching pair has the same colour; deeper levels cycle
through the first three bracket colours of your theme. Quoted and escaped
brackets do not affect the nesting. Colours update as you type, and an
unexpected closing bracket uses the theme's unexpected-bracket colour.

To customise this palette, change `editorBracketHighlight.foreground1`,
`editorBracketHighlight.foreground2` and `editorBracketHighlight.foreground3`
in `workbench.colorCustomizations`. Outer JSON brackets retain VS Code's
normal bracket colouring.

The contents of the innermost bracketed expression containing the cursor get a
background highlight, excluding its opening and closing brackets. Moving out of a nested
expression returns the highlight to its enclosing layer. Unbraced action values
act as an outer layer of their own. This also works with multiple cursors.
Quoted, escaped and unmatched brackets do not create layers. The background
uses `editorBracketMatch.background` from your theme.

## Development

There are no runtime dependencies or compilation steps. From the repository
root, run the tokenizer checks with Node.js 18 or newer:

```sh
node --test tools/vscode-commons-layout/test/*.test.js
```

From the repository root, package the extension with:

```sh
KMP_DUPLICATE_LIB_OK=TRUE conda run -n trex python tools/vscode-commons-layout/package.py
```

Each run increments the internal patch version in `package.json` and the VSIX
manifest, overwrites `commons-layout.vsix`, and removes older numbered packages.
The package contains
`package.json`, `extension.js`, `tokens.js`, `fields.js`, this README and the
repository's GPL licence.

Check the packaging workflow with:

```sh
KMP_DUPLICATE_LIB_OK=TRUE conda run -n trex python tools/vscode-commons-layout/test/package_test.py
```

The highlighting API and theme behaviour are described in the
[VS Code semantic highlighting guide](https://code.visualstudio.com/api/language-extensions/semantic-highlight-guide).
