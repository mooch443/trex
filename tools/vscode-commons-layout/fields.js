'use strict';

// Fields consumed by ParseLayoutTypes.cpp, DynamicGUI.cpp and LabeledField.cpp.
const field = (description, value = '""', values) => ({ description, value, values });
const fontFields = {
    size: field('Font size.', '0.75'),
    style: field('Font style.', '"regular"', ['regular', 'bold', 'italic', 'mono']),
    align: field('Text alignment.', '"left"', ['left', 'right', 'center', 'vcenter'])
};
const font = { ...field('Font size, style and alignment.', '{}'), children: fontFields };
const color = field('Colour as [red, green, blue] or [red, green, blue, alpha].', '[255, 255, 255, 255]');
const size = field('Width and height.', '[100, 30]');
const pad = field('Inner padding.', '[0, 0, 0, 0]');
const corners = field('Corner rounding flags and radius.', '["all", 8]');
const action = field('Action command, with optional colon-separated arguments.');
const text = field('Displayed text; may contain commons expressions.');
const disabled = field('Whether this item is disabled.', 'false');
const children = field('Child control objects.', '[]');
const minSize = field('Minimum layout width and height.', '[0, 0]');
const listItemFields = { text, detail: field('Secondary item text.'), tooltip: field('Item tooltip.'), disabled, action };
const itemFields = { pad: field('Item padding [x, y].', '[8, 4]'), font, fill: color, line: color, color, corners };
const labelFields = { line: color, fill: color, color, font, corners };
const listFields = { line: color, fill: color, size };
const objectField = (description, children) => ({ ...field(description, '{}'), children });

const commonFields = {
    type: field('DynamicGUI element type.'),
    name: field('Name used to reference this control.'),
    pos: field('Position [x, y].', '[0, 0]'),
    size, scale: field('Scale [x, y].', '[1, 1]'), origin: field('Origin [x, y].', '[0, 0]'),
    pad, outer_pad: field('Outer padding.', '[0, 0, 0, 0]'),
    fill: color, line: color, color, corners, font,
    max_size: field('Maximum width and height.', '[100, 100]'),
    clickable: field('Whether this control accepts clicks.', 'false'),
    'pointer-events': field('Pointer event policy.', '"all"', ['none', 'hover', 'click', 'drag', 'scroll', 'all']),
    'z-index': field('Drawing order.', '0'),
    modules: field('Registered module names applied to this control.', '[]'),
    hover: action, unhover: action, drag: action, click: action, deselect: action,
    tooltip: field('Tooltip text for controls that support tooltips.')
};

const controlFields = {
    each: { var: field('Collection expression to iterate.'), as: field('Loop item name.', '"i"'),
        do: field('Control template repeated for each item.', '{}') },
    condition: { var: field('Condition expression.'), then: field('Control shown when the condition is true.', '{}'),
        else: field('Control shown when the condition is false.', '{}') },
    vlayout: { children, min_size: minSize,
        align: field('Horizontal alignment of children.', '"left"', ['left', 'center', 'right']) },
    hlayout: { children, min_size: minSize,
        align: field('Vertical alignment of children.', '"top"', ['top', 'center', 'bottom']) },
    floatinglayout: { children, min_size: minSize,
        policy: field('Direction used to pack children.', '"horizontal-first"', ['horizontal-first', 'vertical-first']) },
    gridlayout: { children: field('Rows of cells containing control objects or arrays of controls.', '[]'),
        halign: field('Horizontal cell alignment.', '"left"', ['left', 'center', 'right']),
        valign: field('Vertical cell alignment.', '"top"', ['top', 'center', 'bottom']),
        vertical_clr: color, horizontal_clr: color, cellfillclr: color, celllineclr: color,
        cellfillinterval: field('Cell fill interval.', '1'), mincellsize: field('Minimum cell dimensions.', '[0, 0]') },
    collection: { children },
    line: { from: field('Start point [x, y].', '[0, 0]'), to: field('End point [x, y].', '[100, 0]'),
        points: field('Polyline points.', '[[0, 0], [100, 0]]'), thickness: field('Line thickness.', '1'),
        closed: field('Whether to close a polyline.', 'false') },
    button: { text, disabled, action },
    circle: { radius: field('Circle radius.', '5') },
    text: { text, shadow: field('Text shadow amount.', '0') },
    stext: { text, shadow: field('Text shadow amount.', '0'), fade_out: field('Text fade-out amount.', '0'),
        alpha: field('Text opacity.', '1') },
    rect: {},
    textfield: { text, action: field('Action on Enter; {text} contains the current input.'),
        on_text_changed: field('Action while editing; {text} contains the current input.') },
    checkbox: { text, checked: field('Checked state.', 'false'), action },
    settings: { var: field('Name of the setting to edit.'), invert: field('Invert the setting control.', 'false'),
        cleartext: field('Clear text for a setting combobox.'), desc: field('Setting description.'),
        placeholder: field('Placeholder text.'), postfix: field('Text appended to the setting value.'),
        auto_limit_size: field('Automatically limit the label size.', 'true'),
        always_open: field('Keep the setting dropdown open.', 'false'), highlight_clr: color,
        label: objectField('Label appearance.', { ...labelFields, size }),
        list: objectField('Dropdown list appearance.', { ...listFields, font }),
        preview: objectField('Path-array preview appearance.', { font, fade_out: field('Preview fade-out amount.', '0'),
            color, max_size: size }) },
    combobox: {},
    image: { path: field('Image file path or path expression.') },
    list: { var: field('Collection expression for a dynamic list.'),
        template: objectField('Dynamic list item template.', listItemFields),
        items: { ...field('Static list items.', '[]'), items: listItemFields },
        placeholder: field('Placeholder text.'), alternate: field('Alternate row appearance.', 'false'),
        label: objectField('Folded label appearance.', { ...labelFields, text }),
        list: objectField('List appearance.', listFields), item: objectField('Row appearance.', itemFields),
        detail: objectField('Secondary text appearance.', { color, font }),
        foldable: field('Allow the list to fold.', 'false'), folded: field('Initial folded state.', 'false') },
    taglist: { var: field('Selected tags, as a string-list source.', '[]'),
        catalog: field('Available tags, as a string-list source.', '[]'),
        allow_new: field('Allow new tag names.', 'true'), match_threshold: field('Tag matching threshold.', '1'),
        item: objectField('Tag appearance.', itemFields),
        input: objectField('Tag input appearance.', { size, list_size: size, placeholder: field('Placeholder text.'),
            font, fill: color, line: color, color }), add_action: action, remove_action: action }
};

const rootFields = {
    objects: field('Top-level control objects.', '[]'),
    defaults: objectField('Default layout appearance and variables.', { font, color, fill: color, line: color,
        highlight_clr: color, window_color: color, 'pointer-events': commonFields['pointer-events'], pad,
        outer_pad: commonFields.outer_pad, vars: field('Named expressions available to the layout.', '{}') }),
    locals: field('Layout-local setting declarations.', '{}')
};

function fieldsForObject(object) {
    if (!object || object.kind !== 'object' || object.inLocals) return {};
    const parent = object.parent;
    if (parent) {
        const owner = parent.kind === 'array' ? parent.parent : parent;
        const key = parent.kind === 'array' ? parent.key : object.key;
        const fields = fieldsForObject(owner);
        const nested = Object.hasOwn(fields, key) ? fields[key] : undefined;
        const children = parent.kind === 'array' ? nested?.items : nested?.children;
        if (children) return children;
    }
    if (Object.hasOwn(controlFields, object.type)) return { ...commonFields, ...controlFields[object.type] };
    return !parent && !object.type ? rootFields : {};
}

module.exports = { fieldsForObject };
