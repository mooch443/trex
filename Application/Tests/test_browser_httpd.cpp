#include "gtest/gtest.h"

#include <gui/BrowserBase.h>
#include <gui/BrowserProtocol.h>
#include <gui/DrawStructure.h>
#include <gui/RenderTraversal.h>
#include <gui/types/Entangled.h>
#include <gui/types/Layout.h>
#include <gui/types/StaticText.h>
#include <gui/types/Textfield.h>

using namespace cmn;
using namespace cmn::gui;
using namespace cmn::gui::browser;

namespace {
class WireReader {
    std::span<const uint8_t> _bytes;
    size_t _offset{0};

public:
    explicit WireReader(std::span<const uint8_t> bytes) : _bytes(bytes) {}

    uint8_t u8() {
        EXPECT_LE(_offset + 1, _bytes.size());
        return _bytes[_offset++];
    }

    uint16_t u16() {
        EXPECT_LE(_offset + 2, _bytes.size());
        const auto value = uint16_t(_bytes[_offset]) | (uint16_t(_bytes[_offset + 1]) << 8);
        _offset += 2;
        return value;
    }

    uint32_t u32() {
        EXPECT_LE(_offset + 4, _bytes.size());
        uint32_t value = 0;
        for(size_t i = 0; i < 4; ++i)
            value |= uint32_t(_bytes[_offset + i]) << (i * 8);
        _offset += 4;
        return value;
    }

    uint64_t u64() {
        EXPECT_LE(_offset + 8, _bytes.size());
        uint64_t value = 0;
        for(size_t i = 0; i < 8; ++i)
            value |= uint64_t(_bytes[_offset + i]) << (i * 8);
        _offset += 8;
        return value;
    }

    float f32() {
        return std::bit_cast<float>(u32());
    }

    void skip(size_t size) {
        ASSERT_LE(_offset + size, _bytes.size());
        _offset += size;
    }

    size_t remaining() const {
        return _bytes.size() - _offset;
    }
};

struct SceneWire {
    MessageType type{MessageType::Error};
    uint64_t sequence{0};
    Size2 viewport;
    std::vector<uint32_t> upserts;
    std::vector<uint32_t> removals;
    std::vector<uint32_t> order;
    bool order_present{false};
};

void read_header(WireReader& reader, MessageType expected, uint64_t* sequence = nullptr) {
    EXPECT_EQ(reader.u32(), protocol_magic);
    EXPECT_EQ(reader.u16(), protocol_version);
    EXPECT_EQ(reader.u16(), static_cast<uint16_t>(expected));
    const auto value = reader.u64();
    if(sequence)
        *sequence = value;
}

std::vector<uint32_t> read_order(WireReader& reader) {
    std::vector<uint32_t> order(reader.u32());
    for(auto& id : order)
        id = reader.u32();
    return order;
}

SceneWire read_scene(std::span<const uint8_t> bytes) {
    WireReader reader(bytes);
    EXPECT_EQ(reader.u32(), protocol_magic);
    EXPECT_EQ(reader.u16(), protocol_version);

    SceneWire scene;
    scene.type = static_cast<MessageType>(reader.u16());
    scene.sequence = reader.u64();
    scene.viewport = {reader.f32(), reader.f32()};

    const auto entry_count = reader.u32();
    scene.upserts.reserve(entry_count);
    for(uint32_t i = 0; i < entry_count; ++i) {
        scene.upserts.push_back(reader.u32());
        reader.skip(reader.u32());
    }

    if(scene.type == MessageType::Snapshot) {
        scene.order = read_order(reader);
        scene.order_present = true;
    } else {
        EXPECT_EQ(scene.type, MessageType::Delta);
        const auto removal_count = reader.u32();
        scene.removals.reserve(removal_count);
        for(uint32_t i = 0; i < removal_count; ++i)
            scene.removals.push_back(reader.u32());
        scene.order_present = reader.u8() != 0;
        if(scene.order_present)
            scene.order = read_order(reader);
    }
    EXPECT_EQ(reader.remaining(), 0u);
    return scene;
}

class PacketWriter {
    std::vector<uint8_t> _bytes;

public:
    PacketWriter(MessageType type, uint64_t sequence) {
        u32(protocol_magic);
        u16(protocol_version);
        u16(static_cast<uint16_t>(type));
        u64(sequence);
    }

    void u8(uint8_t value) { _bytes.push_back(value); }
    void u16(uint16_t value) {
        for(size_t i = 0; i < 2; ++i)
            _bytes.push_back(static_cast<uint8_t>(value >> (i * 8)));
    }
    void u32(uint32_t value) {
        for(size_t i = 0; i < 4; ++i)
            _bytes.push_back(static_cast<uint8_t>(value >> (i * 8)));
    }
    void u64(uint64_t value) {
        for(size_t i = 0; i < 8; ++i)
            _bytes.push_back(static_cast<uint8_t>(value >> (i * 8)));
    }
    void f32(float value) { u32(std::bit_cast<uint32_t>(value)); }
    void string(std::string_view value) {
        u32(narrow_cast<uint32_t>(value.size()));
        _bytes.insert(_bytes.end(), value.begin(), value.end());
    }
    const std::vector<uint8_t>& bytes() const { return _bytes; }
};

RenderCommand command_for(Drawable& drawable) {
    Transform transform;
    transform.combine(drawable.global_transform_no_rotation());
    Transform full_transform;
    full_transform.combine(drawable.global_transform());
    return {
        RenderCommand::DEFAULT,
        0,
        &drawable,
        transform,
        full_transform,
        transform.transformRect(Bounds(0, 0, drawable.width(), drawable.height())),
        {}
    };
}

class KeyProbe final : public Drawable {
public:
    std::optional<KeyEvent> down;
    std::optional<KeyEvent> up;

    KeyProbe()
        : Drawable(Type::RECT, Bounds(0, 0, 20, 20))
    {}

protected:
    bool kdown(Event event) override {
        down = event.key;
        return true;
    }

    bool kup(Event event) override {
        up = event.key;
        return true;
    }
};

class MetricBase final : public Base {
    std::string _title;
    Bounds _window;
    Bounds _text;
    Float2_t _spacing;

public:
    MetricBase(Bounds text, Float2_t spacing)
        : _text(text), _spacing(spacing)
    {}

    void paint(DrawStructure&) override {}
    void set_title(std::string title) override { _title = std::move(title); }
    void set_window_size(Size2 size) override { _window << size; }
    void set_window_bounds(Bounds bounds) override { _window = bounds; }
    Bounds get_window_bounds() const override { return _window; }
    const std::string& title() const override { return _title; }
    Bounds text_bounds(const std::string&, Drawable*, const Font&) override { return _text; }
    Float2_t line_spacing(const Font&) override { return _spacing; }
};
}

TEST(BrowserTraversal, PreservesNestedTransformClipAndZOrder) {
    DrawStructure graph(320, 240);
    Rect normal(Box(2, 3, 8, 9), FillClr{Blue});
    Entangled clipped(Box(10, 20, 100, 80));
    clipped.set_scroll_enabled(true);
    auto child = Layout::Make<Rect>{Box(5, 6, 20, 10), FillClr{Red}, ZIndex{3}}();
    clipped.update([&](Entangled& layout) {
        layout.advance_wrap(*child);
    });

    graph.wrap_object(normal);
    graph.wrap_object(clipped);
    const auto commands = RenderTraversal::collect(graph, {
        .scale = {1, 1},
        .viewport = {320, 240},
        .cull = true
    });

    const auto found = std::find_if(commands.begin(), commands.end(), [&](const auto& command) {
        return command.ptr == child.get() && command.type == RenderCommand::DEFAULT;
    });
    ASSERT_NE(found, commands.end());
    EXPECT_TRUE(found->has_clip());
    EXPECT_FLOAT_EQ(found->_clip_rect.x, 10);
    EXPECT_FLOAT_EQ(found->_clip_rect.y, 20);
    EXPECT_FLOAT_EQ(found->_clip_rect.z, 100);
    EXPECT_FLOAT_EQ(found->_clip_rect.w, 110);
    EXPECT_EQ(found->full_transform.transformPoint(Vec2()), Vec2(15, 26));
    EXPECT_EQ(commands.back().ptr, child.get());
}

TEST(BrowserProtocol, StableIdsIdleDeltaRemovalAndOrder) {
    Rect first(Box(10, 20, 30, 40), FillClr{Red});
    Rect second(Box(50, 60, 20, 10), FillClr{Blue});
    auto first_command = command_for(first);
    auto second_command = command_for(second);
    SceneEncoder encoder;

    auto update = encoder.encode({first_command, second_command}, {320, 240});
    ASSERT_TRUE(update.changed);
    ASSERT_TRUE(update.first_snapshot);
    ASSERT_TRUE(update.outbound);
    const auto snapshot = read_scene(*update.outbound);
    ASSERT_EQ(snapshot.type, MessageType::Snapshot);
    ASSERT_EQ(snapshot.upserts.size(), 2u);
    ASSERT_EQ(snapshot.order, snapshot.upserts);
    const auto first_id = snapshot.order[0];
    const auto second_id = snapshot.order[1];

    update = encoder.encode({first_command, second_command}, {320, 240});
    EXPECT_FALSE(update.changed);
    EXPECT_FALSE(update.outbound);
    EXPECT_EQ(update.sequence, snapshot.sequence);

    first.set_fillclr(Green);
    update = encoder.encode({first_command, second_command}, {320, 240});
    ASSERT_TRUE(update.changed);
    const auto changed = read_scene(*update.outbound);
    EXPECT_EQ(changed.type, MessageType::Delta);
    EXPECT_EQ(changed.upserts, std::vector<uint32_t>{first_id});
    EXPECT_TRUE(changed.removals.empty());
    EXPECT_FALSE(changed.order_present);

    update = encoder.encode({second_command, first_command}, {320, 240});
    const auto reordered = read_scene(*update.outbound);
    EXPECT_TRUE(reordered.upserts.empty());
    EXPECT_TRUE(reordered.removals.empty());
    EXPECT_TRUE(reordered.order_present);
    EXPECT_EQ(reordered.order, (std::vector<uint32_t>{second_id, first_id}));

    update = encoder.encode({second_command}, {320, 240});
    const auto removed = read_scene(*update.outbound);
    EXPECT_EQ(removed.removals, std::vector<uint32_t>{first_id});
    EXPECT_EQ(removed.order, std::vector<uint32_t>{second_id});
}

TEST(BrowserProtocol, ImageRevisionIsIndependentFromStyle) {
    auto pixels = Image::Make(4, 4, 4);
    std::fill_n(pixels->data(), pixels->size(), uint8_t(255));
    ExternalImage image(std::move(pixels), Vec2());
    auto command = command_for(image);
    SceneEncoder encoder;

    auto update = encoder.encode({command}, {100, 100});
    ASSERT_EQ(update.images.size(), 1u);
    const auto first_revision = update.images.front().revision;

    image.set_color(Color(120, 140, 160, 180));
    update = encoder.encode({command}, {100, 100});
    EXPECT_TRUE(update.changed);
    EXPECT_TRUE(update.images.empty());

    auto replacement = Image::Make(4, 4, 4);
    std::fill_n(replacement->data(), replacement->size(), uint8_t(127));
    image.update_with(std::move(replacement));
    update = encoder.encode({command}, {100, 100});
    ASSERT_EQ(update.images.size(), 1u);
    EXPECT_NE(update.images.front().revision, first_revision);

    update = encoder.encode({}, {100, 100});
    EXPECT_EQ(update.removed_resources, std::vector<uint32_t>{1});
}

TEST(BrowserImageEncoding, SelectsJpegForOpaqueAndPngForAlpha) {
    Image opaque(32, 32, 4);
    for(size_t i = 0; i < opaque.size(); i += 4) {
        opaque.data()[i + 0] = uint8_t(i % 251);
        opaque.data()[i + 1] = uint8_t((i / 3) % 241);
        opaque.data()[i + 2] = uint8_t((i / 7) % 239);
        opaque.data()[i + 3] = 255;
    }

    const auto low_quality = encode_image(opaque, 10);
    const auto high_quality = encode_image(opaque, 90);
    ASSERT_FALSE(low_quality.png);
    ASSERT_FALSE(high_quality.png);
    ASSERT_GE(low_quality.bytes.size(), 2u);
    ASSERT_GE(high_quality.bytes.size(), 2u);
    EXPECT_EQ(low_quality.bytes[0], 0xff);
    EXPECT_EQ(low_quality.bytes[1], 0xd8);
    EXPECT_NE(low_quality.bytes, high_quality.bytes);

    opaque.data()[3] = 0;
    const auto alpha = encode_image(opaque, 75);
    ASSERT_TRUE(alpha.png);
    const std::array<uint8_t, 8> signature{0x89, 'P', 'N', 'G', 0x0d, 0x0a, 0x1a, 0x0a};
    ASSERT_GE(alpha.bytes.size(), signature.size());
    EXPECT_TRUE(std::equal(signature.begin(), signature.end(), alpha.bytes.begin()));
}

TEST(BrowserInputProtocol, DecodesUnicodeModifiersAndRejectsInvalidPackets) {
    PacketWriter text(MessageType::Input, 42);
    text.u8(static_cast<uint8_t>(InputKind::Text));
    text.u8(Modifier::Shift | Modifier::Control);
    text.u32(0x1f642);
    auto decoded = decode_client_message(text.bytes());
    ASSERT_TRUE(decoded);
    ASSERT_TRUE(decoded->input);
    EXPECT_EQ(decoded->sequence, 42u);
    EXPECT_EQ(decoded->input->kind, InputKind::Text);
    EXPECT_EQ(decoded->input->codepoint, 0x1f642u);
    EXPECT_EQ(decoded->input->modifiers, Modifier::Shift | Modifier::Control);

    PacketWriter key(MessageType::Input, 43);
    key.u8(static_cast<uint8_t>(InputKind::Key));
    key.u8(Modifier::Alt);
    key.u8(1);
    key.u8(1);
    key.string("KeyZ");
    decoded = decode_client_message(key.bytes());
    ASSERT_TRUE(decoded);
    ASSERT_TRUE(decoded->input);
    EXPECT_TRUE(decoded->input->pressed);
    EXPECT_TRUE(decoded->input->repeat);
    EXPECT_EQ(browser_key_code(decoded->input->code), Keyboard::Z);

    PacketWriter invalid(MessageType::Input, 44);
    invalid.u8(static_cast<uint8_t>(InputKind::Text));
    invalid.u8(0);
    invalid.u32(0xd800);
    std::string error;
    EXPECT_FALSE(decode_client_message(invalid.bytes(), &error));
    EXPECT_EQ(error, "Invalid Unicode codepoint.");
}

TEST(BrowserInputProtocol, PreservesModifiersRepeatAndUnicodeThroughDrawStructure) {
    DrawStructure graph(320, 240);
    KeyProbe probe;
    graph.wrap_object(probe);
    graph.select(&probe);

    Event down(EventType::KEY);
    down.key = {Keyboard::Z, true, true, true, true, true, true};
    EXPECT_TRUE(graph.event(down));
    ASSERT_TRUE(probe.down);
    EXPECT_EQ(probe.down->code, Keyboard::Z);
    EXPECT_TRUE(probe.down->pressed);
    EXPECT_TRUE(probe.down->shift);
    EXPECT_TRUE(probe.down->control);
    EXPECT_TRUE(probe.down->alt);
    EXPECT_TRUE(probe.down->system);
    EXPECT_TRUE(probe.down->repeat);

    Event up(EventType::KEY);
    up.key = {Keyboard::Z, false, true, true, true, true, false};
    EXPECT_TRUE(graph.event(up));
    ASSERT_TRUE(probe.up);
    EXPECT_FALSE(probe.up->pressed);
    EXPECT_TRUE(probe.up->control);
    EXPECT_TRUE(probe.up->alt);
    EXPECT_TRUE(probe.up->system);

    Textfield field(Box(0, 0, 200, 30));
    graph.wrap_object(field);
    graph.select(&field);
    Event text(EventType::TEXT_ENTERED);
    text.text = {0, 0x1f642};
    EXPECT_TRUE(graph.event(text));
    EXPECT_EQ(field.text(), "\xf0\x9f\x99\x82");

    // Legacy producers that only populate TextEvent::c must remain valid after
    // adding the full Unicode codepoint field.
    Event ascii(EventType::TEXT_ENTERED);
    ascii.text.c = 'A';
    EXPECT_TRUE(graph.event(ascii));
    EXPECT_EQ(field.text(), "\xf0\x9f\x99\x82"
                            "A");
}

TEST(BrowserProtocol, ReportsConnectedClientCount) {
    const auto message = clients_message(9, 3);
    WireReader reader(message);
    uint64_t sequence = 0;
    read_header(reader, MessageType::Clients, &sequence);
    EXPECT_EQ(sequence, 9u);
    EXPECT_EQ(reader.u32(), 3u);
    EXPECT_EQ(reader.remaining(), 0u);
}

TEST(BrowserViewport, UsesVideoSizeAndMirrorsNative) {
    EXPECT_EQ(logical_viewport_size(std::nullopt, std::nullopt, 1280), Size2(1280, 720));
    EXPECT_EQ(logical_viewport_size(std::nullopt, Size2(-1), 640), Size2(640, 360));
    EXPECT_EQ(logical_viewport_size(std::nullopt, Size2(640, 480), 1280), Size2(640, 480));
    EXPECT_EQ(logical_viewport_size(std::nullopt, Size2(1920, 1080), 960), Size2(960, 540));
    EXPECT_EQ(logical_viewport_size(Size2(1024, 850), Size2(1920, 1080), 640), Size2(1024, 850));
}

TEST(BrowserFonts, SelectsSymbolsAndBoldMonospaceFaces) {
    EXPECT_EQ(font_face_style(Style::Symbols | Style::Bold), uint32_t(Style::Symbols));
    EXPECT_EQ(font_face_style(Style::Monospace | Style::Bold),
              uint32_t(Style::Monospace | Style::Bold));
    EXPECT_EQ(font_face_style(Style::Monospace | Style::Italic), uint32_t(Style::Monospace));
    EXPECT_EQ(font_face_style(Style::Bold | Style::Underlined), uint32_t(Style::Bold));
    EXPECT_EQ(font_face_style(Style::Bold | Style::Italic | Style::StrikeThrough),
              uint32_t(Style::Bold | Style::Italic));
}

TEST(NativeCompatibility, LatestBackendOwnsDefaultTextMetricsForItsLifetime) {
    MetricBase browser(Bounds(1, 2, 30, 40), 50);
    EXPECT_EQ(Base::default_text_bounds("text"), Bounds(1, 2, 30, 40));
    EXPECT_EQ(Base::default_line_spacing(Font{}), 50);

    {
        MetricBase native(Bounds(3, 4, 60, 70), 80);
        EXPECT_EQ(Base::default_text_bounds("text"), Bounds(3, 4, 60, 70));
        EXPECT_EQ(Base::default_line_spacing(Font{}), 80);
    }

    EXPECT_EQ(Base::default_text_bounds("text"), Bounds(1, 2, 30, 40));
    EXPECT_EQ(Base::default_line_spacing(Font{}), 50);
}
