#include <commons.pc.h>
#include "gtest/gtest.h"
#include <gui/GLFWKeyEvent.h>

#define GLFW_INCLUDE_NONE
#include <GLFW/glfw3.h>

using cmn::gui::Keyboard::Codes;
using cmn::gui::detail::translate_glfw_key_event;

TEST(KeyboardEventTest, MapsPrintableKeysUsingLayoutNames) {
    const struct {
        const char* layout;
        int key;
        const char* name;
        Codes expected;
    } cases[] {
        {"QWERTY", GLFW_KEY_Y, "y", Codes::Y},
        {"QWERTY", GLFW_KEY_Z, "z", Codes::Z},
        {"QWERTZ", GLFW_KEY_Y, "z", Codes::Z},
        {"QWERTZ", GLFW_KEY_Z, "y", Codes::Y},
        {"AZERTY", GLFW_KEY_Q, "a", Codes::A},
        {"AZERTY", GLFW_KEY_A, "q", Codes::Q},
        {"AZERTY", GLFW_KEY_W, "z", Codes::Z},
        {"Uppercase name", GLFW_KEY_Y, "Z", Codes::Z},
        {"Digit", GLFW_KEY_1, "1", Codes::Num1},
        {"Scancode-only key", GLFW_KEY_UNKNOWN, "z", Codes::Z}
    };

    for(const auto& test : cases) {
        SCOPED_TRACE(test.layout);
        SCOPED_TRACE(test.key);
        std::unordered_map<int, Codes> pressed_key_codes;
        const auto event = translate_glfw_key_event(
            test.key, 100, GLFW_PRESS, 0, test.name, pressed_key_codes);
        EXPECT_EQ(event.code, test.expected);
        EXPECT_TRUE(event.pressed);
        EXPECT_FALSE(event.shift);
    }
}

TEST(KeyboardEventTest, MapsPunctuationByCharacter) {
    const struct {
        const char* name;
        Codes expected;
    } cases[] {
        {"'", Codes::Quote},
        {"\"", Codes::Quote},
        {"[", Codes::LBracket},
        {"]", Codes::RBracket},
        {"\\", Codes::BackSlash},
        {"`", Codes::Tilde},
        {"~", Codes::Tilde},
        {",", Codes::Comma},
        {";", Codes::SemiColon},
        {"+", Codes::Add},
        {"-", Codes::Subtract}
    };

    for(const auto& test : cases) {
        SCOPED_TRACE(test.name);
        std::unordered_map<int, Codes> pressed_key_codes;
        const auto event = translate_glfw_key_event(
            GLFW_KEY_WORLD_1, 100, GLFW_PRESS, 0, test.name, pressed_key_codes);
        EXPECT_EQ(event.code, test.expected);
    }
}

TEST(KeyboardEventTest, FallsBackWhenLayoutNameIsUnavailable) {
    const struct {
        int key;
        Codes expected;
    } cases[] {
        {GLFW_KEY_Y, Codes::Y},
        {GLFW_KEY_Z, Codes::Z},
        {GLFW_KEY_APOSTROPHE, Codes::Quote},
        {GLFW_KEY_GRAVE_ACCENT, Codes::Tilde},
        {GLFW_KEY_UNKNOWN, Codes::Unknown}
    };

    for(const auto& test : cases) {
        SCOPED_TRACE(test.key);
        std::unordered_map<int, Codes> pressed_key_codes;
        const auto event = translate_glfw_key_event(
            test.key, 100, GLFW_PRESS, 0, nullptr, pressed_key_codes);
        EXPECT_EQ(event.code, test.expected);
    }
}

TEST(KeyboardEventTest, DoesNotUsePhysicalLettersForUnsupportedCharacters) {
    for(const char* name : {"", "ab", "\xC3\xA4", "(", ")"}) {
        SCOPED_TRACE(name);
        std::unordered_map<int, Codes> pressed_key_codes;
        const auto event = translate_glfw_key_event(
            GLFW_KEY_A, 100, GLFW_PRESS, 0, name, pressed_key_codes);
        EXPECT_EQ(event.code, Codes::Unknown);
    }
}

TEST(KeyboardEventTest, PreservesSpecialAndKeypadKeys) {
    const struct {
        int key;
        const char* name;
        Codes expected;
    } cases[] {
        {GLFW_KEY_SPACE, nullptr, Codes::Space},
        {GLFW_KEY_ESCAPE, nullptr, Codes::Escape},
        {GLFW_KEY_LEFT, nullptr, Codes::Left},
        {GLFW_KEY_RIGHT, nullptr, Codes::Right},
        {GLFW_KEY_F11, nullptr, Codes::F11},
        {GLFW_KEY_KP_1, "1", Codes::Numpad1},
        {GLFW_KEY_KP_ADD, "+", Codes::Add},
        {GLFW_KEY_KP_ENTER, nullptr, Codes::Return}
    };

    for(const auto& test : cases) {
        SCOPED_TRACE(test.key);
        std::unordered_map<int, Codes> pressed_key_codes;
        const auto event = translate_glfw_key_event(
            test.key, 100, GLFW_PRESS, 0, test.name, pressed_key_codes);
        EXPECT_EQ(event.code, test.expected);
    }
}

TEST(KeyboardEventTest, PreservesLocalizedLettersWithShortcutModifiers) {
    for(const int mods : {0, GLFW_MOD_SHIFT, GLFW_MOD_CONTROL, GLFW_MOD_SUPER,
                         GLFW_MOD_ALT, GLFW_MOD_CONTROL | GLFW_MOD_SHIFT,
                         GLFW_MOD_SUPER | GLFW_MOD_SHIFT})
    {
        SCOPED_TRACE(mods);
        std::unordered_map<int, Codes> pressed_key_codes;
        const auto event = translate_glfw_key_event(
            GLFW_KEY_Y, 100, GLFW_PRESS, mods, "z", pressed_key_codes);
        EXPECT_EQ(event.code, Codes::Z);
        EXPECT_EQ(event.shift, (mods & GLFW_MOD_SHIFT) != 0);
        EXPECT_TRUE(event.pressed);
    }

    for(const int key : {GLFW_KEY_LEFT_SHIFT, GLFW_KEY_RIGHT_SHIFT}) {
        SCOPED_TRACE(key);
        std::unordered_map<int, Codes> pressed_key_codes;
        const auto event = translate_glfw_key_event(
            key, 100, GLFW_PRESS, GLFW_MOD_SHIFT, nullptr, pressed_key_codes);
        EXPECT_EQ(event.code, key == GLFW_KEY_LEFT_SHIFT ? Codes::LShift : Codes::RShift);
        EXPECT_TRUE(event.shift);
    }
}

TEST(KeyboardEventTest, RetainsPressCodeUntilReleaseAcrossLayoutChanges) {
    std::unordered_map<int, Codes> pressed_key_codes;
    const auto press = translate_glfw_key_event(
        GLFW_KEY_Y, 100, GLFW_PRESS, 0, "z", pressed_key_codes);
    ASSERT_EQ(press.code, Codes::Z);
    EXPECT_TRUE(press.pressed);

    const auto repeat = translate_glfw_key_event(
        GLFW_KEY_Y, 100, GLFW_REPEAT, GLFW_MOD_SHIFT, "y", pressed_key_codes);
    EXPECT_EQ(repeat.code, Codes::Z);
    EXPECT_TRUE(repeat.pressed);
    EXPECT_TRUE(repeat.shift);

    const auto release = translate_glfw_key_event(
        GLFW_KEY_Y, 100, GLFW_RELEASE, 0, "y", pressed_key_codes);
    EXPECT_EQ(release.code, Codes::Z);
    EXPECT_FALSE(release.pressed);
    EXPECT_FALSE(release.shift);
    EXPECT_TRUE(pressed_key_codes.empty());

    const auto next_press = translate_glfw_key_event(
        GLFW_KEY_Y, 100, GLFW_PRESS, 0, "y", pressed_key_codes);
    EXPECT_EQ(next_press.code, Codes::Y);
    EXPECT_TRUE(next_press.pressed);
}

TEST(KeyboardEventTest, TracksSimultaneouslyHeldScancodesIndependently) {
    std::unordered_map<int, Codes> pressed_key_codes;
    const auto first = translate_glfw_key_event(
        GLFW_KEY_UNKNOWN, 100, GLFW_PRESS, 0, "y", pressed_key_codes);
    const auto second = translate_glfw_key_event(
        GLFW_KEY_UNKNOWN, 101, GLFW_PRESS, 0, "z", pressed_key_codes);
    ASSERT_EQ(first.code, Codes::Y);
    ASSERT_EQ(second.code, Codes::Z);

    const auto first_release = translate_glfw_key_event(
        GLFW_KEY_UNKNOWN, 100, GLFW_RELEASE, 0, "z", pressed_key_codes);
    EXPECT_EQ(first_release.code, Codes::Y);
    EXPECT_FALSE(first_release.pressed);

    const auto second_repeat = translate_glfw_key_event(
        GLFW_KEY_UNKNOWN, 101, GLFW_REPEAT, 0, "y", pressed_key_codes);
    EXPECT_EQ(second_repeat.code, Codes::Z);
    EXPECT_TRUE(second_repeat.pressed);

    const auto second_release = translate_glfw_key_event(
        GLFW_KEY_UNKNOWN, 101, GLFW_RELEASE, 0, "y", pressed_key_codes);
    EXPECT_EQ(second_release.code, Codes::Z);
    EXPECT_FALSE(second_release.pressed);
    EXPECT_TRUE(pressed_key_codes.empty());
}
