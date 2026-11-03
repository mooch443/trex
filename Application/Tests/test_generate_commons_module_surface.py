import unittest

from generate_commons_module_surface import parse_macro_invocations, parse_regular_symbols


class CommonsModuleSurfaceTest(unittest.TestCase):
    def test_namespace_prefixed_enum_preserves_following_declarations(self):
        source = """
ENUM_CLASS(cmn::gui::,
           Type, NONE, RECT)
namespace cmn::gui {
class Drawable {};
}
"""
        macros, _ = parse_macro_invocations("gui/types/Drawable.h", "commons.gui", source)
        regular, _ = parse_regular_symbols("gui/types/Drawable.h", "commons.gui", source)
        self.assertEqual([symbol.qualified for symbol in macros], ["cmn::gui::Type::Class"])
        self.assertIn("cmn::gui::Drawable", {symbol.qualified for symbol in regular})

    def test_attribute_alias_preserves_following_class(self):
        source = """
namespace cmn::gui {
NUMBER_ALIAS(LineClosed, bool)
class Line final {};
}
"""
        macros, _ = parse_macro_invocations("gui/GuiTypes.h", "commons.gui", source)
        regular, _ = parse_regular_symbols("gui/GuiTypes.h", "commons.gui", source)
        self.assertEqual(
            {symbol.qualified for symbol in macros},
            {"cmn::gui::LineClosed", "cmn::gui::LineClosedTag"},
        )
        self.assertIn("cmn::gui::Line", {symbol.qualified for symbol in regular})


if __name__ == "__main__":
    unittest.main()
