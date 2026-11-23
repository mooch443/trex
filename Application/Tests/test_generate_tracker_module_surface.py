import unittest

from generate_tracker_module_surface import parse_macro_invocations, parse_regular_symbols


class TrackerModuleSurfaceTest(unittest.TestCase):
    def test_qualified_enum_preserves_following_alias(self):
        source = """
ENUM_CLASS(track::detect::,
           ObjectDetectionType, yolo)
namespace track::detect {
using ObjectDetectionType_t = ObjectDetectionType::Class;
}
"""
        macros, _ = parse_macro_invocations("core/default_config.h", "trex.core", source)
        regular, _ = parse_regular_symbols("core/default_config.h", "trex.core", source)
        self.assertEqual([symbol.qualified for symbol in macros], ["track::detect::ObjectDetectionType::Class"])
        self.assertIn("track::detect::ObjectDetectionType_t", {symbol.qualified for symbol in regular})


if __name__ == "__main__":
    unittest.main()
