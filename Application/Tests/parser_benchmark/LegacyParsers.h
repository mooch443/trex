#pragma once
#include <commons.pc.h>
#include <gui/dyn/Context.h>
#include <gui/dyn/State.h>
#include <gui/dyn/VarProps.h>

// The snapshots own their parsers, string helpers and the built-ins exercised by
// the benchmark. Context/State and GUI value types are shared with production.
namespace cmn::benchmark_head {
namespace pattern {
#include "LegacyPattern.inc"
}
namespace dyn {
std::string parse_text_snapshot(const std::string_view&, const gui::dyn::Context&, gui::dyn::State&);
void configure(gui::dyn::Context&);
}
}

namespace cmn::benchmark_older {
namespace pattern {
#include "LegacyPattern.inc"
}
namespace dyn {
std::string parse_text_snapshot(const std::string_view&, const gui::dyn::Context&, gui::dyn::State&);
void configure(gui::dyn::Context&);
}
}

