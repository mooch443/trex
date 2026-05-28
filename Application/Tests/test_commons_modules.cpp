#include <gtest/gtest.h>

#include <misc/EnumClass.h>
#include <misc/create_struct.h>
#include <gui/ControlsAttributes.h>

import commons;
import commons.misc;
import commons.file;
import commons.types;
import commons.processing;
import commons.video;
import commons.gui;
import commons.gui.dyn;
#if WITH_MHD
import commons.http;
#endif

#include "generated/commons_module_smoke_types.inc"
#include "generated/commons_module_smoke_misc.inc"
#include "generated/commons_module_smoke_file.inc"
#include "generated/commons_module_smoke_processing.inc"
#include "generated/commons_module_smoke_video.inc"
#include "generated/commons_module_smoke_gui.inc"
#include "generated/commons_module_smoke_gui_dyn.inc"
#include "generated/commons_module_smoke_http.inc"

namespace commons_macro_smoke {

namespace {
ENUM_CLASS(CommonsModuleSmokeEnum, alpha, beta)
ENUM_CLASS_HAS_DOCS(CommonsModuleSmokeEnum)
ENUM_CLASS_DOCS(CommonsModuleSmokeEnum, "alpha", "beta")
CREATE_STRUCT(CommonsModuleSmokeStruct, (int, value), (std::string, label))
}

namespace gui_attr_test {
using namespace ::cmn::gui::attr;
ATTRIBUTE_ALIAS(CommonsModuleSmokeAttribute, int)
NUMBER_ALIAS(CommonsModuleSmokeNumber, int)
}

using ::commons_macro_smoke::CommonsModuleSmokeEnum::Class;
using ::commons_macro_smoke::CommonsModuleSmokeStruct;
using ::commons_macro_smoke::gui_attr_test::CommonsModuleSmokeAttribute;
using ::commons_macro_smoke::gui_attr_test::CommonsModuleSmokeNumber;

}

TEST(CommonsModules, ExhaustiveSurfaceCompiles) {
    SUCCEED();
}
