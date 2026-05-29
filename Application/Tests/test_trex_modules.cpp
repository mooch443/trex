#include <gtest/gtest.h>

import trex;
import trex.core;
import trex.data;
import trex.tracking;
import trex.ml;
import trex.ui;

#include "generated/trex_module_smoke_core.inc"
#include "generated/trex_module_smoke_data.inc"
#include "generated/trex_module_smoke_tracking.inc"
#include "generated/trex_module_smoke_ml.inc"
#include "generated/trex_module_smoke_ui.inc"

TEST(TRexModules, ExhaustiveSurfaceCompiles) {
    SUCCEED();
}
