module;

#define IN_MODULE_INTERFACE 1
#include <ui/TrackingState.h>

export module trex.ui;
export import trex.ml;
export import commons.gui;

export namespace cmn::gui {
using ::cmn::gui::TrackingState;
using ::cmn::gui::VIControllerImpl;
}