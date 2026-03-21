module;

#define IN_MODULE_INTERFACE 1
#include <ml/VisualIdentification.h>

export module trex.ml;
export import trex.tracking;

export namespace Python::TrainingMode {
using ::Python::TrainingMode::Class;
}

export namespace Python {
using ::Python::VINetwork;
}