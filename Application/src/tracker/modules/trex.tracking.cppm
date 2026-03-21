module;

#define IN_MODULE_INTERFACE 1
#include <tracking/Results.h>

export module trex.tracking;
export import trex.detect;

export namespace track {
using ::track::Results;
}