module;

#define IN_MODULE_INTERFACE 1
#include <detect/Detection.h>

export module trex.detect;
export import trex.data;

export namespace track::detect {
using ::track::detect::BackendHooks;
}

export namespace track {
using ::track::Detection;
}