module;

#define IN_MODULE_INTERFACE 1
#include <data/MotionRecord.h>

export module trex.data;
export import trex.core;

export namespace track {
using ::track::FrameProperties;
using ::track::MotionRecord;
using ::track::Units;
}