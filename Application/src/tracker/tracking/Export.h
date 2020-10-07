#pragma once

#include <types.h>

namespace track {
    class Tracker;
    void export_data(Tracker& tracker, long_t fdx, const Rangel& range);
}
