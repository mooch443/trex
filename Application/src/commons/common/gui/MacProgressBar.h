#pragma once

#ifdef __APPLE__

namespace gui {
class MacProgressBar {
public:
    static void set_percent(double v);
    static void set_visible(bool v);
};
}

#endif
