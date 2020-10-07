#ifndef _RESULTS_H
#define _RESULTS_H

#include <file/CSVExport.h>
#include <tracking/Tracker.h>
#include <file/Path.h>
#include <misc/EventAnalysis.h>

namespace track {
    class Results {
        Tracker& _tracker;
        
    public:
        Results(Tracker& tracker) : _tracker(tracker)
            {}
        
        bool save(const file::Path& filename) const;
        bool save_events(const file::Path& filename, std::atomic<float>& percent) const;
    };
}

#endif
