#pragma once

#include <types.h>
#include <file/Path.h>

namespace track {
    class Tracker;
    void export_data(Tracker& tracker, long_t fdx, const Rangel& range);

    inline void temporary_save(file::Path path, std::function<void(file::Path)> fn) {
        /**
         * There is sometimes a problem when people save to network harddrives.
         * The first NPY file will not finish writing / sync completely before the next one starts.
         * This leads to "does not contain a ZIP file" exception and terminates the saving process.
         * Instead, we move the file to a temporary folder first (on our local harddrive) and then
         * move it.
         * (Only if a /tmp/ folder exists though.)
         */
        
        file::Path final_path = path;
        file::Path tmp_path, use_path;
        
#ifdef WIN32
        char chPath[MAX_PATH];
        if (GetTempPath(MAX_PATH, chPath))
            tmp_path = chPath;
#else
        tmp_path = "/tmp";
#endif
        
        if(tmp_path.exists()) {
            if(access(tmp_path.c_str(), W_OK) == 0)
                use_path = tmp_path / path.filename();
        }
        
        try {
            fn(use_path);
            
            if(final_path != use_path) {
                if(!use_path.move_to(final_path))
                    U_EXCEPTION("Cannot move file '%S' to '%S'.", &use_path.str(), &final_path.str());
            }
            
        } catch(...) {
            // there will be a utils exception, so its printed out already
        }
    }
}
