#ifndef _SPLIT_BLOB_H
#define _SPLIT_BLOB_H

#include <misc/GlobalSettings.h>
#include <misc/PVBlob.h>

#define DEBUG_ME false

namespace track { class SplitBlob; }

//! This class tries to find multiple blobs within a big blob.
//  One blob represents one individual.
class track::SplitBlob {
    struct ResultProp {
        float fitness;
        float ratio;
        int threshold;
        std::vector<pv::BlobPtr> blobs;
        //std::vector<std::vector<uchar>> pixels;
        
        std::string toStr() const {
            return "t:"+Meta::toStr(threshold)+" obj:"+Meta::toStr(blobs.size())+" r:"+Meta::toStr(ratio);
        }
        static std::string class_name() {
            return "SplitBlob::ResultProp";
        }
    };
    
    enum Action {
        KEEP,
        KEEP_ABORT,
        REMOVE,
        ABORT
    };
    
private:
    cv::Mat _original, _original_grey;
    size_t max_objects;
    
    // parameters
    pv::BlobPtr _blob;
    std::vector<uchar> _diff_px;
    
public:
    SplitBlob(const Background& average, pv::BlobPtr blob);
    
    /**
     * @param presumed_nr number of individuals to find
     * @param blob The big blob containing potentially more than 1 blob
     * @param pixels greyscale pixel values for all the lines in blob
     * @param average background image of the whole arena
     * @return either an empty list if the blob cannot be split, or
     * a number of Blobs that seem to be two individuals. Also returns
     * every Blob paired with its grey value array
     */
    std::vector<pv::BlobPtr> split(size_t presumed_nr);
    
private:
    size_t apply_threshold(int threshold, std::vector<pv::BlobPtr> &output);
    Action evaluate_result_single(std::vector<pv::BlobPtr>&);
    Action evaluate_result_multiple(size_t presumed_nr, float first_size, std::vector<pv::BlobPtr>&, ResultProp&);
    
#if DEBUG_ME
    void display_match(const std::pair<const int, std::vector<pv::BlobPtr>>&);
#endif
};

#endif
