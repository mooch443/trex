#include "Individual.h"
#include <misc/GlobalSettings.h>
#include <misc/Timer.h>
#include <tracking/Tracker.h>
#include <tracking/DebugDrawing.h>
#include <algorithm>
#include <misc/ReverseAdapter.h>
#include <cnpy/cnpy.h>
#include <tracking/VisualField.h>
#include <misc/pretty.h>
#include <processing/PadImage.h>
#include <misc/cnpy_wrapper.h>
#include <misc/CircularGraph.h>

#include <misc/SoftException.h>
#include <gui/Graph.h>
#include <tracking/CategorizeDatastore.h>
#include <file/DataLocation.h>
#include <tracking/TrackingHelper.h>
#include <tracking/IndividualManager.h>
#include <misc/FOI.h>

#if !COMMONS_NO_PYTHON
#include <misc/PythonWrapper.h>
#include <tracking/RecTask.h>
#endif

#define NAN_SAFE_NORMALIZE(X, Y) { \
    const auto L = X.length(); \
    if(L) Y = X / L; \
    else  Y = Vec2(0); \
}

using namespace track;
using prob_t = track::Match::prob_t;

void Individual::shutdown() {
    
}

#if !COMMONS_NO_PYTHON
IDaverage Individual::qrcode_at(Frame_t segment_start) const {
    std::unique_lock guard(_qrcode_mutex);
    auto it = _qrcode_identities.find(segment_start);
    if(it != _qrcode_identities.end()) {
        return it->second;
    }

    return { -1, -1, 0 };
}

ska::bytell_hash_map<Frame_t, IDaverage> Individual::qrcodes() const {
    std::unique_lock guard(_qrcode_mutex);
    return _qrcode_identities;
}
#endif

#if !COMMONS_NO_PYTHON
bool Individual::add_qrcode(Frame_t frame, pv::BlobPtr&& tag) {
    auto seg = tracklet_for(frame);
    if (!seg) {
        //FormatWarning("Cannot add tag to ", _identity, " because no tracklet is found for frame ", frame);
        return false;
    }

    auto pos = tag->bounds().pos() + tag->bounds().size() * 0.5;
    auto idx = seg->basic_stuff(frame);
    assert(idx != -1);
    auto& basic = basic_stuff()[(size_t)idx];
    auto my_bounds = basic->blob.calculate_bounds();

    if (my_bounds.contains(pos)) {
        //Print("adding tag at ", tag->bounds(), " to individual ", _identity, " at ", my_bounds, " for tracklet ",*seg);
        auto &&[pos, image] = tag->gray_image(nullptr, Bounds(-1, -1, -1, -1), 0);
        if (image->cols != 32 || image->rows != 32)
            FormatWarning("Image dimensions are wrong ", image->bounds());
        else
            _qrcodes[seg->start()].emplace_back( QRCode { frame, std::move(tag) } );
        //tf::imshow(_identity.name(), image->get());
        
        auto check_qrcode = [this](Frame_t frame, const std::shared_ptr<TrackletInformation>& tracklet)
        {
            static const bool tags_recognize = SETTING(tags_recognize).value<bool>();
            
            const bool tracklet_ended = tracklet && tracklet->start() != _last_requested_tracklet;
            if(tracklet_ended) {
                //Print("individual:",identity().ID(), " segment:",tracklet->start()," ended before ", frame);
                _last_requested_qrcode.invalidate();
                _last_requested_tracklet = tracklet->start();
            }

        #if !COMMONS_NO_PYTHON
            //! do we need to start a new segment?
            if (tags_recognize && !_qrcodes.empty() && !_tracklets.empty()) {
                auto tracklet = _tracklets.back();
                
                if(tracklet_ended // either the tracklet ended
                    || !_last_requested_qrcode.valid()
                    || (RecTask::can_take_more() // or we have not requested a code yet
                            && _last_requested_qrcode + Frame_t(Frame_t::number_t(5.f * (float)SLOW_SETTING(frame_rate))) < frame) // or the last time has been at least a second ago
                   )
                {
                    auto it = _qrcodes.find(tracklet->start());
                    if(it != _qrcodes.end()) {
                        //Print("[update] at ", frame," ", tracklet ? tracklet->range : Range<Frame_t>(), " individual:", identity().ID(), " with ended:", tracklet_ended, " lastqrcodepred:", _last_predicted_id, " lastqrframe:",_last_requested_qrcode, " images:", it->second.size());
                        
                        if(it->second.size() > 2 || tracklet_ended) {
                            RecTask task;
                            {
                                task._tracklet_start = tracklet->start(),
                                    task.individual = identity().ID(),
                                    task._optional = !tracklet_ended,
                                    task._fdx = identity().ID();
                            }

                            task._callback = [this, range = tracklet->range, N = it->second.size(), tracklet = tracklet](Predictions&& prediction) {
                                //Print("got callback in ", _identity.ID(), " (", prediction.individual, ")");
                                
                                //Print("\t",range, " individual ", identity().ID(), " has ", N, " images. ended=", tracklet_ended, " got callback with pred=", prediction.best_id);

                                std::unique_lock guard(_qrcode_mutex);
                                _qrcode_identities[prediction._tracklet_start] = { prediction.best_id, prediction.p, (uint32_t)prediction._frames.size() };
                                _last_predicted_id = prediction.best_id;
                                _last_predicted_frame = prediction._tracklet_start;
                                for(size_t i=0; i<prediction._frames.size(); ++i) {
                                    auto bix = tracklet->basic_stuff( prediction._frames[i]);
                                    if(bix != -1) {
                                        auto &basic = basic_stuff().at(bix);
                                        basic->blob.pred = blob::Prediction{
                                            .clid = uint8_t(prediction._ids[i]),
                                            .p = uint8_t(float(prediction.p) * 255.f)
                                        };
                                    }
                                }
                            };

                            auto fill = [it, ID = identity(), tracklet](RecTask& task)
                            {
                                size_t step = it->second.size() / 100;
                                size_t i = 0;

                                for(auto& [frame, blob] : it->second) {
                                  if(step > 0 && i++ % step != 0) {
                                      continue;
                                  }
                                  auto ptr = std::get<1>(blob->gray_image(nullptr, Bounds(-1, -1, -1, -1), 0));
                                  //tf::imshow("push", ptr->get());
                                  task._frames.push_back(frame);
                                  task._images.push_back(std::move(ptr));
                                }

                                //if(it->second.size() > 1)
                                //  Print("sampling from ", it->second.size(), " to ",task._images.size(), " images of individual ", ID," at frame ", frameIndex," which started at ", tracklet->start(),".");
                            };
                            
                            auto callback = [&]() {
                                static const bool tags_save_predictions = SETTING(tags_save_predictions).value<bool>();
                                if(!tags_save_predictions)
                                    return;
                                
                                std::unique_lock guard(_qrcode_mutex);
                                if(!_last_predicted_frame.valid()
                                   || _last_predicted_frame != tracklet->start())
                                    return;
                                
                                static const auto filename = (std::string)SETTING(filename).value<file::Path>().filename();
                                file::Path output = file::DataLocation::parse("output", "tags_"+filename) / Meta::toStr(_last_predicted_id);
                                
                                if(!output.exists())
                                    return;
                                
                                auto prefix = Meta::toStr(identity().ID()) + "." + Meta::toStr(tracklet->start());
                                if(!(output / prefix).exists())
                                    return;
                                
                                auto files = (output / prefix).find_files();
                                
                                // delete files that already existed for this individual AND segment
                                for(auto &f : files) {
                                    f.delete_file();
                                    //Print("\tdeleting file ", f);
                                }
                            };
                            
                            // if we can add this code, update the last requested
                            if(RecTask::add(std::move(task), fill, callback)) {
                                //cmn::Print("Have ", it->second.size(), " QRCodes for tracklet ", *tracklet, " in individual:", identity().ID(), " ", tracklet_ended);

                                std::unique_lock guard(_qrcode_mutex);
                                _last_requested_qrcode = frame;
                                
                            } //else
                                //Print("\t",tracklet->range, " individual:", identity().ID(), " rejected ",it->second.size()," images.");
                                
                        } //else {
                            //Print("\t",tracklet->range, " individual:", identity().ID(), " not enough images ",it->second.size(),".");
                        //}
                        
                    } //else if(tracklet_ended && tracklet->length() > 2) {
                    //    Print("\t",tracklet->range, " individual:", identity().ID(), " does not have QRCodes.");
                    //}
                }
            }
        #endif
        };
        
        check_qrcode(frame, seg);
        
        return true;
    }

    return false;
}
#endif

void Individual::add_tag_image(tags::Tag&& tag) {
    assert(tag.frame.valid());
    auto && [range, first] = get_tracklet(tag.frame);
    
    auto &set = _best_images[range.start];
    if(set.size() > 3) {
        Frame_t last;
        for (auto& tag : set) {
            if (tag.frame > last)
                last = tag.frame;
        }
        
        if(cmn::abs((last - tag.frame).get()) >= SLOW_SETTING(frame_rate)) {
            if(set.size() > 6)
                set.erase(--set.end());
        } else
            return;
    }
    set.insert(std::move(tag));
}

Individual::tracklet_map::const_iterator Individual::find_tracklet_with_start(Frame_t frame) const {
    return find_frame_in_sorted_tracklets(tracklets().begin(), tracklets().end(), frame);
}

FrameRange Individual::get_tracklet(Frame_t frameIndex) const {
    auto it = std::lower_bound(_tracklets.begin(), _tracklets.end(), frameIndex, [](const auto& ptr, Frame_t frame) {
        return ptr->start() < frame;
    });
    if(it != _tracklets.end()) {
        if(it != _tracklets.begin() && (*it)->start() != frameIndex)
            --it;
        assert((*it)->start() <= frameIndex || (*it)->start() == start_frame());
        return *it->get();
    }
    
    if(!_tracklets.empty() && (*_tracklets.rbegin())->start() <= frameIndex)
        return *_tracklets.rbegin()->get();
    
    return FrameRange();
}

FrameRange Individual::get_recognition_segment(Frame_t frameIndex) const {
    auto it = _recognition_tracklets.lower_bound(frameIndex);
    if(it != _recognition_tracklets.end()) {
        if(it != _recognition_tracklets.begin() && it->first != frameIndex)
            --it;
        //assert(it->first <= frameIndex || it->first == start_frame());
        return it->second;
    }
    
    if(!_recognition_tracklets.empty() && _recognition_tracklets.rbegin()->second.start() <= frameIndex)
        return _recognition_tracklets.rbegin()->second;
    
    return FrameRange();
}

FrameRange Individual::get_tracklet_safe(Frame_t frameIndex) const {
    auto tracklet = get_tracklet(frameIndex);
    if(not tracklet.contains(frameIndex))
        return FrameRange();
    
    return tracklet;
}

FrameRange Individual::get_recognition_segment_safe(Frame_t frameIndex) const {
    auto tracklet = get_recognition_segment(frameIndex);
    if(not tracklet.contains(frameIndex))
        return FrameRange();
    
    return tracklet;
}

const std::multiset<tags::Tag>* Individual::has_tag_images_for(Frame_t frameIndex) const {
    if(not frameIndex.valid())
        return nullptr;
    
    auto range = get_tracklet(frameIndex);
    
    auto min_frame = Frame_t(std::numeric_limits<Frame_t::number_t>::max());
    const std::multiset<tags::Tag>* image = nullptr;
    for(auto && [frame, ptr] : _best_images) {
        if(range.contains(frame) && frame < min_frame) {
            min_frame = frame;
            image = &ptr;
        }
    }
    
    if(image != nullptr) {
        //return true;
    }
    
    return image;
}

struct FrameNumber {
    Frame_t frame;
};

inline bool operator<(const std::shared_ptr<track::TrackletInformation>& ptr, const FrameNumber& frame) {
    assert(ptr != nullptr);
    return ptr->end() < frame.frame;
}

inline bool operator<(const FrameNumber& frame, const std::shared_ptr<track::TrackletInformation>& ptr) {
    assert(ptr != nullptr);
    return frame.frame < ptr->start();
}

bool Individual::has(Frame_t frame) const {
    if(not _startFrame.valid() or not frame.valid())
        return false;
    if(frame < _startFrame || frame > _endFrame)
        return false;
    
    return std::binary_search(_tracklets.begin(), _tracklets.end(), FrameNumber{frame});
}

decltype(Individual::_tracklets)::const_iterator Individual::iterator_for(Frame_t frameIndex) const {
    if(empty())
        return _tracklets.end();
    
    if(not frameIndex.valid()
       || frameIndex < _startFrame)
       //|| frameIndex > _endFrame)
    {
        return _tracklets.end();
    }
    if(frameIndex == _startFrame) {
        return _tracklets.begin();
    }
    if(frameIndex == _endFrame)
        return --_tracklets.end();
    
    auto begin = _tracklets.begin();
    auto end = _tracklets.end();
    //assert(frameIndex <= (*--_tracklets.end())->range.end);
    
    auto it = std::lower_bound(begin, end, frameIndex, [](const auto& ptr, Frame_t frame){
        return ptr->start() < frame;
    });
    if(it != end) {
        if((*it)->start() > frameIndex) {
            if(it == begin)
                it = end;
            else
                --it;
        }
        
    } else if(!_tracklets.empty()) {
        --it;
    }
    
    assert(it == _tracklets.end() || (*it)->start() <= frameIndex);
    return it;
}

std::shared_ptr<TrackletInformation> Individual::tracklet_for(Frame_t frameIndex) const {
    if(not frameIndex.valid())
        return nullptr;
    
    if(empty())
        return nullptr;
    if(frameIndex < _startFrame || frameIndex > _endFrame)
        return nullptr;
    
    auto it = iterator_for(frameIndex);
    return it == _tracklets.end() || not (*it)->contains(frameIndex) ? nullptr : *it;
}

#ifndef NDEBUG
#define SEGMENT_ACCESS(INDEXARRAY, INDEX) INDEXARRAY . at( INDEX )
#else
#define SEGMENT_ACCESS(INDEXARRAY, INDEX) INDEXARRAY [ INDEX ]
#endif

BasicStuff* Individual::basic_stuff(Frame_t frameIndex) const {
    auto tracklet = tracklet_for(frameIndex);
    if(tracklet)
        return SEGMENT_ACCESS(_basic_stuff, tracklet->basic_stuff(frameIndex)).get(); //_basic_stuff.at( tracklet->basic_stuff(frameIndex) );
    return nullptr;
}

PostureStuff* Individual::posture_stuff(Frame_t frameIndex) const {
    auto tracklet = tracklet_for(frameIndex);
    if(tracklet) {
        auto index = tracklet->posture_stuff(frameIndex);
        return index != -1 ? SEGMENT_ACCESS(_posture_stuff, index).get() : nullptr;
        //return index != -1 ? _posture_stuff.at( index ) : nullptr;
    }
    return nullptr;
}

std::tuple<BasicStuff*, PostureStuff*> Individual::all_stuff(Frame_t frameIndex) const {
    auto tracklet = tracklet_for(frameIndex);
    if(tracklet) {
        auto basic_index = tracklet->basic_stuff(frameIndex);
        auto posture_index = tracklet->posture_stuff(frameIndex);
        return {
            basic_index != -1 ? SEGMENT_ACCESS(_basic_stuff, basic_index).get() : nullptr,
            posture_index != -1 ? SEGMENT_ACCESS(_posture_stuff, posture_index).get() : nullptr
        };
        //return index != -1 ? _posture_stuff.at( index ) : nullptr;
    }
    return {nullptr, nullptr};
}

int64_t Individual::thresholded_size(Frame_t frameIndex) const {
    auto ptr = basic_stuff(frameIndex);
    return ptr ? narrow_cast<int64_t>(ptr->thresholded_size) : -1;
}

const MotionRecord* Individual::centroid(Frame_t frameIndex) const {
    auto ptr = basic_stuff(frameIndex);
    return ptr ? &ptr->centroid : nullptr;
}

const MotionRecord* Individual::centroid_weighted(Frame_t frameIndex) const {
    auto ptr = basic_stuff(frameIndex);
    return ptr ? &ptr->centroid : nullptr;
}

const MotionRecord* Individual::head(Frame_t frameIndex) const {
    auto ptr = posture_stuff(frameIndex);
    return ptr ? ptr->head.get() : nullptr;
}

const MotionRecord* Individual::centroid_posture(Frame_t frameIndex) const {
    auto ptr = posture_stuff(frameIndex);
    return ptr ? ptr->centroid_posture.get() : nullptr;
}

/*MotionRecord* Individual::centroid(Frame_t frameIndex) {
    auto ptr = basic_stuff(frameIndex);
    return ptr ? &ptr->centroid : nullptr;
}

MotionRecord* Individual::centroid_weighted(Frame_t frameIndex) {
    auto ptr = basic_stuff(frameIndex);
    return ptr ? &ptr->centroid : nullptr;
}

MotionRecord* Individual::head(Frame_t frameIndex) {
    auto ptr = posture_stuff(frameIndex);
    return ptr ? ptr->head.get() : nullptr;
}

MotionRecord* Individual::centroid_posture(Frame_t frameIndex) {
    auto ptr = posture_stuff(frameIndex);
    return ptr ? ptr->centroid_posture.get() : nullptr;
}*/

pv::BlobPtr Individual::blob(Frame_t frameIndex) const {
    auto tracklet = tracklet_for(frameIndex);
    if(tracklet)
        return SEGMENT_ACCESS(_basic_stuff, tracklet->basic_stuff(frameIndex))->blob.unpack();
    return nullptr;
}

pv::CompressedBlob* Individual::compressed_blob(Frame_t frameIndex) const {
    auto tracklet = tracklet_for(frameIndex);
    if(tracklet)
        return &SEGMENT_ACCESS(_basic_stuff, tracklet->basic_stuff(frameIndex))->blob;
    return nullptr;
}

Midline::Ptr Individual::midline(Frame_t frameIndex) const {
    auto posture = posture_stuff(frameIndex);
    if(not posture)
        return nullptr;
    
    return calculate_midline_for(*posture);
}

const Midline* Individual::pp_midline(Frame_t frameIndex) const {
    auto ptr = posture_stuff(frameIndex);
    return ptr ? ptr->cached_pp_midline.get() : nullptr;
}

Midline::Ptr Individual::fixed_midline(Frame_t frameIndex) const {
    auto mid = pp_midline(frameIndex);
    if(mid == nullptr || midline_length() <= 0 || _local_cache._midline_samples == 0)
        return nullptr;
    
    MovementInformation movement;
    if(FAST_SETTING(posture_direction_smoothing) > 1) {
        movement = calculate_previous_vector(frameIndex);
    }

    auto fixed = std::make_unique<Midline>(*mid);
    fixed->post_process(movement, DebugInfo{frameIndex, identity().ID(), false});
    fixed = fixed->normalize(midline_length());
    
    return fixed;
}

const MinimalOutline* Individual::outline(Frame_t frameIndex) const {
    auto ptr = posture_stuff(frameIndex);
    return ptr ? &ptr->outline : nullptr;
}

Individual::Individual(std::optional<Identity>&& id)
    : _identity(id.has_value()
                ? std::move(id.value())
                : Identity::Make({}))
{ }

Individual::~Individual() {
#if !COMMONS_NO_PYTHON
    RecTask::remove(identity().ID());
#endif

    if(!Tracker::instance())
        return;
    
    remove_frame(start_frame());
    
    //! TODO: MISSING remove_invidiual
    //if(Tracker::recognition())
    //    Tracker::recognition()->remove_individual(this);
#ifndef NDEBUG
    Print("Deleting individual ", identity().ID());
#endif
}

void Individual::unregister_delete_callback(void* ptr) {
    std::unique_lock guard(_delete_callback_mutex);
    _delete_callbacks.erase(ptr);
}

void Individual::register_delete_callback(void* ptr, const std::function<void(Individual*)>& lambda) {
    std::unique_lock guard(_delete_callback_mutex);
    _delete_callbacks[ptr] = lambda;
}

#define MAP_SIZE(MAP, A, B) ((sizeof(A) + sizeof(B) + 32 + sizeof(B*)) * (MAP).size() + 24)
#define KB(x)   ((size_t) (x) << 10)
#define MB(x)   ((size_t) (x) << 20)

void Individual::add_manual_match(Frame_t) {
    //assert(frameIndex <= _endFrame && frameIndex >= _startFrame);
    //_manually_matched.insert(frameIndex);
}

void Individual::add_automatic_match(Frame_t frameIndex) {
    //assert(frameIndex <= _endFrame && frameIndex >= _startFrame);
    automatically_matched.insert(frameIndex);
}

bool Individual::is_manual_match(Frame_t ) const {
    return false;
    //return _manually_matched.find(frameIndex) != _manually_matched.end();
}

bool Individual::is_automatic_match(Frame_t frameIndex) const {
    return automatically_matched.find(frameIndex) != automatically_matched.end();
}

bool Individual::recently_manually_matched(Frame_t frameIndex) const {
    for(auto frame = frameIndex; frame >= _startFrame && frame >= frameIndex - Frame_t(SLOW_SETTING(frame_rate) / 2u); --frame) {
        if(is_manual_match(frame))
            return true;
    }
    
    return false;
}

void Individual::remove_frame(Frame_t frameIndex) {
    {
        decltype(_delete_callbacks) callbacks;
        {
            std::unique_lock guard(_delete_callback_mutex);
            callbacks = _delete_callbacks;
            _delete_callbacks.clear();
        }

        for (auto& f : callbacks)
            f.second(this);
    }
    
    if (not _endFrame.valid()
        or frameIndex > _endFrame)
    {
        return;
    }

    if(frameIndex <= start_frame())
        _hints.clear();
    else
        _hints.remove_after(frameIndex);

    {
        auto it = added_postures.begin();
        while (it != added_postures.end() && *it < frameIndex) {
            ++it;
        }
        added_postures.erase(it, added_postures.end());
    }
    
    auto check_integrity = [this](){
        auto it = _posture_stuff.begin();
        for(auto & seg : _tracklets) {
            Frame_t offset{0};
            for(auto id : seg->posture_index) {
                if(id != -1) {
                    if(it == _posture_stuff.end())
                        throw U_EXCEPTION("Ended in ",seg->range.start,"-",seg->range.end,".");
                    if((*it)->frame != seg->start() + offset) {
                        throw U_EXCEPTION("Frame ",(*it)->frame," from posture is != ",seg->start() + offset,"");
                    }
                    ++it;
                }
                ++offset;
            }
        }
    };
    
    check_integrity();
    
    if(!_recognition_tracklets.empty()) {
        auto it = --_recognition_tracklets.end();
        while (it->second.start() >= frameIndex) {
            if(it == _recognition_tracklets.begin())
                break;
            --it;
        }
        
        if(it->second.range.end < frameIndex)
            ++it;
        else if(it->second.range.start < frameIndex) {
            it->second.range.end = frameIndex - 1_f;
            assert(it->second.range.start <= it->second.range.end);
        }
    }
    
    if(!_tracklets.empty()) {
        auto it = --_tracklets.end();
        while ((*it)->range.start >= frameIndex) {
            if(it == _tracklets.begin())
                break;
            --it;
        }
        
        bool shortened_posture_index = false, shortened_basic_index = false;
        
        if((*it)->range.end < frameIndex)
            ++it;
        else if((*it)->range.start < frameIndex) {
#ifndef NDEBUG
            Print("(",identity().ID(),") need to shorten tracklet ",(*it)->range.start,"-",(*it)->range.end," to fit frame ",frameIndex);
#endif
            (*it)->range.end = frameIndex - 1_f;
            assert((*it)->range.start <= (*it)->range.end);
            
            (*it)->basic_index.resize((*it)->length().get());
            _basic_stuff.resize((*it)->basic_index.back() + 1);
            _matched_using.resize(_basic_stuff.size());
            shortened_basic_index = true;
            
            for (auto kit = (*it)->posture_index.begin(); kit != (*it)->posture_index.end(); ++kit) {
                if(*kit != -1 && _posture_stuff.at(*kit)->frame >= frameIndex) {
#ifndef NDEBUG
                    auto ff = _posture_stuff.at(*kit)->frame;
#endif
                    assert(*kit < (long long)_posture_stuff.size());
                    _posture_stuff.resize(*kit);
#ifndef NDEBUG
                    Print("(", identity().ID(),")\tposture_stuff.back == ", _posture_stuff.empty() ? Frame_t() : _posture_stuff.back()->frame," (kit = ",ff,")");
#endif
                    
                    (*it)->posture_index.erase(kit, (*it)->posture_index.end());
#ifndef NDEBUG
                        for (auto kit = (*it)->posture_index.rbegin(); kit != (*it)->posture_index.rend(); ++kit) {
                            if(*kit != -1) {
                                auto o = std::distance(kit, (*it)->posture_index.rend());
                                assert(_posture_stuff.at(*kit)->frame == (*it)->start() + Frame_t(o - 1));
                                break;
                            }
                        }
                    
#endif
                    shortened_posture_index = true;
                    break;
                }
            }
            
            ++it;
            
#ifndef NDEBUG
            if(!shortened_posture_index && it == _tracklets.end())
                Print("Individual ", identity().ID()," did not have any postures after ",frameIndex);
#endif
        }
        
        if(it != _tracklets.end()) {
#ifndef NDEBUG
            Print("(", identity().ID(),") found that we need to delete everything after and including ", (*it)->range.start,"-",(*it)->range.end);
#endif
            
            if(!shortened_basic_index && !(*it)->basic_index.empty()) {
                _basic_stuff.resize((*it)->basic_index.front());
                _matched_using.resize(_basic_stuff.size());
            }
            
            if(!shortened_posture_index) {
                auto start = it;
                while(start != _tracklets.end()) {
                    for (auto kit = (*start)->posture_index.begin(); kit != (*it)->posture_index.end(); ++kit) {
                        if(*kit != -1) {
                            assert(*kit < (long long)_posture_stuff.size());
                            _posture_stuff.resize(*kit);
#ifndef NDEBUG
                            Print("(", identity().ID(),")\tposture_stuff.back == ",_posture_stuff.empty() ? Frame_t() : _posture_stuff.back()->frame);
#endif
                            shortened_posture_index = true;
                            break;
                        }
                    }
                    
                    if(shortened_posture_index)
                        break;
                    ++start;
                }
                
#ifndef NDEBUG
                if(!shortened_posture_index)
                    Print("Individual ", identity().ID()," did not have any postures after ",frameIndex);
#endif
            }
            
            _tracklets.erase(it, _tracklets.end());
            
        } else if(!shortened_posture_index) {
#ifndef NDEBUG
            Print("Individual ", identity().ID()," does not delete any frames.");
#endif
        }
//#ifndef NDEBUG
        check_integrity();
//#endif
    }
    
    while(!_best_images.empty()) {
        auto it = --_best_images.end();
        if(it->first < frameIndex)
            break;
        
        _best_images.erase(it);
    }
    
    while(!average_recognition_tracklet.empty()) {
        auto it = --average_recognition_tracklet.end();
        auto current = it->first;
        
        if(current < frameIndex)
            break;
        
        average_recognition_tracklet.erase(it);
    }
    
    /*while(!_manually_matched.empty()) {
        auto kit = --_manually_matched.end();
        if(*kit < frameIndex)
            break;
        _manually_matched.erase(kit);
    }*/
    
    while(!automatically_matched.empty()) {
        auto kit = --automatically_matched.end();
        if(*kit < frameIndex)
            break;
        automatically_matched.erase(kit);
    }
    
    for (auto i=frameIndex; i<=_endFrame; ++i) {
        if(_custom_data.count(i)) {
            for(auto &pair : _custom_data.at(i)) {
                pair.second.second(pair.second.first);
            }
            _custom_data.erase(i);
        }
    }
    
    _endFrame.invalidate();
    if(_startFrame >= frameIndex)
        _startFrame.invalidate();
    else {
        if(!_basic_stuff.empty())
            _endFrame = _basic_stuff.back()->frame;
        else
            _startFrame.invalidate();
    }
    
    _average_recognition.clear();
    _average_recognition_samples = 0;
    
    if(!average_recognition_tracklet.empty())
        calculate_average_tracklet_id();
    
    _local_cache.regenerate(this);
}

/*void Individual::add(Frame_t frameIndex, float time, Blob *blob, const cv::Mat &, const std::vector<HorizontalLine> &, const std::vector<uchar> &)
{
    if(!add(frameIndex, time, blob))
        return;
    
    //_blob_indices[frameIndex] = blob_index;
    
    if(identity().ID() != 0)
        return;
}*/

void Individual::LocalCache::clear() {
    _v_samples.clear();
    //_current_velocities.clear();
    _current_velocity = Vec2(0);
    
    _outline_size = 0;
    _midline_length = 0;
    _outline_samples = 0;
    _midline_samples = 0;
}

void Individual::LocalCache::regenerate(Individual* fish) {
    //Timer timer;
    clear();
    
    for(auto && basic : fish->_basic_stuff) {
        // make sure we dont get an infinite loop
        //assert(!_current_velocities.empty() || basic->frame == fish->start_frame());
        if(basic)
            add(basic->frame, &basic->centroid);
    }
    
    for(auto & p : fish->_posture_stuff) {
        if(p->cached_pp_midline && !p->cached_pp_midline->empty()) {
            add(*p);
        }
    }
    
}

Float2_t Individual::midline_length() const {
    return _local_cache._midline_samples == 0
        ? GlobalSettings::invalid()
        : (_local_cache._midline_length / _local_cache._midline_samples * 1.1_F);
}
size_t Individual::midline_samples() const { return _local_cache._midline_samples; }
Float2_t Individual::outline_size() const {
    return _local_cache._outline_samples == 0
                ? GlobalSettings::invalid()
                : (_local_cache._outline_size / _local_cache._outline_samples);
}

Vec2 Individual::LocalCache::add(Frame_t /*frameIndex*/, const track::MotionRecord *current) {
    const auto frame_rate = track::slow::frame_rate;
    const size_t maximum_samples = max(3.f, frame_rate * 0.1f);
    
    //Print("frame_rate: ", frame_rate, "slow::", slow::frame_rate, " at ", (int*)&slow::frame_rate);
    
    auto raw_velocity = current->v<Units::CM_AND_SECONDS>();

    // use interpolated velocity if available to correct the detected body angle
    // angles that we get from blob.orientation() can be inverted
    // and we wouldnt know it. compare to velocity angle and see
    // if the difference is big. if so, flip it.
    auto v = _v_samples.empty()
        ? raw_velocity
        : (_current_velocity / float(_v_samples.size()));
    
    if(raw_velocity.length() > 0.1f) {
        _v_samples.push_back(raw_velocity);
        _current_velocity += _v_samples.back();
    }
    
    if(_v_samples.size() >= maximum_samples) {
        _current_velocity -= _v_samples.front();
        _v_samples.erase(_v_samples.begin());
    }
    
    /*_current_velocities[frameIndex] = _v_samples.empty()
        ? raw_velocity
        : (_current_velocity / float(_v_samples.size()));*/
    
    return v;
}

void Individual::LocalCache::add(const PostureStuff& stuff) {
    if(stuff.outline) {
        _outline_size += stuff.outline.size();
        ++_outline_samples;
    }
    
    if(stuff.midline_length.has_value()) {
        _midline_length += stuff.midline_length.value();
        ++_midline_samples;
    }
}

blob::Pose Individual::pose_window(Frame_t start, Frame_t end, Frame_t ref) const {
    start = saturate(start, start_frame(), end_frame());
    end = saturate(end, start_frame(), end_frame()) + 1_f;
    if(not ref.valid() || ref < start || ref >= end) {
        ref = start;
    }
    
    FrameRange range(Range<Frame_t>(start, end));
    std::vector<const blob::Pose*> collection;
    collection.reserve(range.length().get());
    int64_t ref_index = -1;
    
    iterate_frames(range.range, [&](Frame_t idx, const std::shared_ptr<TrackletInformation> &, const BasicStuff * basic, const PostureStuff *)
    {
        if(not basic->blob.pred.pose.empty()) {
            if(idx == ref) {
                ref_index = collection.size();
            }
            
            collection.push_back(&basic->blob.pred.pose);
        }
        return true;
    });
    
    if(collection.empty())
        return blob::Pose{};
    
    std::vector<size_t> empty_indexes;
    if(not collection.empty()) {
        if(ref_index == -1) {
            return {};
        }
        auto& ref_pose = collection.at(ref_index);
        for(size_t i=0; i<ref_pose->size(); ++i) {
            auto & pt = ref_pose->point(i);
            if(not pt.valid())
                empty_indexes.push_back(i);
        }
    }
    
    using namespace blob;
    return Pose::mean<GaussianTemporalWeighting<Focus::Center>>([&](size_t index){
        return *collection.at(index);
    }, collection.size(), empty_indexes);
}

int64_t Individual::add(const AssignInfo& info, const pv::Blob& blob, prob_t current_prob)
{
    const auto frameIndex = info.frame->index();
    if (has(frameIndex))
        return -1;
    
    if (_startFrame.valid() && frameIndex >= _startFrame && frameIndex <= _endFrame)
        throw UtilsException("Cannot add intermediate frames out of order.");
    
    // find valid previous frame
    //!TODO: can probably use tracklet ptr here
    auto prev_frame = frameIndex > Tracker::analysis_range().start()
                        ? frameIndex - 1_f
                        : Frame_t();
    const MotionRecord* prev_prop = nullptr;
    if(!empty()) {
        if(prev_frame.valid()) {
            auto previous = find_frame(prev_frame);
            if(previous) {
                prev_frame = previous->frame;
                prev_prop = &previous->centroid;
            }
        }
        
        if (_startFrame > frameIndex) {
            _startFrame = frameIndex;
        }
        
        if (_endFrame < frameIndex) {
            _endFrame = frameIndex;
        }
        
    } else {
        _startFrame = _endFrame = frameIndex;
    }
    
    _hints.push(frameIndex, info.f_prop);
    
    auto stuff = std::make_unique<BasicStuff>();
    stuff->centroid.init(prev_prop, info.frame->time, blob.center(), blob.orientation());
    
    auto v = _local_cache.add(frameIndex, &stuff->centroid);
    
    auto angle = normalize_angle(cmn::atan2(v.y, v.x));
    auto diff = cmn::abs(angle_difference(angle, stuff->centroid.angle()));
    auto diff2 = cmn::abs(angle_difference(angle, normalize_angle(stuff->centroid.angle() + (float)M_PI)));
    
    //if(identity().ID() == Individual::currentID)
    
    if(diff >= diff2) {
        stuff->centroid.flip(prev_prop);
        //if(identity().ID() == Individual::currentID)
    }
    
    stuff->frame = frameIndex;
    stuff->thresholded_size = blob.raw_recount(-1);//,
    stuff->blob = blob;
    
    //const auto ft = FAST_SETTING(track_threshold);
    //assert(blob->last_recount_threshold() == ft); *Tracker::background());
    
    //!TODO: can use previous tracklet here
    //if(prev_props)
    //    prev_props = centroid_weighted(prev_frame);
    
    //stuff->weighted_centroid = new MotionRecord(prev_props, time, centroid_point, current->angle());
    //push_to_segments(frameIndex, prev_frame);
    
    auto cached = info.frame->cached(identity().ID());
    prob_t p{current_prob};
    if(current_prob == -1 && cached) {
        p = probability(SLOW_SETTING(track_consistent_categories)
                        ? info.frame->label(blob.blob_id())
                        : MaybeLabel{},
                        *cached,
                        frameIndex,
                        stuff->blob);//.p;
    }
#ifndef NDEBUG
    else if(current_prob == -1) {
        FormatWarning("No current probability for individual ", identity(), " in frame ", frameIndex);
    }
#endif
    
    auto tracklet = update_add_tracklet(frameIndex, info.f_prop, info.f_prev_prop, stuff->centroid, prev_frame, &stuff->blob, p);
    
    // add BasicStuff index to segment
    auto index = narrow_cast<long_t>(_basic_stuff.size());
    tracklet->add_basic_at(frameIndex, index);
    if(!_basic_stuff.empty() && stuff->frame < _basic_stuff.back()->frame)
        throw SoftException("(", identity(),") Added basic stuff for frame ", stuff->frame, " after frame ", _basic_stuff.back()->frame,".");
    _basic_stuff.emplace_back(std::move(stuff));
    _matched_using.push_back(info.match_mode);
    
    const auto video_length = Tracker::analysis_range().end();
    if(frameIndex >= video_length) {
        update_midlines(&_hints);
    }
    
    return int64_t(index);
}

std::optional<default_config::matching_mode_t::Class> Individual::matched_using(size_t known_index) const
{
    if(known_index >= _matched_using.size()) {
        return std::nullopt;
    }
    return _matched_using[known_index];
}

void Individual::_iterate_frames(const Range<Frame_t>& segment, const std::function<bool(Frame_t frame, const std::shared_ptr<TrackletInformation>&, const BasicStuff*, const PostureStuff*)>& fn) const {
    auto fit = iterator_for(segment.start);
    auto end = _tracklets.end();
    
    for (auto frame = segment.start; frame<=segment.end && fit != end; ++frame) {
        while(fit != end && (*fit)->range.end < frame)
            ++fit;
        if(fit == end)
            break;
        
        if(fit != end && (*fit)->contains(frame)) {
            auto bid = (*fit)->basic_stuff(frame);
            auto pid = (*fit)->posture_stuff(frame);
            
            auto& basic = SEGMENT_ACCESS(_basic_stuff, bid);
            if(!fn(frame, *fit, basic.get(), pid != -1 ? SEGMENT_ACCESS(_posture_stuff, pid).get() : nullptr))
                break;
        }
    }
}

template<typename Enum>
Enum operator |(Enum lhs, Enum rhs)
{
    static_assert(std::is_enum<Enum>::value,
                  "template parameter is not an enum type");

    using underlying = typename std::underlying_type<Enum>::type;

    return static_cast<Enum> (
        static_cast<underlying>(lhs) |
        static_cast<underlying>(rhs)
    );
}

template<typename T>
T operator *(const T& lhs, Reasons rhs)
{
    using underlying = typename std::underlying_type<Reasons>::type;

    return static_cast<T> (
        lhs * static_cast<underlying>(rhs)
    );
}

template<typename T>
T operator *(Reasons rhs, const T& lhs)
{
    using underlying = typename std::underlying_type<Reasons>::type;

    return static_cast<T> (
        lhs * static_cast<underlying>(rhs)
    );
}

template<typename Enum, typename T>
T& operator |=(T &lhs, Enum rhs)
{
    using underlying = typename std::underlying_type<Enum>::type;
    lhs = static_cast<T> (
        lhs |
        static_cast<underlying>(rhs)
    );

    return lhs;
}

TrackletInformation* Individual::update_add_tracklet(const Frame_t frameIndex, const FrameProperties* props, const FrameProperties* prev_props, const MotionRecord& current, Frame_t prev_frame, const pv::CompressedBlob* blob, prob_t current_prob)
{
    //! find a tracklet this (potentially) belongs to
    const std::shared_ptr<TrackletInformation>* tracklet = nullptr;
    if(!_tracklets.empty()) {
        const auto &last = *_tracklets.rbegin();
        
        // check whether we found the right one
        // (it can only be the last one, or no one)
        if(last->end() >= frameIndex - 1_f)
            tracklet = &last;
        // else this frame does not actually belong within the found segment
    }
    
    assert(Tracker::properties(frameIndex) == props);
    assert(Tracker::properties(frameIndex - 1_f) == prev_props);
    
    double tdelta = props && prev_props
        ? props->time() - prev_props->time()
        : 0;
    
    const auto track_trusted_probability = SLOW_SETTING(track_trusted_probability);
    const auto tracklet_punish_timedelta = SLOW_SETTING(tracklet_punish_timedelta);
    const auto huge_timestamp_seconds = SLOW_SETTING(huge_timestamp_seconds);
    const auto tracklet_punish_speeding = SLOW_SETTING(tracklet_punish_speeding);
    const auto tracklet_max_length = SLOW_SETTING(tracklet_max_length);
    const auto frame_rate = SLOW_SETTING(frame_rate);
    
    uint32_t error_code = 0;
    error_code |= Reasons::FramesSkipped         * uint32_t(prev_frame != frameIndex - 1_f);
    error_code |= Reasons::ProbabilityTooSmall   * uint32_t(current_prob != -1 && current_prob < track_trusted_probability);
    error_code |= Reasons::TimestampTooDifferent * uint32_t(tracklet_punish_timedelta && tdelta >= huge_timestamp_seconds);
    error_code |= Reasons::ManualMatch           * uint32_t(is_manual_match(frameIndex));
    error_code |= Reasons::NoBlob                * uint32_t(!blob);
    error_code |= Reasons::WeirdDistance         * uint32_t(tracklet_punish_speeding && current.speed<Units::CM_AND_SECONDS>() >= weird_distance());
    error_code |= Reasons::MaxSegmentLength      * uint32_t(tracklet_max_length > 0 && tracklet && *tracklet && (*tracklet)->length().get() / float(frame_rate) >= tracklet_max_length);
    
    const bool tracklet_ended = error_code != 0;

    if(frameIndex == _startFrame || tracklet_ended) {
        if(!_tracklets.empty()) {
            _tracklets.back()->error_code = error_code;
        }

        return _tracklets.emplace_back(std::make_shared<TrackletInformation>(Range<Frame_t>(frameIndex, frameIndex), !blob || blob->split() ? Frame_t() : frameIndex)).get();
        
    } else if(prev_frame == frameIndex - 1_f) {
        assert(!_tracklets.empty());
        tracklet = &(*_tracklets.rbegin());
        (*tracklet)->range.end = frameIndex;
        if(!(*tracklet)->first_usable.valid() && blob && !blob->split())
            (*tracklet)->first_usable = frameIndex;
    } // else... nothing

    return tracklet ? tracklet->get() : nullptr;
}

Float2_t Individual::weird_distance() {
    const auto track_max_speed = SLOW_SETTING(track_max_speed);
    return track_max_speed * 0.99_F;
}

void Individual::clear_post_processing() {
    for(auto & stuff : _posture_stuff) {
        if(stuff->head)
            stuff->head = nullptr;
        //stuff->midline = nullptr;
        stuff->posture_original_angle.reset();
        stuff->midline_length.reset();
        stuff->midline_angle.reset();
    }
    for(auto && [frame, custom] : _custom_data) {
        for(auto it = custom.begin(); it!=custom.end();) {
            if(it->first == VisualField::custom_id) {
                auto [ptr, fn] = it->second;
                fn(ptr);
                it = custom.erase(it);
            } else
                ++it;
        }
    }
}

void Individual::update_midlines(const CacheHints* hints) {
    /*if(FAST_SETTING(posture_direction_smoothing) == 0) {
        update_frame_with_posture(frameIndex);
    }*/
    
    const auto smooth_range = Frame_t(FAST_SETTING(posture_direction_smoothing));
    const auto video_length = Tracker::analysis_range().end();
    auto end_frame = Tracker::end_frame();
    
    //! find the first frame that needs to be cached, but hasnt been yet
    auto it = _posture_stuff.rbegin(), last_found = _posture_stuff.rend();
    for (; it != _posture_stuff.rend(); ++it) {
        assert(*it != nullptr);
        
        if((smooth_range == 0_f || video_length == end_frame || (*it)->frame <= end_frame.try_sub(smooth_range)) && (*it)->cached_pp_midline)
        {
            if((*it)->cached()) {
                break;
            } else
                last_found = it;
        }
    }
    
    it = last_found;
    if(it != _posture_stuff.rend()) {
        //long_t last_frame = (*it)->frame;
        for (; ; --it) {
            assert(*it != nullptr);
            
            if((smooth_range == 0_f || video_length == end_frame || (*it)->frame <= end_frame.try_sub(smooth_range)) && (*it)->cached_pp_midline)
            {
                assert((*it)->cached_pp_midline->original_angle() != FLT_MAX
                       && (*it)->cached_pp_midline->original_angle() != std::numeric_limits<Float2_t>::infinity());
                (*it)->posture_original_angle = (*it)->cached_pp_midline->original_angle();
                (*it)->midline_angle = (*it)->cached_pp_midline->angle();
                (*it)->midline_length = (*it)->cached_pp_midline->len();

                auto basic = basic_stuff((*it)->frame);
                auto base_it = it.base() - 1;
                update_frame_with_posture(*basic, base_it, hints);
            }
            
            if(it == _posture_stuff.rbegin())
                break;
        }
    }
    
    /*long_t last_frame = start_frame()-1;
    if(!_posture_stuff.empty()) {
        last_frame = _posture_stuff.rbegin()->frame;
    }
    
    
    auto it = _cached_pp_midlines.rbegin();
    std::set<idx_t> frames;
    for (; it != _cached_pp_midlines.rend(); ++it)
    {
        if (it->first > _cached_pp_midlines.rbegin()->first - smooth_range && end_frame() < video_length)
            continue;
        
        if (it->first <= last_frame)
            break;
        
        frames.insert(it->first);
    }
    for (auto &frame: frames) {
        if(_posture_original_angles.count(frame) == 0) {
            _posture_original_angles[frame] = _cached_pp_midlines.at(frame)->original_angle();
        }
    }
    for (auto &frame: frames) {
        update_frame_with_posture(frame);
    }*/
}

Midline::Ptr Individual::calculate_midline_for(const PostureStuff &posture) const
{
    //if(!posture || !basic)
    //    return nullptr;
    
    auto &ptr = posture.cached_pp_midline;
    //auto &blob = basic.blob;
    //basic.pixels = nullptr;
    
    Midline::Ptr midline;
    
    if(ptr) {
        //Timer timer;
        midline = std::make_unique<Midline>(*ptr);
        
        MovementInformation movement;
        if(SLOW_SETTING(posture_direction_smoothing) > 1) {
            movement = calculate_previous_vector(posture.frame);
        }
        
        midline->post_process(movement, DebugInfo{posture.frame, identity().ID(), false});
        if(!midline->is_normalized())
            midline = midline->normalize();
#ifndef NDEBUG
        else if(size_t(_warned_normalized_midline.elapsed())%5 == 0) {
            FormatWarning(identity().ID()," has a pre-normalized midline in frame ",posture.frame,". not normalizing it again.");
        }
#endif
        
    }
    
    return midline;
}

/*Midline::Ptr Individual::update_frame_with_posture(const BasicStuff>& basic, const std::shared_ptr<PostureStuff>& posture, const CacheHints* hints) {
    auto it = std::partition_point(_posture_stuff.begin(), _posture_stuff.end(), [c = posture->frame](const std::shared_ptr<PostureStuff>& other) {
        return other->frame < c;
    });

    if (it == _posture_stuff.end())
        throw U_EXCEPTION("Cannot find the posture we are talking about (", posture->frame, ").");

    return update_frame_with_posture(basic, it, hints);
}*/

Midline::Ptr Individual::update_frame_with_posture(BasicStuff& basic, const decltype(Individual::_posture_stuff)::const_iterator& posture_it, const CacheHints* hints) {
    auto &posture = **posture_it;
    auto &ptr = posture.cached_pp_midline;
    auto &blob = basic.blob;
    //basic.pixels = nullptr;
    
    Midline::Ptr midline;
    
    if(ptr) {
        midline = calculate_midline_for(posture);
        auto &outline = posture.outline;
        auto &c = basic.centroid;
        
        if(!midline)
            return nullptr;
        
        const size_t head_index = cmn::min(midline->segments().size() - 1u, size_t(roundf(midline->segments().size() * FAST_SETTING(posture_head_percentage))));
        
        const auto bounds = blob.calculate_bounds();
        const auto pt = midline->real_point(bounds, head_index);
        
        const PostureStuff* previous = nullptr;
        if (posture_it != _posture_stuff.begin()) {
            previous = (*(posture_it - 1)).get();
        }

        auto prop = Tracker::properties(posture.frame, hints);
        if(not prop)
            throw InvalidArgumentException("Cannot find info on frame ", posture.frame);
        
        posture.head = std::make_unique<MotionRecord>();
        posture.head->init(previous ? previous->head.get() : nullptr,
                           prop->time(),
                           pt,
                           midline->angle());
        
         //ptr//.outline().original_angle();
#if DEBUG_ORIENTATION
        _why_orientation[frame] = OrientationProperties(
                                                        frame,
                                                        midline->original_angle(),
                                                        midline->inverted_because_previous() //ptr.outline().inverted_because_previous()
                                                        );
#endif
        
        // see if the centroid angle has to be inverted, because
        // we now see its actually flipped (image moments are only [0,2pi] without sign)
        // because the posture angle is more than 60 degrees off
        {
            if(angle_between_vectors(Vec2(cos(c.angle()), sin(c.angle())),
                                     Vec2(cos(midline->angle()), sin(midline->angle())))
               > RADIANS(60))
            {
                c.flip(previous ? previous->head.get() : nullptr);
            }
        }
        
        // calculate midline centroid
        Vec2 centroid_point(0, 0);
        if(midline && not midline->segments().empty()) {
            centroid_point = midline->real_point(bounds, min(midline->segments().size() / 2u, midline->segments().size() - 1u));
            
        } else {
            auto points = outline.uncompress();

            for (auto &p : points) {
                centroid_point += p;
            }
            centroid_point /= float(points.size());
            centroid_point += bounds.pos();
        }
        
        posture.centroid_posture = std::make_unique<MotionRecord>();
        posture.centroid_posture->init(previous
                                        ? previous->centroid_posture.get()
                                        : nullptr,
                                       prop->time(),
                                       centroid_point,
                                       midline->angle());
        posture.midline_angle = midline->angle();
        posture.midline_length = midline->len();
        
        assert(posture.midline_length.has_value());
        /*{
            if (not posture.posture_original_angle.has_value()
                && posture.cached_pp_midline)
            {
                assert(posture.cached_pp_midline->original_angle() != FLT_MAX
                       && posture.cached_pp_midline->original_angle() != std::numeric_limits<Float2_t>::infinity());*/
        //auto oangle = posture.cached_pp_midline->original_angle();
        auto nangle = midline->original_angle();
        assert(nangle != FLT_MAX
               && nangle != std::numeric_limits<Float2_t>::infinity());
        posture.posture_original_angle = nangle;
            /*}
        }*/
        
        _local_cache.add(posture);
    }
    
    return midline;
}

template<class RandAccessIter, typename T = typename RandAccessIter::value_type>
T static_median(RandAccessIter begin, RandAccessIter end) {
    if(begin == end){ throw std::invalid_argument("Median of empty vector."); }
  std::size_t size = end - begin;
  std::size_t middleIdx = size/2;
  RandAccessIter target = begin + middleIdx;
  std::nth_element(begin, target, end);

  if(size % 2 != 0){ //Odd number of elements
    return *target;
  }else{            //Even number of elements
    double a = *target;
    RandAccessIter targetNeighbor= target-1;
    std::nth_element(begin, targetNeighbor, end);
    return (a+*targetNeighbor)/2.0;
  }
}

CacheHints::CacheHints(size_t size) {
    clear(size);
}

template<class T>
auto insert_at(std::vector<T>& vector, T&& element) {
    return vector.insert(std::upper_bound(vector.begin(), vector.end(), element), std::move(element));
}

struct CompareByFrame {
    constexpr bool operator()(const FrameProperties* A, const FrameProperties* B) {
        return (!A && B) || (A && B && A->frame() < B->frame());
    }
    constexpr bool operator()(const FrameProperties* A, const Frame_t& B) {
        return !A || A->frame() < B;
    }
    constexpr bool operator()(const Frame_t& A, const FrameProperties* B) {
        return B && A < B->frame();
    }
};

void CacheHints::remove_after(Frame_t index) {
    auto here = std::lower_bound(_last_second.begin(), _last_second.end(), index, CompareByFrame{});
    if(here == _last_second.end())
        return;
    std::fill(here, _last_second.end(), nullptr);
    std::rotate(_last_second.begin(), here, _last_second.end());
}

void CacheHints::push(Frame_t index, const FrameProperties* ptr) {
    auto here = std::upper_bound(_last_second.begin(), _last_second.end(), index, CompareByFrame{});
    if (_last_second.size() > 1) {
        if (here == _last_second.end() || !*here || (*here)->frame() < index) {
            // have to insert past the end -> rotate
            here = std::rotate(_last_second.begin(), ++_last_second.begin(), _last_second.end());

        }
        else {
            if (here == _last_second.begin()) {
                if (*here != nullptr)
                    return; // the vector is already full and this is older (so dont add it)
            }
            else if (*(here - 1) != nullptr) {
                // rotate everything thats right of our element to the end
                here = std::rotate(_last_second.begin(), ++_last_second.begin(), here + 1);
            }
            else
                --here;
        }

        *here = ptr;
    }
    else if (!_last_second.empty())
        _last_second.back() = ptr;
    else
        _last_second.push_back(ptr);
}

/*void CacheHints::push_front(Frame_t index, const FrameProperties* ptr) {
    //assert(current == -1 || current - (long_t)_last_second.size() == index + 1);
    assert(!ptr || ptr->frame == index);
    assert(!_last_second.empty());
    
    if(_last_second.front() == nullptr) {
        auto front = std::upper_bound(_last_second.begin(), _last_second.end(), (const track::FrameProperties*)0);
        assert(front != _last_second.begin());
        --front;
        
        assert(front != _last_second.end());
        *front = ptr;
        
    } else {
        --current;
        
        if(_last_second.size() > 1)
            std::rotate(_last_second.rbegin(), ++_last_second.rbegin(), _last_second.rend());
        
        _last_second.front() = ptr;
    }
    
    if(_last_second.back())
        current = _last_second.back()->frame;
    else current.invalidate();
}*/

size_t CacheHints::size() const {
    return _last_second.size();
}

bool CacheHints::full() const {
    return _last_second.empty() || (_last_second.front() != nullptr && _last_second.back() != nullptr);
}

void CacheHints::clear(size_t size) {
    if (size == 0 && (SLOW_SETTING(frame_rate) < 0 || SLOW_SETTING(frame_rate) == uint32_t(-1))) {
#ifndef NDEBUG
        FormatExcept("Size=", size," frame_rate=", SLOW_SETTING(frame_rate),"");
#endif
        _last_second.resize(0);
    } else {
        _last_second.resize(size > 0 ? size : SLOW_SETTING(frame_rate));
    }
    std::fill(_last_second.begin(), _last_second.end(), nullptr);
    current.invalidate();
}

template<class T, class U>
typename std::vector<T>::const_iterator find_in_sorted(const std::vector<T>& vector, const U& v) {
    auto it = std::lower_bound(vector.begin(),
                               vector.end(),
                               v,
                [](auto& l, auto& r){ return !l || l->frame() < r; });
    return it == vector.end() || (*it)->frame() == v ? it : vector.end();
}

const FrameProperties* CacheHints::properties(Frame_t index) const {
    if(!index.valid() || _last_second.empty() || !_last_second.back() || index > _last_second.back()->frame()) //|| (idx = size_t((current - index).get())) >= size())
        return nullptr;
    
    if(_last_second.back()->frame() == index)
        return _last_second.back();
    
    auto it = find_in_sorted(_last_second, index);
    if(it == _last_second.end())
        return nullptr;
    else if((*it)->frame() == index)
        return *it;
    
    return nullptr;
}

tl::expected<IndividualCache, const char*> Individual::cache_for_frame(const FrameProperties* previous, Frame_t frameIndex, double time, const CacheHints* hints) const {
    if(not frameIndex.valid())
        return tl::unexpected("Invalid frame in cache_for_frame.");
    if (not _startFrame.valid())
        return tl::unexpected("The individual is empty, there is no cache for next frame.");
    if (frameIndex <= _startFrame)
        return tl::unexpected("Cannot cache for a frame before the individuals first frame after the start frame.");
    
    assert(not empty());
    IndividualCache cache;
    
    // find the first frame thats set for the individual
    const auto it = iterator_for(frameIndex - 1_f);
    
    //! collect samples from previous segments
    //bool manually_matched_segment = false;
    bool last_frame_manual = false;
    cache.last_seen_px = Vec2(-FLT_MAX);
    cache.current_category = std::nullopt;
    const auto cm_per_pixel = SLOW_SETTING(cm_per_pixel);
    const auto consistent_categories = SLOW_SETTING(track_consistent_categories);
    const auto track_max_speed_px = SLOW_SETTING(track_max_speed) / cm_per_pixel;
    const auto frame_rate = SLOW_SETTING(frame_rate);
    const auto track_max_reassign_time = SLOW_SETTING(track_max_reassign_time);
    
    //auto tracklet = get_tracklet(frameIndex-1);
    if(it != _tracklets.end()) {
        long_t bdx = -1;
        
        if((*it)->contains(frameIndex - 1_f)) {
            // is a valid segment
            //if(is_manual_match((*it)->start()) && frameIndex - (*it)->start() < Frame_t(frame_rate * 0.05))
            //    manually_matched_segment = true;
            
            bdx = (*it)->basic_stuff(frameIndex - 1_f);
            assert(bdx != -1);
            
        } else if((*it)->end() < frameIndex) {
            bdx = (*it)->basic_stuff((*it)->end());
            
        } else if(it != _tracklets.begin()) {
            auto copy = it;
            --copy;
            
            assert((*copy)->end() < frameIndex);
            bdx = (*copy)->basic_stuff((*copy)->end());
        }
        
        if(bdx != -1)
            cache.last_seen_px = _basic_stuff.at(bdx)->centroid.pos<Units::PX_AND_SECONDS>();
        
    } else if(!_tracklets.empty()) {
        assert(frameIndex > (*_tracklets.rbegin())->end());
        auto bdx = (*_tracklets.rbegin())->basic_stuff((*_tracklets.rbegin())->end());
        assert(bdx != -1);
        cache.last_seen_px = _basic_stuff.at(bdx)->centroid.pos<Units::PX_AND_SECONDS>();
    }
    
#ifndef NDEBUG
    if(cache.last_seen_px.x == -FLT_MAX)
        FormatWarning("No previous position for fish ", identity().ID()," in frame ",frameIndex,".");
#endif
    
    // find posture stuff and basic stuff for previous frame
    long_t bdx = -1, pdx = -1;
    if(!_tracklets.empty()) {
        if(it != _tracklets.end()
           && (*it)->contains(frameIndex - 1_f))
        {
            bdx = (*it)->basic_stuff(frameIndex - 1_f);
            pdx = (*it)->posture_stuff(frameIndex - 1_f);
            
        } else {
            if(it != _tracklets.end() && (*it)->end() <= frameIndex - 1_f) {
                bdx = (*it)->basic_stuff((*it)->end());
                pdx = (*it)->posture_stuff((*it)->end());
            }
            else if(frameIndex <= _startFrame && _startFrame.valid()) {
                bdx = (*_tracklets.begin())->basic_stuff(_startFrame);
                pdx = (*_tracklets.begin())->posture_stuff(_startFrame);
            } else if(frameIndex >= _endFrame && _endFrame >= _startFrame && _endFrame.valid()) {
                bdx = (*_tracklets.rbegin())->basic_stuff(_endFrame);
                pdx = (*_tracklets.rbegin())->posture_stuff(_endFrame);
            } else
                Print("Nothing to be found for ",frameIndex - 1_f);
        }
    }
    
    auto pp = bdx != -1 ? _basic_stuff.at(bdx).get() : nullptr;
    auto pp_posture = pdx != -1 ? _posture_stuff.at(pdx).get() : nullptr;
    
    /*auto _pp = find_frame(frameIndex-1);
    if(pp != _pp) {
        Print("Frame ",frameIndex,", individual ",identity().ID(),": ",_pp ? _pp->frame : -1," != ",pp ? pp->frame : -1);
    }*/
    
    //auto pp = find_frame(frameIndex-1);
    //auto && [pp, pp_posture] = all_stuff(frameIndex - 1);
    //auto pp = find_frame(frameIndex-1);
    assert(pp); // fish is not empty, find_frame should at least return _startFrame
    
    //auto props = Tracker::properties(frameIndex);
    assert(not previous || frameIndex - 1_f == previous->frame());
    auto prev_props = previous;//Tracker::properties(frameIndex - 1_f, hints);
    if(!prev_props) {
        if(!Tracker::instance()->frames().empty()) {
            auto it = Tracker::instance()->frames().rbegin();
            while(it != Tracker::instance()->frames().rend() && (*it)->frame() >= frameIndex)
            {
                ++it;
            }
            
            if(it != Tracker::instance()->frames().rend())
                prev_props = (*it).get();
        }
    }
    
    cache.previous_frame = pp ? pp->frame : (frameIndex - 1_f);
    auto pp_props = pp && pp->frame == (frameIndex - 1_f) && prev_props
        ? prev_props
        : Tracker::properties(cache.previous_frame, hints);
    assert(!prev_props || prev_props->time() != time);
    
    float ptime = pp_props ? pp_props->time() : (- ((double)frameIndex.get() - (double)cache.previous_frame.get()) * 1 / double(frame_rate) + time);
    
    assert(ptime >= 0);
    assert(time >= ptime);
    if(time - ptime >= track_max_reassign_time) {
        assert(frameIndex.valid() && cache.previous_frame.valid());
        assert(frameIndex > cache.previous_frame);
        ptime = (- ((double)frameIndex.get() - (double)cache.previous_frame.get()) * 1 / double(frame_rate) + time);
    }
    //prev_props ? prev_props->time : ((frameIndex - (frameIndex - 1)) / double(SLOW_SETTING(frame_rate)) + time);
    
    assert(time >= ptime);
    auto tdelta = time - ptime;//pp.first < frameIndex ? (time - ptime) : time;
    cache.local_tdelta = prev_props ? time - prev_props->time() : 0;
    
    if(tdelta == 0) {
        long_t bdx = -1, pdx = -1;
        
        if(!_tracklets.empty()) {
            if(it != _tracklets.end()
               && (*it)->contains(frameIndex - 1_f))
            {
                bdx = (*it)->basic_stuff(frameIndex - 1_f);
                pdx = (*it)->posture_stuff(frameIndex - 1_f);
                
                Print("#1 ", bdx, " ", pdx, " ", (*it)->range, " contains ", frameIndex - 1_f);
                
            } else {
                if(it != _tracklets.end() && (*it)->end() <= frameIndex - 1_f) {
                    bdx = (*it)->basic_stuff((*it)->end());
                    pdx = (*it)->posture_stuff((*it)->end());
                    
                    Print("#2 ", bdx, " ", pdx, " ", (*it)->end(), " <= ", frameIndex - 1_f);
                }
                else if(frameIndex <= _startFrame && _startFrame.valid()) {
                    bdx = (*_tracklets.begin())->basic_stuff(_startFrame);
                    pdx = (*_tracklets.begin())->posture_stuff(_startFrame);
                    
                    Print("#3 ", bdx, " ", pdx, " ", frameIndex, " <= ", _startFrame);
                    
                } else if(frameIndex >= _endFrame && _endFrame >= _startFrame && _endFrame.valid()) {
                    bdx = (*_tracklets.rbegin())->basic_stuff(_endFrame);
                    pdx = (*_tracklets.rbegin())->posture_stuff(_endFrame);
                    
                    Print("#4 ", bdx, " ", pdx, " ", frameIndex, " >= ", _endFrame, "  && ", _endFrame, " >= ", _startFrame);
                    
                } else
                    Print("Nothing to be found for ",frameIndex - 1_f);
            }
        }
        
        throw U_EXCEPTION("No time difference between ",frameIndex," and ",cache.previous_frame," in calculate_next_positions. pp->frame=", pp ? pp->frame : Frame_t(), " ");
    }
    
    auto raw = Vec2(0.0, 0.0);
    auto raw_acc = Vec2(0, 0);
    
    int used_frames = 0;
    //Median<size_t> size_median;
    //float weights = 0;
    
    //! Collect recent number of valid samples within $t - \mathrm{fps} <= \dot{t} <= t$, where all distances between segments must not be reassigned ($\Delta t < fps * T_mathrm{max}$).
    size_t N = 0;
    if(it != _tracklets.end()) {
        //assert((*it)->contains(frameIndex));
        
        const auto lower_limit = frameIndex.try_sub(Frame_t{frame_rate});
        auto previous_frame = frameIndex;
        const auto time_limit = Frame_t(Frame_t::number_t(frame_rate * track_max_reassign_time));
        
        auto copy = it;
        while(true) {
            if((*copy)->end() < lower_limit) {
                break;
            }
            
            if(previous_frame.try_sub((*copy)->end()) > time_limit)
            {
                break;
            }
            
            auto start = (*copy)->start();
            if(start < lower_limit)
                start = lower_limit;
            auto end = (*copy)->end();
            if(end > frameIndex)
                end = frameIndex;
                
            previous_frame = start;
            
            N += max(Frame_t::number_t(0), (end - start).get() + 1);
            if(copy == _tracklets.begin())
                break;
            --copy;
        }
    } else {
        //thread_print("** ", frameIndex, ": ", identity().ID()," no segments.");
    }
    
    
    //! retrieve a number (6) of samples from previous frames in order
    //! to predict the current direction etc.
    auto recent_number_samples = N;
    
    Range<Frame_t> range(max(_startFrame, cache.previous_frame.try_sub(6_f)),
                         cache.previous_frame);
    std::vector<prob_t> average_speed;
    average_speed.reserve(range.length().get() + 1);
    
    //Median<prob_t> average_speed;
    Vec2 previous_v;
    const MotionRecord* previous_p = nullptr;
    double previous_t = 0;
    Frame_t previous_f;

#if !COMMONS_NO_PYTHON
    std::unordered_map<int, size_t> labels;
    size_t samples = 0;
    
    if(consistent_categories
       && cache.previous_frame.valid())
    {
        std::shared_lock guard(Categorize::DataStore::range_mutex());
        iterate_frames(Range<Frame_t>{
                max(_startFrame,
                    cache.previous_frame.try_sub(Frame_t(frame_rate * 2u))),
                cache.previous_frame
            }, [&labels, &samples](auto frame, const auto&, auto basic, auto) -> bool
        {
            auto ldx = Categorize::DataStore::_ranged_label_unsafe(frame, basic->blob.blob_id());
            if(ldx.has_value()) {
                ++labels[ldx.value()];
                ++samples;
            }
            return true;
        });
    }
#endif

    // cm/s / (cm/px)
    // (cm/s)^2 / (cm/px)^2 = (cm^2/s^2) / (cm^2/px^2) = 1 * px^2/s^2
    const auto track_max_px_sq = SQR(track_max_speed_px);
    const FrameProperties *properties = nullptr;
    auto end = Tracker::instance()->frames().end();
    auto iterator = end;
    
    iterate_frames(range, [&](Frame_t frame, const std::shared_ptr<TrackletInformation> &, const BasicStuff* basic, auto) -> bool
    {
        if(is_manual_match(frame)) {
            last_frame_manual = true;
            return true;
        }
        
        const FrameProperties* c_props = nullptr;
        if(iterator != end && ++iterator != end && (*iterator)->frame() == frame) {
            c_props = (*iterator).get();
        } else {
            iterator = Tracker::instance()->properties_iterator(frame/*, hints*/);
            if(iterator != end)
                c_props = (*iterator).get();
        }
        
        auto &h = basic->centroid;
        if(!previous_p) {
            properties = c_props;
            
            previous_p = &h;
            previous_t = c_props ? c_props->time() : 0;
            previous_f = frame;
            return true;
        }
        
        auto p_props = properties && properties->frame() == frame - 1_f
                        ? properties
                        : Tracker::properties(frame - 1_f, hints);
        properties = c_props;
        
        if (c_props && p_props && previous_p) {//(he || h)) {
            double tdelta = c_props->time() - p_props->time();
            
            if(tdelta > prob_t(1))
                return true;
            
            //! \mathbf{v}_i(t) = \mathbf{p}_i'(t) = \frac{\delta}{\delta t} \mathbf{p}_i(t)
            auto v = (h.pos<Units::PX_AND_SECONDS>() - previous_p->pos<Units::PX_AND_SECONDS>()) / (c_props->time() - previous_t);
            auto L_sq = v.sqlength();
            
            //! \hat{\mathbf{v}}_i(t) =
            //!     \mathbf{v}_i(t) *
            //!     \begin{cases}
            //!         1                                       & \mathrm{if} \norm{\mathbf{v}_i(t)} \le D_\mathrm{max} \\
            //!         D_\mathrm{max} / \norm{\mathbf{v}_i(t)} & \mathrm{otherwise}
            //!     \end{cases}
            if(L_sq >= track_max_px_sq) {
                v *= track_max_speed_px / sqrt(L_sq);
                L_sq = track_max_px_sq;
            }
            
            assert(!std::isnan(v.x));
            raw += v;
            average_speed.push_back(L_sq);
            
            //! \mathbf{a}_i(t) = \frac{\delta}{\delta t} \hat{\mathbf{v}}_i(t)
            if(tdelta > 0 && (previous_v.x != 0 || previous_v.y != 0))
                raw_acc += (v - previous_v) / tdelta;
            
            previous_v = v;
            previous_p = &h;
            previous_t = c_props->time();
            previous_f = frame;
            
            used_frames++;
        }
        
        return used_frames <= 5;
    });
    
    if(used_frames) {
        //! \mean{\mathbf{d}_i}(t) = \frac{1}{F(t)-F(\tau)+5} \sum_{k \in [F(\tau)-5, F(t)]} \hat{\mathbf{v}}_i(\Tau(k))
        raw /= prob_t(used_frames);
        
        //! \mean{\mathbf{a}}_i(t) = \mathbf{U}\left( \frac{1}{F(t)-F(\tau)+5} \sum_{k \in [F(\tau)-5, F(t)]} \mathbf{a}_i(\Tau(k)) \right)
        raw_acc /= prob_t(used_frames);
    }

#if !COMMONS_NO_PYTHON
    double max_samples = 0, mid = -1;
    for(auto & [l, n] : labels) {
        auto N = n / double(samples);
        if(N > max_samples) {
            max_samples = N;
            mid = l;
        }
    }
    
    if(mid < 0)
        cache.current_category = std::nullopt;
    else
        cache.current_category = MaybeLabel{narrow_cast<uint16_t>(mid)};
#else
    cache.current_category = std::nullopt;
#endif
    
    const MotionRecord* c = pp ? &pp->centroid : nullptr; //centroid_weighted(cache.previous_frame);
    
    //! \mean{s}_{i}(t) = \underset{k \in [F(\tau)-5, F(t)]}{\median} \norm{\hat{\mathbf{v}}_i(\Tau(k))}
    prob_t speed = max(0.6f, sqrt(used_frames ? static_median(average_speed.begin(), average_speed.end()) : 0));
    
    //! \lambda
    const float lambda = SQR(SQR(max(0, min(1, FAST_SETTING(track_speed_decay)))));
    
    //! \mean{\mathbf{d}_i}(t)
    Vec2 direction;
    NAN_SAFE_NORMALIZE(raw, direction)
    
    Vec2 est;
    prob_t last_used = ptime;
    auto pprops = Tracker::properties(cache.previous_frame - 1_f, hints);
    if(pprops)
        last_used = pprops->time();
    
    NAN_SAFE_NORMALIZE(raw_acc, raw_acc)
    
    if(used_frames > 0 && lambda < 1) {
        for (auto f = cache.previous_frame; f < frameIndex; ++f) {
            auto props = Tracker::properties(f, hints);
            if(props) {
                //! \Tau'(k)
                prob_t tdelta = props->time() - last_used;
                last_used = props->time();
                
                //! w(f) = \frac{1 + \lambda^4}{1 + \lambda^4 \max\left\{ 1, f - F(\tau_i) + 1 \right\}}
                prob_t weight = (1 + lambda) / (1 + lambda * max(1_f, f - cache.previous_frame + 1_f).get());
                //if(weight <= 0.0001)
                //    break;
                
                //! \dot{\mathbf{p}}_i(t) = s_i(t) \sum_{k\in [F(\tau_i), F(t)-1]} w(k) \left(\mean{\mathbf{d}_i}(t) + \Tau'(k) * \mean{\mathbf{a}}_i(t) \right)
                est += weight * tdelta * (speed * (direction + tdelta * raw_acc));
                //weights += weight;
            }
        }
    }
    
    if(c)
        est += c->pos<Units::PX_AND_SECONDS>();
    
    auto h = c;
    if(FAST_SETTING(calculate_posture)) {
        if(pp_posture && pp_posture->centroid_posture)
            h = pp_posture->centroid_posture.get();
    }
    
    //cache.speed = h ? h->speed<Units::CM_AND_SECONDS>() : 0;
    //cache.h = h;
    cache.estimated_px = est;
    
    if(not FAST_SETTING(track_time_probability_enabled)
       || last_frame_manual)
    {
        cache.time_probability = 1;
    } else if(tdelta > SLOW_SETTING(track_max_reassign_time)) {
        cache.time_probability = 0;
    } else {
        cache.time_probability = time_probability(tdelta, cache.previous_frame, recent_number_samples);
    }
    
    //thread_print("** ",frameIndex, ": ", identity().ID(), " => t=",cache.time_probability, " previous=", cache.previous_frame, " recent=",recent_number_samples, " tdelta=", tdelta);
    //PPFrame::Log("** ",frameIndex, ": ", identity().ID(), " => t=",cache.time_probability, " previous=", cache.previous_frame, " recent=",recent_number_samples, " tdelta=", tdelta);
    cache.valid_frame = !h || last_frame_manual;
    
    assert(!std::isnan(cache.estimated_px.x) 
           && !std::isnan(cache.estimated_px.y));
    
    return cache;
}

struct TimeCache {
    
};

prob_t Individual::time_probability(double tdelta, const Frame_t& previous_frame, size_t recent_number_samples) {
    
    /// handled in cache_for_frame:
    //if(!FAST_SETTING(track_time_probability_enabled))
    //    return 1;
    //if (cache.tdelta > SLOW_SETTING(track_max_reassign_time))
    //    return 0.0;
    
    //if(cache.last_frame_manual)
    //    return 1;
    
    const float Tdelta = 1.f / float(SLOW_SETTING(frame_rate));
    
    // make sure that very low frame rates work
    //! F_\mathrm{min} = \min\left\{\frac{1}{T_\Delta}, 5\right\}
    const uint32_t minimum_frames = min(SLOW_SETTING(frame_rate), 5u);
    
    //! R_i(t) = \norm{ \givenset[\Big]{ \Tau(k) | F(t) - T_\Delta^{-1} \leq k \leq t \wedge \Tau(k) - \Tau(k-1) \leq T_\mathrm{max}} }
    
    /**
        \begin{equation} \label{eq:time_prob}
        T_i(t) = \left(1 - \min\left\{ 1, \frac{\max\left\{ 0, \tau_i - t - T_\Delta \right\}} {T_\mathrm{max}} \right\}\right) * \begin{cases}
                \min\left\{ 1, \frac{R_i(\tau_i) - 1}{F_\mathrm{min}} + P_\mathrm{min} \right\} & F(\tau_i) \geq F(t_0) + F_\mathrm{min}\\
                1 & \mathrm{otherwise}
            \end{cases}
        \end{equation}
     */
    
    float p = 1.0f - min(1.0f, max(0, (tdelta - Tdelta) / SLOW_SETTING(track_max_reassign_time)));
    if(previous_frame >= Tracker::start_frame() + Frame_t(minimum_frames))
        p *= min(1.f, float(recent_number_samples - 1) / float(minimum_frames) + FAST_SETTING(match_min_probability));
    
    return p * 0.75 + 0.25;
}

#include <tracking/Tracker.h>

inline Float2_t adiffangle(const Vec2& A, const Vec2& B) {
    // cross A.X*B.Y-A.Y*B.X;
    // atan2(norm(cross(a,b)), dot(a,b))
    // B.x*A.y-B.y*A.x
    
    // angle = atan2(A.y, A.x) - atan2(B.y, B.x);
    // angle = -atan2(B.x * A.y - B.y * A.x, dot(B, A))
    //        where dot = B.x * A.x  + B.y * A.y
    return -atan2(-B.y*A.x+B.x*A.y, B.x*A.x+B.y*A.y);
}

prob_t Individual::position_probability(const IndividualCache cache, Frame_t frameIndex, size_t, const Vec2& position, const Vec2& blob_center)
{
    UNUSED(frameIndex)
#ifndef NDEBUG
    // individual_empty would be set
    //if (frameIndex <= _startFrame)
    //    throw U_EXCEPTION("Cannot calculate probability for a frame thats previous to all known frames.");
#endif
    
    // $\tau$ is the time (s) of the most recent frame assigned to individual $i$
    // $\dot{p}_i(t)$ is the projected position for individual $i$ in the current frame
    
    //! S_{i,b}(t) &= \left(1 + \frac{\norm{ (\mathbf{p}_b(\tau_i) - \dot{\mathbf{p}}_i(t)) / (\tau_i - t) }}{ D_{\mathrm{max}}}\right)^{-2}
    
    Vec2 velocity;
    //if(cache.local_tdelta != 0)
        velocity = (cache.local_tdelta != 0) * (position - cache.estimated_px) / cache.local_tdelta;
    assert(!std::isnan(velocity.x) && !std::isnan(velocity.y));
    
    auto speed = velocity.length() / SLOW_SETTING(track_max_speed) * SLOW_SETTING(cm_per_pixel);
    speed = 1 / SQR(1 + speed);
    
    /*if((frameIndex >= 48181 && identity().ID() == 368) || frameIndex == 48182)
        Debug("Frame %d: Fish%d estimate:%f,%f pos:%f,%f velocity:%f,%f p:%f (raw %f)",
              frameIndex,
              identity().ID(),
              cache.estimated_px.x * cache.cm_per_pixel,
              cache.estimated_px.y * cache.cm_per_pixel,
              position.x, position.y,
              velocity.x, velocity.y,
              speed,
              length(velocity) / cache.track_max_speed);*/
    
    // additional condition, if blobs are apart more than a pixel,
    // check for their angular difference
    if(not cache.valid_frame)
        return speed;
    
    /*
         \begin{equation} \label{eq:speed_prob}
             S_{i}\given{t | B_j} = \left(1 + \frac{\norm{ \left(\mathbf{p}_{B_j}(t) - \dot{\mathbf{p}}_i(t) \right) / (\tau_i - t) }}{ D_{\mathrm{max}}}\right)^{-2}
         \end{equation}
         
         $$ \mathbf{a} = \dot{\mathbf{p}}_i(t) - \mathbf{p}_i(\tau_i)  $$
         $$ \mathbf{b} = \mathbf{p}_{B_j}(t) - \mathbf{p}_i(\tau_i) $$
         
         \begin{equation} \label{eq:angle_prob}
             A_{i}\given{t,\tau_i | B_j } =
             \begin{cases}
                 1 - \frac{1}{\pi}\left|\atantwo\left\{\norm{ \mathbf{a}\times \mathbf{b} }, \mathbf{a}\cdot \mathbf{b}\right\}\right| & \mathrm{if} \norm{\mathbf{a}} > 1 \wedge \norm{\mathbf{b}} > 1 \\
                 1 & \mathrm{otherwise}
             \end{cases}
         \end{equation}
    */
    auto line_center_last = blob_center - cache.last_seen_px;
    auto line_est_last = cache.estimated_px - cache.last_seen_px;
    
    if(line_center_last.sqlength() > 1
       && line_est_last.sqlength() > 1)
    {
        float a = adiffangle(line_center_last, line_est_last);
        assert(!std::isnan(a));
        
        a = abs(a / M_PI);
        a = 0.9 + SQR(1 - a) * 0.1;
        
        return speed * a;
    }
    
    return speed;
}

/*prob_t Individual::size_probability(const IndividualCache& cache, long_t, size_t num_pixels) const {
    if(cache.size_average <= 0)
        return 1.f;
    
    return max(0.5, 1 - 0.25 * (SQR(cmn::abs(min(2, num_pixels / cache.size_average) - 1))));
}*/

Probability Individual::probability(MaybeLabel label, const IndividualCache& cache, Frame_t frameIndex, const pv::CompressedBlob& blob) {
    auto bounds = blob.calculate_bounds();
    return probability(label, cache, frameIndex, bounds.pos() + bounds.size() * 0.5, blob.num_pixels());
}

Probability Individual::probability(MaybeLabel label, const IndividualCache& cache, Frame_t frameIndex, const pv::Blob& blob) {
    return probability(label, cache, frameIndex, blob.bounds().pos() + blob.bounds().size() * 0.5, blob.num_pixels());
}

Probability Individual::probability(MaybeLabel label, const IndividualCache& cache, Frame_t frameIndex, const Vec2& position, size_t pixels) {
    //individual_empty is set in this case
    //assert(frameIndex >= _startFrame);
    
    //if (frameIndex < _startFrame)
    //    throw U_EXCEPTION("Cannot calculate probability for a frame thats previous to all known frames.");
    //assert(!cache.individual_empty);
    //if (//cache.consistent_categories &&
    //    cache.current_category != -1)
    //{
        //auto l = Categorize::DataStore::ranged_label(Frame_t(frameIndex), blob);
        //if(identity().ID() == 38)
        //    FormatWarning("Frame ",frameIndex,": blob ",blob.blob_id()," -> ",l ? l->name.c_str() : "N/A"," (",l ? l->id : -1,") and previous is ",cache.current_category);
        if (label.has_value()
            && cache.current_category.has_value()
            && label.value() != cache.current_category.value())
        {
            //if(identity().ID() == 38)
             //   FormatWarning("Frame ", frameIndex,": current category does not match for blob ",blob.blob_id());
            //return Probability{ 0, 0, 0, 0 };
            return 0;
        }
    //}

    const Vec2& blob_pos = position;
    //auto && [ p_position, p_speed, p_angle ] = 
    auto p_position =    position_probability(cache, frameIndex, pixels, blob_pos, position);
    //thread_print("** ", frameIndex, ": ", identity().ID(), " + ", blob_pos, " => ", p_position * cache.time_probability, " t=", cache.time_probability);
    
    /**
         \begin{equation} \label{eq:combined_prob}
            P_{i} \given[\big]{t,\tau_i | B_j } =  S_{i} \given*{t | B_j} * \left(1 - \omega_1 \left(1 + A_{i} \given*{t,\tau_i | B_j } \right) \right) * \left(1 - \omega_2 \left( 1 +  T_{i}(t,\tau_i) \right) \right)
         \end{equation}
     */
    //return {
    return p_position * cache.time_probability;
    //    cache.time_probability,
    //    p_position,
    //    p_angle
    //};
}

const BasicStuff* Individual::find_frame(Frame_t frameIndex) const
{
    if(empty()) {
        return nullptr;
    }
    
    if(frameIndex <= _startFrame)
        return _basic_stuff.front().get();
    if(frameIndex >= _endFrame)
        return _basic_stuff.back().get();
    
    auto end = _tracklets.end();
    auto it = std::lower_bound(_tracklets.begin(), end, frameIndex, [](const auto& ptr, Frame_t frame){
        return ptr->start() < frame;
    });
    
    if(it == end) { // we are out of range, return last
        auto idx = _tracklets.back()->basic_stuff(frameIndex);
        if(idx != -1)
            return _basic_stuff[ idx ].get();
        else
            return _basic_stuff.back().get();
    }
    
    int32_t index = (int32_t)_basic_stuff.size()-1;
    if((*it)->start() > frameIndex) {
        if(it != _tracklets.begin()) {
            // it is either in between segments (no frame)
            // or inside the previous segment
            --it;
            
            if((*it)->contains(frameIndex)) {
                index = (*it)->basic_stuff(frameIndex);
            } else {
                index = (*it)->basic_index.back();
            }
            
        } else {
            // it is located before our first startFrame
            // this should not happen
            //index = it->second->basic_index.front();
            throw U_EXCEPTION("(",identity().ID(),") frame ",frameIndex,": cannot find basic_stuff after finding tracklet ",(*it)->start(),"-",(*it)->end(),"");
        }
        
    } else {
        if((*it)->contains(frameIndex)) {
            index = (*it)->basic_stuff(frameIndex);
        } else {
            assert((*it)->start() == frameIndex);
            index = (*it)->basic_index.front();
        }
    }
    
    assert(index >= 0 && (uint64_t)index < _basic_stuff.size());
    return _basic_stuff[ index ].get();
}

MovementInformation Individual::calculate_previous_vector(Frame_t frameIndex) const {
    const auto min_samples = Frame_t(FAST_SETTING(posture_direction_smoothing));
    MovementInformation movement;
    
    if(!centroid(frameIndex)) {
        return movement;
    }
    
    std::vector<Frame_t> all_frames;
    std::vector<Float2_t> all_angles;
    std::vector<Vec2> all_head_positions;
    
    Range<Frame_t> range(max(start_frame(), frameIndex.try_sub(min_samples)), min(end_frame(), frameIndex));
    
    for(auto i = range.start; i < range.end; ++i) {
        auto [basic, posture] = this->all_stuff(i);
        if(not basic || not posture || not posture->midline_angle.has_value())
            continue;
        
        auto pp = pp_midline(i);
        if(not pp)
            continue;
        
        auto angle = posture->midline_angle.value();
        auto oangle = pp->original_angle();
        auto L = posture->midline_length.value();
        
        all_head_positions.push_back(- Vec2{
            cos(angle),
            sin(angle)
        } * L * 0.5_F);
        
        all_angles.push_back(oangle);
        
        movement.directions.push_back(Vec2{
            cos(oangle),
            sin(oangle)
        }.normalize());
        
        all_frames.push_back(basic->frame);
    }
    
    for(auto &d : movement.directions) {
        movement.direction += d;
    }
    if(not movement.directions.empty()) {
        movement.direction /= movement.directions.size();
        movement.direction = movement.direction.normalize();
    }
    //movement.direction = previous_direction.normalize();
    //movement.direction = position_sum.normalize();
    
    return movement;
}

//void Individual::clear_training_data() {
    //_training_data.clear();
//}

#if DEBUG_ORIENTATION
OrientationProperties Individual::why_orientation(Frame_t frame) const {
    if(_why_orientation.find(frame) == _why_orientation.end())
        return OrientationProperties();
    return _why_orientation.at(frame);
}
#endif

void Individual::save_posture(const BasicStuff& basic,
                              const PoseMidlineIndexes& pose_midline_indexes,
                              Frame_t frameIndex,
                              pv::BlobPtr&& pixels)
{//Image::Ptr greyscale) {
    /*auto c = centroid(frameIndex);
    auto direction = c->v();
    direction /= ::length(direction);*/
    
    tl::expected<posture::Result, const char*> result = tl::unexpected("Unknown state, cannot calculate posture.");
    
    assert(pixels);
    if(not pixels->prediction().pose.empty()) {
        result = posture::calculate_posture(frameIndex, basic, pixels->prediction().pose, pose_midline_indexes);
    } else if(pixels->prediction().outlines.has_original_outline()) {
        result = posture::calculate_posture(frameIndex, basic, pixels->prediction().outlines);
    }
    else
        result = posture::calculate_posture(frameIndex, pixels.get());
    
    if(not result) {
#ifndef NDEBUG
        FormatWarning("Cannot compute posture for ", identity(), " in frame ", frameIndex, ": ", result.error());
#endif
        return;
    }
    
    auto &ptr = result.value();
    assert(not ptr.outline.empty());
    
    auto tracklet = tracklet_for(frameIndex);
    if(!tracklet)
        throw U_EXCEPTION("save_posture cannot find frame ",frameIndex,".");
    if(!tracklet->contains(frameIndex))
        throw U_EXCEPTION("save_posture found tracklet (",tracklet->start(),"-",tracklet->end(),"), but does not contain ",frameIndex,".");
    
    auto stuff = std::make_unique<PostureStuff>();
    stuff->frame = frameIndex;
    stuff->outline = std::move(ptr.outline);
    
    if(auto midline = std::move(ptr.midline);
       midline && !midline->empty())
    {
        stuff->cached_pp_midline = std::move(midline);
    }
    
    tracklet->add_posture_at(std::move(stuff), this);
    update_midlines(nullptr);
}

Vec2 Individual::weighted_centroid(const pv::Blob& blob, const std::vector<uchar>& pixels) {
    // calculate midline centroid
    Vec2 centroid_point(0, 0);
    float weights = 0;
    
    float minimum = FLT_MAX, maximum = -FLT_MAX;
    for(auto p : pixels) {
        if(p < minimum)
            minimum = p;
        if(p > maximum)
            maximum = p;
    }
    
    auto it = pixels.begin();
    assert(maximum >= minimum);
    
    for(auto &h : blob.hor_lines()) {
        for(short x = h.x0; x<= h.x1; x++) {
            float weight = 1.f - (float(*it)-minimum) / float(maximum - minimum + 1);
            centroid_point += Vec2(x, h.y) * weight;
            weights += weight;
            ++it;
        }
    }
    
    assert(weights > 0);
    return centroid_point / weights;
}

bool Individual::evaluate_fitness() const {
	if(frame_count() <= 25)
		return false;
		
	// check posture
    if(FAST_SETTING(calculate_posture)) {
		for(auto i = start_frame(); i < end_frame(); ++i) {
            const auto m = midline(i);
			if(m && m->size() != FAST_SETTING(midline_resolution))
				return false;
		}
	}
	
	return true;
}

/*void Individual::recognition_segment(Frame_t frame, const std::tuple<size_t, std::map<long_t, float>>& map) {
    average_recognition_tracklet[frame] = map;
}*/

void Individual::clear_recognition() {
    average_recognition_tracklet.clear();
    average_processed_tracklet.clear();
}

void log(FILE* f, const char* cmd, ...) {
    UNUSED(f);
    UNUSED(cmd);
#if !defined(NDEBUG) && false
    if(!f) return;
    
    std::string output;
    
    va_list args;
    va_start(args, cmd);
    
    DEBUG::ParseFormatString(output, cmd, args);
    va_end(args);
    
    output += "\n";
    fwrite(output.c_str(), sizeof(char), output.length(), f);
#endif
}

std::map<Frame_t, FrameRange> split_tracklet_by_probability(const Individual* fish, const TrackletInformation& segment)
{
    auto for_frame = [fish](Frame_t frame) -> std::tuple<long_t, float> {
        std::map<long_t, std::tuple<long_t, float>> samples;
        
        auto blob = fish->compressed_blob(frame);
        if(!blob)
            return {-1.f, 0.f};
        
        float max_id = -1;
        float max_p = 0;
        
        auto pred = Tracker::instance()->find_prediction(frame, blob->blob_id());
        if(pred) {
            auto map = track::prediction2map(*pred);
            for (auto && [fdx, p] : map) {
                if(p > max_p) {
                    max_p = p;
                    max_id = fdx.get();
                }
            }
        }
        
        return {max_id, max_p};
    };
    
    auto median_of = [](const std::deque<long_t>& ids, const std::deque<Frame_t>& frames) -> std::tuple<long_t, Frame_t, std::map<long_t, Frame_t>>
    {
        if(ids.empty())
            return { -1, {}, {} };
        
        std::map<long_t, Frame_t> first_frame, last_frame;
        Median<long_t> median;
        auto fit = frames.begin();
        for(auto it=ids.begin(); it != ids.end(); ++it, ++fit) {
            if(first_frame.find(*it) == first_frame.end())
                first_frame[*it] = *fit;
            last_frame[*it] = *fit;
            median.addNumber(*it);
        }
        
        long_t id = median.getValue();
        return { id, first_frame.at(id), last_frame };
    };
    
    std::deque<long_t> median_ids;
    std::deque<Frame_t> median_frame;
    std::deque<float> median_ps;
    
    Range<Frame_t> current_range(segment.start(), segment.start());
    long_t current_id = -1;
    
    current_range.end.invalidate();
    
    std::map<Frame_t, FrameRange> result;
    std::map<Frame_t, long_t> assigned_ids;
    std::vector<std::tuple<Range<Frame_t>, long_t>> debug_ids;
    
    const size_t N = SLOW_SETTING(frame_rate) * 2u;
    const Frame_t min_samples { Frame_t(Frame_t::number_t(SLOW_SETTING(frame_rate))) };
    
    for(auto i = segment.start(); i < segment.end(); ++i) {
        auto && [id, p] = for_frame(i);
        if(id != -1) {
            median_ids.push_back(id);
            median_ps.push_back(p);
            median_frame.push_back(i);
            
            if(median_ids.size() > N) median_ids.pop_front();
            if(median_ps.size() > N) median_ps.pop_front();
            if(median_frame.size() > N) median_frame.pop_front();
        }
        
        if(median_ids.size() < N)
            continue;
        
        auto && [median, frame, last_frame] = median_of(median_ids, median_frame);
            
        if( /*(id == -1 && current_id != -1) ||*/  median == -1 // no id present (NaN)
           //|| current_range.end < i - 1 // more than one frame apart
           || (i > segment.start() && median != current_id) // median id has changed (and its not the first frame)
           )
        {
            if(current_id != -1) {
                frame = max((last_frame.find(current_id) != last_frame.end() ? last_frame.at(current_id) : current_range.end) + 1_f, frame - 1_f);
                assert(frame - 1_f >= segment.start());
                current_range.end = min(frame - 1_f, current_range.end); // current_range.end might be greater than frame-1, but we arent interested in where the median changed. we want to know where the new median value first occurred
                if(current_range.end >= current_range.start && current_range.length() >= min_samples) {
                    if(!assigned_ids.empty() && assigned_ids.rbegin()->second == current_id) {
                        result.rbegin()->second.range.end = current_range.end;
                    } else {
                        result[current_range.start] = FrameRange(current_range, current_range.start == segment.start() ? segment.first_usable : Frame_t());
                        assigned_ids[current_range.start] = current_id;
                        debug_ids.push_back({current_range, current_id});
                    }
                } else
                    frame = current_range.start;
            }
            
            current_id = median;
            current_range.start = frame;
            current_range.end = i;
            
        } else if(median == current_id) {
            current_range.end = i;
        }
    }
    
    if(current_id != -1 && current_range.length() >= min_samples) {
        if(!assigned_ids.empty() && assigned_ids.rbegin()->second == current_id) {
            result.rbegin()->second.range.end = current_range.end;
        } else {
            result[current_range.start] = FrameRange(current_range, current_range.start == segment.start() ? segment.first_usable : Frame_t());
            debug_ids.push_back({current_range, current_id});
        }
    }
    
    if(!result.empty() && result.rbegin()->second.range.end < segment.end()) //&& result.rbegin()->second.range.end >= segment.end() - (long_t)N)
        result.rbegin()->second.range.end = segment.end();
    if(!result.empty() && result.begin()->second.range.start > segment.start()) {
        auto range = result.begin()->second;
        result.erase(result.begin());
        result[segment.start()] = FrameRange(Range<Frame_t>(segment.start(), range.end()), segment.first_usable);
    }
        
    
    if(result.size() == 1) {
        return {};
    } else {
        return result;
    }
}

void Individual::calculate_average_tracklet_id() {
    _average_recognition_samples = 0;
    _average_recognition.clear();
    
    std::map<Idx_t, size_t> samples;
    const Frame_t frame_limit(SLOW_SETTING(frame_rate) * 2u);
    
    for(auto & tracklet : _tracklets) {
        auto && [n, vector] = average_recognition(tracklet->start());
        _average_recognition_samples += n;
        
        for(auto && [fdx, p] : vector) {
            _average_recognition[fdx] += p * n;
            samples[fdx] += n;
        }
    }
    
    std::map<FrameRange, std::set<long_t>> splits;
/*#ifndef NDEBUG
    std::string file = identity().name()+".log";
    FILE* f = fopen(file.c_str(), "wb");
    if(!f)
       FormatError("Cannot open file ", file," for writing ('",identity(),"')");
#else*/
    FILE* f = nullptr;
//#endif
    
    // anything thats below 2 seconds + at least 10% more with a different id, is considered lame and unimportant
    std::map<Frame_t, FrameRange> processed_tracklets;
    
    for(auto & tracklet : _tracklets) {
        if(tracklet->length() < frame_limit) {
            processed_tracklets[tracklet->start()] = *tracklet;
            log(f, "Tracklet ",tracklet->start(), "-", tracklet->end()," shorter than ", frame_limit);
            continue;
        }
        
        log(f, "Checking tracklet ",tracklet->start(), "-", tracklet->end()," (L=", tracklet->length(), ")");
        auto split_up = split_tracklet_by_probability(this, *tracklet);
        
        if(split_up.empty()) {
            processed_tracklets[tracklet->start()] = *tracklet;
        } else {
            assert(split_up.begin()->second.start() == tracklet->start());
            assert(split_up.rbegin()->second.end() == tracklet->end());
            
            log(f, "Split tracklet ", *tracklet, " into ", split_up);
            
            auto prev_end = tracklet->start() - 1_f;
            for(auto && [start, range] : split_up) {
                assert(start == range.start() && ((not prev_end.valid() && start == start_frame()) || (prev_end.valid() && prev_end + 1_f == start)));
                prev_end = range.end();
                processed_tracklets[start] = range;
                
                FOI::add(FOI(start, {FOI::fdx_t(identity().ID())}, "split_up"));
            }
        }
    }
    
    if(f)
        fclose(f);
    
    if(!splits.empty()) {
        auto str = Meta::toStr(splits);
        FormatWarning("Found tracklets for fish ", identity().ID()," that have to be split:\n",str);
    }
    
    _recognition_tracklets = processed_tracklets;
    
    for(auto && [fdx, n] : samples) {
        if(n > 0)
            _average_recognition[fdx] /= float(n);
    }
}

std::tuple<bool, FrameRange> Individual::frame_has_segment_recognition(Frame_t frameIndex) const {
    if(not frameIndex.valid() || empty() || frameIndex > _endFrame || frameIndex < _startFrame)
        return {false, FrameRange()};
    
    auto range = get_tracklet(frameIndex);
    auto & tracklet = range.range;
    return { tracklet.contains(frameIndex) && average_recognition_tracklet.find(tracklet.start) != average_recognition_tracklet.end(), range };
}

std::tuple<bool, FrameRange> Individual::has_processed_tracklet(Frame_t frameIndex) const {
    if(not frameIndex.valid() || empty() || frameIndex > _endFrame || frameIndex < _startFrame)
        return {false, FrameRange()};
    
    auto range = get_recognition_segment(frameIndex);
    auto & tracklet = range.range;
    return { tracklet.contains(frameIndex) && average_processed_tracklet.find(tracklet.start) != average_processed_tracklet.end(), range };
}

/*const decltype(Individual::average_recognition_tracklet)::mapped_type& Individual::average_recognition(long_t segment_start) const {
    return average_recognition_tracklet.at(segment_start);
}*/

const decltype(Individual::average_recognition_tracklet)::mapped_type Individual::processed_recognition(Frame_t segment_start) {
    auto it = average_processed_tracklet.find(segment_start);
    if(it == average_processed_tracklet.end()) {
        //! acquire write access
        LockGuard guard(w_t{}, "average_recognition_tracklet");
        
        // average cannot be found for given segment. try to calculate it...
        auto sit = _recognition_tracklets.find(segment_start);
        if(sit == _recognition_tracklets.end())
            throw U_EXCEPTION("Cannot find tracklet starting at ",segment_start," for fish ",identity().raw_name(),".");
        
        const auto &[ segment, usable] = sit->second;
        
        if(segment.end >= _endFrame && Tracker::end_frame() != Tracker::analysis_range().end()) {
            return {0, {}};
        }
        
        std::map<Idx_t, std::tuple<long_t, float>> samples;
        size_t overall = 0;

        for (auto i = segment.start; i < segment.end; ++i) {
            auto blob = this->blob(i);
            if (!blob)
                continue;

            auto pred = Tracker::instance()->find_prediction(i, blob->blob_id());
            if(pred) {
                auto map = track::prediction2map(*pred);
                ++overall;

                for (auto&& [fdx, p] : map) {
                    ++std::get<0>(samples[Idx_t(fdx)]);
                    std::get<1>(samples[Idx_t(fdx)]) += p;
                }
            }
        }
        
        if(overall > 0) {
            std::map<Idx_t, float> average;
            float s = 0;
            for (auto && [key, value] : samples) {
                float n = std::get<0>(value);
                average[key] = n > 0 ? std::get<1>(value) / n : 0;
                s += average[key];
            }
            
            if(s > 0.001)
                average_processed_tracklet[segment_start] = {overall, average};
            else {
                Print("Not using fish ",identity().ID()," tracklet ",segment_start,"-",segment.end," because sum is ",s);
                return {0, {}};
            }
        } else
            return {0, {}};
    }
    
    return average_processed_tracklet.at(segment_start);
}

const decltype(Individual::average_recognition_tracklet)::mapped_type Individual::average_recognition(Frame_t segment_start) {
    auto it = average_recognition_tracklet.find(segment_start);
    if(it == average_recognition_tracklet.end()) {
        // average cannot be found for given segment. try to calculate it...
        auto sit = std::upper_bound(_tracklets.begin(), _tracklets.end(), segment_start, [](Frame_t frame, const auto& ptr) {
            return frame < ptr->start();
        });
        if((sit == _tracklets.end() || sit != _tracklets.begin()) && (*(--sit))->start() == segment_start)
        {
            //! found the segment
        } else
            throw U_EXCEPTION("Cannot find tracklet starting at ",segment_start," for fish ",identity().raw_name(),".");
        
        const auto && [segment, usable] = (FrameRange)*sit->get();
        
        if(segment.end >= _endFrame && Tracker::end_frame() + 1_f != Frame_t(narrow_cast<Frame_t::number_t>(SETTING(video_length).value<uint64_t>()))) {
            return {0, {}};
        }
        
        std::map<Idx_t, std::tuple<long_t, float>> samples;
        size_t overall = 0;
        
        for(auto i = segment.start; i < segment.end; ++i) {
            auto blob = this->blob(i);
            if(!blob)
                continue;
            
            auto pred = Tracker::instance()->find_prediction(i, blob->blob_id());
            if(pred) {
                auto map = track::prediction2map(*pred);
                ++overall;
                
                for (auto && [fdx, p] : map) {
                    ++std::get<0>(samples[Idx_t(fdx)]);
                    std::get<1>(samples[Idx_t(fdx)]) += p;
                }
            }
        }
        
        if(overall > 0) {
            std::map<Idx_t, float> average;
            float s = 0;
            for (auto && [key, value] : samples) {
                float n = std::get<0>(value);
                average[key] = n > 0 ? std::get<1>(value) / n : 0;
                s += average[key];
            }
            
            if(s > 0.001)
                average_recognition_tracklet[segment_start] = {overall, average};
            else {
                Print("Not using fish ",identity().ID()," tracklet ",segment_start,"-",segment.end," because sum is ",s);
                return {0, {}};
            }
        } else
            return {0, {}};
    }
    
    return average_recognition_tracklet.at(segment_start);
}

std::tuple<size_t, Idx_t, float> Individual::average_recognition_identity(Frame_t segment_start) const {
    auto it = average_recognition_tracklet.find(segment_start);
    if(it == average_recognition_tracklet.end()) {
        return {0, Idx_t(), 0};
    }
    
    Idx_t mdx;
    float mdx_p = 0;
    
    for(auto && [fdx, p] : std::get<1>(it->second)) {
        if(!mdx.valid() || p > mdx_p) {
            mdx_p = p;
            mdx = fdx;
        }
    }
    
    return {std::get<0>(it->second), mdx, mdx_p};
}

void Individual::add_custom_data(Frame_t frame, long_t id, void* ptr, std::function<void(void*)> fn_delete) {
    LockGuard guard(w_t{}, "add_custom_data");
    auto it = _custom_data[frame].find(id);
    if(it != _custom_data[frame].end()) {
        FormatWarning("Custom data with id ", id," already present in frame ",frame,".");
        it->second.second(it->second.first);
    }
    _custom_data[frame][id] = { ptr, fn_delete };
}

void * Individual::custom_data(Frame_t frame, long_t id) const {
    LockGuard guard(ro_t{}, "custom_data");
    auto it = _custom_data.find(frame);
    if(it == _custom_data.end())
        return NULL;
    
    auto it1 = it->second.find(id);
    if(it1 != it->second.end()) {
        return it1->second.first;
    }
    
    return NULL;
}

void Individual::save_visual_field(const file::Path& path, Range<Frame_t> range, const std::function<void(float, const std::string&)>& update, bool blocking) const {
    if(range.empty())
        range = Range<Frame_t>(_startFrame, _endFrame);
    
    
    std::vector<float> depth, body_part;
    std::vector<long_t> ids;
    std::vector<Vec2> fish_pos, eye_pos;
    std::vector<float> fish_angle, eye_angle;
    std::vector<Vec2> visible_points;
    std::vector<Frame_t::number_t> frames;
    
    size_t len = 0;

    iterate_frames(range, [&](Frame_t, const std::shared_ptr<TrackletInformation>&, auto, auto posture) -> bool
    {
        if (!posture || !posture->head)
            return true;
        ++len;
        return true;
    });

    Print("Saving to ",path," (",len," frames in range ",range.start,"-",range.end,")");

    size_t vres = VisualField::field_resolution * VisualField::layers;
    size_t eye_len = len * vres;
    
    frames.reserve(len);
    depth.reserve(eye_len * 2);
    body_part.reserve(depth.size());
    ids.reserve(depth.size());
    
    fish_pos.reserve(len);
    fish_angle.reserve(len);
    visible_points.reserve(eye_len * 2);
    
    eye_angle.reserve(len * 2);
    eye_pos.reserve(len * 2);
    
    std::shared_ptr<LockGuard> guard;
    
    iterate_frames(range, [&](Frame_t frame, const std::shared_ptr<TrackletInformation> &, auto basic, auto posture) -> bool
    {
        if(blocking)
            guard = std::make_shared<LockGuard>(ro_t{}, "new VisualField");
        if(!posture || !posture->head)
            return true;
        
        bool owned = false;
        VisualField* ptr = (VisualField*)custom_data(frame, VisualField::custom_id);
        if(!ptr && basic) {
            ptr = new VisualField(identity().ID(), frame, *basic, posture, false);
            owned = true;
        }
        
        if(ptr) {
            assert(ptr->eyes().size() == 2);

            frames.push_back(frame.get());
            
            fish_pos.emplace_back(ptr->fish_pos());
            fish_angle.push_back(ptr->fish_angle());
            
            for(long_t j=0; j<2; j++) {
                auto &e = ptr->eyes()[j];
                
                eye_angle.push_back(e.angle);
                eye_pos.emplace_back(e.pos);
                
                visible_points.insert(visible_points.end(), e._visible_points.begin(), e._visible_points.begin() + vres);
                depth.insert(depth.end(), e._depth.begin(), e._depth.begin() + vres);
                body_part.insert(body_part.end(), e._visible_head_distance.begin(), e._visible_head_distance.begin() + vres);
                ids.insert(ids.end(), e._visible_ids.begin(), e._visible_ids.begin() + vres);
            }
            
            if(owned)
                delete ptr;
        }
        
        if(frame.get() % 1000 == 0) {
            update((frame - range.start).get() / (float)(range.end - range.start).get() * 0.5, "");
            Print(frame," / ",range.end);
        }
        
        return true;
    });
    
    Print("Saving depth...");
    FileSize fileSize(depth.size() * sizeof(decltype(depth)::value_type)
                      + visible_points.size() * sizeof(decltype(visible_points)::value_type)
                      + ids.size() * sizeof(decltype(ids)::value_type)
                      + body_part.size() * sizeof(decltype(body_part)::value_type));
    update(0.5, "writing files ("+Meta::toStr(fileSize)+")");

    bool use_npz = fileSize.bytes < 1.5 * 1000 * 1000 * 1000;
    if (!use_npz) {
        Timer save_timer;
        cnpy::npy_save(path.str() + "_depth.npy", depth.data(), {
            len,
            2u,
            VisualField::layers,
            vres / VisualField::layers
        });

        FileSize per_second(double(depth.size() * sizeof(decltype(depth)::value_type)) / save_timer.elapsed());
        auto str = Meta::toStr(per_second) + "/s";
        Print("saved depth @ ", str.c_str());

        update(1 / 3. * 0.5 + 0.5, "writing files (" + Meta::toStr(fileSize) + ") @ ~" + str);

        cnpy::npy_save(path.str() + "_ids.npy", ids.data(), {
            len,
            2,
            VisualField::layers,
            vres / VisualField::layers
        });

        update(2 / 3. * 0.5 + 0.5, "");
        cnpy::npy_save(path.str() + "_body_part.npy", body_part.data(), {
            len,
            2,
            VisualField::layers,
            vres / VisualField::layers
        });
    }
    else {
        Timer save_timer;
        cnpy::npz_save(path.str() + ".npz", "depth", depth.data(), {
            len,
            2,
            VisualField::layers,
            vres / VisualField::layers
        });
        
        cnpy::npz_save(path.str() + ".npz", "visible_points", (const Float2_t*)visible_points.data(), {
            len,
            2,
            VisualField::layers,
            vres / VisualField::layers,
            2
        }, "a");

        FileSize per_second(double(depth.size() * sizeof(decltype(depth)::value_type)) / save_timer.elapsed());
        auto str = Meta::toStr(per_second) + "/s";
        Print("saved depth @ ", str.c_str());

        update(1 / 3. * 0.5 + 0.5, "writing files (" + Meta::toStr(fileSize) + ") @ ~" + str);

        cnpy::npz_save(path.str() + ".npz", "ids", ids.data(), {
            len,
            2,
            VisualField::layers,
            vres / VisualField::layers
        }, "a");

        update(2 / 3. * 0.5 + 0.5, "");
        cnpy::npz_save(path.str() + ".npz", "body_part", body_part.data(), {
            len,
            2,
            VisualField::layers,
            vres / VisualField::layers
        }, "a");
    }
    
    update(2.8/3. * 0.5 + 0.5, "");
    std::vector<int> colors;
    IndividualManager::transform_all([&colors](auto fdx, auto fish) {
        colors.push_back(fdx.get());
        colors.push_back(fish->identity().color().r);
        colors.push_back(fish->identity().color().g);
        colors.push_back(fish->identity().color().b);
    });
    
    try {
        file::Path meta_path = use_npz ? (path.str() + ".npz") : (path.str() + "_meta.npz");

        cmn::npz_save(meta_path.str(), "colors", colors.data(), {
            colors.size() / 4, 4
        }, use_npz ? "a" : "w");
        
        cmn::npz_save(meta_path.str(), "fov_range", std::vector<double>{-VisualField::symmetric_fov, VisualField::symmetric_fov}, "a");
        cmn::npz_save(meta_path.str(), "frame_range", std::vector<long_t>{(long_t)range.start.get(), (long_t)range.end.get()}, "a");
        
        assert(fish_pos.size() == len);
        cmn::npz_save(meta_path.str(), "fish_pos", (const Float2_t*)fish_pos.data(), {len, 2}, "a");
        
        assert(fish_angle.size() == len);
        cmn::npz_save(meta_path.str(), "fish_angle", fish_angle.data(), {len}, "a");
        
        assert(eye_pos.size() == 2 * len);
        cmn::npz_save(meta_path.str(), "eye_pos", (const Float2_t*)eye_pos.data(), {len, 2, 2}, "a");
        
        assert(eye_angle.size() == 2 * len);
        cmn::npz_save(meta_path.str(), "eye_angle", eye_angle.data(), { len, 2 }, "a");

        assert(frames.size() == len);
        cmn::npz_save(meta_path.str(), "frames", frames.data(), { len }, "a");
        
        if(!use_npz)
            Print("Saved visual field metadata to ",meta_path.str()," and image data to ",path.str()+"_*.npy",".");
        else
            Print("Saved to ",path.str()+".npz",".");

    } catch(...) {
        // there will be a utils exception, so its printed out already
    }
}

std::string Individual::toStr() const {
    //std::stringstream ss;
    //ss << "Individual<" << _identity.ID() << " frames:" << _centroid.size() << " pos:" << head(_endFrame)->pos() << ">";
    return _identity.name();
}

bool Individual::empty() const noexcept {
    return not _startFrame.valid();
}
