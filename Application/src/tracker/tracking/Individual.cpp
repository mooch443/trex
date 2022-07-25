#include "Individual.h"
#include <misc/GlobalSettings.h>
#include <misc/Timer.h>
#include <tracking/Tracker.h>
#include <tracking/DebugDrawing.h>
#include <algorithm>
#include <misc/ReverseAdapter.h>
#include <tracking/Recognition.h>
#include <cnpy/cnpy.h>
#include <tracking/VisualField.h>
#include <misc/pretty.h>
#include <processing/PadImage.h>
#include <misc/cnpy_wrapper.h>
#include <misc/CircularGraph.h>
#include <misc/metastring.h>
#include <misc/SoftException.h>
#include <gui/Graph.h>
#include <tracking/Categorize.h>

#define NAN_SAFE_NORMALIZE(X, Y) { \
    const auto L = X.length(); \
    if(L) Y = X / L; \
    else  Y = Vec2(0); \
}

using namespace track;
using prob_t = track::Match::prob_t;

std::atomic<uint32_t> RUNNING_ID(0);

void Identity::set_running_id(Idx_t value) { RUNNING_ID = value; }
Idx_t Identity::running_id() { return Idx_t(RUNNING_ID.load()); }

Identity::Identity(Idx_t myID)
    : _color(myID.valid() ? ColorWheel(myID).next() : ColorWheel(RUNNING_ID).next()), _myID(myID.valid() ? myID : Idx_t(RUNNING_ID++)), _name(Meta::toStr(_myID))
{
    if(myID.valid() && RUNNING_ID < myID) {
        RUNNING_ID = Idx_t(myID + 1);
    }
}

const std::string& Identity::raw_name() {
    auto names = Settings::get<Settings::individual_names>();
    auto it = names->find(_myID);
    if(it != names->end()) {
        _name = it->second;
    }
    return _name;
}

std::string Identity::name() const {
    {
        auto names = Settings::get<Settings::individual_names>();
        auto it = names->find(_myID);
        if(it != names->end()) {
            return it->second;
        }
    }
    return FAST_SETTINGS(individual_prefix) + raw_name();
}

std::string Identity::raw_name() const {
    auto names = Settings::get<Settings::individual_names>();
    auto it = names->find(_myID);
    if(it != names->end()) {
        return it->second;
    }
    return _name;
}

#if !COMMONS_NO_PYTHON
struct Predictions {
    Frame_t  _segment_start;
    Idx_t individual;
    std::vector<Frame_t> _frames;
    std::vector<int64_t> _ids;
    int64_t best_id;
    float p;
};

struct RecTask {
    inline static constexpr auto tagwork = "pretrained_tagwork";
    inline static std::atomic_bool _terminate = false;
    inline static std::mutex _mutex;
    inline static std::condition_variable _variable;
    inline static std::vector<RecTask> _queue;

    Frame_t _segment_start;
    Idx_t individual;
    std::vector<Frame_t> _frames;
    std::vector<Image::UPtr> _images;

    std::function<void(Predictions&&)> _callback;
    bool _optional;
    Idx_t _fdx;
    inline static Idx_t _current_fdx;

    static void thread() {
        static std::atomic<size_t> counted{ 0 };

        set_thread_name("RecTask::update_thread");
        print("RecTask::update_thread begun");

        RecTask::init();

        std::unique_lock guard(_mutex);
        while(!_terminate || !_queue.empty()) {
            while(!_queue.empty()) {
                auto task = std::move(_queue.back());
                _queue.erase(--_queue.end());
                //if(!_queue.empty())
                
                if(!_queue.empty() && _queue.size() % 10 == 0 && _terminate) {
                    //print("task ", counted.load(), " -> ", _queue.size(), " tasks left (frame: ", task._frames.back(), ")");
                    
                    /*std::unordered_set<std::tuple<Idx_t, Frame_t>> segments;
                    std::map<Idx_t, std::vector<Frame_t>> histo;
                    for(auto &t : _queue) {
                        histo[t._fdx].push_back(t._segment_start);
                        if(segments.contains({t._fdx, t._segment_start})) {
                            //print("\talready contains ", t._fdx, " and ", t._segment_start, " (", t._frames.size(), ").");
                        } else
                            segments.insert({t._fdx, t._segment_start});
                    }
                    
                    print("\t-> ",histo);*/
                }
                
                _current_fdx = task._fdx;
                
                //print("[task] individual:", task._fdx, " segment:", task._segment_start, " _queue:", _queue.size());

                guard.unlock();
                RecTask::update(std::move(task));
                guard.lock();

                ++counted;
                _current_fdx = Idx_t();
            }

            _variable.wait_for(guard, std::chrono::milliseconds(1));
        }

        print("RecTask::update_thread ended");
    }

    static void init();
    static bool add(RecTask&& task, const std::function<void(RecTask&)>& fill, const std::function<void()>& callback) {
        std::unique_lock guard(_mutex);
        static std::once_flag flag;

        std::call_once(flag, []() {
            _update_thread = std::make_unique<std::thread>(RecTask::thread);
        });
        
        if(callback)
            callback();

        for(auto it = _queue.begin(); it != _queue.end(); ) {
            if(it->_fdx != task._fdx
               || it->_segment_start != task._segment_start)
            {
                ++it;
                continue;
            }
            
            _queue.erase(it);
            
            fill(task);
            //if(task._images.size() < 5)
            //    return false;
            
            //print("[fill] individual:", task._fdx, " segment:", task._segment_start, " size:", task._images.size());
            _queue.emplace_back(std::move(task));
            
            return true;
        }
        
        //if(task._optional && _queue.size() >= 50 && !_terminate)
        //    return false;
        
        fill(task);
        
        //print("[fill'] individual:", task._fdx, " segment:", task._segment_start, " size:", task._images.size());
        if(task._images.size() < 5)
            return false;
        
        _queue.emplace_back(std::move(task));
        _variable.notify_one();

        return true;
    }

    static void update(RecTask&& task) {
        Predictions result{
            ._segment_start = task._segment_start,
            .individual = task.individual,
            ._frames = std::move(task._frames)
        };

        auto receive = [&](std::vector<float> values) {
            //print("received ", values.size(), " ids for ", task._images.size(), " images.");
            result._ids.assign(values.begin(), values.end());
            //print(result._ids);

            std::unordered_map<int, int> _best_id;
            for(auto i : result._ids)
                _best_id[i]++;

            int maximum = -1;
            int max_key = -1;
            int N = result._ids.size();
            for(auto& [k, v] : _best_id) {
                if(v > maximum) {
                    maximum = v;
                    max_key = k;
                }
            }

            result.best_id = max_key;
            result.p = float(maximum) / float(N);
            //print("\t",result._segment_start,": individual ", result.individual, " is ", max_key, " with p:", result.p, " (", task._images.size(), " samples)");

            static const bool tags_save_predictions = SETTING(tags_save_predictions).value<bool>();
            if(tags_save_predictions) {
                static std::atomic<int64_t> saved_index{ 0 };
                static const auto filename = (std::string)SETTING(filename).value<file::Path>().filename();

                //if(result.p <= 0.7)
                {
                    file::Path output = pv::DataLocation::parse("output", "tags_"+filename) / Meta::toStr(max_key);
                    if(!output.exists())
                        output.create_folder();
                    
                    auto prefix = Meta::toStr(result.individual) + "." + Meta::toStr(result._segment_start);
                    if(!(output / prefix).exists())
                        (output / prefix).create_folder();
                    
                    auto files = (output / prefix).find_files();
                    
                    // delete files that already existed for this individual AND segment
                    for(auto &f : files) {
                        if(utils::beginsWith((std::string)f.filename(), prefix))
                            f.delete_file();
                    }
                    
                    // save example image
                    if(!task._images.empty())
                        cv::imwrite((output / prefix).str() + ".png", task._images.front()->get());
                    
                    output = output / prefix / (Meta::toStr(saved_index.load()) + ".");

                    print("\t\t-> exporting ", task._images.size()," guesses to ", output);
                    
                    for (size_t i=0; i<task._images.size(); ++i) {
                        cv::imwrite(output.str() + Meta::toStr(i) + ".png", task._images[i]->get());
                    }
                }
                
                ++saved_index;
            }
        };

        auto apply = [&]() -> bool {
            PythonIntegration::set_variable("tag_images", task._images, tagwork);
            PythonIntegration::set_function("receive", receive, tagwork);
            PythonIntegration::run(tagwork, "predict");
            PythonIntegration::set_function("receive", [](std::vector<float> v) {
                FormatError("Illegal call. ", v.size());
                }, tagwork);

            return true;
        };

        auto res = PythonIntegration::async_python_function(std::move(apply)).get();
        if(!res) {
            FormatError("There was an error during apply for ", result.individual);
        }
        //else
        //    print("Predicted values for ", result.individual, " result: ", result._ids.size());

        task._callback(std::move(result));
    }

    inline static std::unique_ptr<std::thread> _update_thread;

    static void remove(Idx_t fdx) {
        std::unique_lock guard(_mutex);
        for(auto it = _queue.begin(); it != _queue.end(); ) {
            if(it->_fdx == fdx) {
                it = _queue.erase(it);
            } else
                ++it;
        }

        while(_current_fdx.valid() && fdx == _current_fdx) {
            // we are currently processing an individual
            _variable.wait_for(guard, std::chrono::milliseconds(1));
        }
    }

    static void deinit() {
        if(_terminate)
            return;

        _terminate = true;
        _variable.notify_all();

        if(_update_thread) {
            _update_thread->join();
            _update_thread = nullptr;
        }
    }
};

void Individual::shutdown() {
    RecTask::deinit();
}

void RecTask::init() {
    Recognition::fix_python(true);

    PythonIntegration::ensure_started().get();
    //Recognition::check_learning_module(true);
    PythonIntegration::async_python_function([]()->bool {return true; });
    auto res = PythonIntegration::async_python_function([&]() -> bool {
        try {
            PythonIntegration::import_module(tagwork);
            PythonIntegration::set_variable("width", 32, tagwork);
            PythonIntegration::set_variable("height", 32, tagwork);
            PythonIntegration::run(tagwork, "init");
            print("tagging initialized.");
            return true;
        } catch(...) {
            return false;
        }
        }).get();

        if(res) {
            print("Initialized tagging successfully.");
        } else
            FormatError("Error during tagging initialization.");
}

Individual::IDaverage Individual::qrcode_at(Frame_t segment_start) const {
    std::unique_lock guard(_qrcode_mutex);
    auto it = _qrcode_identities.find(segment_start);
    if(it != _qrcode_identities.end()) {
        return it->second;
    }

    return { -1, -1, 0 };
}

ska::bytell_hash_map<Frame_t, Individual::IDaverage> Individual::qrcodes() const {
    std::unique_lock guard(_qrcode_mutex);
    return _qrcode_identities;
}
#endif

#if !COMMONS_NO_PYTHON
bool Individual::add_qrcode(Frame_t frame, pv::BlobPtr&& tag) {
    auto seg = segment_for(frame);
    if (!seg) {
        //FormatWarning("Cannot add tag to ", _identity, " because no segment is found for frame ", frame);
        return false;
    }

    auto pos = tag->bounds().pos() + tag->bounds().size() * 0.5;
    auto idx = seg->basic_stuff(frame);
    assert(idx != -1);
    auto& basic = basic_stuff()[(size_t)idx];
    auto my_bounds = basic->blob.calculate_bounds();

    if (my_bounds.contains(pos)) {
        //print("adding tag at ", tag->bounds(), " to individual ", _identity, " at ", my_bounds, " for segment ",*seg);
        auto &&[pos, image] = tag->image(nullptr, Bounds(-1, -1, -1, -1), 0);
        if (image->cols != 32 || image->rows != 32)
            FormatWarning("Image dimensions are wrong ", image->bounds());
        else
            _qrcodes[seg->start()].emplace_back( QRCode { frame, std::move(tag) } );
        //tf::imshow(_identity.name(), image->get());
        
        auto check_qrcode = [this](Frame_t frame, const std::shared_ptr<SegmentInformation>& segment)
        {
            static const bool tags_recognize = SETTING(tags_recognize).value<bool>();
            
            const bool segment_ended = segment && segment->start() != _last_requested_segment;
            if(segment_ended) {
                //print("individual:",identity().ID(), " segment:",segment->start()," ended before ", frame);
                _last_requested_qrcode.invalidate();
                _last_requested_segment = segment->start();
            }

        #if !COMMONS_NO_PYTHON
            //! do we need to start a new segment?
            if (tags_recognize && !_qrcodes.empty() && !_frame_segments.empty()) {
                auto segment = _frame_segments.back();
                
                if(segment_ended // either the segment ended
                    || !_last_requested_qrcode.valid() // or we have not requested a code yet
                    || _last_requested_qrcode + Frame_t(5.f * (float)FAST_SETTINGS(frame_rate)) < frame // or the last time has been at least a second ago
                   )
                {
                    auto it = _qrcodes.find(segment->start());
                    if(it != _qrcodes.end()) {
                        //print("[update] at ", frame," ", segment ? segment->range : Range<Frame_t>(), " individual:", identity().ID(), " with ended:", segment_ended, " lastqrcodepred:", _last_predicted_id, " lastqrframe:",_last_requested_qrcode, " images:", it->second.size());
                        
                        if(it->second.size() > 2 || segment_ended) {
                            RecTask task{
                                ._segment_start = segment->start(),
                                .individual = identity().ID(),
                                ._optional = !segment_ended,
                                ._fdx = identity().ID()
                            };

                            task._callback = [this, range = segment->range, N = it->second.size()](Predictions&& prediction) {
                                //print("got callback in ", _identity.ID(), " (", prediction.individual, ")");
                                
                                //print("\t",range, " individual ", identity().ID(), " has ", N, " images. ended=", segment_ended, " got callback with pred=", prediction.best_id);

                                std::unique_lock guard(_qrcode_mutex);
                                _qrcode_identities[prediction._segment_start] = { prediction.best_id, prediction.p, (uint32_t)prediction._frames.size() };
                                _last_predicted_id = prediction.best_id;
                                _last_predicted_frame = prediction._segment_start;
                            };

                            auto fill = [it, ID = identity(), segment](RecTask& task)
                            {
                                size_t step = it->second.size() / 100;
                                size_t i = 0;

                                for(auto& [frame, blob] : it->second) {
                                  if(step > 0 && i++ % step != 0) {
                                      continue;
                                  }
                                  auto ptr = std::get<1>(blob->image(nullptr, Bounds(-1, -1, -1, -1), 0));
                                  //tf::imshow("push", ptr->get());
                                  task._frames.push_back(frame);
                                  task._images.push_back(std::move(ptr));
                                }

                                //if(it->second.size() > 1)
                                //  print("sampling from ", it->second.size(), " to ",task._images.size(), " images of individual ", ID," at frame ", frameIndex," which started at ", segment->start(),".");
                            };
                            
                            auto callback = [&]() {
                                static const bool tags_save_predictions = SETTING(tags_save_predictions).value<bool>();
                                if(!tags_save_predictions)
                                    return;
                                
                                std::unique_lock guard(_qrcode_mutex);
                                if(!_last_predicted_frame.valid()
                                   || _last_predicted_frame != segment->start())
                                    return;
                                
                                static const auto filename = (std::string)SETTING(filename).value<file::Path>().filename();
                                file::Path output = pv::DataLocation::parse("output", "tags_"+filename) / Meta::toStr(_last_predicted_id);
                                
                                if(!output.exists())
                                    return;
                                
                                auto prefix = Meta::toStr(identity().ID()) + "." + Meta::toStr(segment->start());
                                if(!(output / prefix).exists())
                                    return;
                                
                                auto files = (output / prefix).find_files();
                                
                                // delete files that already existed for this individual AND segment
                                for(auto &f : files) {
                                    f.delete_file();
                                    //print("\tdeleting file ", f);
                                }
                            };
                            
                            // if we can add this code, update the last requested
                            if(RecTask::add(std::move(task), fill, callback)) {
                                //cmn::print("Have ", it->second.size(), " QRCodes for segment ", *segment, " in individual:", identity().ID(), " ", segment_ended);

                                std::unique_lock guard(_qrcode_mutex);
                                _last_requested_qrcode = frame;
                                
                            } //else
                                //print("\t",segment->range, " individual:", identity().ID(), " rejected ",it->second.size()," images.");
                                
                        } //else {
                            //print("\t",segment->range, " individual:", identity().ID(), " not enough images ",it->second.size(),".");
                        //}
                        
                    } //else if(segment_ended && segment->length() > 2) {
                    //    print("\t",segment->range, " individual:", identity().ID(), " does not have QRCodes.");
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

void Individual::add_tag_image(const tags::Tag& tag) {
    assert(tag.frame.valid());
    auto && [range, first] = get_segment(tag.frame);
    
    auto &set = _best_images[range.start];
    if(set.size() > 3) {
        Frame_t last;
        for (auto& tag : set) {
            if (tag.frame > last)
                last = tag.frame;
        }
        
        if(cmn::abs((last - tag.frame).get()) >= FAST_SETTINGS(frame_rate)) {
            if(set.size() > 6)
                set.erase(--set.end());
        } else
            return;
    }
    set.insert(tag);
}

Individual::segment_map::const_iterator Individual::find_segment_with_start(Frame_t frame) const {
    return find_frame_in_sorted_segments(frame_segments().begin(), frame_segments().end(), frame);
}

FrameRange Individual::get_segment(Frame_t frameIndex) const {
    auto it = std::lower_bound(_frame_segments.begin(), _frame_segments.end(), frameIndex, [](const auto& ptr, Frame_t frame) {
        return ptr->start() < frame;
    });
    if(it != _frame_segments.end()) {
        if(it != _frame_segments.begin() && (*it)->start() != frameIndex)
            --it;
        assert((*it)->start() <= frameIndex || (*it)->start() == start_frame());
        return *it->get();
    }
    
    if(!_frame_segments.empty() && (*_frame_segments.rbegin())->start() <= frameIndex)
        return *_frame_segments.rbegin()->get();
    
    return FrameRange();
}

FrameRange Individual::get_recognition_segment(Frame_t frameIndex) const {
    auto it = _recognition_segments.lower_bound(frameIndex);
    if(it != _recognition_segments.end()) {
        if(it != _recognition_segments.begin() && it->first != frameIndex)
            --it;
        //assert(it->first <= frameIndex || it->first == start_frame());
        return it->second;
    }
    
    if(!_recognition_segments.empty() && _recognition_segments.rbegin()->second.start() <= frameIndex)
        return _recognition_segments.rbegin()->second;
    
    return FrameRange();
}

FrameRange Individual::get_segment_safe(Frame_t frameIndex) const {
    auto segment = get_segment(frameIndex);
    if(!segment.contains(frameIndex))
        return FrameRange();
    
    return segment;
}

FrameRange Individual::get_recognition_segment_safe(Frame_t frameIndex) const {
    auto segment = get_recognition_segment(frameIndex);
    if(!segment.contains(frameIndex))
        return FrameRange();
    
    return segment;
}

const std::multiset<tags::Tag>* Individual::has_tag_images_for(Frame_t frameIndex) const {
    auto range = get_segment(frameIndex);
    
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

inline bool operator<(const std::shared_ptr<track::Individual::SegmentInformation>& ptr, const FrameNumber& frame) {
    assert(ptr != nullptr);
    return ptr->end() < frame.frame;
}

inline bool operator<(const FrameNumber& frame, const std::shared_ptr<track::Individual::SegmentInformation>& ptr) {
    assert(ptr != nullptr);
    return frame.frame < ptr->start();
}

bool Individual::has(Frame_t frame) const {
    if(frame < _startFrame || frame > _endFrame)
        return false;
    
    return std::binary_search(_frame_segments.begin(), _frame_segments.end(), FrameNumber{frame});
}

void Individual::SegmentInformation::add_basic_at(Frame_t frame, long_t gdx) {
    UNUSED(frame);
    assert(end() == frame);
    basic_index.push_back(gdx);
}

#ifndef NDEBUG
#define SEGMENT_ACCESS(INDEXARRAY, INDEX) INDEXARRAY . at( INDEX )
#define FRAME_SEGMENT_ACCESS(INDEXARRAY, INDEX) INDEXARRAY . at( ( INDEX ).get() )
#else
#define SEGMENT_ACCESS(INDEXARRAY, INDEX) INDEXARRAY [ INDEX ]
#define FRAME_SEGMENT_ACCESS(INDEXARRAY, INDEX) INDEXARRAY [ ( INDEX ).get() ]
#endif

void Individual::SegmentInformation::add_posture_at(std::unique_ptr<PostureStuff>&& stuff, Individual* fish) {//long_t gdx) {
    long_t L = length();
    if(posture_index.size() != size_t(L)) {
        long_t prev = posture_index.size();
        posture_index.resize(L);
        for (long_t i=prev; i<L; ++i) {
            SEGMENT_ACCESS(posture_index, i) = -1;
        }
    }
    
    long_t gdx = fish->_posture_stuff.size();
    
    if(fish->added_postures.find(stuff->frame) == fish->added_postures.end()) {
        fish->added_postures.insert(stuff->frame);
    } else {
        print(fish->added_postures);
        throw SoftException("(", fish->identity(),") Posture for frame ",stuff->frame," already added.");
    }
    
    if(!fish->_posture_stuff.empty() && stuff->frame < fish->_posture_stuff.back()->frame)
        throw SoftException("(", fish->identity().ID(),") Adding frame ", stuff->frame," after frame ", fish->_last_posture_added);
    
    fish->_last_posture_added = stuff->frame;
    FRAME_SEGMENT_ACCESS(posture_index, stuff->frame - start()) = gdx;

    fish->_posture_stuff.push_back(std::move(stuff));
}

long_t Individual::SegmentInformation::basic_stuff(Frame_t frame) const {
    //assert(frame >= start() && frame <= end() && size_t(frame-start()) < basic_index.size());
    if(frame < start() || frame > end() || (size_t)(frame-start()).get() >= basic_index.size())
        return -1;
    return FRAME_SEGMENT_ACCESS(basic_index, frame - start());
}

long_t Individual::SegmentInformation::posture_stuff(Frame_t frame) const {
    if(posture_index.empty() || !contains(frame)
       || (posture_index.size() < basic_index.size() && (size_t)(frame - start()).get() >= posture_index.size() ))
        return -1;
    assert(frame >= start() && frame <= end() && (size_t)(frame-start()).get() < posture_index.size());
    return FRAME_SEGMENT_ACCESS(posture_index, frame - start()); //posture_index .at( frame - start() );
}

decltype(Individual::_frame_segments)::const_iterator Individual::iterator_for(Frame_t frameIndex) const {
    if(frameIndex < _startFrame || frameIndex > _endFrame)
        return _frame_segments.end();
    if(frameIndex == _startFrame)
        return _frame_segments.begin();
    
    auto it = std::lower_bound(_frame_segments.begin(), _frame_segments.end(), frameIndex, [](const auto& ptr, Frame_t frame){
        return ptr->start() < frame;
    });
    if(it != _frame_segments.end()) {
        if((*it)->start() > frameIndex) {
            if(it == _frame_segments.begin())
                it = _frame_segments.end();
            else
                --it;
        }
        
    } else if(!_frame_segments.empty()) {
        --it;
    }
    
    assert(it == _frame_segments.end() || (*it)->start() <= frameIndex);
    return it;
}

std::shared_ptr<Individual::SegmentInformation> Individual::segment_for(Frame_t frameIndex) const {
    if(frameIndex < _startFrame || frameIndex > _endFrame)
        return nullptr;
    
    auto it = iterator_for(frameIndex);
    return it == _frame_segments.end() || !(*it)->contains(frameIndex) ? nullptr : *it;
}

Individual::PostureStuff::~PostureStuff() {
    if(head) delete head;
    if(centroid_posture) delete centroid_posture;
}

Individual::BasicStuff* Individual::basic_stuff(Frame_t frameIndex) const {
    auto segment = segment_for(frameIndex);
    if(segment)
        return SEGMENT_ACCESS(_basic_stuff, segment->basic_stuff(frameIndex)).get(); //_basic_stuff.at( segment->basic_stuff(frameIndex) );
    return nullptr;
}

Individual::PostureStuff* Individual::posture_stuff(Frame_t frameIndex) const {
    auto segment = segment_for(frameIndex);
    if(segment) {
        auto index = segment->posture_stuff(frameIndex);
        return index != -1 ? SEGMENT_ACCESS(_posture_stuff, index).get() : nullptr;
        //return index != -1 ? _posture_stuff.at( index ) : nullptr;
    }
    return nullptr;
}

std::tuple<Individual::BasicStuff*, Individual::PostureStuff*> Individual::all_stuff(Frame_t frameIndex) const {
    auto segment = segment_for(frameIndex);
    if(segment) {
        auto basic_index = segment->basic_stuff(frameIndex);
        auto posture_index = segment->posture_stuff(frameIndex);
        return {
            basic_index != -1 ? SEGMENT_ACCESS(_basic_stuff, basic_index).get() : nullptr,
            posture_index != -1 ? SEGMENT_ACCESS(_posture_stuff, posture_index).get() : nullptr
        };
        //return index != -1 ? _posture_stuff.at( index ) : nullptr;
    }
    return {nullptr, nullptr};
}

long_t Individual::thresholded_size(Frame_t frameIndex) const {
    auto ptr = basic_stuff(frameIndex);
    return ptr ? ptr->thresholded_size : -1;
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
    return ptr ? ptr->head : nullptr;
}

const MotionRecord* Individual::centroid_posture(Frame_t frameIndex) const {
    auto ptr = posture_stuff(frameIndex);
    return ptr ? ptr->centroid_posture : nullptr;
}

MotionRecord* Individual::centroid(Frame_t frameIndex) {
    auto ptr = basic_stuff(frameIndex);
    return ptr ? &ptr->centroid : nullptr;
}

MotionRecord* Individual::centroid_weighted(Frame_t frameIndex) {
    auto ptr = basic_stuff(frameIndex);
    return ptr ? &ptr->centroid : nullptr;
}

MotionRecord* Individual::head(Frame_t frameIndex) {
    auto ptr = posture_stuff(frameIndex);
    return ptr ? ptr->head : nullptr;
}

MotionRecord* Individual::centroid_posture(Frame_t frameIndex) {
    auto ptr = posture_stuff(frameIndex);
    return ptr ? ptr->centroid_posture : nullptr;
}

pv::BlobPtr Individual::blob(Frame_t frameIndex) const {
    auto segment = segment_for(frameIndex);
    if(segment)
        return SEGMENT_ACCESS(_basic_stuff, segment->basic_stuff(frameIndex))->blob.unpack();
    return nullptr;
}

pv::CompressedBlob* Individual::compressed_blob(Frame_t frameIndex) const {
    auto segment = segment_for(frameIndex);
    if(segment)
        return &SEGMENT_ACCESS(_basic_stuff, segment->basic_stuff(frameIndex))->blob;
    return nullptr;
}

const Midline::Ptr Individual::midline(Frame_t frameIndex) const {
    auto && [basic, posture] = all_stuff(frameIndex);
    if(!posture)
        return nullptr;
    
    return calculate_midline_for(*basic, *posture);
}

const Midline::Ptr Individual::pp_midline(Frame_t frameIndex) const {
    auto ptr = posture_stuff(frameIndex);
    return ptr ? ptr->cached_pp_midline : nullptr;
}

Midline::Ptr Individual::fixed_midline(Frame_t frameIndex) const {
    auto mid = pp_midline(frameIndex);
    if(mid == nullptr || midline_length() <= 0 || _local_cache._midline_samples == 0)
        return nullptr;
    
    MovementInformation movement;
    if(FAST_SETTINGS(posture_direction_smoothing) > 1) {
        auto && [samples, hist, index, mov] = calculate_previous_vector(frameIndex);
        movement = mov;
    }

    auto fixed = std::make_shared<Midline>(*mid);
    fixed->post_process(movement, DebugInfo{frameIndex, identity().ID(), false});
    fixed = fixed->normalize(midline_length());
    
    return fixed;
}

MinimalOutline::Ptr Individual::outline(Frame_t frameIndex) const {
    auto ptr = posture_stuff(frameIndex);
    return ptr ? ptr->outline : nullptr;
}

Individual::Individual(Identity&& id)
    : _identity(std::move(id))
{
    if(FAST_SETTINGS(manual_identities).count(identity().ID()))
        identity().set_manual(true);
    //else
    //    throw U_EXCEPTION("Hey");
}

Individual::~Individual() {
#if !COMMONS_NO_PYTHON
    RecTask::remove(identity().ID());
#endif

    if(!Tracker::instance())
        return;
    
    remove_frame(start_frame());
    
    if(Tracker::recognition())
        Tracker::recognition()->remove_individual(this);
#ifndef NDEBUG
    print("Deleting individual ", identity().ID());
#endif
}

void Individual::unregister_delete_callback(void* ptr) {
    if(!Tracker::instance())
        return;
    Tracker::LockGuard guard("unregister_delete_callback");
    _delete_callbacks.erase(ptr);
}

void Individual::register_delete_callback(void* ptr, const std::function<void(Individual*)>& lambda) {
    if(!Tracker::instance())
        return;
    
    Tracker::LockGuard guard("register_delete_callback");
    _delete_callbacks[ptr] = lambda;
}

#define MAP_SIZE(MAP, A, B) ((sizeof(A) + sizeof(B) + 32 + sizeof(B*)) * (MAP).size() + 24)
#define KB(x)   ((size_t) (x) << 10)
#define MB(x)   ((size_t) (x) << 20)

void Individual::add_manual_match(Frame_t frameIndex) {
    //assert(frameIndex <= _endFrame && frameIndex >= _startFrame);
    _manually_matched.insert(frameIndex);
}

void Individual::add_automatic_match(Frame_t frameIndex) {
    //assert(frameIndex <= _endFrame && frameIndex >= _startFrame);
    automatically_matched.insert(frameIndex);
}

bool Individual::is_manual_match(Frame_t frameIndex) const {
    return _manually_matched.find(frameIndex) != _manually_matched.end();
}

bool Individual::is_automatic_match(Frame_t frameIndex) const {
    return automatically_matched.find(frameIndex) != automatically_matched.end();
}

bool Individual::recently_manually_matched(Frame_t frameIndex) const {
    for(auto frame = frameIndex; frame >= _startFrame && frame >= frameIndex - Frame_t(FAST_SETTINGS(frame_rate) * 0.5); --frame) {
        if(_manually_matched.find(frame) != _manually_matched.end())
            return true;
    }
    
    return false;
}

void Individual::remove_frame(Frame_t frameIndex) {
    if (frameIndex > _endFrame)
        return;

    {
        auto callbacks = _delete_callbacks;
        for (auto& f : callbacks)
            f.second(this);
        _delete_callbacks.clear();
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
        for(auto & seg : _frame_segments) {
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
    
    if(!_recognition_segments.empty()) {
        auto it = --_recognition_segments.end();
        while (it->second.start() >= frameIndex) {
            if(it == _recognition_segments.begin())
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
    
    if(!_frame_segments.empty()) {
        auto it = --_frame_segments.end();
        while ((*it)->range.start >= frameIndex) {
            if(it == _frame_segments.begin())
                break;
            --it;
        }
        
        bool shortened_posture_index = false, shortened_basic_index = false;
        
        if((*it)->range.end < frameIndex)
            ++it;
        else if((*it)->range.start < frameIndex) {
#ifndef NDEBUG
            print("(",identity().ID(),") need to shorten segment ",(*it)->range.start,"-",(*it)->range.end," to fit frame ",frameIndex);
#endif
            (*it)->range.end = frameIndex - 1_f;
            assert((*it)->range.start <= (*it)->range.end);
            
            (*it)->basic_index.resize((*it)->length());
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
                    print("(", identity().ID(),")\tposture_stuff.back == ", _posture_stuff.empty() ? Frame_t() : _posture_stuff.back()->frame," (kit = ",ff,")");
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
            if(!shortened_posture_index && it == _frame_segments.end())
                print("Individual ", identity().ID()," did not have any postures after ",frameIndex);
#endif
        }
        
        if(it != _frame_segments.end()) {
#ifndef NDEBUG
            print("(", identity().ID(),") found that we need to delete everything after and including ", (*it)->range.start,"-",(*it)->range.end);
#endif
            
            if(!shortened_basic_index && !(*it)->basic_index.empty()) {
                _basic_stuff.resize((*it)->basic_index.front());
                _matched_using.resize(_basic_stuff.size());
            }
            
            if(!shortened_posture_index) {
                auto start = it;
                while(start != _frame_segments.end()) {
                    for (auto kit = (*start)->posture_index.begin(); kit != (*it)->posture_index.end(); ++kit) {
                        if(*kit != -1) {
                            assert(*kit < (long long)_posture_stuff.size());
                            _posture_stuff.resize(*kit);
#ifndef NDEBUG
                            print("(", identity().ID(),")\tposture_stuff.back == ",_posture_stuff.empty() ? Frame_t() : _posture_stuff.back()->frame);
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
                    print("Individual ", identity().ID()," did not have any postures after ",frameIndex);
#endif
            }
            
            _frame_segments.erase(it, _frame_segments.end());
            
        } else if(!shortened_posture_index) {
#ifndef NDEBUG
            print("Individual ", identity().ID()," does not delete any frames.");
#endif
        }
//#ifndef NDEBUG
        check_integrity();
//#endif
    }
    
    /*while(!_frame_segments.empty()) {
        auto it = --_frame_segments.end();
        auto &segment = *it->second;
        auto & [current, usable] = (FrameRange&)segment;
        
        assert(segment.basic_index.empty() || *segment.basic_index.rbegin() == (long_t)_basic_stuff.size()-1);
        //assert(segment.posture_index.empty() || *segment.posture_index.rbegin() == (long_t)_posture_stuff.size()-1);
        
        if(frameIndex > current.start && frameIndex <= current.end) {
            // contains the frame and more than the frame,
            // so shorten it and break
            //size_t offset = current.length() - segment.posture_index.size();
            current.end = frameIndex - 1;
            
            _basic_stuff.resize(_basic_stuff.size() - (segment.basic_index.size() - current.length()));
            
            segment.basic_index.resize(current.length());
            if(!segment.posture_index.empty()) {
                long_t last_found_gdx = _posture_stuff.size();
                for(auto kit = segment.posture_index.rbegin(); kit != segment.posture_index.rend(); ++kit) {
                    if(*kit != -1 && SEGMENT_ACCESS(_posture_stuff, *kit)->frame <= frameIndex ) {
                        last_found_gdx = SEGMENT_ACCESS(_posture_stuff, *kit)->frame < frameIndex ? (*kit + 1) : (*kit);
                        assert(SEGMENT_ACCESS(_posture_stuff, *kit)->frame >= frameIndex);
                        break;
                    }
                }
                
                print("#1 (", identity().ID(),") resizing posture_stuff ", _posture_stuff.size()," -> ",last_found_gdx);
                
                segment.posture_index.resize(current.length());
                _posture_stuff.resize(last_found_gdx);
            }
            
            break;
        }
        
        // it might still be [start, start+n] at this point,
        // so break only if its actually smaller
        if(current.start < frameIndex)
            break;
        
        _basic_stuff.resize(_basic_stuff.size() - segment.basic_index.size());
        
        long_t last_found_gdx = _posture_stuff.size();
        for(auto kit = segment.posture_index.begin(); kit != segment.posture_index.end(); ++kit) {
            if(*kit != -1) {
                last_found_gdx = *kit;
                break;
            }
        }
        
        print("#2 (", identity().ID(),") resizing posture_stuff ", _posture_stuff.size()," -> ",last_found_gdx);
        _posture_stuff.resize(last_found_gdx);
        
        _frame_segments.erase(it);
    }*/
    
    /*while(!_training_data.empty()) {
        auto it = --_training_data.end();
        if(it->first < frameIndex)
            break;
        
        _training_data.erase(it);
    }*/
    
    while(!_best_images.empty()) {
        auto it = --_best_images.end();
        if(it->first < frameIndex)
            break;
        
        _best_images.erase(it);
    }
    
    while(!average_recognition_segment.empty()) {
        auto it = --average_recognition_segment.end();
        auto current = it->first;
        
        if(current < frameIndex)
            break;
        
        average_recognition_segment.erase(it);
    }
    
    while(!_manually_matched.empty()) {
        auto kit = --_manually_matched.end();
        if(*kit < frameIndex)
            break;
        _manually_matched.erase(kit);
    }
    
    while(!automatically_matched.empty()) {
        auto kit = --automatically_matched.end();
        if(*kit < frameIndex)
            break;
        automatically_matched.erase(kit);
    }
    
    /*for (long_t i=frameIndex-FAST_SETTINGS(posture_direction_smoothing); i<frameIndex; i++) {
        if (_midlines.count(i))
            _midlines.erase(i);
        
        MotionRecord* ptr;
        if((ptr = head(i))) {
            delete ptr;
            _head.erase(i);
        }
        if((ptr = centroid_posture(i))) {
            delete ptr;
            _centroid_posture.erase(i);
        }
    }*/
    
    for (auto i=frameIndex; i<=_endFrame; ++i) {
        /*auto c = centroid(i);
        if(c) {
            delete c;
            _centroid.erase(i);
        }
        
        if(centroid_weighted(i)) {
            delete _weighted_centroid.at(i);
            _weighted_centroid.erase(i);
        }
            
        if(head(i)) {
            delete _head.at(i);
            _head.erase(i);
        }
            
        if(centroid_posture(i)) {
            delete _centroid_posture.at(i);
            _centroid_posture.erase(i);
        }
        
        if(_thresholded_size.find(i) != _thresholded_size.end())
            _thresholded_size.erase(i);
        
        if(_blobs.count(i)) {
            _blobs.erase(i);
        }
            
        if (_midlines.count(i))
            _midlines.erase(i);
            
        if(_outlines.count(i))
            _outlines.erase(i);*/
            
        if(_custom_data.count(i)) {
            for(auto &pair : _custom_data.at(i)) {
                pair.second.second(pair.second.first);
            }
            _custom_data.erase(i);
        }
            
        /*if(_posture_original_angles.count(i)) {
            _posture_original_angles.erase(i);
        }
            
        if(_cached_fixed_midlines.count(i))
            _cached_fixed_midlines.erase(i);
        
        if(_cached_pp_midlines.count(i))
            _cached_pp_midlines.erase(i);*/
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
    
    if(!average_recognition_segment.empty())
        calculate_average_recognition();
    
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
    _current_velocities.clear();
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
        assert(!_current_velocities.empty() || basic->frame == fish->start_frame());
        add(basic->frame, &basic->centroid);
    }
    
    for(auto & p : fish->_posture_stuff) {
        if(p->cached_pp_midline && !p->cached_pp_midline->empty()) {
            add(*p);
        }
    }
    
}

float Individual::midline_length() const {
    return _local_cache._midline_samples == 0
        ? gui::Graph::invalid()
        : (_local_cache._midline_length / _local_cache._midline_samples * 1.1f);
}
size_t Individual::midline_samples() const { return _local_cache._midline_samples; }
float Individual::outline_size() const { return _local_cache._outline_samples == 0 ? gui::Graph::invalid() : (_local_cache._outline_size / _local_cache._outline_samples); }

Vec2 Individual::LocalCache::add(Frame_t frameIndex, const track::MotionRecord *current) {
    const size_t maximum_samples = max(3.f, FAST_SETTINGS(frame_rate)*0.1f);
    
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
    
    _current_velocities[frameIndex] = _v_samples.empty()
        ? raw_velocity
        : (_current_velocity / float(_v_samples.size()));
    
    return v;
}

void Individual::LocalCache::add(const PostureStuff& stuff) {
    if(stuff.outline) {
        _outline_size += stuff.outline->size();
        ++_outline_samples;
    }
    
    if(stuff.midline_length != PostureStuff::infinity) {
        _midline_length += stuff.midline_length;
        ++_midline_samples;
    }
}

int64_t Individual::add(const FrameProperties* props, Frame_t frameIndex, const PPFrame& frame, const pv::BlobPtr& blob, prob_t current_prob, default_config::matching_mode_t::Class match_mode)
{
    if (has(frameIndex))
        return -1;
    
    if (frameIndex >= _startFrame && frameIndex <= _endFrame)
        throw UtilsException("Cannot add intermediate frames out of order.");
    
    if (_startFrame > frameIndex || !_startFrame.valid()) {
        _startFrame = frameIndex;
    }
    
    if (_endFrame < frameIndex || !_endFrame.valid()) {
        _endFrame = frameIndex;
    }
    
    // find valid previous frame
    //!TODO: can probably use segment ptr here
    auto prev_frame = frameIndex - 1_f;
    const MotionRecord* prev_prop = nullptr;
    if(!empty()) {
        if(frameIndex > _startFrame) {
            auto &previous = find_frame(prev_frame);
            if(previous) {
                prev_frame = previous->frame;
                prev_prop = &previous->centroid;
            }
        }
    }
    
    _hints.push(frameIndex, props);
    
    auto stuff = std::make_unique<BasicStuff>();
    stuff->centroid.init(prev_prop, frame.time, blob->center(), blob->orientation());
    
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
    stuff->blob = blob;
    stuff->pixels = blob;
    stuff->thresholded_size = blob->recount(FAST_SETTINGS(track_threshold), *Tracker::instance()->background()) / SQR(FAST_SETTINGS(cm_per_pixel));
    
    //auto centroid_point = weighted_centroid(*blob, *blob->pixels());
    
    //!TODO: can use previous segment here
    //if(prev_props)
    //    prev_props = centroid_weighted(prev_frame);
    
    //stuff->weighted_centroid = new MotionRecord(prev_props, time, centroid_point, current->angle());
    //push_to_segments(frameIndex, prev_frame);
    
    auto cached = frame.cached(identity().ID());
    prob_t p{current_prob};
    if(current_prob == -1 && cached) {
        if(cached->individual_empty /* || frameIndex < start_frame() */)
            p = 0;
        else
            p = probability(cached->consistent_categories ? frame.label(blob) : -1,
                            *cached,
                            frameIndex,
                            stuff->blob);//.p;
    }
    
    auto segment = update_add_segment(frameIndex, stuff->centroid, prev_frame, &stuff->blob, p);
    
    // add BasicStuff index to segment
    auto index = _basic_stuff.size();
    segment->add_basic_at(frameIndex, index);
    if(!_basic_stuff.empty() && stuff->frame < _basic_stuff.back()->frame)
        throw SoftException("(", identity(),") Added basic stuff for frame ", stuff->frame, " after frame ", _basic_stuff.back()->frame,".");
    _basic_stuff.push_back(std::move(stuff));
    _matched_using.push_back(match_mode);
    
    const auto video_length = Tracker::analysis_range().end;
    if(frameIndex >= video_length) {
        update_midlines(&_hints);
    }
    
    return int64_t(index);
}

void Individual::iterate_frames(const Range<Frame_t>& segment, const std::function<bool(Frame_t frame, const std::shared_ptr<SegmentInformation>&, const Individual::BasicStuff*, const Individual::PostureStuff*)>& fn) const {
    auto fit = iterator_for(segment.start);
    auto end = _frame_segments.end();
    
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

std::shared_ptr<Individual::SegmentInformation> Individual::update_add_segment(Frame_t frameIndex, const MotionRecord& current, Frame_t prev_frame, const pv::CompressedBlob* blob, prob_t current_prob)
{
    //! find a segment this (potentially) belongs to
    std::shared_ptr<SegmentInformation> segment = nullptr;
    if(!_frame_segments.empty()) {
        segment = *_frame_segments.rbegin();
        
        // check whether we found the right one
        // (it can only be the last one, or no one)
        if(segment->end() < frameIndex - 1_f)
            segment = nullptr; // this frame does not actually belong
    }
    
    auto prop = Tracker::properties(frameIndex);
    auto prev_prop = Tracker::properties(frameIndex - 1_f);
    
    double tdelta = prop && prev_prop ? prop->time - prev_prop->time : 0;
    uint32_t error_code = 0;
    error_code |= Reasons::LostForOneFrame       * uint32_t(prev_frame != frameIndex - 1_f);
    error_code |= Reasons::ProbabilityTooSmall   * uint32_t(current_prob != -1 && current_prob < FAST_SETTINGS(track_trusted_probability));
    error_code |= Reasons::TimestampTooDifferent * uint32_t(FAST_SETTINGS(huge_timestamp_ends_segment) && tdelta >= FAST_SETTINGS(huge_timestamp_seconds));
    error_code |= Reasons::ManualMatch           * uint32_t(is_manual_match(frameIndex));
    error_code |= Reasons::NoBlob                * uint32_t(!blob);
    error_code |= Reasons::WeirdDistance         * uint32_t(FAST_SETTINGS(track_end_segment_for_speed) && current.speed<Units::CM_AND_SECONDS>() >= weird_distance());
    error_code |= Reasons::MaxSegmentLength      * uint32_t(FAST_SETTINGS(track_segment_max_length) > 0 && segment && segment->length() / float(FAST_SETTINGS(frame_rate)) >= FAST_SETTINGS(track_segment_max_length));
    
    const bool segment_ended = error_code != 0;

    if(frameIndex == _startFrame || segment_ended) {
        if(!_frame_segments.empty()) {
            _frame_segments.back()->error_code = error_code;
        }
        
        /*if(frameIndex != _startFrame) {
            print("reasoning for ", identity(), " in ", frameIndex, ":");
            print(prev_frame != frameIndex - 1_f, " prev: ", prev_frame, " frameIndex: ", frameIndex);
            print(current_prob != -1 && current_prob < FAST_SETTINGS(track_trusted_probability), " current_prob:", current_prob, " trusted = ", FAST_SETTINGS(track_trusted_probability));
            print(FAST_SETTINGS(huge_timestamp_ends_segment) && tdelta >= FAST_SETTINGS(huge_timestamp_seconds), " flag: ", FAST_SETTINGS(huge_timestamp_ends_segment), " -> ", tdelta, " >= ", FAST_SETTINGS(huge_timestamp_seconds));
            print(is_manual_match(frameIndex), " is manual match ", frameIndex, " = ", is_manual_match(frameIndex));
            print(!blob, " blob: ", blob);
            print(FAST_SETTINGS(track_end_segment_for_speed) && current.speed<Units::CM_AND_SECONDS>() >= weird_distance(), " max speed: ", FAST_SETTINGS(track_end_segment_for_speed), " speed = ", current.speed<Units::CM_AND_SECONDS>(), " weird: ", weird_distance());
            if(segment) print(FAST_SETTINGS(track_segment_max_length) > 0 && segment && segment->length() / float(FAST_SETTINGS(frame_rate)) >= FAST_SETTINGS(track_segment_max_length), " segment max length = ", FAST_SETTINGS(track_segment_max_length), " segment percent: ", segment->length() / float(FAST_SETTINGS(frame_rate)));
            print("");
        }*/

        segment = std::make_shared<SegmentInformation>(Range<Frame_t>(frameIndex, frameIndex), !blob || blob->split() ? Frame_t() : frameIndex);
        _frame_segments.push_back(segment);
        
#if !COMMONS_NO_PYTHON
        //! Update recognition if enabled
        if(Recognition::recognition_enabled())
            Recognition::notify();
#endif
        
    } else if(prev_frame == frameIndex - 1_f) {
        assert(!_frame_segments.empty());
        segment = *_frame_segments.rbegin();
        segment->range.end = frameIndex;
        if(!segment->first_usable.valid() && blob && !blob->split())
            segment->first_usable = frameIndex;

    } // else... nothing

    return segment;
}

float Individual::weird_distance() {
    return FAST_SETTINGS(track_max_speed) * 0.99;
}

void Individual::clear_post_processing() {
    for(auto & stuff : _posture_stuff) {
        if(stuff->head) {
            delete stuff->head;
            stuff->head = nullptr;
        }
        //stuff->midline = nullptr;
        stuff->posture_original_angle = PostureStuff::infinity;
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
    /*if(FAST_SETTINGS(posture_direction_smoothing) == 0) {
        update_frame_with_posture(frameIndex);
    }*/
    
    const auto smooth_range = Frame_t(FAST_SETTINGS(posture_direction_smoothing));
    const auto video_length = Tracker::analysis_range().end;
    auto end_frame = Tracker::end_frame();
    
    //! find the first frame that needs to be cached, but hasnt been yet
    auto it = _posture_stuff.rbegin(), last_found = _posture_stuff.rend();
    for (; it != _posture_stuff.rend(); ++it) {
        if((smooth_range == 0_f || video_length == end_frame || (*it)->frame <= end_frame - smooth_range) && (*it)->cached_pp_midline)
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
            if((smooth_range == 0_f || video_length == end_frame || (*it)->frame <= end_frame - smooth_range) && (*it)->cached_pp_midline)
            {
                (*it)->posture_original_angle = (*it)->cached_pp_midline->original_angle();

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

Midline::Ptr Individual::calculate_midline_for(const BasicStuff &basic, const PostureStuff &posture) const
{
    //if(!posture || !basic)
    //    return nullptr;
    
    auto &ptr = posture.cached_pp_midline;
    auto &blob = basic.blob;
    //basic.pixels = nullptr;
    
    Midline::Ptr midline;
    
    if(ptr) {
        //Timer timer;
        midline = std::make_unique<Midline>(*ptr);
        
        MovementInformation movement;
        //movement.position = blob->bounds().pos();
        
        if(FAST_SETTINGS(posture_direction_smoothing) > 1) {
            auto && [samples, hist, index, mov] = calculate_previous_vector(posture.frame);
            movement = mov;
        }
        
        midline->post_process(movement, DebugInfo{posture.frame, identity().ID(), false});
        if(!midline->is_normalized())
            midline = midline->normalize();
        else if(size_t(_warned_normalized_midline.elapsed())%5 == 0) {
#ifndef NDEBUG
            FormatWarning(identity().ID()," has a pre-normalized midline in frame ",posture.frame,". not normalizing it again.");
#endif
        }
        
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
        midline = calculate_midline_for(basic, posture);
        auto &outline = posture.outline;
        auto &c = basic.centroid;
        
        if(!midline)
            return nullptr;
        
        size_t head_index = cmn::min(midline->segments().size() - 1u, size_t(roundf(midline->segments().size() * FAST_SETTINGS(posture_head_percentage))));
        auto pt = midline->segments().at(head_index).pos;
        
        float angle = midline->angle() + M_PI;
        float x = (pt.x * cmn::cos(angle) - pt.y * cmn::sin(angle));
        float y = (pt.x * cmn::sin(angle) + pt.y * cmn::cos(angle));
        
        auto bounds = blob.calculate_bounds();
        
        pt = Vec2(x, y);
        pt += bounds.pos() + midline->offset();
        
        const PostureStuff* previous = nullptr;
        if (posture_it != _posture_stuff.begin()) {
            previous = (*(posture_it - 1)).get();
        }

        auto prop = Tracker::properties(posture.frame, hints);
        assert(prop);
        posture.head = new MotionRecord;
        posture.head->init(previous ? previous->head : nullptr, prop->time, pt, midline->angle());
        
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
                c.flip(previous ? previous->head : nullptr);
            }
        }
        
        // calculate midline centroid
        Vec2 centroid_point(0, 0);
        auto points = outline->uncompress();
        
        for (auto &p : points) {
            centroid_point += p;
        }
        centroid_point /= float(points.size());
        centroid_point += bounds.pos();
        
        posture.centroid_posture = new MotionRecord;
        posture.centroid_posture->init(previous ? previous->centroid_posture : nullptr, prop->time, centroid_point, midline->angle());
        posture.midline_angle = midline->angle();
        posture.midline_length = midline->len();
        
        _local_cache.add(posture);
    }
    
    return midline;
}

/*void Individual::push_to_segments(Frame_t frameIndex, long_t prev_frame) {
    // conditions of when to start/end frame segments
    double tdelta = Tracker::has(frameIndex) && Tracker::has(frameIndex-1) ? Tracker::properties(frameIndex).time - Tracker::properties(frameIndex-1).time : 0;
    
    auto segment = segment_for(frameIndex);
    auto current = centroid_weighted(frameIndex);
    auto blob = this->blob(frameIndex);
    
    if(frameIndex == _startFrame
       || prev_frame != frameIndex-1
       || (FAST_SETTINGS(huge_timestamp_ends_segment) && tdelta >= FAST_SETTINGS(huge_timestamp_seconds))
       || is_manual_match(frameIndex)
       || (current && current->speed() >= weird_distance())
       || !blob)
    {
        _frame_segments[frameIndex] = std::make_shared<SegmentInformation>(Range<Frame_t>(frameIndex, frameIndex), !blob || blob->split() ? -1 : frameIndex);
        
        update_midlines();
        
        // guard has been unlocked:
        //! Update recognition if enabled
        if(Recognition::recognition_enabled())
            Recognition::notify();
        
    } else if(prev_frame == frameIndex-1) {
        assert(!_frame_segments.empty());
        auto segment = _frame_segments.rbegin()->second;
        segment->range.end = frameIndex;
        if(segment->first_usable == -1 && blob && !blob->split())
            segment->first_usable = frameIndex;
        
        const long_t video_length = Tracker::analysis_range().end;
        if(frameIndex >= video_length) {
            update_midlines();
        }
    }
}*/

#include <random>

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
        return (!A && B) || (A && B && A->frame < B->frame);
    }
    constexpr bool operator()(const FrameProperties* A, const Frame_t& B) {
        return !A || A->frame < B;
    }
    constexpr bool operator()(const Frame_t& A, const FrameProperties* B) {
        return B && A < B->frame;
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
        if (here == _last_second.end() || !*here || (*here)->frame < index) {
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
    
    /*assert(!current.valid() || current < index);
    assert(!ptr || ptr->frame == index);
    assert(!_last_second.empty());
    current = index;
    
#ifndef NDEBUG
    auto tmp = _last_second.front();
#endif
    if(_last_second.size() > 1)
        std::rotate(_last_second.begin(), ++_last_second.begin(), _last_second.end());
#ifndef NDEBUG
    assert(tmp == _last_second.back());
#endif
    
    _last_second.back() = ptr;*/
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
    if (size == 0 && FAST_SETTINGS(frame_rate) < 0) {
#ifndef NDEBUG
        FormatExcept("Size=", size," frame_rate=", FAST_SETTINGS(frame_rate),"");
#endif
        _last_second.resize(0);
    } else {
        _last_second.resize(size > 0 ? size : FAST_SETTINGS(frame_rate));
    }
    std::fill(_last_second.begin(), _last_second.end(), nullptr);
    current.invalidate();
}

template<class T, class U>
typename std::vector<T>::const_iterator find_in_sorted(const std::vector<T>& vector, const U& v) {
    auto it = std::lower_bound(vector.begin(),
                               vector.end(),
                               v,
                    [](auto& l, auto& r){ return !l || l->frame < r; });
    return it == vector.end() || (*it)->frame == v ? it : vector.end();
}

const FrameProperties* CacheHints::properties(Frame_t index) const {
    if(!index.valid() || _last_second.empty() || !_last_second.back() || index > _last_second.back()->frame) //|| (idx = size_t((current - index).get())) >= size())
        return nullptr;
    
    if(_last_second.back()->frame == index)
        return _last_second.back();
    
    auto it = find_in_sorted(_last_second, index);
    if(it == _last_second.end())
        return nullptr;
    else if((*it)->frame == index)
        return *it;
    
    return nullptr;
}

IndividualCache Individual::cache_for_frame(Frame_t frameIndex, double time, const CacheHints* hints) const {
    IndividualCache cache;
    cache._idx = Idx_t(identity().ID());
    if(empty()) {
        cache.individual_empty = true;
        return cache;
    }
    
    cache.individual_empty = false;
    
    if (frameIndex < _startFrame)
        throw U_EXCEPTION("Cant cache for frame before start frame");
        //return cache;//centroid(_startFrame)->pos(PX_AND_SECONDS);
    
    // find the first frame thats set for the individual
    auto it = iterator_for(frameIndex - 1_f);
    
    //! collect samples from previous segments
    bool manually_matched_segment = false;
    cache.last_frame_manual = false;
    cache.last_seen_px = Vec2(-FLT_MAX);
    cache.current_category = -1;
    cache.cm_per_pixel = FAST_SETTINGS(cm_per_pixel);
    cache.consistent_categories = FAST_SETTINGS(track_consistent_categories);
    cache.track_max_speed_px = FAST_SETTINGS(track_max_speed) / cache.cm_per_pixel;
    const auto frame_rate = FAST_SETTINGS(frame_rate);
    const auto track_max_reassign_time = FAST_SETTINGS(track_max_reassign_time);
    
    //auto segment = get_segment(frameIndex-1);
    if(it != _frame_segments.end()) {
        long_t bdx = -1;
        
        if((*it)->contains(frameIndex - 1_f)) {
            // is a valid segment
            if(is_manual_match((*it)->start()) && frameIndex - (*it)->start() < Frame_t(frame_rate * 0.05))
                manually_matched_segment = true;
            
            bdx = (*it)->basic_stuff(frameIndex - 1_f);
            assert(bdx != -1);
            
        } else if((*it)->end() < frameIndex) {
            bdx = (*it)->basic_stuff((*it)->end());
            
        } else if(it != _frame_segments.begin()) {
            auto copy = it;
            --copy;
            
            assert((*copy)->end() < frameIndex);
            bdx = (*copy)->basic_stuff((*copy)->end());
        }
        
        if(bdx != -1)
            cache.last_seen_px = _basic_stuff.at(bdx)->centroid.pos<Units::PX_AND_SECONDS>();
        
    } else if(!_frame_segments.empty()) {
        assert(frameIndex > (*_frame_segments.rbegin())->end());
        auto bdx = (*_frame_segments.rbegin())->basic_stuff((*_frame_segments.rbegin())->end());
        assert(bdx != -1);
        cache.last_seen_px = _basic_stuff.at(bdx)->centroid.pos<Units::PX_AND_SECONDS>();
    }
    
#ifndef NDEBUG
    if(cache.last_seen_px.x == -FLT_MAX)
        FormatWarning("No previous position for fish ", identity().ID()," in frame ",frameIndex,".");
#endif
    
    // find posture stuff and basic stuff for previous frame
    long_t bdx = -1, pdx = -1;
    if(!_frame_segments.empty()) {
        if(it != _frame_segments.end()
           && (*it)->contains(frameIndex - 1_f))
        {
            bdx = (*it)->basic_stuff(frameIndex - 1_f);
            pdx = (*it)->posture_stuff(frameIndex - 1_f);
            
        } else {
            if(it != _frame_segments.end() && (*it)->end() <= frameIndex - 1_f) {
                bdx = (*it)->basic_stuff((*it)->end());
                pdx = (*it)->posture_stuff((*it)->end());
            }
            else if(frameIndex <= _startFrame && _startFrame.valid()) {
                bdx = (*_frame_segments.begin())->basic_stuff(_startFrame);
                pdx = (*_frame_segments.begin())->posture_stuff(_startFrame);
            } else if(frameIndex >= _endFrame && _endFrame >= _startFrame && _endFrame.valid()) {
                bdx = (*_frame_segments.rbegin())->basic_stuff(_endFrame);
                pdx = (*_frame_segments.rbegin())->posture_stuff(_endFrame);
            } else
                print("Nothing to be found for ",frameIndex - 1_f);
        }
    }
    
    auto pp = bdx != -1 ? _basic_stuff.at(bdx).get() : nullptr;
    auto pp_posture = pdx != -1 ? _posture_stuff.at(pdx).get() : nullptr;
    
    /*auto _pp = find_frame(frameIndex-1);
    if(pp != _pp) {
        print("Frame ",frameIndex,", individual ",identity().ID(),": ",_pp ? _pp->frame : -1," != ",pp ? pp->frame : -1);
    }*/
    
    //auto pp = find_frame(frameIndex-1);
    //auto && [pp, pp_posture] = all_stuff(frameIndex - 1);
    //auto pp = find_frame(frameIndex-1);
    assert(pp); // fish is not empty, find_frame should at least return _startFrame
    
    //auto props = Tracker::properties(frameIndex);
    auto prev_props = Tracker::properties(frameIndex - 1_f, hints);
    if(!prev_props) {
        if(!Tracker::instance()->frames().empty()) {
            auto it = Tracker::instance()->frames().rbegin();
            while(it != Tracker::instance()->frames().rend() && (*it)->frame >= frameIndex)
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
    assert(!prev_props || prev_props->time != time);
    
    float ptime = pp_props ? pp_props->time : (- (frameIndex - cache.previous_frame).get() * 1 / double(frame_rate) + time);
    if(time - ptime >= track_max_reassign_time) {
        ptime = (- (frameIndex - cache.previous_frame).get() * 1 / double(frame_rate) + time);
    }
    //prev_props ? prev_props->time : ((frameIndex - (frameIndex - 1)) / double(FAST_SETTINGS(frame_rate)) + time);
    
    cache.tdelta = time - ptime;//pp.first < frameIndex ? (time - ptime) : time;
    cache.local_tdelta = prev_props ? time - prev_props->time : 0;
    
    auto raw = Vec2(0.0, 0.0);
    auto raw_acc = Vec2(0, 0);
    
    int used_frames = 0;
    //Median<size_t> size_median;
    float weights = 0;
    
    //! Collect recent number of valid samples within $t - \mathrm{fps} <= \dot{t} <= t$, where all distances between segments must not be reassigned ($\Delta t < fps * T_mathrm{max}$).
    size_t N = 0;
    if(it != _frame_segments.end()) {
        const auto lower_limit = max(0_f, frameIndex - Frame_t(frame_rate));
        auto previous_frame = frameIndex;
        const auto time_limit = Frame_t(frame_rate * track_max_reassign_time);
        
        while(true) {
            if((*it)->end() < lower_limit)
                break;
            
            if(previous_frame - (*it)->end() > time_limit)
            {
                break;
            }
            
            auto start = (*it)->start();
            if(start < lower_limit)
                start = lower_limit;
            
            previous_frame = start;
            
            N += max(0, ((*it)->end() - start).get() + 1);
            if(it == _frame_segments.begin())
                break;
            --it;
        }
    }
    
    
    //! retrieve a number (6) of samples from previous frames in order
    //! to predict the current direction etc.
    auto recent_number_samples = N;
    
    Range<Frame_t> range(max(_startFrame, cache.previous_frame - 6_f), cache.previous_frame);
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
    
    if(cache.consistent_categories) {
        std::shared_lock guard(Categorize::DataStore::range_mutex());
        iterate_frames(Range<Frame_t>(max(_startFrame, cache.previous_frame - Frame_t(frame_rate * 2)), cache.previous_frame), [&labels, &samples](auto frame, const auto&, auto basic, auto) -> bool
        {
            auto ldx = Categorize::DataStore::_ranged_label_unsafe(frame, basic->blob.blob_id());
            if(ldx != -1) {
                ++labels[ldx];
                ++samples;
            }
            return true;
        });
    }
#endif

    // cm/s / (cm/px)
    // (cm/s)^2 / (cm/px)^2 = (cm^2/s^2) / (cm^2/px^2) = 1 * px^2/s^2
    const auto track_max_px_sq = SQR(cache.track_max_speed_px);
    const FrameProperties *properties = nullptr;
    auto end = Tracker::instance()->frames().end();
    auto iterator = end;
    
    iterate_frames(range, [&](Frame_t frame, const std::shared_ptr<SegmentInformation> &, const Individual::BasicStuff* basic, auto) -> bool
    {
        if(is_manual_match(frame)) {
            cache.last_frame_manual = true;
            return true;
        }
        
        const FrameProperties* c_props = nullptr;
        if(iterator != end && ++iterator != end && (*iterator)->frame == frame) {
            c_props = (*iterator).get();
        } else {
            iterator = Tracker::properties_iterator(frame/*, hints*/);
            if(iterator != end)
                c_props = (*iterator).get();
        }
        
        auto &h = basic->centroid;
        if(!previous_p) {
            properties = c_props;
            
            previous_p = &h;
            previous_t = c_props ? c_props->time : 0;
            previous_f = frame;
            return true;
        }
        
        auto p_props = properties && properties->frame == frame - 1_f
                        ? properties
                        : Tracker::properties(frame - 1_f, hints);
        properties = c_props;
        
        if (c_props && p_props && previous_p) {//(he || h)) {
            double tdelta = c_props->time - p_props->time;
            
            if(tdelta > prob_t(1))
                return true;
            
            //! \mathbf{v}_i(t) = \mathbf{p}_i'(t) = \frac{\delta}{\delta t} \mathbf{p}_i(t)
            auto v = (h.pos<Units::PX_AND_SECONDS>() - previous_p->pos<Units::PX_AND_SECONDS>()) / (c_props->time - previous_t);
            auto L_sq = v.sqlength();
            
            //! \hat{\mathbf{v}}_i(t) =
            //!     \mathbf{v}_i(t) *
            //!     \begin{cases}
            //!         1                                       & \mathrm{if} \norm{\mathbf{v}_i(t)} \le D_\mathrm{max} \\
            //!         D_\mathrm{max} / \norm{\mathbf{v}_i(t)} & \mathrm{otherwise}
            //!     \end{cases}
            if(L_sq >= track_max_px_sq) {
                v *= cache.track_max_speed_px / sqrt(L_sq);
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
            previous_t = c_props->time;
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
    
    cache.current_category = int(mid);
#else
    cache.current_category = -1;
#endif
    
    const MotionRecord* c = pp ? &pp->centroid : nullptr; //centroid_weighted(cache.previous_frame);
    
    //! \mean{s}_{i}(t) = \underset{k \in [F(\tau)-5, F(t)]}{\median} \norm{\hat{\mathbf{v}}_i(\Tau(k))}
    prob_t speed = max(0.6f, sqrt(used_frames ? static_median(average_speed.begin(), average_speed.end()) : 0));
    
    if(cache.tdelta == 0)
        throw U_EXCEPTION("No time difference between ",frameIndex," and ",cache.previous_frame," in calculate_next_positions.");
    
    //! \lambda
    const float lambda = SQR(SQR(max(0, min(1, FAST_SETTINGS(track_speed_decay)))));
    
    //! \mean{\mathbf{d}_i}(t)
    Vec2 direction;
    NAN_SAFE_NORMALIZE(raw, direction)
    
    Vec2 est;
    prob_t last_used = ptime;
    auto pprops = Tracker::properties(cache.previous_frame - 1_f, hints);
    if(pprops)
        last_used = pprops->time;
    
    NAN_SAFE_NORMALIZE(raw_acc, raw_acc)
    
    if(used_frames > 0 && lambda < 1) {
        for (auto f = cache.previous_frame; f < frameIndex; ++f) {
            auto props = Tracker::properties(f, hints);
            if(props) {
                //! \Tau'(k)
                prob_t tdelta = props->time - last_used;
                last_used = props->time;
                
                //! w(f) = \frac{1 + \lambda^4}{1 + \lambda^4 \max\left\{ 1, f - F(\tau_i) + 1 \right\}}
                prob_t weight = (1 + lambda) / (1 + lambda * max(1_f, f - cache.previous_frame + 1_f).get());
                //if(weight <= 0.0001)
                //    break;
                
                //! \dot{\mathbf{p}}_i(t) = s_i(t) \sum_{k\in [F(\tau_i), F(t)-1]} w(k) \left(\mean{\mathbf{d}_i}(t) + \Tau'(k) * \mean{\mathbf{a}}_i(t) \right)
                est += weight * tdelta * (speed * (direction + tdelta * raw_acc));
                weights += weight;
            }
        }
    }
    
    if(c)
        est += c->pos<Units::PX_AND_SECONDS>();
    
    auto h = c;
    if(FAST_SETTINGS(calculate_posture)) {
        if(pp_posture && pp_posture->centroid_posture)
            h = pp_posture->centroid_posture;
    }
    
    cache.speed = h ? h->speed<Units::CM_AND_SECONDS>() : 0;
    cache.h = h;
    cache.estimated_px = est;
    cache.time_probability = time_probability(cache, recent_number_samples);
    
    assert(!std::isnan(cache.estimated_px.x) && !std::isnan(cache.estimated_px.y));
    cache.valid = true;
    
    return cache;
}

prob_t Individual::time_probability(const IndividualCache& cache, size_t recent_number_samples) const {
    if(!FAST_SETTINGS(track_time_probability_enabled))
        return 1;
    if (cache.tdelta > FAST_SETTINGS(track_max_reassign_time))
        return 0.0;
    
    if(cache.last_frame_manual)
        return 1;
    
    const float Tdelta = 1.f / float(FAST_SETTINGS(frame_rate));
    
    // make sure that very low frame rates work
    //! F_\mathrm{min} = \min\left\{\frac{1}{T_\Delta}, 5\right\}
    const float minimum_frames = (float)min(FAST_SETTINGS(frame_rate), 5);
    
    //! R_i(t) = \norm{ \givenset[\Big]{ \Tau(k) | F(t) - T_\Delta^{-1} \leq k \leq t \wedge \Tau(k) - \Tau(k-1) \leq T_\mathrm{max}} }
    
    /**
        \begin{equation} \label{eq:time_prob}
        T_i(t) = \left(1 - \min\left\{ 1, \frac{\max\left\{ 0, \tau_i - t - T_\Delta \right\}} {T_\mathrm{max}} \right\}\right) * \begin{cases}
                \min\left\{ 1, \frac{R_i(\tau_i) - 1}{F_\mathrm{min}} + P_\mathrm{min} \right\} & F(\tau_i) \geq F(t_0) + F_\mathrm{min}\\
                1 & \mathrm{otherwise}
            \end{cases}
        \end{equation}
     */
    
    float p = 1.0f - min(1.0f, max(0, (cache.tdelta - Tdelta) / FAST_SETTINGS(track_max_reassign_time)));
    if(cache.previous_frame >= Tracker::start_frame() + Frame_t(minimum_frames))
        p *= min(1.f, float(recent_number_samples - 1) / minimum_frames + FAST_SETTINGS(matching_probability_threshold));
    
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

prob_t Individual::position_probability(const IndividualCache& cache, Frame_t frameIndex, size_t, const Vec2& position, const Vec2& blob_center) const
{
    UNUSED(frameIndex)
#ifndef NDEBUG
    if (frameIndex <= _startFrame)
        throw U_EXCEPTION("Cannot calculate probability for a frame thats previous to all known frames.");
#endif
    
    // $\tau$ is the time (s) of the most recent frame assigned to individual $i$
    // $\dot{p}_i(t)$ is the projected position for individual $i$ in the current frame
    
    //! S_{i,b}(t) &= \left(1 + \frac{\norm{ (\mathbf{p}_b(\tau_i) - \dot{\mathbf{p}}_i(t)) / (\tau_i - t) }}{ D_{\mathrm{max}}}\right)^{-2}
    
    Vec2 velocity;
    if(cache.local_tdelta != 0)
        velocity = (position - cache.estimated_px) / cache.local_tdelta;
    assert(!std::isnan(velocity.x) && !std::isnan(velocity.y));
    
    auto speed = velocity.length() / cache.track_max_speed_px;
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
    if(cache.h && !cache.last_frame_manual) {
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
    }
    
    return speed;
}

/*prob_t Individual::size_probability(const IndividualCache& cache, long_t, size_t num_pixels) const {
    if(cache.size_average <= 0)
        return 1.f;
    
    return max(0.5, 1 - 0.25 * (SQR(cmn::abs(min(2, num_pixels / cache.size_average) - 1))));
}*/

Individual::Probability Individual::probability(int label, const IndividualCache& cache, Frame_t frameIndex, const pv::CompressedBlob& blob) const {
    auto bounds = blob.calculate_bounds();
    return probability(label, cache, frameIndex, bounds.pos() + bounds.size() * 0.5, blob.num_pixels());
}

Individual::Probability Individual::probability(int label, const IndividualCache& cache, Frame_t frameIndex, const pv::BlobPtr& blob) const {
    return probability(label, cache, frameIndex, blob->bounds().pos() + blob->bounds().size() * 0.5, blob->num_pixels());
}

Individual::Probability Individual::probability(int label, const IndividualCache& cache, Frame_t frameIndex, const Vec2& position, size_t pixels) const {
    assert(frameIndex >= _startFrame);
    //if (frameIndex < _startFrame)
    //    throw U_EXCEPTION("Cannot calculate probability for a frame thats previous to all known frames.");
    assert(!cache.individual_empty);

    if (cache.consistent_categories && cache.current_category != -1) {
        //auto l = Categorize::DataStore::ranged_label(Frame_t(frameIndex), blob);
        //if(identity().ID() == 38)
        //    FormatWarning("Frame ",frameIndex,": blob ",blob.blob_id()," -> ",l ? l->name.c_str() : "N/A"," (",l ? l->id : -1,") and previous is ",cache.current_category);
        if (label != -1) {
            if (label != cache.current_category) {
                //if(identity().ID() == 38)
                 //   FormatWarning("Frame ", frameIndex,": current category does not match for blob ",blob.blob_id());
                //return Probability{ 0, 0, 0, 0 };
                return 0;
            }
        }
    }

    const Vec2& blob_pos = position;
    //auto && [ p_position, p_speed, p_angle ] = 
    auto p_position =    position_probability(cache, frameIndex, pixels, blob_pos, position);
    
    /**
         \begin{equation} \label{eq:combined_prob}
            P_{i} \given[\big]{t,\tau_i | B_j } =  S_{i} \given*{t | B_j} * \left(1 - \omega_1 \left(1 + A_{i} \given*{t,\tau_i | B_j } \right) \right) * \left(1 - \omega_2 \left( 1 +  T_{i}(t,\tau_i) \right) \right)
         \end{equation}
     */
    //return {
    return (cache.last_frame_manual ? 1.0f : 1.0f) * p_position * cache.time_probability;
    //    cache.time_probability,
    //    p_position,
    //    p_angle
    //};
}

const std::unique_ptr<Individual::BasicStuff>& Individual::find_frame(Frame_t frameIndex) const
{
    if(!empty()) {
        if(frameIndex <= _startFrame)
            return _basic_stuff.front();
        if(frameIndex >= _endFrame)
            return _basic_stuff.back();
        
        auto end = _frame_segments.end();
        auto it = std::lower_bound(_frame_segments.begin(), end, frameIndex, [](const auto& ptr, Frame_t frame){
            return ptr->start() < frame;
        });
        
        if(it == end) { // we are out of range, return last
            auto idx = _frame_segments.back()->basic_stuff(frameIndex);
            if(idx != -1)
                return _basic_stuff[ idx ];
            else
                return _basic_stuff.back();
        }
        
        int32_t index = (int32_t)_basic_stuff.size()-1;
        if((*it)->start() > frameIndex) {
            if(it != _frame_segments.begin()) {
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
                throw U_EXCEPTION("(",identity().ID(),") frame ",frameIndex,": cannot find basic_stuff after finding segment ",(*it)->start(),"-",(*it)->end(),"");
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
        return _basic_stuff[ index ];
    }
    
    throw U_EXCEPTION("Cannot find a frame for ",frameIndex,"");
}

std::tuple<std::vector<std::tuple<float, float>>, std::vector<float>, size_t, MovementInformation> Individual::calculate_previous_vector(Frame_t frameIndex) const {
    const auto min_samples = Frame_t(FAST_SETTINGS(posture_direction_smoothing));
    std::vector<float> tmp;
    std::vector<std::tuple<float, float>> samples;
    MovementInformation movement;
    std::vector<float> hist;
    float space_width = M_PI * 2;
    float bin_size = RADIANS(5);
    hist.resize(space_width / bin_size);
    
    if(!centroid(frameIndex)) {
        return {samples, hist, 0, movement};
    }
    
    std::vector<Frame_t> all_frames;
    std::vector<float> all_angles;
    std::vector<Vec2> all_head_positions;
    float position_sum = 0;
    float position_samples = 0;
    
    Range<Frame_t> range(max(start_frame(), frameIndex - min_samples), min(end_frame(), frameIndex + min_samples));
    
    iterate_frames(range, [&](Frame_t frame, const auto&, auto basic, auto posture) -> bool
    {
        if(frame == range.start) {
            movement.directions.push_back(Vec2(0));
            return true;
        }
        
        if(posture && posture->midline_length != PostureStuff::infinity) {
            assert(posture->posture_original_angle != PostureStuff::infinity);
            all_head_positions.push_back(-Vec2(cos(posture->midline_angle), sin(posture->midline_angle)) * posture->midline_length * 0.5);
            //auto post = head(it->first);
            //all_head_positions.push_back(post->pos(PX_AND_SECONDS, true) - centroid(it->first)->pos(PX_AND_SECONDS, true));
            all_angles.push_back(posture->posture_original_angle);
            
            movement.directions.push_back(Vec2(cos(posture->posture_original_angle), sin(posture->posture_original_angle)).normalize());
            all_frames.push_back(basic->frame);
            
            position_sum += posture->midline_length * 0.5;//all_head_positions.back().length();
            ++position_samples;
        }
        
        return true;
    });
    
    position_sum /= position_samples;
    
    //auto str = Meta::toStr(all_head_positions);
    //for(size_t i=0; i<all_head_positions.size(); ++i) {
    //}
    
    Vec2 last_head(gui::Graph::invalid());
    if(!all_head_positions.empty()) {
        last_head = all_head_positions.front();
    }
    
    if(frameIndex > start_frame()) {
        auto props = Tracker::properties(frameIndex);
        auto cache = cache_for_frame(frameIndex, props->time);
        movement.position = cache.estimated_px;
        movement.velocity = Vec2(position_sum);
    }
    //movement.position = last_head;
    //movement.velocity = centroid(frameIndex)->v(PX_AND_SECONDS, true);
    
    
    for(size_t i=0; i<all_angles.size(); ++i) {
        auto angle = all_angles.at(i) + bin_size * 0.5;
        //assert(it->second >= -M_PI && it->second <= M_PI);
        
        while(angle >= space_width) {
            angle -= space_width;
        }
        while (angle < 0) {
            angle += space_width;
        }
        
        //float w = (1 - float(all_frames.at(i)) / min_samples) * 0.9 + 0.1;
        //w *= w;
        float w = 1;
        
        size_t bin = angle / space_width * hist.size();
        hist.at(bin) += w;
        
        samples.push_back(std::tuple<float, float>{(float)w, (float)all_angles.at(i)});
    }
    
    Outline::smooth_array(hist, tmp);
    Outline::smooth_array(tmp, hist);
    Outline::smooth_array(hist, tmp);
    Outline::smooth_array(tmp, hist);
    
    //auto str = Meta::toStr(all_angles);
    
    Vec2 previous_direction;
    
    auto && [maptr, miptr] = periodic::find_peaks(std::make_shared<std::vector<Float2_t>>(hist.begin(), hist.end()), 0, {}, periodic::PeakMode::FIND_POINTY);
    float max_len = 0;
    Vec2 max_index;
    float angle = -1;
    float m_hist = 0;
    size_t idx = 0;
    for(auto &maximum : *maptr) {
        auto len = abs(maximum.integral);
        if(len > max_len) { //maximum.range.length() > max_len) {
            max_len = len;
            max_index = maximum.position;
            m_hist = maximum.position.y;
            angle = (maximum.position.x + 0.5) * bin_size;
            idx = maximum.position.x;
        }
    }
    
    /*Vec2 result;
    if(angle != -1) {
        float samples = 0;
        for(auto &maximum : *maptr) {
            float angle = (maximum.position.x + 0.5) * bin_size;
            float w = (abs(maximum.integral) / max_len);
            result += Vec2(cos(angle), sin(angle)) * w;
            samples += w;
            //middle += maximum.position * (abs(maximum.integral) / max_len);
        }
    
        if(samples > 0) {
            result /= samples;
            previous_direction = result;
        }
        //angle = atan2(result);
    }*/
    
    /*float m_hist = 0;
    size_t idx = 0;
    float angle = -1;
    for (size_t i=0; i<hist.size(); ++i) {
        if(hist.at(i) > m_hist) {
            m_hist = hist.at(i);
            idx = i;
            angle = (i + 0.5) * bin_size;
        }
    }*/
    
    if(angle != -1 && m_hist > 0) {
        previous_direction = Vec2(cos(angle), sin(angle));
    } else previous_direction = Vec2(0);
    
    movement.direction = previous_direction.normalize();
    //movement.direction = position_sum.normalize();
    
    return {samples, hist, idx, movement};
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

void Individual::save_posture(const BasicStuff& stuff, Frame_t frameIndex) {//Image::Ptr greyscale) {
    /*auto c = centroid(frameIndex);
    auto direction = c->v();
    direction /= ::length(direction);*/
    
    assert(stuff.pixels);
    Posture ptr(frameIndex, identity().ID());
    ptr.calculate_posture(frameIndex, stuff.pixels);
    //ptr.calculate_posture(frameIndex, greyscale->get(), previous_direction);
    
    if(ptr.outline_empty() /*|| !ptr.normalized_midline()*/) {
        return;
    }
    
    /*std::pair<int64_t, long_t> gui_show_fish = SETTING(gui_show_fish);
    if(gui_show_fish.first == identity().ID() && frameIndex == gui_show_fish.second)
    {
        auto && [pos, greyscale] = blob->difference_image(*Tracker::instance()->background(), 0);
        auto mat = greyscale->get();
        
        print("Frame ", frameIndex);
        
        DebugDrawing draw(Vec2(), Vec2(), "draw_debug", int(max(1.f, 500.f/greyscale->cols)), greyscale->cols,greyscale->rows);
        draw.paint(ptr, mat);
        
        cv::putText(mat, std::to_string(frameIndex), cv::Point(10, 10), cv::FONT_HERSHEY_PLAIN, 0.8, cv::Scalar(255));
        tf::imshow("greyscale", mat);
        SETTING(analysis_paused) = true;
    }*/
    
	if(!ptr.outline_empty()) {
        const auto &midline = ptr.normalized_midline();
		/*if(midline && midline->size() != FAST_SETTINGS(midline_resolution)) {
            FormatWarning("Posture error (",midline->size()," segments) in ",_identity.ID()," at frame ",frameIndex,".");
		}*/
        
        auto segment = segment_for(frameIndex);
        if(!segment)
            throw U_EXCEPTION("save_posture cannot find frame ",frameIndex,".");
        if(!segment->contains(frameIndex))
            throw U_EXCEPTION("save_posture found segment (",segment->start(),"-",segment->end(),"), but does not contain ",frameIndex,".");
        
        auto stuff = std::make_unique<PostureStuff>();
        stuff->frame = frameIndex;
        
        if(!ptr.outline_empty())
            stuff->outline = std::make_shared<MinimalOutline>(ptr.outline());
        
        if(midline && !midline->empty()) {
            //if(!FAST_SETTINGS(midline_samples) || _midline_length.added() < FAST_SETTINGS(midline_samples))
            //    _midline_length.addNumber(midline->len());
            
            //if(!FAST_SETTINGS(midline_samples) || _outline_size.added() < FAST_SETTINGS(midline_samples))
            //    _outline_size.addNumber(_outlines[frameIndex]->size());
            
            //auto copy = std::make_shared<Midline>(*midline);
            stuff->cached_pp_midline = midline;

            if (stuff && stuff->midline_length != PostureStuff::infinity) {
                if (stuff->posture_original_angle == PostureStuff::infinity && stuff->cached_pp_midline)
                    stuff->posture_original_angle = stuff->cached_pp_midline->original_angle();
            }
        }
        
        segment->add_posture_at(std::move(stuff), this);
        update_midlines(nullptr);
	}
}

Vec2 Individual::weighted_centroid(const Blob& blob, const std::vector<uchar>& pixels) {
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

std::tuple<Image::UPtr, Vec2> normalize_image(const cv::Mat& mask, const cv::Mat& image, const gui::Transform &midline_transform, const pv::BlobPtr& blob, float midline_length, const Size2 &output_size, bool use_legacy) {
    cv::Mat padded;
    
    if(midline_length < 0) {
        static Timer timer;
        if(timer.elapsed() > 1) { // dont spam messages
            FormatWarning("[Individual::calculate_normalized_diff_image] invalid midline_length");
            timer.reset();
        }
        return {nullptr, Vec2()};
    }
        //throw std::invalid_argument("[Individual::calculate_normalized_diff_image] invalid midline_length");
    
    
    if(!output_size.empty())
        padded = cv::Mat::zeros(output_size.height, output_size.width, CV_8UC1);
    else
        image.copyTo(padded);
    assert(padded.isContinuous());
    
    auto size = Size2(padded.size());
    auto scale = FAST_SETTINGS(recognition_image_scale);
    //Vec2 pos = size * 0.5 + Vec2(midline_length * 0.4);
    
    gui::Transform tr;
    if(use_legacy) {
        tr.translate(size * 0.5);
        tr.scale(Vec2(scale));
        tr.translate(Vec2(-midline_length * 0.5, 0));
        
    } else {
        tr.translate(size * 0.5);
        tr.scale(Vec2(scale));
        tr.translate(Vec2(midline_length * 0.4));
    }
    tr.combine(midline_transform);
    
    auto t = tr.toCV();
    
    image.copyTo(image, mask);
    //tf::imshow("before", image);
    
    cv::warpAffine(image, padded, t, (cv::Size)size, cv::INTER_LINEAR, cv::BORDER_CONSTANT);
    //resize_image(padded, SETTING(recognition_image_scale).value<float>());
    
    //tf::imshow("after", padded);
    int left = 0, right = 0, top = 0, bottom = 0;
    
    if(!output_size.empty()) {
        if(padded.cols < output_size.width) {
            left = roundf(output_size.width - padded.cols);
            right = left / 2;
            left -= right;
        }
        
        if(padded.rows < output_size.height) {
            top = roundf(output_size.height - padded.rows);
            bottom = top / 2;
            top -= bottom;
        }
        
        if(left || right || top || bottom)
            cv::copyMakeBorder(padded, padded, top, bottom, left, right, cv::BORDER_CONSTANT, 0);
        
        assert(padded.cols >= output_size.width && padded.rows >= output_size.height);
        if(padded.cols > output_size.width || padded.rows > output_size.height) {
            left = padded.cols - output_size.width;
            right = left / 2;
            left -= right;
            
            top = padded.rows - output_size.height;
            bottom = top / 2;
            top -= bottom;
            
            padded(Bounds(left, top, padded.cols - left - right, padded.rows - top - bottom)).copyTo(padded);
        }
    }
    
    if(!output_size.empty() && (padded.cols != output_size.width || padded.rows != output_size.height))
        throw U_EXCEPTION("Padded size differs from expected size (",padded.cols,"x",padded.rows," != ",output_size.width,"x",output_size.height,")");
    
    auto i = tr.getInverse();
    auto pt = i.transformPoint(left, top);
    return { Image::Make(padded), pt };
}

std::tuple<Image::UPtr, Vec2> Individual::calculate_normalized_image(const gui::Transform &midline_transform, const pv::BlobPtr& blob, float midline_length, const Size2 &output_size, bool use_legacy) {
    cv::Mat mask, image;
    if(!blob->pixels())
        throw std::invalid_argument("[Individual::calculate_normalized_diff_image] The blob has to contain pixels.");
    imageFromLines(blob->hor_lines(), &mask, &image, NULL, blob->pixels().get(), 0, &Tracker::average(), 0);
    
    return normalize_image(mask, image, midline_transform, blob, midline_length, output_size, use_legacy);
}

std::tuple<Image::UPtr, Vec2> Individual::calculate_normalized_diff_image(const gui::Transform &midline_transform, const pv::BlobPtr& blob, float midline_length, const Size2 &output_size, bool use_legacy) {
    cv::Mat mask, image;
    if(!blob->pixels())
        throw std::invalid_argument("[Individual::calculate_normalized_diff_image] The blob has to contain pixels.");
    imageFromLines(blob->hor_lines(), &mask, NULL, &image, blob->pixels().get(), 0, &Tracker::average(), 0);
    
    return normalize_image(mask, image, midline_transform, blob, midline_length, output_size, use_legacy);
}

std::tuple<Image::UPtr, Vec2> Individual::calculate_diff_image(pv::BlobPtr blob, const Size2& output_size) {
    cv::Mat mask, image;
    cv::Mat padded;
    
    if(!blob->pixels())
        throw std::invalid_argument("[Individual::calculate_diff_image] The blob has to contain pixels.");
    imageFromLines(blob->hor_lines(), &mask, NULL, &image, blob->pixels().get(), 0, &Tracker::average(), 0);
    image.copyTo(padded, mask);
    
    auto scale = FAST_SETTINGS(recognition_image_scale);
    if(scale != 1)
        resize_image(padded, scale);
    
    Bounds bounds(blob->bounds().pos(), blob->bounds().size() + blob->bounds().pos());
    
    if(!output_size.empty()) {
        int left = 0, right = 0, top = 0, bottom = 0;
        if(padded.cols < output_size.width) {
            left = roundf(output_size.width - padded.cols);
            right = left / 2;
            left -= right;
        }
        
        if(padded.rows < output_size.height) {
            top = roundf(output_size.height - padded.rows);
            bottom = top / 2;
            top -= bottom;
        }
        
        if(left || right || top || bottom) {
            bounds.x -= left;
            bounds.y -= top;
            bounds.width += right;
            bounds.height += bottom;
            
            cv::copyMakeBorder(padded, padded, top, bottom, left, right, cv::BORDER_CONSTANT, 0);
        }
        
        bounds << Size2(bounds.size() - bounds.pos());
        
        assert(padded.cols >= output_size.width && padded.rows >= output_size.height);
        if(padded.cols > output_size.width || padded.rows > output_size.height) {
            left = padded.cols - output_size.width;
            right = left / 2;
            left -= right;
            
            top = padded.rows - output_size.height;
            bottom = top / 2;
            top -= bottom;
            
            Bounds cut(left, top, padded.cols - left - right, padded.rows - top - bottom);
            
            bounds.x += cut.x;
            bounds.y += cut.y;
            bounds.width = cut.width;
            bounds.height = cut.height;
            
            padded(cut).copyTo(padded);
        }
    }
    
    if(!output_size.empty() && (padded.cols != output_size.width || padded.rows != output_size.height))
        throw U_EXCEPTION("Padded size differs from expected size (",padded.cols,"x",padded.rows," != ",output_size.width,"x",output_size.height,")");
    
    
    return { Image::Make(padded), bounds.pos() };
}

bool Individual::evaluate_fitness() const {
	if(frame_count() <= 25)
		return false;
		
	// check posture
    if(FAST_SETTINGS(calculate_posture)) {
		for(auto i = start_frame(); i < end_frame(); ++i) {
            const auto m = midline(i);
			if(m && m->size() != FAST_SETTINGS(midline_resolution))
				return false;
		}
	}
	
	return true;
}

/*void Individual::recognition_segment(Frame_t frame, const std::tuple<size_t, std::map<long_t, float>>& map) {
    average_recognition_segment[frame] = map;
}*/

void Individual::clear_recognition() {
    average_recognition_segment.clear();
    average_processed_segment.clear();
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

std::map<Frame_t, FrameRange> split_segment_by_probability(const Individual* fish, const Individual::SegmentInformation& segment)
{
    if (!Tracker::recognition() || Tracker::recognition()->data().empty())
        return {};

    auto for_frame = [fish](Frame_t frame) -> std::tuple<long_t, float> {
        std::map<long_t, std::tuple<long_t, float>> samples;
        
        auto blob = fish->compressed_blob(frame);
        if(!blob)
            return {-1.f, 0.f};
        
        auto raw = Tracker::recognition()->ps_raw(frame, blob->blob_id());
        float max_id = -1;
        float max_p = 0;
        
        if(!raw.empty()) {
            for (auto && [fdx, p] : raw) {
                if(p > max_p) {
                    max_p = p;
                    max_id = fdx;
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
    
    const size_t N = FAST_SETTINGS(frame_rate) * 2;
    const Frame_t min_samples { Frame_t(N * 0.5) };
    
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

void Individual::calculate_average_recognition() {
    _average_recognition_samples = 0;
    _average_recognition.clear();
    
    std::map<Idx_t, size_t> samples;
    const float frame_limit = FAST_SETTINGS(frame_rate) * 2;
    
    for(auto & segment : _frame_segments) {
        auto && [n, vector] = average_recognition(segment->start());
        _average_recognition_samples += n;
        
        for(auto && [fdx, p] : vector) {
            _average_recognition[fdx] += p * n;
            samples[fdx] += n;
        }
    }
    
    std::map<FrameRange, std::set<long_t>> splits;
#ifndef NDEBUG
    std::string file = identity().name()+".log";
    FILE* f = fopen(file.c_str(), "wb");
    if(!f)
       FormatError("Cannot open file ", file," for writing ('",identity(),"')");
#else
    FILE* f = nullptr;
#endif
    
    // anything thats below 2 seconds + at least 10% more with a different id, is considered lame and unimportant
    std::map<Frame_t, FrameRange> processed_segments;
    
    for(auto & segment : _frame_segments) {
        if(segment->length() < frame_limit) {
            processed_segments[segment->start()] = *segment;
            log(f, "Segment %d-%d shorter than %f", segment->start(), segment->end(), frame_limit);
            continue;
        }
        
        log(f, "Checking segment %d-%d (L=%d)", segment->start(), segment->end(), segment->length());
        auto split_up = split_segment_by_probability(this, *segment);
        
        if(split_up.empty()) {
            processed_segments[segment->start()] = *segment;
        } else {
            assert(split_up.begin()->second.start() == segment->start());
            assert(split_up.rbegin()->second.end() == segment->end());
            
            auto prev_end = segment->start() - 1_f;
            for(auto && [start, range] : split_up) {
                assert(start == range.start() && prev_end + 1_f == start);
                prev_end = range.end();
                processed_segments[start] = range;
                
                FOI::add(FOI(start, {FOI::fdx_t(identity().ID())}, "split_up"));
            }
        }
    }
    
    if(f)
        fclose(f);
    
    if(!splits.empty()) {
        auto str = Meta::toStr(splits);
        FormatWarning("Found frame segments for fish ", identity().ID()," that have to be split:\n",str);
    }
    
    _recognition_segments = processed_segments;
    
    for(auto && [fdx, n] : samples) {
        if(n > 0)
            _average_recognition[fdx] /= float(n);
    }
}

std::tuple<bool, FrameRange> Individual::frame_has_segment_recognition(Frame_t frameIndex) const {
    if(frameIndex > _endFrame || frameIndex < _startFrame)
        return {false, FrameRange()};
    
    auto range = get_segment(frameIndex);
    auto & segment = range.range;
    return { segment.contains(frameIndex) && average_recognition_segment.find(segment.start) != average_recognition_segment.end(), range };
}

std::tuple<bool, FrameRange> Individual::has_processed_segment(Frame_t frameIndex) const {
    if(frameIndex > _endFrame || frameIndex < _startFrame)
        return {false, FrameRange()};
    
    auto range = get_recognition_segment(frameIndex);
    auto & segment = range.range;
    return { segment.contains(frameIndex) && average_processed_segment.find(segment.start) != average_processed_segment.end(), range };
}

/*const decltype(Individual::average_recognition_segment)::mapped_type& Individual::average_recognition(long_t segment_start) const {
    return average_recognition_segment.at(segment_start);
}*/

const decltype(Individual::average_recognition_segment)::mapped_type Individual::processed_recognition(Frame_t segment_start) {
    auto it = average_processed_segment.find(segment_start);
    if(it == average_processed_segment.end()) {
        // average cannot be found for given segment. try to calculate it...
        auto sit = _recognition_segments.find(segment_start);
        if(sit == _recognition_segments.end())
            throw U_EXCEPTION("Cannot find segment starting at ",segment_start," for fish ",identity().raw_name(),".");
        
        const auto &[ segment, usable] = sit->second;
        
        if(segment.end >= _endFrame && Tracker::instance()->end_frame() != Tracker::analysis_range().end) {
            return {0, {}};
        }
        
        std::map<Idx_t, std::tuple<long_t, float>> samples;
        size_t overall = 0;

        if (!Tracker::recognition()->data().empty()) {
            for (auto i = segment.start; i < segment.end; ++i) {
                auto blob = this->blob(i);
                if (!blob)
                    continue;

                auto raw = Tracker::recognition()->ps_raw(i, blob->blob_id());
                if (!raw.empty()) {
                    ++overall;

                    for (auto&& [fdx, p] : raw) {
                        ++std::get<0>(samples[Idx_t(fdx)]);
                        std::get<1>(samples[Idx_t(fdx)]) += p;
                    }
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
                average_processed_segment[segment_start] = {overall, average};
            else {
                print("Not using fish ",identity().ID()," segment ",segment_start,"-",segment.end," because sum is ",s);
                return {0, {}};
            }
        } else
            return {0, {}};
    }
    
    return average_processed_segment.at(segment_start);
}

const decltype(Individual::average_recognition_segment)::mapped_type Individual::average_recognition(Frame_t segment_start) {
    auto it = average_recognition_segment.find(segment_start);
    if(it == average_recognition_segment.end()) {
        if (Tracker::recognition()->data().empty())
            return { 0,{} };

        // average cannot be found for given segment. try to calculate it...
        auto sit = std::upper_bound(_frame_segments.begin(), _frame_segments.end(), segment_start, [](Frame_t frame, const auto& ptr) {
            return frame < ptr->start();
        });
        if((sit == _frame_segments.end() || sit != _frame_segments.begin()) && (*(--sit))->start() == segment_start)
        {
            //! found the segment
        } else
            throw U_EXCEPTION("Cannot find segment starting at ",segment_start," for fish ",identity().raw_name(),".");
        
        const auto && [segment, usable] = (FrameRange)*sit->get();
        
        if(segment.end >= _endFrame && Tracker::instance()->end_frame() + 1_f != Frame_t(narrow_cast<Frame_t::number_t>(SETTING(video_length).value<uint64_t>()))) {
            return {0, {}};
        }
        
        std::map<Idx_t, std::tuple<long_t, float>> samples;
        size_t overall = 0;
        
        for(auto i = segment.start; i < segment.end; ++i) {
            auto blob = this->blob(i);
            if(!blob)
                continue;
            
            auto raw = Tracker::recognition()->ps_raw(i, blob->blob_id());
            if(!raw.empty()) {
                ++overall;
                
                for (auto && [fdx, p] : raw) {
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
                average_recognition_segment[segment_start] = {overall, average};
            else {
                print("Not using fish ",identity().ID()," segment ",segment_start,"-",segment.end," because sum is ",s);
                return {0, {}};
            }
        } else
            return {0, {}};
    }
    
    return average_recognition_segment.at(segment_start);
}

std::tuple<size_t, Idx_t, float> Individual::average_recognition_identity(Frame_t segment_start) const {
    auto it = average_recognition_segment.find(segment_start);
    if(it == average_recognition_segment.end()) {
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

void Individual::save_visual_field(const file::Path& path, Range<Frame_t> range, const std::function<void(float, const std::string&)>& update, bool blocking) {
    if(range.empty())
        range = Range<Frame_t>(_startFrame, _endFrame);
    
    
    std::vector<float> depth, body_part;
    std::vector<long_t> ids;
    std::vector<Vec2> fish_pos, eye_pos;
    std::vector<float> fish_angle, eye_angle;
    std::vector<Frame_t::number_t> frames;
    
    size_t len = 0;

    iterate_frames(range, [&](Frame_t, const std::shared_ptr<SegmentInformation>&, auto, auto posture) -> bool
    {
        if (!posture || !posture->head)
            return true;
        ++len;
        return true;
    });

    print("Saving to ",path," (",len," frames in range ",range.start,"-",range.end,")");

    size_t vres = VisualField::field_resolution * VisualField::layers;
    size_t eye_len = len * vres;
    
    frames.reserve(len);
    depth.reserve(eye_len * 2);
    body_part.reserve(depth.size());
    ids.reserve(depth.size());
    
    fish_pos.reserve(len);
    fish_angle.reserve(len);
    
    eye_angle.reserve(len * 2);
    eye_pos.reserve(len * 2);
    
    std::shared_ptr<Tracker::LockGuard> guard;
    
    iterate_frames(range, [&](Frame_t frame, const std::shared_ptr<SegmentInformation> &, auto basic, auto posture) -> bool
    {
        if(blocking)
            guard = std::make_shared<Tracker::LockGuard>("new VisualField");
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
            
            fish_pos.push_back(ptr->fish_pos());
            fish_angle.push_back(ptr->fish_angle());
            
            for(long_t j=0; j<2; j++) {
                auto &e = ptr->eyes()[j];
                
                eye_angle.push_back(e.angle);
                eye_pos.push_back(e.pos);
                
                depth.insert(depth.end(), e._depth.begin(), e._depth.begin() + vres);
                body_part.insert(body_part.end(), e._visible_head_distance.begin(), e._visible_head_distance.begin() + vres);
                ids.insert(ids.end(), e._visible_ids.begin(), e._visible_ids.begin() + vres);
            }
            
            if(owned)
                delete ptr;
        }
        
        if(frame.get() % 1000 == 0) {
            update((frame - range.start).get() / (float)(range.end - range.start).get() * 0.5, "");
            print(frame," / ",range.end);
        }
        
        return true;
    });
    
    print("Saving depth...");
    FileSize fileSize(depth.size() * sizeof(decltype(depth)::value_type)
                      + ids.size() * sizeof(decltype(ids)::value_type)
                      + body_part.size() * sizeof(decltype(body_part)::value_type));
    update(0.5, "writing files ("+Meta::toStr(fileSize)+")");

    bool use_npz = fileSize.bytes < 1.5 * 1000 * 1000 * 1000;
    if (!use_npz) {
        Timer save_timer;
        cnpy::npy_save(path.str() + "_depth.npy", depth.data(), {
            len,
            2,
            VisualField::layers,
            vres / VisualField::layers
        });

        FileSize per_second(double(depth.size() * sizeof(decltype(depth)::value_type)) / save_timer.elapsed());
        auto str = Meta::toStr(per_second) + "/s";
        print("saved depth @ ", str.c_str());

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

        FileSize per_second(double(depth.size() * sizeof(decltype(depth)::value_type)) / save_timer.elapsed());
        auto str = Meta::toStr(per_second) + "/s";
        print("saved depth @ ", str.c_str());

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
    for(auto && [fdx, fish] : Tracker::instance()->_individuals) {
        colors.push_back((int)fish->identity().ID());
        colors.push_back(fish->identity().color().r);
        colors.push_back(fish->identity().color().g);
        colors.push_back(fish->identity().color().b);
    }
    
    try {
        file::Path meta_path = use_npz ? (path.str() + ".npz") : (path.str() + "_meta.npz");

        cmn::npz_save(meta_path.str(), "colors", colors.data(), {
            colors.size() / 4, 4
        }, use_npz ? "a" : "w");
        
        cmn::npz_save(meta_path.str(), "fov_range", std::vector<double>{-VisualField::symmetric_fov, VisualField::symmetric_fov}, "a");
        cmn::npz_save(meta_path.str(), "frame_range", std::vector<long_t>{range.start.get(), range.end.get()}, "a");
        
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
            print("Saved visual field metadata to ",meta_path.str()," and image data to ",path.str()+"_*.npy",".");
        else
            print("Saved to ",path.str()+".npz",".");

    } catch(...) {
        // there will be a utils exception, so its printed out already
    }
}

/*size_t Individual::memory_size() const {
    std::map<std::string, FileSize> sizes;
    size_t overall = 0;
    
    overall = sizeof(Individual);
    overall += sizeof(decltype(_manually_matched)::value_type) * _manually_matched.size();
    overall += sizeof(decltype(_frame_segments)::value_type) * _frame_segments.size();
    
    sizes["misc"] = overall;
    
    overall = sizeof(decltype(_blobs)::value_type) * _blobs.size();
    for(auto && [frame, blob] : _blobs)
        overall += blob->memory_size();
    
    sizes["blobs"] = overall;
    
    overall = 0;
    
    foreach([&](auto& c){
        overall += c.second->memory_size() + sizeof(long_t);
    }, _centroid, _head, _centroid_posture, _weighted_centroid);
    
    //overall += sizeof(decltype(_current_velocities)::value_type) * _current_velocities.size();
    //overall += sizeof(decltype(_v_samples)::value_type) * _v_samples.size();
    overall += sizeof(_local_cache)
            + sizeof(decltype(_local_cache._current_velocities)::value_type) * _local_cache._current_velocities.size()
            + sizeof(decltype(_local_cache._v_samples)) * _local_cache._v_samples.size();
    
    sizes["physical_properties"] = overall;
    
    overall = sizeof(decltype(_midlines)::value_type) * _midlines.size();
    foreach([&](auto &pair) {
        overall += pair.second->memory_size();
    }, _midlines);
    sizes["midlines"] = overall;
    
    overall = sizeof(decltype(_outlines)::value_type) * _outlines.size();
    foreach([&](auto &pair) {
        overall += pair.second->memory_size();
    }, _outlines);
    sizes["outlines"] = overall;
    
    overall = sizeof(decltype(_posture_original_angles)::value_type) * _posture_original_angles.size();
    //overall += sizeof(float) * _outline_size.getHeapSize();
    //overall += sizeof(float) * _midline_length.getHeapSize();
    
    foreach([&](auto &pair) {
        overall += pair.second->memory_size();
    }, _cached_fixed_midlines);
    sizes["erasable_cache"] = overall;
    
    overall = 0;
    overall += sizeof(decltype(_custom_data)::value_type) * _custom_data.size();
    size_t objects = 0;
    
    for(auto && [key, map] : _custom_data) {
        overall += sizeof(decltype(map)::value_type) * map.size();
        objects += map.size();
    }
    
    sizes["custom"] = overall;
    
    auto str = prettify_array(Meta::toStr(sizes));
    
    overall = 0;
    foreach([&](auto &v){
        overall += v.second.bytes;
    }, sizes);
    
    auto fs = Meta::toStr(FileSize{overall});
    
    print("Fish ",identity().ID(),": ",str," (overall: ",fs," + ",objects," custom objects)");
    
    return overall;
}*/

std::string Individual::toStr() const {
    //std::stringstream ss;
    //ss << "Individual<" << _identity.ID() << " frames:" << _centroid.size() << " pos:" << head(_endFrame)->pos() << ">";
    return _identity.name();
}

