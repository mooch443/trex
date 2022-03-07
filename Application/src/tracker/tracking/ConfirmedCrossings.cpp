#include "ConfirmedCrossings.h"
#include <gui/DrawStructure.h>
#include <gui/types/StaticText.h>
#include <misc/metastring.h>
#include <misc/default_settings.h>
#include <tracking/Tracker.h>
#include <misc/cnpy_wrapper.h>
#include <pv.h>

namespace track {
static std::map<Frame_t, std::set<FOI>> _fois;
static std::mutex _mutex;
static std::deque<FOIStatus> _current_queue;
static std::deque<FOIStatus> _previous;
static std::set<FOI> _confirmed;
static std::set<FOI> _wrong;
static std::shared_ptr<gui::StaticText> _text;
static std::string _review;

void ConfirmedCrossings::draw(gui::DrawStructure & base, Frame_t frame) {
    auto _started = started();
    if(!_started && _review.empty())
        return;
    
    using namespace gui;
    if(!_text) {
        _text = std::make_shared<StaticText>("", Vec2(200));
        _text->set_clickable(true);
        _text->set_draggable();
    }
    
    std::lock_guard<std::mutex> guard(_mutex);
    FOIStatus last = _previous.empty() ? FOIStatus() : _previous.back();
    
    std::string str = "<h2>Validation</h2>\n";
    size_t undecided = 0;
    for(auto && [foi, status] : _current_queue)
        if(status == DecisionStatus::UNDECIDED)
            ++undecided;
    for(auto && [foi, status] : _previous)
        if(status == DecisionStatus::UNDECIDED)
            ++undecided;
    
    str += undecided == 0 ? "" : "Undecided: <nr>"+Meta::toStr(undecided)+"</nr>\n";
    
    if (_started) {
        str += "<b>navigate using M / N keys.\nselected:"+settings::htmlify(Meta::toStr(last.foi));
        if(last.status != DecisionStatus::UNDECIDED) {
            if(last.status == DecisionStatus::CONFIRMED) {
                str += "\n<nr>CONFIRMED</nr>";
            } else
                str += "\n<str>WRONG</str>";
        }
        
        if((frame <= last.foi.frames().end || (!_current_queue.empty() && frame < _current_queue.front().foi.frames().start - 1_f)) && frame >= last.foi.frames().start) {
            _text->set_background(Color(0,50,0,50));
        }
        else
            _text->set_background(Color(50,0,0,50));
    } else
        _text->set_background(Black.alpha(50));

    if(!_wrong.empty() || !_confirmed.empty())
    {
        double percent = double(_confirmed.size()) / double(_wrong.size() + _confirmed.size()) * 100;
        
        _review = "\n\n<h3>REVIEW</h3>"
            "<nr>Confirmed</nr>: <nr>"+Meta::toStr(_confirmed.size())+"</nr>\n"
            "<str>Wrong:</str>: <nr>"+Meta::toStr(_wrong.size())+"</nr>\n"
            "---------------------\n"
            "<b>Sum</b>: <nr>"+Meta::toStr(_wrong.size() + _confirmed.size())+"</nr>  (<nr>"+Meta::toStr(percent)+"%</nr> correct)"
        ;
    }
    
    if(!_review.empty())
        str += _review;
    
    _text->set_txt(str);
    _text->set_scale(base.scale().reciprocal());
    base.wrap_object(*_text);
}

void ConfirmedCrossings::set_wrong() {
    std::lock_guard<std::mutex> guard(_mutex);
    if(_previous.empty())
        return;
    
    _previous.back().status = DecisionStatus::WRONG;
    _wrong.insert(_previous.back().foi);
    
    auto it = _confirmed.find(_previous.back().foi);
    if(it != _confirmed.end())
        _confirmed.erase(it);
}

void ConfirmedCrossings::set_confirmed() {
    std::lock_guard<std::mutex> guard(_mutex);
    if(_previous.empty())
        return;
    
    _previous.back().status = DecisionStatus::CONFIRMED;
    _confirmed.insert(_previous.back().foi);
    
    auto it = _wrong.find(_previous.back().foi);
    if(it != _wrong.end())
        _wrong.erase(it);
}

bool ConfirmedCrossings::next(FOIStatus& foi) {
    std::lock_guard<std::mutex> guard(_mutex);

    if(_current_queue.empty() && !_previous.empty() && _previous.size() != _wrong.size() + _confirmed.size())
    {
        // still undecided fois in _previous
        for(auto it = _previous.begin(); it != _previous.end();) {
            if(it->status == DecisionStatus::UNDECIDED) {
                _current_queue.push_back(*it);
                it = _previous.erase(it);
            } else
                ++it;
        }
    }
    
    if (_current_queue.empty()) {
        foi = FOI();
        
        if(!_wrong.empty() || !_confirmed.empty()) {
            DebugHeader("Review summary");
            double percent = double(_confirmed.size()) / double(_wrong.size() + _confirmed.size()) * 100;
            print(_confirmed.size(), " confirmed");
            print(_wrong.size(), " wrong");
            print(_wrong.size() + _confirmed.size(), " instances (", dec<2>(percent),"% confirmed)");
            DebugHeader("============== ");
            
            std::vector<Frame_t> rows;
            for (auto && [foi, status] : _previous) {
                rows.insert(rows.end(), {
                    foi.frames().start,
                    foi.frames().end,
                    Frame_t((Frame_t::number_t)status)
                });
                
                const auto range = arange<long_t>(0, FAST_SETTINGS(track_max_individuals)-1);
                for(auto id : range) {
                    if(foi.fdx().count(FOI::fdx_t(id)))
                        rows.push_back(1_f);
                    else
                        rows.push_back(0_f);
                }
            }
            
            std::vector<size_t> shape{
                _previous.size(),
                size_t(3 + FAST_SETTINGS(track_max_individuals))
            };
            
            assert(rows.size() == shape[0] * shape[1]);
            file::Path path = pv::DataLocation::parse("output", SETTING(filename).value<file::Path>().str()+"_confirmations.npz");
            try {
                cmn::npz_save(path.str(), "data", rows.data(), shape);
                DebugHeader("Saved to ", path, ".");
            } catch(...) {
                FormatExcept("Exception while saving to ",path,".");
            }
            
            _wrong.clear();
            _confirmed.clear();
        }
        
        return false;
    }

    foi = _current_queue.front();
    _current_queue.pop_front();
    _previous.push_back(foi);
    return true;
}

bool ConfirmedCrossings::previous(FOIStatus& foi) {
    std::lock_guard<std::mutex> guard(_mutex);
    if (!_previous.empty()) {
        foi = _previous.back();
        _previous.pop_back();
        _current_queue.push_front(foi);
        return true;
    }
    return false;
}

void ConfirmedCrossings::start() {
    std::lock_guard<std::mutex> guard(_mutex);
    if (!_current_queue.empty())
        return;

    _confirmed.clear();
    _wrong.clear();
    _previous.clear();
    
    _review = "";

    for (auto&& [key, set] : _fois) {
        for (auto& foi : set)
            _current_queue.push_back(foi);
    }
}

bool ConfirmedCrossings::started() {
    std::lock_guard<std::mutex> guard(_mutex);
    if (!_current_queue.empty() || !_confirmed.empty() || !_wrong.empty())
        return true;
    return false;
}

void ConfirmedCrossings::add_foi(const FOI& foi) {
    if(!foi.valid())
        return;
    
    if(foi.name(foi.id()) != "correcting") {
        return;
    }
    
    std::lock_guard<std::mutex> guard(_mutex);
    _fois[foi.frames().start].insert(foi);
}

void ConfirmedCrossings::remove_frames(Frame_t frame, long_t id) {
    std::lock_guard<std::mutex> guard(_mutex);
    _current_queue.clear(); // clear current queue if it exists

    for (auto&& [key, set] : _fois) {
        auto copy = set;

        for (auto& k : set) {
            if (id == -1 || k.id() == id) {
                if (k.frames().end >= frame)
                    copy.erase(k);
            }
        }

        set = copy;
    }
}

bool ConfirmedCrossings::is_foi_confirmed(const FOI& foi) {
    std::lock_guard<std::mutex> guard(_mutex);
    if (_confirmed.find(foi) != _confirmed.end()) {
        return true;
    }
	return false;
}

bool ConfirmedCrossings::is_foi_wrong(const FOI& foi) {
    std::lock_guard<std::mutex> guard(_mutex);
    if (_wrong.find(foi) != _wrong.end()) {
        return true;
    }
    return false;
}

bool ConfirmedCrossings::confirmation_available() {
    std::lock_guard<std::mutex> guard(_mutex);
    return !_confirmed.empty() || !_wrong.empty();
}
}
