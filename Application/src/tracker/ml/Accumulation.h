#pragma once

#if !TREX_NO_PYTHON
#include <tracking/Tracker.h>
#include <tracking/DatasetQuality.h>
#include <gui/types/Layout.h>
#include <tracking/TrainingData.h>
#include <tracking/VisualIdentification.h>
#include <pv.h>

namespace cmn::gui {
class Graph;
class HorizontalLayout;
class IMGUIBase;
class StaticText;
class ExternalImage;
}

namespace Python {
class VINetwork;
}

namespace track {
namespace TrainingMode = ::Python::TrainingMode;

ENUM_CLASS(AccumulationStatus, Added, Cached, Failed, None)
ENUM_CLASS(AccumulationReason, NoUniqueIDs, ProbabilityTooLow, NotEnoughImages, TrainingFailed, UniquenessTooLow, Skipped, None)

template<typename K, typename V>
using hash_map = std::map<K, V>;
template<typename K>
using hash_set = std::set<K>;

class Accumulation {
protected:
    struct Result {
        FrameRange _range;
        AccumulationStatus::Class _success;
        AccumulationReason::Class _reasoning;
        std::string _reason;
        float _best_uniqueness;
        float _uniqueness_after_step;
        std::string _training_stop;
        size_t _num_ranges_added;
        
        Result(FrameRange range = FrameRange(), float uniqueness_after = -1, AccumulationStatus::Class success = AccumulationStatus::None, AccumulationReason::Class reason = AccumulationReason::None, const std::string& r = "")
            : _range(range),
              _success(success),
              _reasoning(reason),
              _reason(r),
              _best_uniqueness(-1),
              _uniqueness_after_step(uniqueness_after),
              _num_ranges_added(0)
        {
            
        }
        std::string toStr() const;
        static std::string class_name() { return "Accumulation::Result"; }
    };
    
    TrainingMode::Class _mode;
    std::vector<Range<Frame_t>> _trained;
    std::shared_ptr<TrainingData> _collected_data, _generated_data;
    std::shared_ptr<TrainingData> _discrimination_data;
    std::vector<Image::SPtr> _disc_images;
    std::map<Frame_t, Range<size_t>> _disc_frame_map;
    std::vector<Frame_t> _checked_ranges_output;
    hash_map<Frame_t, float> unique_map, temp_unique;
    std::map<Range<Frame_t>, std::tuple<double, FrameRange>> assigned_unique_averages;
    size_t _accumulation_step;
    size_t _counted_steps, _last_step;
    std::vector<file::Path> _coverage_paths;
    std::vector<float> _uniquenesses;
    GETTER(std::vector<Result>, accumulation_results);
    GETTER_SETTER(std::string, last_stop_reason);
    
    /**
     * The following stuff is needed for the GUI elements in WorkProgress.
     */
    
    //! per-class accuracy data
    std::vector<float> _current_per_class;
    
    //! average uniqueness per identity
    std::vector<float> _uniqueness_per_class;
    
    //! a point for all uniqueness values in the current run (should be as long as the number of epochs)
    std::vector<float> _current_uniqueness_history;
    
    gui::derived_ptr<gui::StaticText> _textarea;
    gui::derived_ptr<gui::Graph> _graph;
    gui::derived_ptr<gui::HorizontalLayout> _layout;
    gui::derived_ptr<gui::VerticalLayout> _layout_rows;
    gui::derived_ptr<gui::ExternalImage> _coverage_image;
    gui::derived_ptr<gui::Entangled> _dots;
    
    std::mutex _coverage_mutex;
    Image::Ptr _raw_coverage;
    std::shared_ptr<pv::File> _video{nullptr};
    gui::IMGUIBase* _base{nullptr};
    
public:
    Accumulation(std::shared_ptr<pv::File>&& video, gui::IMGUIBase* base, TrainingMode::Class);
    ~Accumulation();
    bool start();
    
    struct Status {
        bool busy{false};
        float percent{-1};
        size_t failed_blobs{0};
        
        auto operator<=>(const Status& other) const = default;
    };
    
    static Status& status();
    
    static void register_apply_callback(std::function<void()>&&);
    static void register_apply_callback(std::function<void(double)>&&);
    static void on_terminate();
    
    static float good_uniqueness();
    static std::map<Frame_t, std::set<Idx_t>> generate_individuals_per_frame(const Range<Frame_t>& range, TrainingData* data, std::map<Idx_t, std::set<std::shared_ptr<SegmentInformation>>>*);
    std::tuple<bool, std::map<Idx_t, Idx_t>> check_additional_range(const Range<Frame_t>& range, TrainingData& data, bool check_length, DatasetQuality::Quality);
    void confirm_weights();
    void update_coverage(const TrainingData& data);
    
    static std::tuple<float, hash_map<Frame_t, float>, float> calculate_uniqueness(bool internal, const std::vector<Image::SPtr>&, const std::map<Frame_t, Range<size_t>>&);
    static std::tuple<std::shared_ptr<TrainingData>, std::vector<Image::SPtr>, std::map<Frame_t, Range<size_t>>> generate_discrimination_data(pv::File& video, const std::shared_ptr<TrainingData>& source = nullptr);
    static void setup();
    static void unsetup();
    static Accumulation* current();
    
private:
    
    Range<Frame_t> _initial_range;
    std::map<Frame_t, std::set<Idx_t>> individuals_per_frame;
    //std::map<long_t, std::set<std::shared_ptr<SegmentInformation>>> overall_coverage;
    std::vector<Range<Frame_t>> _added_ranges;
    std::vector<Range<Frame_t>> _next_ranges;
    float current_best;
    
    std::shared_ptr<TrainingData::DataRange> current_salt;
    
    void end_a_step(Result reason);
    float best_uniqueness() const;
    float accepted_uniqueness(float base = -1) const;
    float step_calculate_uniqueness();

    friend class Recognition;
    friend class Python::VINetwork;
    void set_per_class_accuracy(const std::vector<float>& v);
    void set_uniqueness_history(const std::vector<float>& v);
    std::vector<float> per_class_accuracy() const;
    std::vector<float> uniqueness_history() const;
    void update_display(gui::Entangled& e, const std::string& text);
};

}
#endif

