#pragma once

#include <tracking/Tracker.h>
#include <tracking/Recognition.h>
#include <tracking/DatasetQuality.h>
#include <gui/types/Layout.h>

namespace gui {
class Graph;
class HorizontalLayout;
}

namespace track {
ENUM_CLASS(AccumulationStatus, Added, Cached, Failed, None)
ENUM_CLASS(AccumulationReason, NoUniqueIDs, ProbabilityTooLow, NotEnoughImages, TrainingFailed, UniquenessTooLow, None)

class Accumulation {
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
        operator MetaObject() const;
        static std::string class_name() { return "Accumulation::Result"; }
    };
    
    TrainingMode::Class _mode;
    std::vector<Rangel> _trained;
    std::shared_ptr<TrainingData> _collected_data, _generated_data;
    std::shared_ptr<TrainingData> _discrimination_data;
    std::vector<Image::Ptr> _disc_images;
    std::map<long_t, Range<size_t>> _disc_frame_map;
    std::vector<long_t> _checked_ranges_output;
    std::map<uint32_t, float> unique_map, temp_unique;
    std::map<Rangel, std::tuple<double, FrameRange>> assigned_unique_averages;
    size_t _accumulation_step;
    size_t _counted_steps, _last_step;
    std::vector<file::Path> _coverage_paths;
    std::vector<float> _uniquenesses;
    GETTER(std::vector<Result>, accumulation_results)
    GETTER_SETTER(std::string, last_stop_reason)
    
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
    std::unique_ptr<Image> _raw_coverage;
    
public:
    Accumulation(TrainingMode::Class);
    bool start();
    
    static float good_uniqueness();
    std::map<long_t, std::set<long_t>> generate_individuals_per_frame(const Rangel& range, TrainingData* data, std::map<long_t, std::set<std::shared_ptr<Individual::SegmentInformation>>>*);
    std::tuple<bool, std::map<long_t, long_t>> check_additional_range(const Rangel& range, TrainingData& data, bool check_length, DatasetQuality::Quality);
    void confirm_weights();
    void update_coverage(const TrainingData& data);
    
    static std::tuple<float, std::map<uint32_t, float>, float> calculate_uniqueness(bool internal, const std::vector<Image::Ptr>&, const std::map<long_t, Range<size_t>>&);
    static std::tuple<std::shared_ptr<TrainingData>, std::vector<Image::Ptr>, std::map<long_t, Range<size_t>>> generate_discrimination_data(const std::shared_ptr<TrainingData>& source = nullptr);
    static void setup();
    static void unsetup();
    static Accumulation* current();
    
private:
    
    Rangel _initial_range;
    std::map<long_t, std::set<long_t>> individuals_per_frame;
    //std::map<long_t, std::set<std::shared_ptr<Individual::SegmentInformation>>> overall_coverage;
    std::vector<Rangel> _added_ranges;
    std::vector<Rangel> _next_ranges;
    float current_best;
    
    std::shared_ptr<TrainingData::DataRange> current_salt;
    
    void end_a_step(Result reason);
    float best_uniqueness() const;
    float accepted_uniqueness(float base = -1) const;
    float step_calculate_uniqueness();
    
    friend class Recognition;
    void set_per_class_accuracy(const std::vector<float>& v);
    void set_uniqueness_history(const std::vector<float>& v);
    std::vector<float> per_class_accuracy() const;
    std::vector<float> uniqueness_history() const;
    void update_display(gui::Entangled& e, const std::string& text);
};

}
