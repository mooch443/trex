#pragma once

#include <commons.pc.h>
#include <misc/Image.h>
#include <misc/frame_t.h>
#include <tracking/TrainingData.h>
#include <pv.h>

namespace track {

class AccumulationSession {
public:
    virtual ~AccumulationSession() = default;

    virtual float estimate_uniqueness() = 0;
    virtual float accepted_uniqueness_threshold() const = 0;
    virtual void update_last_stop_reason(const std::string&) = 0;
    virtual void update_per_class_accuracy(const std::vector<float>&) = 0;
    virtual void update_uniqueness_history(const std::vector<float>&) = 0;
};

namespace accumulation_runtime {

using DiscriminationData = std::tuple<std::shared_ptr<TrainingData>,
                                      std::vector<Image::SPtr>,
                                      std::map<cmn::Frame_t, cmn::Range<size_t>>>;
using UniquenessMap = std::map<cmn::Frame_t, float>;
using UniquenessCalculation = std::tuple<float, UniquenessMap, float>;
using SetupFn = std::function<void()>;
using TeardownFn = std::function<void()>;
using GenerateDiscriminationDataFn = std::function<DiscriminationData(pv::File&, const std::shared_ptr<TrainingData>&)>;
using CalculateUniquenessFn = std::function<UniquenessCalculation(bool,
                                                                  const std::vector<Image::SPtr>&,
                                                                  const std::map<cmn::Frame_t, cmn::Range<size_t>>&,
                                                                  const std::unique_lock<std::mutex>*)>;

void set_current(AccumulationSession* session);
AccumulationSession* current();

void register_setup(SetupFn fn);
void register_unsetup(TeardownFn fn);
void register_generate_discrimination_data(GenerateDiscriminationDataFn fn);
void register_calculate_uniqueness(CalculateUniquenessFn fn);

void setup();
void unsetup();
DiscriminationData generate_discrimination_data(pv::File& video, const std::shared_ptr<TrainingData>& source = nullptr);
UniquenessCalculation calculate_uniqueness(bool internal,
                                           const std::vector<Image::SPtr>& images,
                                           const std::map<cmn::Frame_t, cmn::Range<size_t>>& map_indexes,
                                           const std::unique_lock<std::mutex>* guard = nullptr);

}

}
