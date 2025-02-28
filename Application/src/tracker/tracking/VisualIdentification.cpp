#include "VisualIdentification.h"
#include <misc/PythonWrapper.h>
#include <python/GPURecognition.h>
#include <misc/frame_t.h>
#include <misc/PVBlob.h>
#include <tracking/Tracker.h>
#include <ml/Accumulation.h>
#include <misc/create_struct.h>
#include <misc/cnpy_wrapper.h>
#include <file/DataLocation.h>
#include <tracking/IndividualManager.h>
#include <python/ModuleProxy.h>

namespace Python {

namespace py = Python;

using namespace cmn;
using namespace track;

inline static std::mutex _instance_mutex;
inline static std::unique_ptr<VINetwork> _instance;

static constexpr auto module_name = "visual_recognition_torch";

CREATE_STRUCT(Settings,
    (Size2, individual_image_size)
)

#define PSetting(NAME) Settings :: get<Settings::Variables:: NAME>()

std::string VINetwork::Average::toStr() const {
    return "<s:"+Meta::toStr(samples)+" v:"+Meta::toStr(values)+">";
}

const std::unique_ptr<VINetwork>& VINetwork::instance() {
    std::unique_lock guard(_instance_mutex);
    if(!_instance)
        _instance = std::unique_ptr<VINetwork>(new VINetwork);
    return _instance;
}

VINetwork::VINetwork()
    : _network("VINetwork")
{
    Settings::init();
    
    _network.setup = [this](){
        py::schedule([this]() {
            setup(false);
        }).get();
    };
    _network.unsetup = [](){
        
    };
}

bool VINetwork::is_good(const track::BasicStuff *basic,
                        const track::PostureStuff *posture)
{
    return basic && (!FAST_SETTING(calculate_posture) || posture);
}

VINetwork::Status VINetwork::status(){
    return instance()->_status;
}

file::Path VINetwork::network_path() {
    file::Path filename = SETTING(filename).value<file::Path>().filename();
    filename = filename.has_extension("pv")
                    ? filename.remove_extension()
                    : filename;
    filename = file::DataLocation::parse("output", filename.str() + "_weights");
    return filename;
}

void VINetwork::set_status(Status status) {
    instance()->_status = status;
}

void VINetwork::add_percent_callback(
     const char *identifier,
     std::function<void (float, const std::string &)> && fn)
{
    std::unique_lock guard(instance()->_data_mutex);
    if(instance()->_callbacks.count(identifier)) {
        FormatWarning("Identifier ", std::string(identifier), " already present in callbacks.");
    }
    
    instance()->_callbacks[identifier] = std::move(fn);
}

void VINetwork::setup(bool force) {
    using py = PythonIntegration;
    
    auto result = py::check_module(module_name);
    if(result || force || py::is_none("classes", module_name)) {
        uint32_t N = FAST_SETTING(track_max_individuals) ? FAST_SETTING(track_max_individuals) : 1u;
        std::vector<uint32_t> ids;
        ids.resize(N);
        
        for(uint32_t i=0; i<N; ++i)
            ids[i] = i;
        
        uint64_t batch_size = ids.size(); // compute the next highest power of 2 of 32-bit v
        batch_size = max(batch_size, 64u);
        
        if(batch_size < 128)
            batch_size = next_pow2<uint64_t>(batch_size);
        else
            batch_size = 128;
        
        auto version = SETTING(visual_identification_version).value<default_config::visual_identification_version_t::Class>();
        Print("network version: ", version);
    
        auto device = SETTING(gpu_torch_device).value<default_config::gpu_torch_device_t::Class>().toStr();
        if(device == "automatic")
            py::unset_function("device", module_name);
        else
            py::set_variable("device", device, module_name);
        
        py::set_variable("accumulation_step", (long_t)-1, module_name);
        py::set_variable("network_version", version.toStr(), module_name);
        py::set_variable("classes", ids, module_name);
        py::set_variable("image_width", int(PSetting(individual_image_size).width), module_name);
        py::set_variable("image_height", int(PSetting(individual_image_size).height), module_name);
        py::set_variable("learning_rate", SETTING(gpu_learning_rate).value<float>(), module_name);
        py::set_variable("batch_size", (long_t)batch_size, module_name);
        py::set_variable("video_length", narrow_cast<long_t>(SETTING(video_length).value<uint64_t>()), module_name);
        py::set_variable("verbosity", int(SETTING(gpu_verbosity).value<default_config::gpu_verbosity_t::Class>().value()));
        
        auto filename = VINetwork::network_path();
        try {
            if(!filename.remove_filename().exists()) {
                if(filename.remove_filename().create_folder())
                    Print("Created folder ",filename.remove_filename().str());
                else
                    Print("Error creating folder for ",filename.str());
            }
        } catch(...) {
            Print("Error creating folder for ",filename.str());
        }
        
        auto image_mode = Background::image_mode();
        py::set_variable("image_channels", (long_t)required_channels(image_mode), module_name);
        py::set_variable("output_path", filename.str(), module_name);
        py::set_variable("output_prefix", SETTING(output_prefix).value<std::string>(), module_name);
        py::set_variable("filename", (std::string)SETTING(filename).value<file::Path>().filename(), module_name);
        
        if(!py::valid("model", module_name)) {
            py::run(module_name, "reinitialize_network");
        }
    }
        
    set_work_variables(result || force);
}

void VINetwork::set_abort_training(std::function<bool ()> abort_function) {
    instance()->abort_function = abort_function;
}
void VINetwork::set_skip_button(std::function<bool ()> skip_function) {
    instance()->skip_function = skip_function;
}

void VINetwork::set_work_variables(bool force) {
    using py = track::PythonIntegration;
    
    if(force || py::is_none("update_work_percent", module_name)) {
        py::set_function("get_abort_training", (std::function<bool()>)[this]() -> bool {
            return abort_function ? abort_function() : false;
        }, module_name);
        py::set_function("get_skip_step", (std::function<bool()>)[this]() -> bool {
            return skip_function ? skip_function() : false;
        }, module_name);
        
        py::set_function("estimate_uniqueness", (std::function<float(void)>)[](void) -> float {
            if(Accumulation::current())
                return Accumulation::current()->step_calculate_uniqueness();
            FormatWarning("There is currently no accumulation in progress.");
            return 0;
            
        }, module_name);
        py::set_function("acceptable_uniqueness", (std::function<float(void)>)[](void) -> float {
            if(Accumulation::current())
                return SETTING(accumulation_sufficient_uniqueness).value<float>();
            FormatWarning("There is currently no accumulation in progress.");
            return -1;
            
        }, module_name);
        py::set_function("accepted_uniqueness", (std::function<float(void)>)[](void) -> float {
            if(Accumulation::current())
                return Accumulation::current()->accepted_uniqueness();
            FormatWarning("There is currently no accumulation in progress.");
            return -1;
            
        }, module_name);
        py::set_function("update_work_percent", [this](float x) {
            for(auto& [id, c] : _callbacks)
                c(x, "");
        }, module_name);
        py::set_function("update_work_description", [this](std::string x) {
            for(auto& [id, c] : _callbacks)
                c(-1, x);
            
        }, module_name);
        py::set_function("set_stop_reason", [](std::string x) {
            if(Accumulation::current()) {
                Accumulation::current()->set_last_stop_reason(x);
            } else
                FormatWarning("No accumulation object set.");
        }, module_name);
        py::set_function("set_per_class_accuracy", [](std::vector<float> x) {
            if(Accumulation::current()) {
                Accumulation::current()->set_per_class_accuracy(x);
            } else
                FormatWarning("No accumulation object set.");
        }, module_name);
        py::set_function("set_uniqueness_history", [](std::vector<float> x) {
            if(Accumulation::current()) {
                Accumulation::current()->set_uniqueness_history(x);
            } else
                FormatWarning("No accumulation object set.");
        }, module_name);
    }
}

void VINetwork::unset_work_variables() {
    try {
        py::schedule([](){
            using py = track::PythonIntegration;
            try {
                py::unset_function("update_work_percent", module_name);
                py::unset_function("update_work_description", module_name);
                py::unset_function("set_stop_reason", module_name);
                py::unset_function("set_per_class_accuracy", module_name);
                py::unset_function("set_uniqueness_history", module_name);
                py::unset_function("get_abort_training", module_name);
                py::unset_function("get_skip_step", module_name);
            } catch(...) {
                FormatExcept("Failed to unset some variables.");
            }
            
        }).get();
    } catch(const std::future_error& e) {
        throw SoftException("Failed to unsetup python (", std::string(e.what()),")");
    }
}

void VINetwork::reinitialize_internal() {
    using py = PythonIntegration;
    py::check_correct_thread_id();
    setup(true);
    py::run(module_name, "reinitialize_network");
    _status.weights_valid = false;
    _status.busy = false;
}

void VINetwork::load_weights_internal() {
    using py = track::PythonIntegration;
    reinitialize_internal();
    
    try {
        py::run(module_name, "load_weights");
        Print("\tReloaded weights.");
        
    } catch(...) {
        throw;
    }
}

void VINetwork::load_weights() {
    py::schedule(PackagedTask{
        ._network = &_network,
        ._task = PromisedTask([this](){
            load_weights_internal();
        }),
        ._can_run_before_init = true
    }).get();
}


bool VINetwork::weights_available() {
    auto filename = network_path();
    return filename.add_extension("pth").exists();
}


void VINetwork::set_variables_internal(auto && images, callback_t && callback)
{
    using py = PythonIntegration;
    py::check_correct_thread_id();
    
    try {
        if(images.size() == 0) {
            Print("Empty images array.");
            callback(std::vector<std::vector<float>>{},std::vector<float>{});
            return;
        }
        
        py::set_variable("images", images, module_name);
        py::set_function("receive", std::move(callback), module_name);
        py::run(module_name, "predict");
        py::unset_function("receive", module_name);
        py::unset_function("images", module_name);
        
    } catch(const SoftExceptionImpl& e) {
        FormatWarning("Runtime exception: ", e.what());
        throw;
    }
}

void VINetwork::set_variables(std::vector<cmn::Image::Ptr> && images, callback_t&& callback) {
    set_variables_internal(std::move(images), std::move(callback));
}

void VINetwork::set_variables(std::vector<cmn::Image::SPtr> && images, callback_t&& callback)
{
    set_variables_internal(std::move(images), std::move(callback));
}

bool VINetwork::train(std::shared_ptr<TrainingData> data,
                      const FrameRange& global_range,
                      TrainingMode::Class load_results,
                      uchar gpu_max_epochs,
                      bool dont_save,
                      float *worst_accuracy_per_class,
                      int accumulation_step)
{
    bool success = false;
    float best_accuracy_worst_class = worst_accuracy_per_class ? *worst_accuracy_per_class : -1;
    if(worst_accuracy_per_class)
        *worst_accuracy_per_class = -1;
    
    //! TODO: MISSING set progress to zero
    /*if(GUI::instance()) {
        GUI::work().set_progress("training", 0);
    }*/
    
    std::future<bool> future;
    
    // try doing everything in-memory without saving it
    if(load_results == TrainingMode::Restart)
        Print("Beginning training for ", data->size()," images.");
    else if(load_results == TrainingMode::Continue)
        Print("Continuing training (", data->size()," images)");
    else if(load_results == TrainingMode::Apply)
        Print("Just loading weights (", data->size()," images)");
    else if(load_results == TrainingMode::Accumulate)
        Print("Accumulating and training on more segments (", data->size()," images)");
    else
        throw U_EXCEPTION("Unknown training mode ",load_results," in train_internally");
    
    if(load_results == TrainingMode::Continue
       && _last_training_data != nullptr
       && !dont_save)
    {
        // we already have previous training data, but now we want to continue
        // see if they overlap. if they dont overlap, join the datasets
        
        if(_last_training_data->normalized() != data->normalized()) {
            FormatWarning("Cannot combine normalized and unnormalized datasets.");
            
        } else if(!_last_training_data->empty() && !data->empty()) {
            // the range is not empty for both, so we can actually compare
            auto strme = Meta::toStr(*data);
            auto strold = Meta::toStr(*_last_training_data);
            {
                
                // TODO: only merge those that dont overlap with anything else
                // they dont overlap -> join
                Print("Last training data (",strold,") does not overlap with new training data (",strme,"). Attempting to join them.");
                
                // check the accuracy of the given segment
                {
                    Print("Seems alright. Gonna merge now...");
                    data->merge_with(_last_training_data);
                }
            }
            
        } else {
            FormatWarning("There were no ranges set for one of the TrainingDatas.");
        }
    }
    
    if(!dont_save) {
        Print("Saving last training data ptr...");
        _last_training_data = data;
    }
    
    std::promise<bool> promise;
    future = promise.get_future();
    
    auto schedule = py::schedule(PackagedTask{
        ._network = &_network,
        ._task = PromisedTask([
            promise = std::move(promise),
            data, load_results,
            gpu_max_epochs,dont_save, &best_accuracy_worst_class,
                worst_accuracy_per_class, accumulation_step,
                &global_range, this]() mutable
         {
            try {
                using py = PythonIntegration;
                //check_learning_module();
                
                std::vector<Idx_t> classes(data->all_classes().begin(), data->all_classes().end());
                auto joined_data = data->join_split_data();
                
                if(number_classes() > classes.size()) {
                    std::set<Idx_t> missing;
                    for(auto id : Tracker::identities()) {
                        if(std::find(classes.begin(), classes.end(), id) == classes.end())
                            missing.insert(id);
                    }
                    
                    //! abort training process since not all identities have been found
                    throw SoftException("Not all identities are represented in the training data (missing: ",missing,").");
                }
                
                //! decide whether to reload the network
                if(is_in(load_results, TrainingMode::LoadWeights, TrainingMode::Apply))
                    load_weights_internal();
                else if(load_results == TrainingMode::Restart)
                    reinitialize_internal();
                
                py::set_variable("X", joined_data.training_images, module_name);
                py::set_variable("Y", joined_data.training_ids, module_name);
                
                if(joined_data.training_images.size() != joined_data.training_ids.size()) {
                    throw U_EXCEPTION("Training image array size ",joined_data.training_images.size()," != ids array size ",joined_data.training_ids.size(),"");
                }
                
                py::set_variable("X_val", joined_data.validation_images, module_name);
                py::set_variable("Y_val", joined_data.validation_ids, module_name);
                
                if(joined_data.validation_images.size() != joined_data.validation_ids.size()) {
                    throw U_EXCEPTION("Validation image array size ",joined_data.validation_images.size()," != ids array size ",joined_data.validation_ids.size(),"");
                }
                
                py::set_variable("global_tracklet", std::vector<long_t>{
                    global_range.empty() ? -1 : (long_t)global_range.start().get(),
                    global_range.empty() ? -1 : (long_t)global_range.end().get() }, module_name);
                py::set_variable("accumulation_step", (long_t)accumulation_step, module_name);
                py::set_variable("classes", classes, module_name);
                py::set_variable("save_weights_after", false, module_name);
                //load_results != TrainingMode::Accumulate, module_name);
                
                Print("Pushing ", (joined_data.validation_images.size() + joined_data.training_images.size())," images (",FileSize((joined_data.validation_images.size() + joined_data.training_images.size()) * PSetting(individual_image_size).width * PSetting(individual_image_size).height * 4),") to python...");
                
                uchar setting_max_epochs = int(SETTING(gpu_max_epochs).value<uchar>());
                py::set_variable("max_epochs", uint64_t(gpu_max_epochs != 0 ? min(setting_max_epochs, gpu_max_epochs) : setting_max_epochs), module_name);
                py::set_variable("min_iterations", long_t(SETTING(gpu_min_iterations).value<uchar>()), module_name);
                py::set_variable("verbosity", int(SETTING(gpu_verbosity).value<default_config::gpu_verbosity_t::Class>().value()), module_name);
                
                auto filename = network_path();
                try {
                    if(!filename.remove_filename().exists()) {
                        if(filename.remove_filename().create_folder())
                            Print("Created folder ",filename.remove_filename().str());
                        else
                            Print("Error creating folder for ",filename.str());
                    }
                } catch(...) {
                    Print("Error creating folder for ",filename.str());
                }
                
                py::set_variable("run_training",
                                 load_results == TrainingMode::Restart
                                 || load_results == TrainingMode::Continue
                                 || load_results == TrainingMode::Accumulate, module_name);
                py::set_variable("best_accuracy_worst_class", (float)best_accuracy_worst_class, module_name);
                best_accuracy_worst_class = -1;
                
                py::set_function("do_save_training_images", (std::function<bool()>)[]() -> bool {
                    return SETTING(visual_identification_save_images).value<bool>();
                }, module_name);
                
                try {
                    _status.busy = true;
                    py::run(module_name, "start_learning");
                    
                    if (skip_function && skip_function())
                        throw SoftException("User skipped.");
                    
                    best_accuracy_worst_class = py::get_variable<float>("best_accuracy_worst_class", module_name);
                    if(worst_accuracy_per_class)
                        *worst_accuracy_per_class = best_accuracy_worst_class;
                    Print("best_accuracy_worst_class = ", best_accuracy_worst_class);
                    
                    //if(!dont_save)
                    _status.weights_valid = true;
                    _status.busy = false;
                    
                    {
                        LockGuard guard(w_t{}, "train_internally");
                        IndividualManager::transform_all([](auto, auto fish) {
                            fish->clear_recognition();
                        });
                    }
                    
                    //! TODO: MISSING probability clearing
                    {
                        /*std::unique_lock probs_guard(_probs_mutex);
                         if(!probs.empty()) {
                         FormatWarning("Re-trained network, so we'll clear everything...");
                         probs.clear();
                         }*/
                    }
                    
                    // save training data
                    file::Path ranges_path(filename.remove_extension());
                    ranges_path = ranges_path.str() + "_training.npz";
                    
                    std::vector<long_t> all_ranges;
                    std::vector<size_t> lengths;
                    std::vector<float> positions;
                    std::vector<float> frames;
                    std::vector<float> ids;
                    std::vector<uchar> images;
                    
                    Size2 resolution(-1);
                    
                    for(auto &d : data->data()) {
                        all_ranges.push_back(d->frames.start.get());
                        all_ranges.push_back(d->frames.end.get());
                        
                        //for(auto && [range, d] : data->data()) {
                        // save per fish
                        for(auto && [id, fish] : d->mappings) {
                            for(size_t i=0; i<fish.images.size(); ++i) {
                                if(resolution.width == -1) {
                                    resolution = Size2(fish.images.at(i)->cols, fish.images.at(i)->rows);
                                    
                                } else if(fish.images.at(i)->cols != resolution.width
                                          || fish.images.at(i)->rows != resolution.height)
                                {
                                    FormatExcept("Image dimensions of ",fish.images.at(i)->cols,"x",fish.images.at(i)->rows," are different from the others (",resolution.width,"x",resolution.height,") in training data for fish ",d->frames.start," in range [",d->frames.end,",%d].");
                                    promise.set_value(false);
                                    return;
                                }
                                
                                images.insert(images.end(), fish.images.at(i)->data(), fish.images.at(i)->data() + size_t(resolution.width * resolution.height));
                                ids.insert(ids.end(), id.get());
                                positions.insert(positions.end(), fish.positions.at(i).x);
                                positions.insert(positions.end(), fish.positions.at(i).y);
                                frames.insert(frames.end(), fish.frame_indexes.at(i).get());
                            }
                        }
                        //}
                    }
                    
                    if(is_in(load_results, TrainingMode::Continue, TrainingMode::Restart)
                       && !dont_save)
                    {
                        FileSize size(images.size());
                        auto ss = size.to_string();
                        Print("Images are ",ss," big. Saving to '",ranges_path.str(),"'.");
                        
                        cmn::npz_save(ranges_path.str(), "ranges", all_ranges.data(), { all_ranges.size() / 2u, 2u }, "w");
                        cmn::npz_save(ranges_path.str(), "positions", positions.data(), {positions.size() / 2u, 2u}, "a");
                        cmn::npz_save(ranges_path.str(), "ids", ids, "a");
                        cmn::npz_save(ranges_path.str(), "frames", frames, "a");
                        cmn::npz_save(ranges_path.str(), "images", images.data(), { ids.size(), (size_t)resolution.height, (size_t)resolution.width }, "a");
                    }
                    
                } catch(const SoftExceptionImpl& e) {
                    Print("Runtime error: ", e.what());
                    throw;
                }
                
                promise.set_value(true);
                
            } catch(const std::exception& ex) {
                FormatExcept("Exception: ", ex.what());
                promise.set_exception(std::current_exception());
            } catch(...) {
                promise.set_exception(std::current_exception());
            }
        }),
        ._can_run_before_init = false
        
    });

    try {
        //! check this future first, to catch first-level-exceptions.
        //! otherwise, the custom promise (moved to the callback) might
        //! already be destroyed!
        schedule.get();
        
        if (!future.valid())
            throw U_EXCEPTION("Future is invalid.");

        if(future.get()) {
            if(best_accuracy_worst_class != -1)
                DebugCallback("Success (train) with best_accuracy_worst_class = ", best_accuracy_worst_class, ".");
            else
                DebugCallback("Success (train) with unspecified accuracy (will only be displayed directly after training).");
            success = true;
        } else
            Print("Training the network failed (",best_accuracy_worst_class,").");
        
    } catch(const SoftExceptionImpl& e) {
        Print("Runtime error: ", e.what());
        throw;
    }
    
    return success;
}

std::vector<float> VINetwork::transform_results(
    const size_t N,
    std::vector<float> && indexes,
    std::vector<std::vector<float>> && values)
{
    std::vector<float> probs;
    const size_t M = FAST_SETTING(track_max_individuals);
    probs.resize(N * M);
    
    size_t i = 0;
    for(int64_t j=0; j<(int64_t)indexes.size(); ++j, ++i) {
        size_t idx = narrow_cast<size_t>(indexes.at((size_t)j));
        if(i < idx) {
            std::fill(probs.begin() + i * M, probs.begin() + idx * M, -1.f);
            i = idx;
        }
        
        std::move(values[idx].begin(), values[idx].end(), probs.begin() + idx * M);
    }
    
    return probs;
}

std::future<void> VINetwork::clear_caches() {
    return py::schedule(PackagedTask{
        ._network = instance() ? &instance()->_network : nullptr,
        ._task = PromisedTask([](){
            Python::ModuleProxy m(module_name, nullptr);
            m.run("clear_caches");
        })
    });
}

std::set<Idx_t> VINetwork::classes() {
    auto identities = IndividualManager::all_ids();
    assert(FAST_SETTING(track_max_individuals) == identities.size());
    return identities;
}

size_t VINetwork::number_classes() {
    return classes().size();
}

/*auto VINetwork::map1d(std::vector<std::vector<float>> && values, const std::vector<float> &indexes) {
}*/

/*auto && [indexes, values] = PythonIntegration::probabilities(images);
//auto str = Meta::toStr(values);

auto time = timer.elapsed();
Print("[GPU] ",dec<2>(values.size() / float(FAST_SETTING(track_max_individuals))),"/",images.size()," values returned in ",dec<2>(time * 1000),"ms");

this->stop_running();

{
    std::lock_guard<std::mutex> guard(_mutex);
    for(int64_t j=0; j<(int64_t)indexes.size(); ++j) {
        size_t i = narrow_cast<size_t>(indexes.at((size_t)j));
        probs[data[i].frame][data[i].blob.blob.blob_id()] = std::vector<float>(values.begin() + j * FAST_SETTING(track_max_individuals), values.begin() + (j + 1) * FAST_SETTING(track_max_individuals));
    }
}
*/

/*auto VINetwork::probabilities(std::vector<Image::Ptr>&& images, auto&& callback)
{
    
    namespace py = pybind11;
    using namespace py::literals;
    
    static std::mutex mutex;
    static std::vector<float> values, indexes;
    
    std::lock_guard<std::mutex> guard(mutex);
    values.clear();
    indexes.clear();
    
    try {
        //check_module(module_name);
        py::module module;
        
        {
            if (check_module(module_name))
                throw SoftException("Had to reload learn_static while in the training process. This is currently unsupported.");

            std::lock_guard<std::mutex> guard(module_mutex);
            if (!_modules.count(module_name))
                throw SoftException("Cannot find 'learn_static'.");

            module = _modules.find(module_name)->second;
        }
        
        module.attr("images") = images;
        / *module.def("receive", [&](py::array_t<float> x, py::array_t<float> idx) {
            std::vector<float> temporary;
            auto array = x.unchecked<2>();
            auto idxes = idx.unchecked<1>();
            
            Print("Copying ", array.size()," data");
            auto ptr = array.data(0,0);
            auto end = ptr + array.size();
            temporary.insert(temporary.end(), ptr, end);
            values = temporary;
            
            ptr = idxes.data(0);
            end = ptr + idxes.size();
            temporary.clear();
            temporary.insert(temporary.end(), ptr, end);
            indexes = temporary;
            x.release();
            idx.release();
            
        }, py::arg("x").noconvert(), py::arg("idx").noconvert());*/
    
        
        //module.attr("predict")();
        
        //module.attr("receive") = py::none();
        //module.attr("images") = py::none();
        //std::string str = utils::read_file("probs.py");
        //py::exec(str, py::globals(), *_locals);
        
        //(*_locals)["images"] = nullptr;
//}

}
