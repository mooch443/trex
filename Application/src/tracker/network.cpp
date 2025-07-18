#include <commons.pc.h>
#include <misc/Image.h>

#include <tracking/Individual.h>
#include <tracking/Tracker.h>
#include <misc/default_config.h>
#include <tracking/Output.h>
#include <misc/PythonWrapper.h>
#include <misc/CommandLine.h>
#include <tracking/ImageExtractor.h>
#include <tracking/VisualIdentification.h>
#include <ml/Accumulation.h>

struct Tmp {
    auto tmp() {
        //return std::tuple(5,5) <=> std::tuple(3,5);
    }
};

int main(int argc, char**argv) {
    CommandLine::init(argc, argv);
    auto&cmd = CommandLine::instance();
    cmd.cd_home();
    
    print("Sizeof transform = ", sizeof(gui::Transform));
    
    DebugHeader("LOADING DEFAULT SETTINGS");
    default_config::get(GlobalSettings::map(), GlobalSettings::docs(), &GlobalSettings::set_access_level);
    default_config::get(GlobalSettings::set_defaults(), GlobalSettings::docs(), &GlobalSettings::set_access_level);
    
    default_config::register_default_locations();
    
    file::Path path("/Users/tristan/Videos/group_1.pv");
    //file::Path path("/Users/tristan/Videos/tmp/20180505_100700_body.pv");
    //file::Path path("/Users/tristan/Videos/locusts/converted/four_patches_tagged_60_locusts_top_right_high_top_left_low_20220610_142144_body.pv");
    SETTING(filename) = path.remove_extension("pv");
    
    auto video = std::make_shared<pv::File>(path, pv::FileMode::READ);
    video->print_info();
    
    file::Path settings_file(path.replace_extension("settings"));
    GlobalSettings::map().set_print_by_default(true);
    DebugHeader("LOADING ", settings_file);
    try {
        auto content = utils::read_file(settings_file.str());
        default_config::load_string_with_deprecations(settings_file, content, GlobalSettings::map(), AccessLevelType::STARTUP);
        
    } catch(const cmn::illegal_syntax& e) {
        FormatError("Illegal syntax in settings file.");
        throw;
    }
    DebugHeader("LOADED ", settings_file);
    
    Tracker tracker(Image::Make(video->average()), *video);
    video->print_info();
    
    Output::TrackingResults results(tracker);
    try {
        results.load();
    } catch(...) {
        print("Loading failed. Analysing instead...");
        
        PPFrame pp;
        pv::Frame frame;
        Timer timer;
        double s = 0;
        for(Frame_t i=0_f; i<video->length(); ++i) {
            video->read_frame(frame, i);
            track::Tracker::preprocess_frame(std::move(frame), pp, nullptr, track::PPFrame::NeedGrid::NoNeed, video->header().resolution, false);
            tracker.add(pp);
            
            s += timer.elapsed();
            if(i.get() % 1000 == 0)
                print(1.0 / (s / double(i.get())), "fps ", i, "/", video->length());
            timer.reset();
        }
    }
    
    namespace py = Python;
    py::init().get();
    
    try {
        auto visual = py::VINetwork::instance().get();
        {
            auto data = std::make_shared<TrainingData>();
            FrameRange range(Range<Frame_t>{0_f, 1000_f});
            auto individuals_per_frame = Accumulation::generate_individuals_per_frame(range.range, data.get(), nullptr);
            if(data->generate("test", *video, individuals_per_frame, [](float) {}, nullptr)) {
                float worst_accuracy_per_class = 0;
                visual->train(data, range, py::TrainingMode::Continue, 5, true, &worst_accuracy_per_class, 0);
            }
        }
        
        using namespace extract;
        Timer timer;
        ImageExtractor features{
            decltype(video)(video),
            [](const Query& q)->bool {
                return !q.basic->blob.split();
            },
            [&](std::vector<Result>&& results) {
                // partial_apply
                std::vector<Image::Ptr> images;
                images.reserve(results.size());
                
                for(auto &&r : results)
                    images.emplace_back(std::move(r.image));
                
                try {
                    std::vector<Idx_t> ids;
                    for (auto &r : results) {
                        ids.push_back(r.fdx);
                    }
                    
                    auto averages = visual->paverages(ids, std::move(images));
                    /*visual->probabilities(std::move(images), [results = std::move(results)](auto&& values, auto&& indexes) mutable {
                        py::VINetwork::transform_results(results.size(), std::move(indexes), std::move(values));
                        print("\tGot response for ", results.size(), " items (with ",values.size()," items and ",indexes.size()," indexes).");
                    }).get();*/
                    print("Got averages for ", results.size(), " extracted images: ", averages);
                    
                } catch(...) {
                    FormatExcept("Prediction failed.");
                    throw;
                }
            },
            [](auto extractor, double percent, bool finished) {
                // callback
                if(finished) {
                    print("All done extracting. Overall pushed ", extractor->pushed_items());
                    SETTING(terminate) = true;
                } else {
                    print("Percent: ", percent * 100, "%");
                }
            },
            extract::Settings{
                .flags = (uint32_t)Flag::RemoveSmallFrames,
                .max_size_bytes = 1000u * 1000u * 1000u / 5u / 10u,
                .image_size = Size2(80,80),
                .num_threads = 5u,
                .normalization = SETTING(individual_image_normalization).value<default_config::individual_image_normalization_t::Class>()
            }
        };
        
        
        while(features.future().wait_for(std::chrono::milliseconds(1)) != std::future_status::ready) {
            tf::show();
        }
        
        print("Left the loop.");
        
        features.future().get();
        print("Took ", timer.elapsed(), "s");
        
    } catch(const SoftExceptionImpl& e) {
        print("Breaking out of main() because of: ", e.what());
    }
    
    py::deinit().get();
    print("----- EOF");
}
