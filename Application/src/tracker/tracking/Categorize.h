#pragma once

#include <misc/Image.h>
#include <tracking/Individual.h>

namespace gui {
class DrawStructure;
}

namespace track {
namespace Categorize {

struct Label {
    using Ptr = std::shared_ptr<Label>;
    std::string name;
    int id;
    
    template<typename... Args>
    static Ptr Make(Args&&...args) {
        return std::make_shared<Label>(std::forward<Args>(args)...);
    }
    
    Label(const std::string& name) : name(name) {
        static int _ID = 0;
        id = _ID++;
    }
};

struct Sample {
    using Ptr = std::shared_ptr<Sample>;
    template<typename... Args>
    static Ptr Make(Args&&...args) {
        return std::make_shared<Sample>(std::forward<Args>(args)...);
    }
    
    Idx_t _fish;
    std::shared_ptr<Individual::SegmentInformation> _segment;
    
    std::vector<long_t> _frames;
    std::vector<Image::Ptr> _images;
    
    Label::Ptr _assigned_label;
    std::map<Label::Ptr, float> _probabilities;
    bool _requested = false;
    
    Sample(Idx_t fish, const decltype(_segment)& segment, std::vector<long_t>&& frames, const std::vector<Image::Ptr>& images);
    
    static const Sample::Ptr& Invalid() {
        static Sample::Ptr invalid(nullptr);
        return invalid;
    }
    
    void set_label(const Label::Ptr& label) {
        if(!label) {
            // unassigning label
            _assigned_label = nullptr;
            return;
        }
        
        if(_assigned_label != nullptr)
            U_EXCEPTION("Replacing label for sample (was already assigned '%s', but now also '%s').", _assigned_label->name.c_str(), label->name.c_str());
        _assigned_label = label;
    }
};

struct DataStore {
    static std::set<std::string> label_names();
    
    static Label::Ptr label(const char* name);
    static Label::Ptr label(int ID);
    
    static const Sample::Ptr& sample(
        const std::shared_ptr<Individual::SegmentInformation>& segment,
        Individual* fish
    );
    static void predict(const std::shared_ptr<Individual::SegmentInformation>& segment,
                        Individual* fish,
                        std::function<void(Sample::Ptr)>&& callback);
    static Sample::Ptr temporary(const std::shared_ptr<Individual::SegmentInformation>& segment,
                                 Individual* fish);
    
    static const Sample::Ptr& random_sample(Idx_t fid);
    static Sample::Ptr get_random();
    
    struct Composition {
        std::unordered_map<std::string, size_t> _numbers;
        std::string toStr() const;
    };
    
    static Composition composition();
};

struct LearningTask {
    enum class Type {
        Prediction,
        Training
    } type;
    
    Sample::Ptr sample;
    std::function<void(const LearningTask&)> callback;
    std::vector<float> result;
};

namespace Work {
void add_task(LearningTask&&);
}

void show();
void hide();
void draw(gui::DrawStructure&);
void terminate();

}
}
