#pragma once
#include <commons/common/commons.pc.h>
#include <tracking/FOI.h>

namespace gui {
class DrawStructure;
}

namespace track {

enum class DecisionStatus {
    CONFIRMED,
    WRONG,
    UNDECIDED
};

struct FOIStatus {
    FOI foi;
    DecisionStatus status;
    
    FOIStatus(const FOI& foi = FOI(), DecisionStatus status = DecisionStatus::UNDECIDED)
        : foi(foi), status(status)
    {}
    
    bool operator==(const FOI& other) const {
        return other == foi;
    }
    
    bool operator==(const FOIStatus& other) const {
        return other.foi == foi && other.status == status;
    }
};

class ConfirmedCrossings {
private:
	ConfirmedCrossings() {}

public:
	static void remove_frames(Frame_t frame, long_t id = -1);
	static bool is_foi_confirmed(const FOI& foi);
	static bool is_foi_wrong(const FOI& foi);
	static bool confirmation_available();
	static void add_foi(const FOI& foi);
    
    static void draw(gui::DrawStructure&, Frame_t frame);

	static void start();
	static void stop();
	static bool next(FOIStatus&);
	static bool previous(FOIStatus&);
	static bool started();
    
    static void set_wrong();
    static void set_confirmed();
};

}
