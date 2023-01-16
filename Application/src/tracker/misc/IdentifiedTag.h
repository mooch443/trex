#pragma once

#include <misc/frame_t.h>
#include <misc/idx_t.h>
#include <misc/vec2.h>
#include <misc/bid.h>
#include <file/DataFormat.h>

namespace track {
namespace tags {
	struct Detection {
		Idx_t id;
		Vec2 pos;
		pv::bid bid;
		float p;
		
		bool valid() const {
			return bid.valid();
		}

		auto operator<=>(const Detection&) const = default;
		bool operator==(const pv::bid&) const;
		bool operator==(const Detection&) const = default;
		size_t hash() const {
			return std::hash<uint32_t>()( bid.operator unsigned int() );
		}
	};

struct Assignment {
    Idx_t id;
    pv::bid bid;
    float p;
    
    Assignment() = default;
    Assignment(Idx_t id, pv::bid bid, float p) : id(id), bid(bid), p(p) {}
    Assignment(Detection&& d)
        : id(d.id), bid(d.bid), p(d.p)
    {}
    
    bool valid() const {
        return bid.valid() && id.valid();
    }

    auto operator<=>(const Assignment&) const = default;
    bool operator==(const pv::bid&) const;
    bool operator==(const Assignment&) const = default;
    size_t hash() const {
        return std::hash<uint32_t>()( bid.operator unsigned int() );
    }
};

	void detected(auto&& collection) {
		for(auto&& [frame, tags] : collection) {
			for(auto&& tag : tags) {
				detected(frame, std::move(tag));
			}
		}
	}
	void detected(Frame_t, Detection&& tag);
	void remove(Frame_t, pv::bid);
	Assignment find(Frame_t, pv::bid);
	//UnorderedVectorSet<std::tuple<float, Assignment>> query(Frame_t frame, const Vec2& pos, float distance);
	bool available();
    
    //! writes to results file / binary format
    void write(Data&);
    //! reads from results file / binary format
    void read(Data&);
}
}
