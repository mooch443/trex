#pragma once

#include <misc/frame_t.h>
#include <misc/idx_t.h>
#include <misc/vec2.h>
#include <misc/PVBlob.h>

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
			return std::hash<std::tuple<uint32_t, float, float>>()({ bid.operator unsigned int(), pos.x, pos.y });
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
	Detection find(Frame_t, pv::bid);
	UnorderedVectorSet<std::tuple<float, Detection>> query(Frame_t frame, const Vec2& pos, float distance);
	bool available();
}
}