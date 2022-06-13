#pragma once

#include <misc/idx_t.h>
#include <misc/vec2.h>

namespace track {
	struct IdentifiedTag {
		Idx_t id;
		Vec2 pos;
		float p;
	};
}