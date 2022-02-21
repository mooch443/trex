#include <types.h>
#include <tracking/MotionRecord.h>

template<typename... Args>
void ASSERT(bool condition, Args... args) {
	if (!condition) {
        throw U_EXCEPTION(std::forward<Args>(args)...);
	}
}

int main() {
	using namespace track;

	FileSize size{ sizeof(MotionRecord) };
	auto str = size.toStr();
    print("Testing physical properties (",str,")");

	for (int i = 1; i <= 10; ++i) {
		SETTING(cm_per_pixel) = float(i);
		SETTING(frame_rate) = int(60);

		Frame_t f0 = 0_f, f1 = 1_f;

		double t0 = 0;
		double t1 = double(f1.get()) / double(SETTING(frame_rate).value<int>());

		Vec2 p0(i, sqrtf(i)), 
			p1((i - 1) / float(i) + 1, i - 1);

		MotionRecord start, next;
		start.init(nullptr, t0, p0, M_PI);
		next .init(&start,  t1, p1, M_PI);

		Debug("(%f,%f) -> (%f,%f) => %fcm/s",
			start.pos<Units::CM_AND_SECONDS>().x,
			start.pos<Units::CM_AND_SECONDS>().y,
			next.pos<Units::CM_AND_SECONDS>().x,
			next.pos<Units::CM_AND_SECONDS>().y,
			next.speed<Units::CM_AND_SECONDS>());

		auto manual = (p1 - p0).length() / (t1 - t0) * SETTING(cm_per_pixel).value<float>();
        print("Manual: ", manual,"cm/s");

		auto epsilon = manual * 0.0001;
		ASSERT(manual - next.speed<Units::CM_AND_SECONDS>() <= epsilon,
			"Difference between manually chosen and automatically calculated speed > %f", epsilon);
	}


	SETTING(cm_per_pixel) = float(2);
	SETTING(frame_rate) = int(60);

	const MotionRecord* previous = nullptr;
	std::vector<MotionRecord> vector;
	Vec2 pos(0, 0);
	auto v = Vec2(1, 1).normalize() * 10;
	auto speed = v.length();

	for (Frame_t frame = 0_f; frame <= 10_f; ++frame) {
		double time = (double)frame.get() / (double)SETTING(frame_rate).value<int>();
		double dt = time - (previous ? previous->time() : 0);
		
		pos = pos + v * dt;
		
		vector.emplace_back();
		vector.back().init(previous, time, pos, M_PI);
		previous = &vector.back();

		Debug("Position: (%f,%f) / (%f,%f) manual:%fcm/s automatic:%fcm/s (dt:%fs)", 
			pos.x * SETTING(cm_per_pixel).value<float>(), pos.y * SETTING(cm_per_pixel).value<float>(),
			previous->pos<Units::CM_AND_SECONDS>().x, previous->pos<Units::CM_AND_SECONDS>().y,
			speed * SETTING(cm_per_pixel).value<float>(),
			previous->speed<Units::CM_AND_SECONDS>(),
			dt);

		auto epsilon = speed * SETTING(cm_per_pixel).value<float>() * 0.0001;
		ASSERT(previous->speed<Units::CM_AND_SECONDS>() - speed * SETTING(cm_per_pixel).value<float>() <= epsilon,
			"Speed was %f, but should have been %f", 
			previous->speed<Units::CM_AND_SECONDS>(), 
			speed * SETTING(cm_per_pixel).value<float>());
	}

	for (auto& v : vector) {
        print("(", v.pos<Units::CM_AND_SECONDS>().x,",",v.pos<Units::CM_AND_SECONDS>().y,")");
	}

	return 0;
}

