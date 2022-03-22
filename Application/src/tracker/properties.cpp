#include <types.h>
#include <gui/IMGUIBase.h>
#include <tracking/MotionRecord.h>
#include <tracking/Tracker.h>
#include <tracker/misc/default_config.h>
#include <gui/GUICache.h>
#include <misc/CommandLine.h>

template<typename... Args>
void ASSERT(bool condition, const Args&... args) {
	if (!condition) {
        throw cmn::U_EXCEPTION(args...);
	}
}

int main(int argc, char**argv) {
	using namespace track;
	default_config::register_default_locations();
	GlobalSettings::map().set_do_print(true);

	CommandLine cmd(argc, argv);
	cmd.cd_home();

	gui::init_errorlog();
	set_thread_name("main");
	srand((uint)time(NULL));

	FILE* log_file = NULL;
	std::mutex log_mutex;

	/**
	 * Set default values for global settings
	 */
	using namespace Output;
	DebugHeader("LOADING DEFAULT SETTINGS");
	default_config::get(GlobalSettings::map(), GlobalSettings::docs(), &GlobalSettings::set_access_level);
	default_config::get(GlobalSettings::set_defaults(), GlobalSettings::docs(), &GlobalSettings::set_access_level);
	GlobalSettings::map().dont_print("gui_frame");
	GlobalSettings::map().dont_print("gui_focus_group");

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

		print(start.pos<Units::CM_AND_SECONDS>(), " -> ", next.pos<Units::CM_AND_SECONDS>()," => ", next.speed<Units::CM_AND_SECONDS>(),"cm/s");

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

		print("Position: ", pos * SETTING(cm_per_pixel).value<float>()," / ", previous->pos<Units::CM_AND_SECONDS>()," manual:", speed * SETTING(cm_per_pixel).value<float>(),"cm/s automatic:", previous->speed<Units::CM_AND_SECONDS>(),"cm/s (dt:",
			dt, "s)");

		auto epsilon = speed * SETTING(cm_per_pixel).value<float>() * 0.0001;
		ASSERT(previous->speed<Units::CM_AND_SECONDS>() - speed * SETTING(cm_per_pixel).value<float>() <= epsilon,
			"Speed was ", previous->speed<Units::CM_AND_SECONDS>(),", but should have been ",
			speed * SETTING(cm_per_pixel).value<float>());
	}

	for (auto& v : vector) {
        print("(", v.pos<Units::CM_AND_SECONDS>().x,",",v.pos<Units::CM_AND_SECONDS>().y,")");
	}

    using namespace gui;
	DrawStructure graph(1024, 1024);
	Timer timer;
    Vec2 last_mouse_pos;
	Entangled e;
    bool terminate = false;
    IMGUIBase* ptr = nullptr;

	SETTING(do_history_split) = false;
	
	GlobalSettings::load_from_string({}, GlobalSettings::map(),
		"blob_size_ranges = [[0.001,1000.07]]\n"
		"blob_split_global_shrink_limit = 0.005\n"
		"blob_split_max_shrink = 0.1\n"
		"gpu_enable_accumulation = false\n"
		"gpu_max_epochs = 5\n"
		"gpu_max_sample_gb = 1\n"
		"midline_stiff_percentage = 0.07\n"
		"outline_resample = 0.75\n"
		"speed_extrapolation = 10\n"
		"track_max_individuals = 0\n"
		"track_max_reassign_time = 0.25\n"
		"track_max_speed = 15\n"
		"cm_per_pixel = 0.008082\n"
		"track_speed_decay = 0.5\n"
		"track_threshold = 25\n", cmn::AccessLevelType::STARTUP);

#if defined(__EMSCRIPTEN__)
	pv::File file("group_1.pv");
#else
	pv::File file("C:/Users/tristan/Videos/group_1.pv");
#endif
	print("Will open... group_1...");
	file.start_reading();
	file.print_info();

	try {
		Tracker tracker;
		print("Added tracker");
		print("Image: ", Size2(file.average()));
		tracker.set_average(std::make_unique<Image>(file.average()));

		GUICache cache{&graph, &file};

		std::atomic<double> fps{ 0.0 };

		//Tracker::set_of_individuals_t active;
		std::thread tracking([&]() {
			PPFrame frame;
			pv::Frame single;
			Frame_t index(0);
			Timer timer;
			double time_per_frame = 0;
			double samples = 0;

			while (!SETTING(terminate)) {
				file.read_frame(single, index.get());

				frame.clear();
				frame.set_index(index);
				frame.frame() = single;
				frame.frame().set_index(index.get());
				frame.frame().set_timestamp(single.timestamp());
				frame.set_index(index);

				track::Tracker::LockGuard guard("update_tracker_queue");
				track::Tracker::preprocess_frame(frame, {}, NULL, NULL, false);
				tracker.add(frame);
				//active = tracker.active_individuals(index - 1_f);
				//print(index);

				if (index.get() < file.length() - 1)
					index += 1_f;
				else
					break;

				++samples;
				time_per_frame += timer.elapsed();
				if (samples >= 100) {
					time_per_frame = time_per_frame / samples;
					samples = 1;
					fps = 1.0 / time_per_frame;
				}
				timer.reset();
			}
		});

		Frame_t index(0);
		//FrameInfo frameinfo;

		IMGUIBase base("BBC Micro owl", graph, [&]() -> bool {
			auto dt = timer.elapsed();
			cache.set_dt(dt);
			timer.reset();

			auto h = graph.text("BBC MicroOwl", Vec2(10, 10), White.alpha(255), Font(1), graph.scale().reciprocal() * gui::interface_scale())->local_bounds().height;
			float w = 10;
			w += graph.text(dec<2>(fps.load()).toStr() + "fps", Vec2(w, 10 + h), Cyan.alpha(200), Font(0.75), graph.scale().reciprocal() * gui::interface_scale())->local_bounds().width + 10;
			graph.text("("+cache.tracked_frames.end.toStr() + "/" + Meta::toStr(file.length() - 1) + ")", Vec2(w, 10 + h), Gray.alpha(200), Font(0.75), graph.scale().reciprocal()* gui::interface_scale())->local_bounds().width + 10;

			e.update([&](Entangled& e) {
				e.add<Rect>(Bounds(graph.mouse_position(), Size2(100, 25)), White, Red);
			});
			graph.wrap_object(e);

			//image.set_pos(last_mouse_pos);
			//graph.wrap_object(image);

			auto scale = graph.scale().reciprocal();
			auto dim = ptr->window_dimensions().mul(scale * gui::interface_scale());
			graph.draw_log_messages();//Bounds(Vec2(0), dim));

			static Timer frame_timer;
			if (frame_timer.elapsed() >= 1.0 / (double)SETTING(frame_rate).value<int>()) {
				index += 1_f;
				frame_timer.reset();
			}

			graph.section("fishbowl", [&](auto&, Section* s) {
				if(cache.frame_idx == index) {
					s->reuse_objects();
					return;
				}

				track::Tracker::LockGuard guard("update", 1);
				if (!guard.locked()) {
					s->reuse_objects();
					return;
				}

				s->set_scale(graph.scale().reciprocal());
				cache.update_data(index);
				//frameinfo.mx = graph.mouse_position().x;
				//frameinfo.frameIndex = index;

				//print("Frame ", index, " active: ",active.size(), ".");
				for (auto fish : Tracker::active_individuals(index)) {
					if (fish->has(index)) {
						auto [basic, posture] = fish->all_stuff(index);
						//print("\t", fish->identity().name(), " has frame ", index, " at ", basic->centroid.pos<Units::PX_AND_SECONDS>());
						graph.circle(basic->centroid.pos<Units::PX_AND_SECONDS>() * 0.25, 5, fish->identity().color(), fish->identity().color().alpha(150));
					}
				}
			});

			cache.on_redraw();

			return !terminate;

		}, [&](const Event& e) {
			if (e.type == EventType::KEY && !e.key.pressed) {
				if (e.key.code == Codes::F && graph.is_key_pressed(Codes::LControl)) {
					ptr->toggle_fullscreen(graph);
				}
				else if (e.key.code == Codes::Escape)
					terminate = true;
			}
		});

		ptr = &base;
		base.loop();

		SETTING(terminate) = true;
		tracking.join();
		
	}
	catch (const UtilsException& e) {
		FormatExcept("Exception: ", e.what());
	}
	catch (...) {
		FormatExcept("Unknown exception.");
	}

    return 0;
}

