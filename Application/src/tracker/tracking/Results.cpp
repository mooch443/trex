#include "Results.h"
#include <file/CSVExport.h>
#include <tracking/Outline.h>
#include <tracking/OutputLibrary.h>
#include <tracking/IndividualManager.h>
#include <tracking/LockGuard.h>
#include <tracking/EventAnalysis.h>
#include <tracking/Individual.h>

using namespace cmn::file;
using namespace Output;
using namespace track;

bool Results::save_events(const Path &filename, std::function<void(float)> set_percent) const {
    LockGuard guard(w_t{}, "save_events");
    auto events = EventAnalysis::events();
    
    size_t overall_frames = 0;
    for (auto &pair : events->map())
        overall_frames += pair.second.events.size();
    
    std::vector<std::string> header = {"event_start", "length", "energy", "angle_change", "velocity_change", "speed_before", "speed_after"};
    size_t written_frames = 0;
    
    for(auto &pair : events->map()) {
        if(!pair.first->evaluate_fitness())
            continue;
        
        std::string f = filename.str() + pair.first->identity().raw_name() + ".csv";
        Print("Exporting fish ", pair.first->identity()," (",pair.first->identity(),") events (",f,")...");
        
        Table table(header);
        
        for(auto &e : pair.second.events) {
            Row row;
            row.add(e.second.begin);
            row.add(e.second.end - e.second.begin);
            row.add(e.second.energy);
            row.add(e.second.direction_change);
            row.add(e.second.acceleration);
            row.add(e.second.speed_before);
            row.add(e.second.speed_after);
            table.add(row);
            
            written_frames++;
            
            if(written_frames%100 == 0 && set_percent)
                set_percent(double(written_frames) / double(overall_frames));
        }
        
        
        CSVExport exp(table);
        exp.save(f);
    }
    
    delete events;
    return true;
}

bool Results::save(const Path &filename) const {
	std::vector<Individual*> individuals;
	std::map<Individual*, bool> fitness;

#define CONTINUE_IF_UNFIT(IND) { if(fitness.at(IND) == false) continue; }

    IndividualManager::transform_all([&](auto, auto fish){
        individuals.push_back(fish);
        fitness[fish] = fish->evaluate_fitness();
    });

	std::vector<std::string> header = {"frame"};
	for (auto &fish : individuals) {
		CONTINUE_IF_UNFIT(fish);
		header.push_back(fish->identity().name());
	}

	for (uint32_t idx = 0; idx < individuals.size(); idx++) {
		auto &fish = individuals.at(idx);
		CONTINUE_IF_UNFIT(fish);
		
		std::vector<std::string> header = {"frame"};
		for (uint32_t i = 0; i < FAST_SETTING(midline_resolution); i++) {
			std::stringstream ss;
			ss << "segment" << i;
			header.push_back(ss.str());
		}

		Table table(header);
		for (auto i = fish->start_frame(); i <= fish->end_frame(); ++i) {
			Row row;
			row.add(i);

			//auto posture = fish->posture(i);
            auto midline = fish->midline(i);
			if (midline) {
				//auto& midline = //posture->outline().normalize_midline();

				Vec2 prev(0, 0);
				for (uint32_t p=0; p<midline->segments().size(); p++) {
					auto& pos = midline->segments().at(p).pos;
					auto line = pos - prev;
					auto angle = cmn::atan2(line.y, line.x);
                    
                    float x = cmn::cos(angle),
                          y = cmn::sin(angle);
                    
                    row.add(x);
                    row.add(y);
					//row.add(normalize_angle(angle));
					prev = pos;
				}

				/*for (auto &m : midline.segments) {
					row.add(m.pos.y);
				}*/

				table.add(row);

			} else {
				for (uint32_t p = 0; p < FAST_SETTING(midline_resolution); p++) {
					row.add("");
				}
			}
		}

		CSVExport e(table);
		e.save(filename.str() + "_posture" + std::to_string(idx) + ".csv");
	}

	return true;
}
