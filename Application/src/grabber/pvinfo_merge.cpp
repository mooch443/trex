#include "pvinfo_merge.h"
#include <pv.h>
#include <processing/Background.h>
#include <misc/SpriteMap.h>
#include <misc/GlobalSettings.h>
#include <misc/default_config.h>
#include <misc/PVBlob.h>
#include <processing/CPULabeling.h>
#include <misc/ranges.h>
#include <file/DataLocation.h>

using namespace cmn;

std::string date_time() {
    time_t rawtime;
    struct tm * timeinfo;
    char buffer[80];
    
    time (&rawtime);
    timeinfo = localtime(&rawtime);
    
    strftime(buffer,sizeof(buffer),"%d-%m-%Y %H:%M:%S",timeinfo);
    std::string str(buffer);
    return str;
}

void initiate_merging(const std::vector<file::Path>& merge_videos, int argc, char**argv) {
    Frame_t min_length;
    std::vector<std::shared_ptr<pv::File>> files;
    std::vector<std::shared_ptr<Background>> backgrounds;
    std::vector<std::shared_ptr<sprite::Map>> configs;
    
    std::map<pv::File*, float> cms_per_pixel;
    Size2 resolution;
    
    file::DataLocation::register_path("merge", [](const sprite::Map& map, file::Path filename) -> file::Path {
        if(!filename.empty() && filename.is_absolute()) {
#ifndef NDEBUG
            if(GlobalSettings::is_runtime_quiet())
                print("Returning absolute path ",filename.str(),". We cannot be sure this is writable.");
#endif
            return filename;
        }
        
        auto path = map.at("merge_dir").value<file::Path>();
        if(path.empty()) {
            return file::DataLocation::parse("input", filename, &map);
        } else
            return path / filename;
    });
    
    for(auto name : merge_videos) {
        name = file::DataLocation::parse("merge", name);
        if((name.has_extension("pv") && not name.is_regular())
           || not name.add_extension("pv").is_regular())
        { // approximation (this is not a true XOR)
            throw U_EXCEPTION("File ",name.str()," cannot be found.");
        }
        
        if(not name.has_extension("pv"))
            name = name.add_extension("pv");
        
        auto file = std::make_shared<pv::File>(name, pv::FileMode::READ);
        files.push_back(file);
        
        if(not min_length.valid() || min_length > file->length())
            min_length = file->length();
        
        resolution += Size2(file->header().resolution);
        backgrounds.push_back(std::make_shared<Background>(Image::Make(file->average()), nullptr));
        //cv::imshow(name.str(), backgrounds.back()->image().get());
        //cv::waitKey(1);
        
        SETTING(filename) = name.remove_extension();
        auto settings_file = file::DataLocation::parse("output_settings");
        if(settings_file.exists()) {
            print("settings for ",name.str()," found");
            auto config = std::make_shared<sprite::Map>();
            GlobalSettings::docs_map_t docs;
            grab::default_config::get(*config, docs, NULL);
            
            GlobalSettings::load_from_string(sprite::MapSource{settings_file}, {}, *config, utils::read_file(settings_file.str()), AccessLevelType::STARTUP);
            if(!file->header().metadata.empty())
                sprite::parse_values(sprite::MapSource{file->filename()}, *config, file->header().metadata);
            if(!config->has("meta_real_width") || config->at("meta_real_width").value<float>() == 0)
                (*config)["meta_real_width"].value<float>(30);
            if(!config->has("cm_per_pixel") || config->at("cm_per_pixel").value<float>() == 0)
                (*config)["cm_per_pixel"] = config->at("meta_real_width").value<float>() / float(file->average().cols);
            
            cms_per_pixel[file.get()] = config->at("cm_per_pixel").value<float>();
            configs.push_back(config);
            
        } else {
            throw U_EXCEPTION("Cant find settings for ",name.str()," at ",settings_file.str());
        }
    }
    
    resolution /= (float)files.size();
    resolution = resolution.map<round>();
    
    cv::Mat average;
    if(SETTING(merge_background).value<file::Path>().empty()) {
        for(auto file : files) {
            if(file->header().resolution.width >= resolution.width
               && file->header().resolution.height >= resolution.height
               && (average.empty() || (file->header().resolution.width > average.cols && file->header().resolution.height > average.rows)))
            {
                file->average()(Bounds(Vec2(), resolution)).copyTo(average);
            }
        }
        
    } else {
        auto path = SETTING(merge_background).value<file::Path>();
        auto raw_path = path;
        path = file::DataLocation::parse("input", path);
        
        if((!path.has_extension()
            && path.add_extension("pv").exists())
           || path.has_extension("pv"))
        {
            pv::File file(path, pv::FileMode::READ);
            file.average().copyTo(average);
            
        } else {
            if(!path.exists()) {
                auto dimensions = Meta::fromStr<Size2>(raw_path.str());
                average = cv::Mat::ones(dimensions.height, dimensions.width, CV_8UC1);
                average *= 255;
                
            } else {
                auto mat = cv::imread(path.str());
                if(mat.channels() > 1) {
                    std::vector<cv::Mat> images;
                    cv::split(mat, images);
                    images[0].copyTo(average);
                } else
                    mat.copyTo(average);
            }
        }
        
        resolution = Size2(average);
    }
    
    Background new_background(Image::Make(average), nullptr);
    
    if(SETTING(frame_rate).value<uint32_t>() == 0){
        if(!files.front()->header().metadata.empty())
            sprite::parse_values(sprite::MapSource{files.front()->filename()}, GlobalSettings::map(), files.front()->header().metadata);
        
        //SETTING(frame_rate) = int(1000 * 1000 / float(frame.timestamp()));
    }
    
    struct Source {
        uint64_t video_index;
        Frame_t frame_index;
        pv::bid blob_id;
    };
    
    SETTING(meta_write_these) = std::vector<std::string>{
        "meta_real_width",
        "meta_source_path",
        "meta_cmd",
        "meta_build",
        "meta_conversion_time",
        "meta_number_merged_videos",
        "frame_rate",
        "meta_video_scale",
        "detect_classes",
        "meta_encoding"
    };
    
    SETTING(meta_conversion_time) = std::string(date_time());
    std::stringstream ss;
    for(int i=0; i<argc; ++i) {
        if(i > 0)
            ss << " ";
        if(argv[i][0] == '-')
            ss << argv[i];
        else
            ss << "'" << argv[i] << "'";
    }
    SETTING(meta_cmd) = ss.str();
    SETTING(meta_source_path) = std::string();
    SETTING(meta_number_merged_videos) = size_t(files.size());
    
    // frame: {blob : source}
    std::map<Frame_t, std::map<pv::bid, Source>> meta;
    if(SETTING(merge_output_path).value<file::Path>().empty())
        SETTING(merge_output_path) = file::Path("merged");
    
    file::Path out_path = file::DataLocation::parse("output", SETTING(merge_output_path).value<file::Path>());
    pv::File output(out_path, pv::FileMode::WRITE | pv::FileMode::OVERWRITE);
    
    output.set_resolution((cv::Size)resolution);
    output.set_average(average);
    
    output.set_start_time(std::chrono::system_clock::now());
    
    //auto start_time = output.header().timestamp;
    print("Writing videos ",files," to '",out_path.c_str(),"' [0,",min_length,"] with resolution (",resolution.width,",",resolution.height,")");
    using namespace track;
    GlobalSettings::map()["cm_per_pixel"].get().set_do_print(false);
    const bool merge_overlapping_blobs = SETTING(merge_overlapping_blobs);
    //const float scaled_video_width = floor(sqrt(resolution.width * resolution.height / float(files.size())));
    
    /*for(uint64_t frame=0; frame<min(1000, min_length); ++frame) {
     pv::Frame f, o;
     if(SETTING(terminate))
     break;
     
     std::vector<pv::BlobPtr> ptrs;
     std::vector<size_t> indexes;
     
     for(size_t vdx = 0; vdx < files.size(); ++vdx) {
     auto &file = files.at(vdx);
     file->read_frame(f, frame);
     if(!vdx) o.set_timestamp(f.timestamp());
     
     Vec2 offset = Vec2();
     auto blob_size_range = configs.at(vdx)->get<Rangef>("blob_size_range").value();
     const int track_threshold = configs.at(vdx)->get<int>("track_threshold").value();
     SETTING(cm_per_pixel) = cms_per_pixel[file.get()];
     
     for(size_t i=0; i<f.n(); ++i) {
     auto b = pv::Blob::Make(f.mask().at(i), f.pixels().at(i));
     auto recount = b->recount(track_threshold, *backgrounds.at(vdx));
     
     if(recount < blob_size_range.start * 0.1 || recount > blob_size_range.end * 5)
     continue;
     
     auto id = b->blob_id();
     
     b->transfer_backgrounds(*backgrounds.at(vdx), new_background, offset);
     b->add_offset(offset);
     
     if(!new_background.bounds().contains(b->bounds()))
     {
     auto str = Meta::toStr(*b);
     FormatWarning(str.c_str()," out of bounds for background ",new_background.bounds().width,"x",new_background.bounds().height);
     
     } else {
     meta[frame][b->blob_id()] = Source{ vdx, frame, id };
     
     
     ptrs.push_back(b);
     indexes.push_back(vdx);
     }
     }
     }
     }*/
    
    auto timestamp_offset = output.length() == 0_f ? timestamp_t(0) : output.last_frame().timestamp();
    merge_mode_t::Class merge_mode = SETTING(merge_mode);
    
    for (Frame_t frame=0_f; frame<min_length; ++frame) {
        pv::Frame f, o;
        if(SETTING(terminate))
            break;
        
        std::vector<pv::BlobPtr> ptrs;
        std::vector<uint64_t> indexes;
        
        for(uint64_t vdx = 0; vdx < files.size(); ++vdx) {
            auto &file = files.at(vdx);
            file->read_frame(f, frame);
            if(!vdx) o.set_timestamp(timestamp_offset + f.timestamp());
            //o.set_timestamp(start_time + f.timestamp() - file->start_timestamp());
            
            Vec2 offset = merge_mode == merge_mode_t::centered ? Vec2((Size2(average) - Size2(file->average())) * 0.5) : Vec2(0);
            Vec2 scale = merge_mode == merge_mode_t::centered ? Vec2(1) : Vec2(Size2(average).div(Size2(file->average())));
            auto blob_size_range = configs.at(vdx)->at("blob_size_range").value<Rangef>();
            const int track_threshold = configs.at(vdx)->at("track_threshold").value<int>();
            SETTING(cm_per_pixel) = cms_per_pixel[file.get()];
            
            for(size_t i=0; i<f.n(); ++i) {
                auto b = f.steal_blob(i);
                auto recount = b->recount(track_threshold, *backgrounds.at(vdx));
                
                if(recount < blob_size_range.start * 0.1 || recount > blob_size_range.end * 5)
                    continue;
                
                auto id = b->blob_id();
                
                b->transfer_backgrounds(*backgrounds.at(vdx), new_background, offset);
                b->scale_coordinates(scale);
                b->add_offset(offset);
                
                if(!new_background.bounds().contains(b->bounds()))
                {
                    auto str = Meta::toStr(*b);
                    FormatWarning(str.c_str()," out of bounds for background ",new_background.bounds().width,"x",new_background.bounds().height);
                    
                } else {
                    meta[frame][b->blob_id()] = Source{ vdx, frame, id };
                    
                    ptrs.push_back(std::move(b));
                    indexes.push_back(vdx);
                }
            }
        }
        
        // collect cliques of potentially overlapping blobs
        std::vector<std::vector<pv::BlobPtr>> cliques;
        std::vector<bool> viewed;
        viewed.resize(ptrs.size());
        
        for(size_t i=0; i<ptrs.size(); ++i) {
            if (viewed[i])
                continue;
            
            std::vector<pv::BlobPtr> clique;
            clique.emplace_back(std::move(ptrs.at(i)));
            viewed[i] = true;
            
            for (size_t j=i+1; j<ptrs.size(); ++j) {
                if(!ptrs.at(i) || viewed.at(j) /*|| indexes.at(j) == indexes.at(i)*/)
                    continue;
                
                if(ptrs.at(i)->bounds().overlaps(ptrs.at(j)->bounds())) {
                    viewed.at(j) = true;
                    clique.emplace_back(std::move(ptrs.at(j)));
                }
            }
            
            cliques.emplace_back(std::move(clique));
        }
        
        for(auto &clique : cliques) {
            if(clique.size() == 1 || !merge_overlapping_blobs) {
                for(auto &&b : clique) {
                    o.add_object(blob::Pair{
                        std::move(b->steal_lines()),
                        std::move(b->pixels()),
                        b->flags()
                    });
                }
                
            } else {
                Bounds bounds(FLT_MAX, FLT_MAX, 0, 0);
                Bounds test(FLT_MAX, FLT_MAX, 0, 0);
                for(auto &b : clique) {
                    bounds.combine(b->bounds());
                    test << (Vec2) min(test.pos(), b->bounds().pos());
                    test << (Size2)max(test.size(), b->bounds().pos() + b->bounds().size());
                }
                test << Size2(test.size() - test.pos());
                
                if(bounds != test)
                    print("why");
                
                assert(!clique.empty());
                
                cv::Mat mat = cv::Mat::zeros(bounds.height, bounds.width, CV_8UC1);
                
                for(auto &b: clique) {
                    auto [pos, image] = b->image(NULL, Bounds(bounds.pos(), Size2(mat)));
                    pos -= bounds.pos();
                    
                    // blend image into combined image
                    for (uint x=0; x<image->cols; ++x) {
                        for (uint y=0; y<image->rows; ++y) {
                            assert(Rangef(0, mat.cols).contains(x + pos.x));
                            assert(Rangef(0, mat.rows).contains(y + pos.y));
                            
                            assert(image->cols * y + x < image->size());
                            
                            auto &pb = mat.at<uchar>(y + pos.y, x + pos.x);
                            auto &pi = image->data()[image->cols * y + x];
                            if(!pb) pb = pi;
                            else {
                                float alphai = pi > 0 ? 1 - pi / 255.f : 0;
                                float alphab = pb > 0 ? 1 - pb / 255.f : 0;
                                
                                pb = saturate((int)roundf((float(pi) * alphai + float(pb) * alphab) / (alphai + alphab)), 0, 255);
                            }
                        }
                    }
                }
                
                auto blobs = CPULabeling::run(mat);
                for(auto && pair : blobs) {
                    for(auto &line : *pair.lines) {
                        line.x0 += bounds.pos().x;
                        line.x1 += bounds.pos().x;
                        line.y += bounds.pos().y;
                    }
                    o.add_object(std::move(pair));
                    //pv::Blob::Make(lines, pixels);
                }
                //cv::imshow("blended", mat);
                //cv::waitKey(10);
            }
        }
        
#ifndef NDEBUG
        for(size_t i=0; i<f.n(); ++i) {
            assert(viewed[i]);
        }
#endif
        
        output.add_individual(std::move(o));
        
        if(frame.get() % size_t(min_length.get() * 0.1) == 0) {
            print("merging ", frame,"/",min_length);
        }
    }
    
    output.close();
    
    {
        pv::File video(output.filename(), pv::FileMode::READ);
        video.print_info();
    }
}
