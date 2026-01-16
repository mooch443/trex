#include <commons.pc.h>
#include <tracker/misc/default_config.h>
#include <misc/GlobalSettings.h>
#include <misc/default_config.h>

#include <algorithm>
#include <cctype>
#include <iostream>
#include <locale>
#include <string>
#include <string_view>
#include <utility>
#include <vector>

namespace {

std::string normalize_doc(std::string_view input) {
    std::string out;
    out.reserve(input.size());
    bool pending_space = false;
    for (unsigned char ch : input) {
        if (std::isspace(ch)) {
            if (!out.empty()) {
                pending_space = true;
            }
            continue;
        }
        if (pending_space) {
            out.push_back(' ');
            pending_space = false;
        }
        out.push_back(static_cast<char>(ch));
    }
    return out;
}

} // namespace

int main(int argc, char** argv) {
    SETTING(quiet) = true;
    (void)argc;
    (void)argv;

    const char* locale = "C";
    std::locale::global(std::locale(locale));

    ::default_config::register_default_locations();
    cmn::GlobalSettings::write([](cmn::Configuration& config) {
        ::default_config::get(config);
    });

    auto docs = cmn::GlobalSettings::read([](const cmn::Configuration& config) {
        return config.docs;
    });

    std::vector<std::pair<std::string, std::string>> rows;
    rows.reserve(docs.size());
    for (const auto& entry : docs) {
        rows.emplace_back(entry.first, normalize_doc(entry.second));
    }

    std::sort(rows.begin(), rows.end(), [](const auto& lhs, const auto& rhs) {
        return lhs.first < rhs.first;
    });

    for (const auto& [key, doc] : rows) {
        std::cout << key;
        //if (!doc.empty()) {
            std::cout << "\t" << doc;
        //}
        std::cout << "\n";
    }

    return 0;
}
