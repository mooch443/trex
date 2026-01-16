#include <commons.pc.h>
#include <misc/stringutils.h>
#include <misc/GlobalSettings.h>

#include <algorithm>
#include <cctype>
#include <fstream>
#include <iostream>
#include <locale>
#include <optional>
#include <string>
#include <unordered_map>
#include <vector>

using namespace cmn;

namespace {

enum class SortKey {
    Importance,
    TermFrequency,
    DocFrequency
};

enum class SearchMethod {
    Compare,
    WithDocs,
    NamesOnly
};

struct Options {
    std::string input_path;
    std::string search_text;
    size_t top = 20;
    size_t min_freq = 1;
    SortKey sort_key = SortKey::Importance;
    SearchMethod search_method = SearchMethod::Compare;
    bool show_top_terms = true;
};

struct TermRow {
    std::string term;
    int term_freq = 0;
    int doc_freq = 0;
    double importance = 0.0;
};

void print_usage(const char* argv0) {
    cmn::Print("Usage:");
    cmn::Print("  ", argv0, " --input <file> [options]");
    cmn::Print("  ", argv0, " <file> [options]");
    cmn::Print("");
    cmn::Print("Options:");
    cmn::Print("  -i, --input <file>     Corpus file (one entry per line). Use '-' for stdin.");
    cmn::Print("                         Line format: <name> <docs...>");
    cmn::Print("  -s, --search <text>    Run text_search and print ranked results.");
    cmn::Print("  -t, --top <N>          Rows to show (0 = all). Default: 20.");
    cmn::Print("  -m, --min-freq <N>     Minimum term frequency to include. Default: 1.");
    cmn::Print("  -b, --by <key>         Sort by: importance|freq|docfreq (idf|tf|df).");
    cmn::Print("  --choose-method <id>   Choose ranking method: compare|with-docs|names-only.");
    cmn::Print("  --no-docs              Alias for --choose-method names-only.");
    cmn::Print("  --no-top-terms         Do not print the 'Top terms' summary and table.");
    cmn::Print("  -h, --help             Show this help.");
}

bool parse_size_t(const std::string& value, size_t* out) {
    try {
        size_t idx = 0;
        unsigned long parsed = std::stoul(value, &idx, 10);
        if (idx != value.size()) {
            return false;
        }
        *out = static_cast<size_t>(parsed);
        return true;
    } catch (...) {
        return false;
    }
}

bool parse_sort_key(const std::string& value, SortKey* out) {
    if (value == "importance" || value == "idf") {
        *out = SortKey::Importance;
        return true;
    }
    if (value == "freq" || value == "tf") {
        *out = SortKey::TermFrequency;
        return true;
    }
    if (value == "docfreq" || value == "df") {
        *out = SortKey::DocFrequency;
        return true;
    }
    return false;
}

bool parse_search_method(const std::string& value, SearchMethod* out) {
    if (value == "compare") {
        *out = SearchMethod::Compare;
        return true;
    }
    if (value == "with-docs" || value == "withdocs" || value == "docs") {
        *out = SearchMethod::WithDocs;
        return true;
    }
    if (value == "names-only" || value == "names" || value == "no-docs" || value == "nodocs") {
        *out = SearchMethod::NamesOnly;
        return true;
    }
    return false;
}

bool parse_args(int argc, char** argv, Options* options) {
    if (argc <= 1) {
        print_usage(argv[0]);
        return false;
    }

    for (int i = 1; i < argc; ++i) {
        std::string arg = argv[i];
        if (arg == "-h" || arg == "--help") {
            print_usage(argv[0]);
            return false;
        }
        if (arg == "-i" || arg == "--input") {
            if (i + 1 >= argc) {
                std::cerr << "Missing value for " << arg << ".\n";
                return false;
            }
            options->input_path = argv[++i];
            continue;
        }
        if (arg == "-s" || arg == "--search") {
            if (i + 1 >= argc) {
                std::cerr << "Missing value for " << arg << ".\n";
                return false;
            }
            options->search_text = argv[++i];
            continue;
        }
        if (arg == "-t" || arg == "--top") {
            if (i + 1 >= argc) {
                std::cerr << "Missing value for " << arg << ".\n";
                return false;
            }
            size_t value = 0;
            if (!parse_size_t(argv[++i], &value)) {
                std::cerr << "Invalid --top value.\n";
                return false;
            }
            options->top = value;
            continue;
        }
        if (arg == "-m" || arg == "--min-freq") {
            if (i + 1 >= argc) {
                std::cerr << "Missing value for " << arg << ".\n";
                return false;
            }
            size_t value = 0;
            if (!parse_size_t(argv[++i], &value)) {
                std::cerr << "Invalid --min-freq value.\n";
                return false;
            }
            options->min_freq = value;
            continue;
        }
        if (arg == "-b" || arg == "--by") {
            if (i + 1 >= argc) {
                std::cerr << "Missing value for " << arg << ".\n";
                return false;
            }
            if (!parse_sort_key(argv[++i], &options->sort_key)) {
                std::cerr << "Invalid --by value.\n";
                return false;
            }
            continue;
        }
        if (arg == "--choose-method") {
            if (i + 1 >= argc) {
                std::cerr << "Missing value for " << arg << ".\n";
                return false;
            }
            if (!parse_search_method(argv[++i], &options->search_method)) {
                std::cerr << "Invalid --choose-method value.\n";
                return false;
            }
            continue;
        }
        if (arg == "--no-docs") {
            options->search_method = SearchMethod::NamesOnly;
            continue;
        }
        if (arg == "--no-top-terms") {
            options->show_top_terms = false;
            continue;
        }
        if (!arg.empty() && arg[0] == '-') {
            std::cerr << "Unknown option: " << arg << "\n";
            return false;
        }
        if (options->input_path.empty()) {
            options->input_path = arg;
        } else {
            std::cerr << "Unexpected argument: " << arg << "\n";
            return false;
        }
    }

    if (options->input_path.empty()) {
        std::cerr << "Missing input file.\n";
        return false;
    }

    return true;
}

std::string trim_copy(std::string value) {
    auto not_space = [](unsigned char ch) { return std::isspace(ch) == 0; };
    auto begin = std::find_if(value.begin(), value.end(), not_space);
    auto end = std::find_if(value.rbegin(), value.rend(), not_space).base();
    if (begin >= end) {
        return "";
    }
    return std::string(begin, end);
}

bool load_corpus(const std::string& input_path,
                 std::vector<std::string>* names,
                 std::vector<std::string>* docs) {
    std::istream* input = nullptr;
    std::ifstream file;
    if (input_path == "-") {
        input = &std::cin;
    } else {
        file.open(input_path);
        if (!file.is_open()) {
            std::cerr << "Unable to open input file: " << input_path << "\n";
            return false;
        }
        input = &file;
    }

    std::string line;
    while (std::getline(*input, line)) {
        line = trim_copy(line);
        if (!line.empty()) {
            size_t split = 0;
            while (split < line.size() && !std::isspace(static_cast<unsigned char>(line[split]))) {
                ++split;
            }
            std::string name = line.substr(0, split);
            std::string doc;
            if (split < line.size()) {
                doc = trim_copy(line.substr(split));
            }
            names->push_back(std::move(name));
            docs->push_back(std::move(doc));
        }
    }

    return true;
}

double metric_value(const TermRow& row, SortKey key) {
    switch (key) {
        case SortKey::TermFrequency:
            return static_cast<double>(row.term_freq);
        case SortKey::DocFrequency:
            return static_cast<double>(row.doc_freq);
        case SortKey::Importance:
        default:
            return row.importance;
    }
}

std::string truncate_term(const std::string& term, size_t width) {
    if (term.size() <= width) {
        return term;
    }
    if (width <= 1) {
        return term.substr(0, width);
    }
    return term.substr(0, width - 1) + "~";
}

std::string make_bar(double value, double max_value, size_t width) {
    if (max_value <= 0.0 || width == 0) {
        return "";
    }
    double ratio = value / max_value;
    if (ratio <= 0.0) {
        return "";
    }
    size_t count = static_cast<size_t>(ratio * static_cast<double>(width));
    if (count == 0) {
        count = 1;
    }
    return std::string(count, '#');
}

const char* sort_key_label(SortKey key) {
    switch (key) {
        case SortKey::TermFrequency:
            return "freq";
        case SortKey::DocFrequency:
            return "docfreq";
        case SortKey::Importance:
        default:
            return "importance";
    }
}

void print_search_row(size_t display_rank,
                      const std::string& name,
                      const std::string& doc,
                      std::optional<size_t> other_rank) {
    if (other_rank.has_value()) {
        if (doc.empty()) {
            cmn::Print(display_rank, " (other ",*other_rank, ")\t[", name, "]");
        } else {
            cmn::Print(display_rank, " (other ",*other_rank, ")\t[", name, "]\t", cmn::utils::ShortenText(doc, 100));
        }
    } else {
        if (doc.empty()) {
            cmn::Print(display_rank, " [", name, "]");
        } else {
            cmn::Print(display_rank, " [", name, "]\t", cmn::utils::ShortenText(doc, 100));
        }
    }
}

} // namespace

int main(int argc, char** argv) {
    const char* locale = "C";
    std::locale::global(std::locale(locale));

    SETTING(quiet) = true;
    
    Options options;
    if (!parse_args(argc, argv, &options)) {
        return 1;
    }

    std::vector<std::string> names;
    std::vector<std::string> docs;
    if (!load_corpus(options.input_path, &names, &docs)) {
        return 1;
    }

    if (names.empty()) {
        std::cerr << "Input corpus is empty.\n";
        return 1;
    }

    PreprocessedDataWithDocs data = preprocess_corpus(names, docs);
    std::unordered_map<std::string, int> term_frequency;
    for (const auto& tokens : data.tokenizedNames) {
        for (const auto& term : tokens) {
            ++term_frequency[term];
        }
    }
    for (const auto& tokens : data.tokenizedDocs) {
        for (const auto& term : tokens) {
            ++term_frequency[term];
        }
    }

    std::vector<TermRow> rows;
    rows.reserve(data.docFrequency.size());
    for (const auto& [term, freq] : data.docFrequency) {
        int tf = 0;
        auto tf_it = term_frequency.find(term);
        if (tf_it != term_frequency.end()) {
            tf = tf_it->second;
        }
        if (static_cast<size_t>(tf) < options.min_freq) {
            continue;
        }
        TermRow row;
        row.term = term;
        row.term_freq = tf;
        row.doc_freq = freq;
        auto imp_it = data.termImportance.find(term);
        if (imp_it != data.termImportance.end()) {
            row.importance = imp_it->second;
        }
        rows.push_back(row);
    }

    auto metric = [&](const TermRow& row) { return metric_value(row, options.sort_key); };
    std::sort(rows.begin(), rows.end(), [&](const TermRow& a, const TermRow& b) {
        double av = metric(a);
        double bv = metric(b);
        if (av > bv) {
            return true;
        }
        if (av < bv) {
            return false;
        }
        return a.term < b.term;
    });

    size_t display_count = options.top == 0 ? rows.size() : std::min(options.top, rows.size());
    const size_t max_term_width = 28;
    size_t term_width = 4;
    for (size_t i = 0; i < display_count; ++i) {
        term_width = std::max(term_width, std::min(rows[i].term.size(), max_term_width));
    }

    double max_metric = 0.0;
    for (size_t i = 0; i < display_count; ++i) {
        max_metric = std::max(max_metric, metric(rows[i]));
    }

    if (options.show_top_terms) {
        cmn::Print("Corpus entries: ", names.size());
        cmn::Print("Unique terms: ", data.docFrequency.size());
        if (options.top == 0) {
            cmn::Print("Top terms by ", sort_key_label(options.sort_key),
                       " (min freq ", options.min_freq, ", all rows)");
        } else {
            cmn::Print("Top terms by ", sort_key_label(options.sort_key),
                       " (min freq ", options.min_freq, ", ", display_count, " rows)");
        }
        cmn::Print("");

        if (display_count == 0) {
            cmn::Print("No terms match the selected filters.");
        } else {
            cmn::Print("term\t", "tf\t", "df\t", "idf\t", "bar");

            for (size_t i = 0; i < display_count; ++i) {
                const TermRow& row = rows[i];
                cmn::Print(truncate_term(row.term, term_width), "\t",
                           row.term_freq, "\t",
                           row.doc_freq, "\t",
                           cmn::dec<3>(row.importance), "\t",
                           make_bar(metric(row), max_metric, 30));
            }
        }
    }

    if (!options.search_text.empty()) {
        std::vector<int> with_docs_indexes;
        std::vector<int> names_only_indexes;

        if (options.search_method == SearchMethod::Compare
            || options.search_method == SearchMethod::WithDocs) {
            with_docs_indexes = text_search(options.search_text, names, docs, data);
        }

        PreprocessedData names_only_data;
        if (options.search_method == SearchMethod::Compare
            || options.search_method == SearchMethod::NamesOnly)
        {
            names_only_data = preprocess_corpus(names);
            names_only_indexes = text_search(options.search_text, names, names_only_data);
        }

        if (options.search_method == SearchMethod::Compare) {
            std::vector<size_t> rank_with_docs(names.size(), 0);
            std::vector<size_t> rank_names_only(names.size(), 0);
            for (size_t i = 0; i < with_docs_indexes.size(); ++i) {
                int idx = with_docs_indexes[i];
                if (idx >= 0 && static_cast<size_t>(idx) < names.size()) {
                    rank_with_docs[static_cast<size_t>(idx)] = i + 1;
                }
            }
            for (size_t i = 0; i < names_only_indexes.size(); ++i) {
                int idx = names_only_indexes[i];
                if (idx >= 0 && static_cast<size_t>(idx) < names.size()) {
                    rank_names_only[static_cast<size_t>(idx)] = i + 1;
                }
            }

            size_t count_with_docs = options.top == 0
                ? with_docs_indexes.size()
                : std::min(options.top, with_docs_indexes.size());
            size_t count_names_only = options.top == 0
                ? names_only_indexes.size()
                : std::min(options.top, names_only_indexes.size());

            cmn::Print("");
            cmn::Print("Comparative search results for ", options.search_text);
            cmn::Print("Method: with-docs (showing names-only rank ",count_with_docs,")");
            for (size_t i = 0; i < count_with_docs; ++i) {
                int index = with_docs_indexes[i];
                if (index < 0 || static_cast<size_t>(index) >= docs.size()) {
                    FormatWarning("Index at ", i, " (", index, ") is outside the range of docs ", docs.size());
                    continue;
                }
                size_t other_rank = rank_names_only[static_cast<size_t>(index)];
                std::optional<size_t> other =
                    other_rank > 0 ? std::optional<size_t>(other_rank) : std::nullopt;
                print_search_row(i + 1, names[index], docs[index], other);
            }

            cmn::Print("");
            cmn::Print("Method: names-only (showing with-docs rank, ",count_names_only,")");
            for (size_t i = 0; i < count_names_only; ++i) {
                int index = names_only_indexes[i];
                if (index < 0 || static_cast<size_t>(index) >= names.size()) {
                    FormatWarning("Index at ", i, " (", index, ") is outside the range of names ", names.size());
                    continue;
                }
                size_t other_rank = rank_with_docs[static_cast<size_t>(index)];
                std::optional<size_t> other =
                    other_rank > 0 ? std::optional<size_t>(other_rank) : std::nullopt;
                print_search_row(i + 1, names[index], docs[index], other);
            }
        } else if (options.search_method == SearchMethod::WithDocs) {
            size_t count = options.top == 0
                ? with_docs_indexes.size()
                : std::min(options.top, with_docs_indexes.size());
            cmn::Print("");
            cmn::Print("Search results (with-docs) for ", options.search_text, " (",
                       count, " shown)");
            for (size_t i = 0; i < count; ++i) {
                int index = with_docs_indexes[i];
                if (index < 0 || static_cast<size_t>(index) >= names.size()) {
                    FormatWarning("Index at ", i, " (", index, ") is outside the range of names ", names.size());
                    continue;
                }
                print_search_row(i + 1, names[index], docs[index], std::nullopt);
            }
        } else {
            size_t count = options.top == 0
                ? names_only_indexes.size()
                : std::min(options.top, names_only_indexes.size());
            cmn::Print("");
            cmn::Print("Search results (names-only) for ", options.search_text, " (",
                       count, " shown)");
            for (size_t i = 0; i < count; ++i) {
                int index = names_only_indexes[i];
                if (index < 0 || static_cast<size_t>(index) >= names.size()) {
                    FormatWarning("Index at ", i, " (", index, ") is outside the range of names ", docs.size());
                    continue;
                }
                print_search_row(i + 1, names[index], docs[index], std::nullopt);
            }
        }
    }

    return 0;
}

