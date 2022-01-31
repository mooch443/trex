#include "Debug.h"
#include "Printable.h"
#include "../miscutils.h"
#include "../stringutils.h"

#include <stdio.h>
#include <stdlib.h>
#include <fcntl.h>
#ifndef WIN32
#include <unistd.h>
#endif

#include "DynamicTree.h"

#define DEBUG_PRINT_STACKTRACES false
#define _XCODE_COLORS_DISABLE true

#ifndef WIN32
#include <unistd.h>
#include <execinfo.h>
#endif

#ifdef __unix__
#include <stdexcept>
#endif

#include <iostream>
#include <exception>
#include <iomanip>
#include <mutex>
#include <sstream>
#include <string>

namespace DEBUG {
    bool& runtime_quiet() {
        static bool runtime_quiet = false;
        return runtime_quiet;
    }
    
    void set_runtime_quiet() {
        runtime_quiet() = true;
    }
    
    std::mutex& debug_mutex() {
        static std::mutex* DEBUG_MUTEX = new std::mutex; // this is never freed, but that's okay :) avoids some release-hierarchy crashes, I think.
        return *DEBUG_MUTEX;
    }
    
#if __APPLE__
    FILE* out = stdout, *err = stderr;
    
    struct RuntimeCheck {
        ~RuntimeCheck() {
            if(out && out != stdout) {
                if(!runtime_quiet())
                    printf("\nPROGRAM TERMINATED\n");
                
                fflush(out); fclose(out);
                fflush(err); fclose(err);
            }
        }
    };
    
    static RuntimeCheck runtimecheck;
#endif
    
    struct DebugCallback {
        std::vector<DEBUG_TYPE> _types;
        std::function<void(const StatusMsg*, const std::string&)> _callback;
    };
    
    static std::vector<DebugCallback*> debug_callbacks;
    static std::vector<std::string> debug_keywords {
        "int", "float", "double", "long", "bool", "string",
        "true", "false"
    };
    
    void* SetDebugCallback(const std::vector<DEBUG_TYPE>& types, const std::function<void(const StatusMsg*, const std::string&)>& callback) {
        debug_mutex().lock();
        
        auto *ptr = new DebugCallback;
        ptr->_types = types;
        ptr->_callback = callback;
        
        debug_callbacks.push_back(ptr);
        
        debug_mutex().unlock();
        
        return (void*)ptr;
    }

void UnsetDebugCallback(void * callback) {
    auto ptr = (DebugCallback*)callback;
    if(!ptr)
        return;
    
    debug_mutex().lock();
    
    auto it = std::find(debug_callbacks.begin(), debug_callbacks.end(), ptr);
    if(it != debug_callbacks.end()) {
        debug_callbacks.erase(it);
        delete ptr;
    } else
        printf("[EXCEPTION] Cannot find debug callback to delete it.\n");
    
    debug_mutex().unlock();
}
    
    
#if __APPLE__
#define XCODE_COLORS "OS_ACTIVITY_DT_MODE"
#endif
    
	// different states of console colors - unix colors are for usual terminals,
	// but xcode can only display another color format (if XcodeColors is installed)
    enum CONSOLE_COLOR_STATE {
        DISABLE_COLORS = 0,
        ENABLE_XCODE,
        ENABLE_UNIX,
        ENABLE_WINDOWS
	} ENABLE_COLORS = CONSOLE_COLOR_STATE::DISABLE_COLORS;
    
    //! the last outputted color - can be global, because we only have one console
    CONSOLE_COLORS LAST_COLOR = CONSOLE_COLORS::BLACK;
    
	//! will be set to true upon initialization by a call to some output message
    bool console_initted = false;
    
    //! changes the current color of the console (inherited to all following characters)
    void setColor(CONSOLE_COLORS value);
    
    // this value will be added to each inserted nodes' offset so that we can
    // use 0 as root node with normal color - have to access the nodeForOffset with
    // -NODE_OFFSET in the end
#define NODE_OFFSET 1
    
    void insert_end(OrderedTree<PARSE_OBJECTS> &tree, TreeNode<PARSE_OBJECTS> **current_node, size_t &i, PARSE_OBJECTS X, int Y)
    {
        if(*current_node) {
            auto node = new TreeNode<PARSE_OBJECTS>(PARSE_OBJECTS::END, (size_t)((int)i + Y + NODE_OFFSET));
            if(!(*current_node)->addChild(node))
                delete node;
        }
        
        while (*current_node && (*current_node)->value != X) {
            *current_node = (*current_node)->parent;
        }
        
        if (!(*current_node))
            *current_node = tree.root();
        if ((*current_node)->value == X)
            *current_node = (*current_node)->parent;
    }
    
    bool insert_single(OrderedTree<PARSE_OBJECTS> &tree, TreeNode<PARSE_OBJECTS> **current_node, TreeNode<PARSE_OBJECTS> *e) {
        if(*current_node) {
            if(!(*current_node)->addChild(e))
                return false;
            
        } else {
            if(!tree.addNode(e))
                return false;
            else
                *current_node = e;
        }
        
        return true;
    }
    
    bool insert_start(OrderedTree<PARSE_OBJECTS> &tree, TreeNode<PARSE_OBJECTS> **current_node, TreeNode<PARSE_OBJECTS> *e) {
        if(insert_single(tree, current_node, e)) {
            *current_node = e;
            return true;
        }
        return false;
    }
    
    // macro for inserting an end object at a given position
    // X is the value that has to be inserted, Y is the offset from current position
#define INSERT_END(X, Y) insert_end(tree, &current_node, i, X, Y)
    
    // inserts a single character with a certain color
#define INSERT_SINGLE(TYPE, POS) auto e = new TreeNode<PARSE_OBJECTS>(TYPE, POS+NODE_OFFSET); if(!insert_single(tree, &current_node, e)) delete e

    // inserts a node into the tree with the given offset and sets it as the
    // current node, so that all following characters will be colored in this way
#define INSERT_START(TYPE, POS) { \
auto e = new TreeNode<PARSE_OBJECTS>(TYPE, POS+NODE_OFFSET); \
if(!insert_start(tree, &current_node, e)) delete e; }
    
	//! will be called when debug functions are used for the first time
    void Init() {
#if __APPLE__
        char *xcode_colors = getenv(XCODE_COLORS);
        if (xcode_colors && (strcmp(xcode_colors, "YES")==0))
        {
            ENABLE_COLORS = CONSOLE_COLOR_STATE::DISABLE_COLORS;
        } else {
            if(isatty(fileno(stdout))) {
                ENABLE_COLORS = CONSOLE_COLOR_STATE::ENABLE_UNIX;
            }
        }
#elif !_WIN32
        if(isatty(fileno(stdout))) {
            ENABLE_COLORS = CONSOLE_COLOR_STATE::ENABLE_UNIX;
        }
#else
        ENABLE_COLORS = CONSOLE_COLOR_STATE::ENABLE_WINDOWS;
#endif
    }
    
#if __APPLE__
    inline void selective_print(const char *cmd, ...) {
        va_list args;
        va_start(args, cmd);
        
        std::string str;
        DEBUG::ParseFormatString(str, cmd, args);
        
        va_end(args);
        
        if(out && out != stdout) {
            fwrite(str.data(), sizeof(char), str.length(), out);
            fflush(out);
        }
        printf("%s", str.c_str());
    }
#else
#define selective_print printf
#endif
    
	void setColor(CONSOLE_COLORS value) {
        if (LAST_COLOR == value) {
            return;
        }
        LAST_COLOR = value;
        
#ifdef _WIN32
        WORD clr = (WORD)value;
        if (value == CONSOLE_COLORS::BLACK) {
            clr = FOREGROUND_RED | FOREGROUND_GREEN | FOREGROUND_BLUE | FOREGROUND_INTENSITY;
        }

        if(ENABLE_COLORS == CONSOLE_COLOR_STATE::ENABLE_WINDOWS) {
            HANDLE hConsole = GetStdHandle(STD_OUTPUT_HANDLE);
            SetConsoleTextAttribute(hConsole, clr);
        }
#else
        
        if (ENABLE_COLORS == CONSOLE_COLOR_STATE::ENABLE_XCODE) {
#define COLOR(R, G, B) { r = R; g = G; b = B; break; }
            
            int r, g, b;
            
            switch (value) {
                case YELLOW: COLOR(150, 150, 0)
                case DARK_RED: COLOR(150, 0, 0)
                case GREEN: COLOR(0, 150, 0)
                case GRAY: COLOR(50, 50, 50)
                case CYAN: COLOR(0, 100, 150)
                case DARK_CYAN: COLOR(0, 50, 100)
                case PINK: COLOR(200, 0, 150)
                case BLUE: COLOR(0, 0, 200)
                case BLACK: COLOR(0, 0, 0)
                case DARK_GRAY: COLOR(50, 50, 50)
                case WHITE: COLOR(130, 130, 130)
                case DARK_GREEN: COLOR(0, 50, 0)
                case PURPLE: COLOR(150, 0, 150)
                case RED: COLOR(200, 0, 0)
                case LIGHT_BLUE: COLOR(65, 162, 176)
                    
                default:
                    COLOR(0, 0, 0);
                    break;
            }
            
#if __APPLE__
            if(out && out != stdout)
                fprintf(out, "\033[fg%d,%d,%d;", r, g, b);
            else
#endif
                printf("\033[fg%d,%d,%d;", r, g, b);
            
#undef COLOR
            
        } else if(ENABLE_COLORS == CONSOLE_COLOR_STATE::ENABLE_UNIX) {
#define COLOR(PREFIX, FINAL) { prefix = PREFIX; final = FINAL; break; }
            
            int final = 0, prefix = 22;
            
            switch (value) {
                case YELLOW: COLOR(1, 33)
                case DARK_RED: COLOR(22, 31)
                case GREEN: COLOR(22, 32)
#ifdef __APPLE__
                case GRAY: COLOR(1, 30)
#else
                case GRAY: COLOR(22, 37)
#endif
                case CYAN: COLOR(1,36)
                case DARK_CYAN: COLOR(22, 36)
                case PINK: COLOR(22,35)
                case BLUE: COLOR(22,34)
                case BLACK: COLOR(22,30)
                case DARK_GRAY: COLOR(1,30)
                case WHITE: COLOR(1, 37)
                case DARK_GREEN: COLOR(1, 32)
                case PURPLE: COLOR(22, 35)
                case RED: COLOR(1, 31)
                case LIGHT_BLUE: COLOR(1, 34)
                    
                default:
                    break;
            }
            
#if __APPLE__
            if(out && out != stdout)
                fprintf(out, "\033[%02d;%dm", prefix, final);
            else
#endif
                printf("\033[%02d;%dm", prefix, final);
#undef COLOR
        }
#endif
	}
    
	void ParseStatusMessage(StatusMsg*msg) {
        {
            std::lock_guard<std::mutex> lock(debug_mutex());
            if( !console_initted)
                Init();
        }
        
#define CURRENT_NODE_IS(VALUE) (current_node && current_node->value == VALUE)
#define CURRENT_NODE_NOT(VALUE) (!current_node || current_node->value != VALUE)
        
#define ALPHANUMERIC(LETTER) ((LETTER >= '0' && LETTER <= '9') || (tolower(LETTER) >= 'a' && tolower(LETTER) <= 'z') || LETTER == '.')
        
		// create tree and root node
		OrderedTree<PARSE_OBJECTS> tree;
		auto current_node = tree.addNode(PARSE_OBJECTS::NORMAL, 0);
        assert(current_node);
        
		// this is where the output goes
		std::stringstream ss;
        
		// iterate through message text and build tree
		long last_non_number = 0, word_start = 0, word_end = 0;
		bool in_word = false;
        
        // where the last string started
        long last_string_start = -1;
        
        char last_string_start_c = 0;
        char prev = 0;
        size_t i = 0;
        
        auto apply = [&](char current) {
            // find all words seperated by illegal characters
            if(ALPHANUMERIC(current) || current == '_') {
                if(!in_word) {
                    word_start = word_end = (long)i;
                    in_word = true;
                    
                } else word_end++;
                
            } else if(in_word) {
                in_word = false;
                
                auto word = msg->buf.substr(word_start, word_end-word_start+1);
                if(contains(debug_keywords, word)) {
                    INSERT_START(PARSE_OBJECTS::KEYWORD, word_start);
                    INSERT_END(PARSE_OBJECTS::KEYWORD, -1);
                }
            }
            
            if (CURRENT_NODE_IS(PARSE_OBJECTS::KEY)) {
                if(!ALPHANUMERIC(current)
                   && current != '_' // allow _ in classnames
                   && (current < '0' || current > '9') ) // allow numbers in class names
                {
                    INSERT_END(PARSE_OBJECTS::KEY, -1);
                }
                
            } else if CURRENT_NODE_IS(PARSE_OBJECTS::HEXDECIMAL) {
                if((current < '0' || current > '9')
                   && (tolower(current) < 'a' || tolower(current) > 'f')
                   && (tolower(current) != 'x')) {
                    INSERT_END(PARSE_OBJECTS::HEXDECIMAL, -1);
                }
                
            } else if ((current < '0' || current > '9')
                       && current != '.' && current != ';')
            {
                // parse numbers
                if(CURRENT_NODE_IS(PARSE_OBJECTS::NUMBER)) {
                    INSERT_END(PARSE_OBJECTS::NUMBER, -1);
                }
                
                last_non_number = i;
                
            }
            else if (current == '0'
                     && i < msg->buf.length()-1
                     && msg->buf[i+1] == 'x'
                     && CURRENT_NODE_NOT(PARSE_OBJECTS::NUMBER))
            {
                INSERT_START(PARSE_OBJECTS::HEXDECIMAL, i);
                
            }
            else if (current >= '0' && current <= '9'
                     && CURRENT_NODE_NOT(PARSE_OBJECTS::NUMBER)
                     && CURRENT_NODE_NOT(PARSE_OBJECTS::STRING)
                     && CURRENT_NODE_NOT(PARSE_OBJECTS::KEY)
                     && CURRENT_NODE_NOT(PARSE_OBJECTS::CLASSNAME))
                //&& last_non_number-word_start+1 <= 0)
            {
                bool validWord = last_non_number-word_start+1 <= 0;
                if(!validWord) {
                    int illegalCount = 0;
                    validWord = true;
                    
                    for (size_t p = (size_t)word_start; p < msg->buf.length(); p++) {
                        if(!ALPHANUMERIC(msg->buf[p]) && current != '_') {
                            // break at word end
                            break;
                        }
                        
                        if(msg->buf[p] < '0' || msg->buf[p] > '9') {
                            illegalCount++;
                            
                            if(illegalCount >= 2) {
                                validWord = false;
                                break;
                            }
                        }
                    }
                }
                
                if(validWord) {
                    size_t pos = (size_t)last_non_number;
                    if (pos && msg->buf[pos] != '.')
                        ++pos;
                    
                    INSERT_START(PARSE_OBJECTS::NUMBER, pos);
                }
            }
            
            switch (current) {
#ifdef _WIN32
                case 'ä': current = 132; break;
                case 'Ä': current = 142; break;
                case 'ö': current = 148; break;
                case 'Ö': current = 153; break;
                case 'ü': current = 129; break;
                case 'Ü': current = 154; break;
                case 'ß': current = 225; break;
#endif
                case '"':
                case '\'':
                    if(prev == '\\')
                        break;
                    
                    if(last_string_start != -1 && last_string_start_c == current) {
                        for(long k=last_string_start; k<=(long)i; k++) {
                            TreeNode<DEBUG::PARSE_OBJECTS> * node = tree.nodeForOffset(k+NODE_OFFSET);
                         
                            if(node && node != tree.root()) {
                                if(current_node && (current_node == node || current_node->is_child_of(node)))
                                    current_node = node->parent;
                                if(!current_node)
                                    current_node = tree.root();
                                
                                // dont use the current_node, if it is a child of node
                                // use parent of node instead
                                auto n = current_node;
                                while(n) {
                                    if (n == node) {
                                        current_node = node->parent;
                                        break;
                                    }
                                    
                                    n = n->parent;
                                }
                                
                                delete node;
                            }
                        }
                        
                        INSERT_START(PARSE_OBJECTS::STRING, last_string_start);
                        INSERT_END(PARSE_OBJECTS::STRING, 0);
                        
                        last_string_start = -1;
                    }
                    else if(last_string_start == -1) {
                        last_string_start = i;
                        last_string_start_c = current;
                    }
                    break;
            }
            
            if CURRENT_NODE_NOT(PARSE_OBJECTS::STRING) {
                switch (current) {
                        
                    case '\r': return false;
                    case '\n': break;
                        
                    case '(':
                    case '[': {
                        INSERT_START(PARSE_OBJECTS::BRACKETS, i);
                        break;
                    }
                        
                    case ')':
                    case ']':
                        INSERT_END(PARSE_OBJECTS::BRACKETS, 0);
                        break;
                        
                    case 0:
                    case ' ': {
                        break;
                    }
                        
                    case '<':
                    case '>':
                    case ':':
                        if (current == '<' || (current == ':' && i+1 < msg->buf.length() && msg->buf[i + 1] != ':' && (size_t)word_start != i)) {
                            TreeNode<DEBUG::PARSE_OBJECTS> *node;
                            long str_start = -1;
                            long off = 0;
                            
                            for(auto k=word_start; k<=word_end; k++) {
                                node = tree.nodeForOffset(k+NODE_OFFSET);
                                if(node && node != tree.root() && node->value != DEBUG::PARSE_OBJECTS::STRING) {
                                    if(str_start>-1) {
                                        if(node->value == DEBUG::PARSE_OBJECTS::END) {
                                            str_start = -1;
                                            off = k-word_start;
                                        }
                                        
                                    } else {
                                        if(current_node && (current_node == node || current_node->is_child_of(node)))
                                            current_node = node->parent;
                                        if(!current_node)
                                            current_node = tree.root();
                                        delete node;
                                    }
                                    
                                } else if(node && node != tree.root() && str_start == -1) {
                                    str_start = k;
                                }
                            }
                            
                            if(str_start != -1)
                                off = word_end-word_start + 1;
                            
                            // only add key if there is something left to be added
                            if(off+word_start < word_end) {
                                DEBUG::PARSE_OBJECTS type =
                                    (current == ':'
                                         && (i >= msg->buf.length() || msg->buf[i+1] != ':')
                                         && (i == 0 || msg->buf[i-1] != ':'))
                                    ? PARSE_OBJECTS::KEY
                                    : PARSE_OBJECTS::CLASSNAME;
                                
                                auto e = tree.addNode(type, word_start+off+NODE_OFFSET);
                                if(e)
                                    e->addChild(new TreeNode<PARSE_OBJECTS>(PARSE_OBJECTS::END, word_end+NODE_OFFSET));
                            }
                        }
                        
                        if (current == ':') {
                            // match ':' pairs
                            if ((i+1 < msg->buf.length() && msg->buf[i + 1] == ':')
                                || (i > 0 && msg->buf[i - 1] == ':'))
                            {
                                INSERT_SINGLE(PARSE_OBJECTS::CLASSSEPERATOR, i);
                            }
                            
                            if (i > 0 && msg->buf[i - 1] == ':') {
                                INSERT_START(PARSE_OBJECTS::KEY, i + 1);
                            }
                        }
                        else if (current == '<' || current == '>') {
                            INSERT_SINGLE(PARSE_OBJECTS::DESCRIPTION, i);
                        }
                        
                        break;
                }
            }
            
            return true;
        };

		for (; i < msg->buf.length(); i++) {
			char current = msg->buf[i];
            
			if(!apply(current))
                continue;
            
			ss << current;
            
            if(prev == '\\')
                prev = 0;
            else
                prev = current;
		}
        
        // end things properly
        apply(0);
        
        INSERT_END(PARSE_OBJECTS::NORMAL, 0);
        INSERT_SINGLE(PARSE_OBJECTS::NORMAL, i);

		// print out string
        std::lock_guard<std::mutex> lock(debug_mutex());
        
#if !_XCODE_COLORS_DISABLE
        if(ENABLE_COLORS != CONSOLE_COLOR_STATE::ENABLE_XCODE)
            printf("                                                                  \r");
#endif
        
        // parse message "header" and print it out
		CONSOLE_COLORS mainClr = CONSOLE_COLORS::RED;
        
		time_t rawtime;
		struct tm * timeinfo;
		char buffer[128];
        
		time(&rawtime);
		timeinfo = localtime(&rawtime);
        
		strftime(buffer, sizeof(buffer), "%H:%M:%S", timeinfo);
        
        bool save_printed = !debug_callbacks.empty();
        if (save_printed) {
            save_printed = false;
            for (auto ptr : debug_callbacks) {
                if (msg->force_callback || contains(ptr->_types, msg->type)) {
                    save_printed = true;
                    break;
                }
            }
        }
        
        std::stringstream printed;
		char c[128] = { 0 };
        
		switch (msg->type) {
            case TYPE_ERROR:
#ifdef _WIN32
				strcpy_s(c, "ERROR ");
#else
                //strlcpy(c, "ERROR ", sizeof(c));
                strcpy(c, "ERROR ");
#endif
                mainClr = CONSOLE_COLORS::RED;
                break;
            case TYPE_EXCEPTION:
#ifdef _WIN32
				strcpy_s(c, "EXCEPTION ");
#else
                strcpy(c, "EXCEPTION ");
                //strlcpy(c, "EXCEPTION ", sizeof(c));
#endif
                mainClr = CONSOLE_COLORS::RED;
                break;
            case TYPE_INFO:
                mainClr = CONSOLE_COLORS::CYAN;
                break;
            case TYPE_WARNING:
#ifdef _WIN32
				strcpy_s(c, "WARNING ");
#else
                strcpy(c, "WARNING ");
				//strlcpy(c, "WARNING ", sizeof(c));
#endif
                mainClr = CONSOLE_COLORS::YELLOW;
                break;
            case TYPE_CUSTOM: {
                std::string add = msg->prefix + " ";
                std::copy(add.begin(), add.end(), c);
                c[add.size()] = 0;
                
                mainClr = msg->color;
                break;
            }
                
            default:
                throw new std::logic_error("Cannot find type "+std::to_string(msg->type)+".");
		}
        
		setColor(mainClr);
        if(msg->line != -1 && msg->file != NULL) {
            selective_print("[%s%s %s:%d] ", c, buffer, msg->file, msg->line);
            printed << "[" << c << buffer << " " << msg->file << ":" << msg->line << "] ";
        } else {
            selective_print("[%s%s] ", c, buffer);
            printed << "[" << c << buffer << "] ";
        }
        
		std::string str = ss.str();
		for (unsigned int i = 0; i < str.length(); i++)
		{
			auto node = tree.nodeForOffset(i+NODE_OFFSET);
			PARSE_OBJECTS o = node ? node->value : PARSE_OBJECTS::NORMAL;
			if (o == PARSE_OBJECTS::END) o = node->parent->value;
            
			switch (o) {
                case PARSE_OBJECTS::BRACKETS:
                    setColor(CONSOLE_COLORS::WHITE);
                    break;
                case PARSE_OBJECTS::CLASSSEPERATOR:
#ifndef __unix__
                    setColor(CONSOLE_COLORS::DARK_GRAY);
#else
                    setColor(CONSOLE_COLORS::WHITE);
#endif
                    break;
                case PARSE_OBJECTS::NORMAL:
#ifdef __APPLE__
                    setColor(CONSOLE_COLORS::BLACK);
#else
                    setColor(CONSOLE_COLORS::WHITE);
#endif
                    break;
                case PARSE_OBJECTS::CLASSNAME:
#ifdef _WIN32
                    setColor(CONSOLE_COLORS::CYAN);
#else
                    setColor(CONSOLE_COLORS::LIGHT_BLUE);
#endif
                    
                    break;
                case PARSE_OBJECTS::NUMBER:
                    setColor(CONSOLE_COLORS::GREEN);
                    break;
                case PARSE_OBJECTS::STRING:
                    setColor(CONSOLE_COLORS::RED);
                    break;
                    
                case PARSE_OBJECTS::DESCRIPTION:
                    setColor(CONSOLE_COLORS::WHITE);
                    break;
                    
                case PARSE_OBJECTS::HEXDECIMAL:
                    setColor(CONSOLE_COLORS::BLUE);
                    break;
                    
                case PARSE_OBJECTS::KEY:
                    setColor(CONSOLE_COLORS::GRAY);
                    break;
                    
                case PARSE_OBJECTS::KEYWORD:
                    setColor(CONSOLE_COLORS::PURPLE);
                    break;
                    
                default:
                    break;
			}
            
            if (str.at(i)) {
                selective_print("%c", str.at(i));
                if(save_printed)
                    printed << str.at(i);
            }
		}
        
        setColor(BLACK);
		selective_print("\n");
        
        //if(save_printed)
        {
            std::string pr = printed.str();
            for (auto ptr : debug_callbacks) {
                if (msg->force_callback || contains(ptr->_types, msg->type)) {
                    ptr->_callback(msg, pr);
                }
            }
        }
        
		//tree.root()->print();
        
#undef INSERT_END
	}
    


enum State {
    text = 0,
    flags,
    width,
    precision,
    length,
    specifier
};

enum Justification {
    RIGHT,
    LEFT
};

struct ArgumentModifiers {
    // flags
    Justification justification;
    bool force_sign;
    bool space_if_no_sign;
    bool fill_padding_with_zeros;
    bool force_decimal_point;
    
    // width
	bool min_characters_enabled;
    size_t min_characters;
    bool use_last_int_for_width;
    
    // precision
    int precision;
    bool use_last_int_for_precision;
    
	// length
	bool long_int;
    
    void reset() {
        // flags
        justification = Justification::RIGHT;
        force_sign = false;
        space_if_no_sign = false;
        force_decimal_point = false;
        fill_padding_with_zeros = false;
        
        // width
		min_characters_enabled = false;
        min_characters = 0;
        use_last_int_for_width = false;
        
        // precision
        precision = -1;
        use_last_int_for_precision = false;
        
		// length
		long_int = false;
    }
};

    std::string format(const char *cmd, ...) {
        va_list args;
        va_start(args, cmd);
        
        std::string str;
        DEBUG::ParseFormatString(str, cmd, args);
        
        va_end(args);
        
        return str;
    }
    
void ParseFormatString(std::string& buffer, const char *cmd, va_list args) {
    std::stringstream format_buffer;
    State state = State::text;
    
    bool in_number = false;
    std::stringstream number;
    
    ArgumentModifiers modifiers;
    modifiers.reset();
    
    for (size_t i = 0, str_len=strlen(cmd); i < str_len; ) {
        if(state == State::text) {
            if(cmd[i] == '%') {
                state = State::flags;
				modifiers.reset();
            } else {
                buffer += cmd[i];
            }
            
            i++;
            
        } else if(state == State::flags) {
            switch (cmd[i]) {
                case '%':
                    buffer += '%';
                    state = State::text;
                    i++;
                    break;
                    
				case '-':
					modifiers.justification = Justification::LEFT;
					i++;
					break;
                    
				case '+':
					modifiers.force_sign = true;
					i++;
					break;
                    
				case ' ':
					modifiers.space_if_no_sign = true;
					i++;
					break;
                    
				case '#':
					modifiers.force_decimal_point = true;
					i++;
					break;
                    
				case '0':
					modifiers.fill_padding_with_zeros = true;
					i++;
					break;
                    
                default:
                    state = State::width;
                    break;
            }
            
        } else if(state == State::width) {
			if (cmd[i] == '*') {
				modifiers.use_last_int_for_width = true;
				i++;
			}
			else if (cmd[i] >= '0' && cmd[i] <= '9') {
				if (!in_number) {
					in_number = true;
					number.str(std::string());
					number.clear();
				}
				number << cmd[i];
				i++;
			}
			else if(in_number) {
				modifiers.min_characters_enabled = true;
                auto s = number.str();
				modifiers.min_characters = strtoul(s.c_str(), NULL, 0);
				in_number = false;
				number.str(std::string());
				number.clear();
                
			}
			else {
				state = State::precision;
			}
            
        } else if(state == State::precision) {
            if(cmd[i] == '.') {
                i++;
                in_number = true;
				number.str(std::string());
				number.clear();
                
            } else if(in_number && cmd[i] >= '0' && cmd[i] <= '9') {
                number << cmd[i];
                i++;
                
			}
			else if (cmd[i] == '*' && in_number) {
				in_number = false;
				number.str(std::string());
				number.clear();
				i++;
                
				modifiers.use_last_int_for_precision = true;
			}
			else if (in_number) {
				modifiers.precision = atoi(number.str().c_str());
				in_number = false;
				number.str(std::string());
				number.clear();
                
            } else {
                state = State::length;
            }
            
        } else if(state == State::length) {
			switch (cmd[i]) {
                case 'l':
                    modifiers.long_int = true;
                    i++;
                    break;
                default:
                    state = State::specifier;
			}
            
        } else if(state == State::specifier) {
            format_buffer.str("");
            
			if (modifiers.min_characters_enabled) {
				format_buffer << std::setw((int)modifiers.min_characters);
			}
            
			if (modifiers.fill_padding_with_zeros) {
				format_buffer << std::setfill('0');
			}
			else {
				format_buffer << std::setfill(' ');
			}
            
			format_buffer << std::resetiosflags(std::ios_base::basefield);
            
            switch (cmd[i]) {
                case '@': {
                    // OBJECT AUTO DETECTION
                    Printable *obj = va_arg(args, Printable*);
                    if(obj) {
                        buffer += obj->toStdString();
                    }
                    
                    break;
                }
                    
                case 'S': {
                    std::string *str = va_arg(args, std::string*);
                    if(str) {
                        std::string copy = *str;
                        
                        for (size_t i=0; i<copy.length(); i++) {
							if (copy.at(i) == 0) {
								copy = std::string(copy.data(), i);
								break;
							}
                        }
                        
                        buffer += copy;
                    }
                    
                    break;
                }
                    
                case 'V': {
                    const std::vector<std::string>* v = va_arg(args, const std::vector<std::string>*);
                    if(v) {
                        const std::vector<std::string>& vec = *v;
                        buffer += "{ ";
                        for(size_t i=0; i<vec.size(); i++) {
                            auto &s = vec.at(i);
                            buffer += '\'' + s + '\'';
                            
                            if(i < vec.size()-1)
                                buffer += ", ";
                        }
                        buffer += " }";
                    }
                    
                    break;
                }
                    
				case 'u': {
					if (modifiers.long_int) {
						unsigned long d = va_arg(args, unsigned long);
                        format_buffer << d;
					}
					else {
						unsigned int d = va_arg(args, unsigned int);
                        format_buffer << d;
					}
					
					break;
				}
                    
				case 'i':
                case 'd': {
					if (modifiers.long_int) {
						long d = va_arg(args, long);
                        format_buffer << d;
					}
					else {
						int d = va_arg(args, int);
                        format_buffer << d;
					}
                    
                    break;
                }
                    
                case 'F':
					format_buffer << std::uppercase;
                    
                case 'f': {
					format_buffer << std::dec;
                    
                    double f = va_arg(args, double);
					std::streamsize size = format_buffer.precision();
					if (modifiers.precision > -1) {
						format_buffer.precision(modifiers.precision);
					}
                    format_buffer << std::fixed << f;
					format_buffer.precision(size);
					format_buffer << std::nouppercase;
                    break;
                }
                    
				case 'g':
				case 'G':
				case 'E':
					format_buffer << std::uppercase;
                    
				case 'e': {
					double f = va_arg(args, double);
					std::streamsize size = format_buffer.precision();
					if (modifiers.precision > -1) {
						format_buffer.precision(modifiers.precision);
					}
					format_buffer << std::scientific << f;
					format_buffer.precision(size);
					format_buffer << std::nouppercase;
					break;
				}
                    
				case 'X':
					format_buffer << std::uppercase;
                    
				case 'x': {
					format_buffer << std::hex;
                    
					if (modifiers.long_int) {
						unsigned long f = va_arg(args, unsigned long);
						format_buffer << f;
					}
					else {
						unsigned int f = va_arg(args, unsigned int);
						format_buffer << f;
					}
                    
					format_buffer << std::nouppercase;
					break;
				}
                    
                case 's': {
                    const char* s = va_arg(args, const char*);
                    format_buffer << s;
                    break;
                }
                    
				case 'c': {
					char c = (char)va_arg(args, int);
					format_buffer << c;
					break;
				}
                    
                default:
                    throw DebugException("Unknown format specifier '%c' in string '%s' at position %d.", cmd[i], cmd, i);
            }
            
            i++;
            state = State::text;
            
            buffer += format_buffer.str();
            
        } else {
            throw DebugException("Unknown state %d.", state);
        }
    }
}
}

#undef MESSAGE_TYPE
#undef MESSAGE_TYPE_DBG

#define MESSAGE_TYPE(NAME, TYPE, FORCE_CALLBACK) \
    void NAME(const char *cmd, ...) { \
        va_list args; \
        va_start(args, cmd); \
        \
        std::string str;\
        DEBUG::ParseFormatString(str, cmd, args); \
        StatusMessage(DEBUG::DEBUG_TYPE::TYPE, str.c_str(), -1, NULL, FORCE_CALLBACK); \
        \
        va_end(args); \
    }

#define STACKTRACE

#if !_WIN32
#if DEBUG_PRINT_STACKTRACES
	#undef STACKTRACE
    #define STACKTRACE void* callstack[128]; \
    int i, frames = backtrace(callstack, 128); \
    char** strs = backtrace_symbols(callstack, frames); \
    for (i = 0; i < frames; ++i) { \
    printf("\t%s\n", strs[i]); \
    } \
    free(strs);
#endif
#endif

#define MESSAGE_TYPE_DBG(NAME, TYPE, PRINTSTACK, FORCE_CALLBACK) \
    void NAME(const char *file, int line, const char *cmd, ...) { \
        va_list args; \
        va_start(args, cmd); \
        \
        std::string str;\
        DEBUG::ParseFormatString(str, cmd, args); \
        StatusMessage(DEBUG::DEBUG_TYPE::TYPE, str, line, file); \
        \
        va_end(args); \
        if(PRINTSTACK) { \
            STACKTRACE \
        } \
    }

MESSAGE_TYPE_DBG(WARNING_, TYPE_WARNING, false, false)
MESSAGE_TYPE_DBG(ERROR_, TYPE_ERROR, true, false)
MESSAGE_TYPE_DBG(EXCEPTION_, TYPE_EXCEPTION, true, false)
MESSAGE_TYPE(Debug, TYPE_INFO, false)
MESSAGE_TYPE(DebugCallback, TYPE_INFO, true)

void DebugHeader(const char *cmd, ...) {
    va_list args;
    va_start(args, cmd);
    
    std::string str;
    DEBUG::ParseFormatString(str, cmd, args);
    Debug(utils::repeat("-", str.length()).c_str());
    StatusMessage(DEBUG::DEBUG_TYPE::TYPE_INFO, str.c_str(), -1, NULL, false);
    Debug(utils::repeat("-", str.length()).c_str());
    
    va_end(args);
}

void StatusMessage(DEBUG::DEBUG_TYPE type,
#if __cpp_lib_string_view
const std::string_view& buf,
#elif __cpp_lib_experimental_string_view
const std::experimental::string_view& buf, 
#else
const std::string& buf,
#endif
int line, const char *file, bool force_callback) {
	DEBUG::StatusMsg msg(type, buf, line, file);
    msg.force_callback = force_callback;
    DEBUG::ParseStatusMessage(&msg);
}

