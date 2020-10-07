#include "pretty.h"
#include <sstream>
#include <queue>

namespace cmn {
    std::string prettify_array(const std::string& text) {
        std::stringstream ss;
        
        std::deque<char> brackets;
        std::deque<char> q;
        std::string indent = "";
        bool skip_white = false;
        
        std::deque<size_t> indent_at;
        
        auto open_brackets = [&](char c) {
            if(!q.empty())
                return;
            
            brackets.push_back(c);
            
            if((c == '[' || c == '{') && brackets.size() == 1 && text.length() >= 50) {
                indent += "  ";
                indent_at.push_back(brackets.size());
                ss << "\n" << indent;
                skip_white = true;
            }
        };
        
        auto close_brackets = [&](char c) -> bool {
            if(!q.empty())
                return false;
            
            bool okay = false;
            if(!brackets.empty()) {
                switch (brackets.back()) {
                    case '(':
                        okay = c == ')';
                        break;
                    case '[':
                        okay = c == ']';
                        break;
                    case '{':
                        okay = c == '}';
                        break;
                    default:
                        break;
                }
            }
            
            if(okay) {
                if(!indent_at.empty() && indent_at.back() == brackets.size()) {
                    indent_at.pop_back();
                    if(!indent.empty())
                        indent = indent.substr(0, indent.length()-2);
                    ss << "\n" << indent;
                }
                
                brackets.pop_back();
                return true;
            }
            
            return false;
        };
        
        auto quotes = [&](char c) {
            if(!q.empty() && q.back() == c)
                q.pop_back();
            else
                q.push_back(c);
        };
        
        for(size_t i=0; i<text.length(); ++i) {
            auto c = text.at(i);
            switch (c) {
                case '}':
                case ']':
                case ')':
                    if(close_brackets(c))
                        break;
            }
            
            /*if(c == ' ' && skip_white) {
                skip_white = false;
                continue;
            }*/
            
            ss << c;
            
            switch(c) {
                case ',':
                    if(q.empty()) {
                        if(!indent_at.empty() && indent_at.back() == brackets.size())  {
                            ss << "\n" << indent;
                            skip_white = true;
                        }
                    }
                    break;
                case '\'':
                case '"':
                    quotes(c);
                    break;
                    
                case '(':
                case '[':
                case '{':
                    open_brackets(c);
                    break;
            }
        }
        
        return ss.str();
    }
}
