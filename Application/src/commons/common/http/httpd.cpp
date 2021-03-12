#include "httpd.h"
#include <misc/GlobalSettings.h>

#if WITH_MHD
#ifdef WIN32
#include <winsock.h>
#else
#include <netinet/in.h>
#include <arpa/inet.h>
#endif

using namespace cmn;

int accept_callback(void *cls,
                             const struct sockaddr *addr,
                     socklen_t addrlen) {
    
    char *s = NULL;
    switch(addr->sa_family) {
        case AF_INET: {
            struct sockaddr_in *addr_in = (struct sockaddr_in *)addr;
            s = (char*)malloc(INET_ADDRSTRLEN);
            inet_ntop(AF_INET, &(addr_in->sin_addr), s, INET_ADDRSTRLEN);
            break;
        }
        case AF_INET6: {
            struct sockaddr_in6 *addr_in6 = (struct sockaddr_in6 *)addr;
            s = (char*)malloc(INET6_ADDRSTRLEN);
            inet_ntop(AF_INET6, &(addr_in6->sin6_addr), s, INET6_ADDRSTRLEN);
            break;
        }
        default:
            break;
    }
    
    if(s == NULL)
        return MHD_NO;
    
    if(SETTING(httpd_accepted_ip).value<std::string>().empty()
       || SETTING(httpd_accepted_ip).value<std::string>() == std::string(s))
    {
        return MHD_YES;
    }
    
    return MHD_NO;
}

Httpd::Httpd(const url_callback& get_image, const std::string& default_page, const init_session & init, const no_access& access) : _default_page(default_page), _get_url(get_image), _init_session(init), _no_access(access)
{
    const int default_port = GlobalSettings::has("httpd_port") ? SETTING(httpd_port).value<int>() : 8080;
    int port = default_port;
    
    daemon = NULL;
    while (daemon == NULL && port < default_port + 8) {
        daemon = MHD_start_daemon(MHD_USE_THREAD_PER_CONNECTION,
                                  port++,
                                  &accept_callback,
                                  NULL,
                                  &(Httpd::ahc_echo),
                                  (void*)this,
                                  //MHD_OPTION_HTTPS_MEM_KEY, key_pem.c_str(),
                                  //MHD_OPTION_HTTPS_MEM_CERT, cert_pem.c_str(),
                                  //MHD_OPTION_HTTPS_MEM_TRUST, root_ca_pem,
                                  MHD_OPTION_END);
    }
    
    if(daemon == NULL)
        Except("Cannot start HTTP daemon. Check your firewall settings (tried ports %d-%d).", default_port, port-1);
    else
        DebugCallback("Started HTTP daemon on port %d.", port-1);
}

Httpd::~Httpd() {
    if(daemon)
        MHD_stop_daemon(daemon);
}

int Httpd::ahc_echo(void * cls,
                    struct MHD_Connection * connection,
                    const char * url,
                    const char * method,
                    const char *,
                    const char * upload_data,
                    size_t * upload_data_size,
                    void ** ptr)
{
    Httpd * _this = static_cast<Httpd*>(cls);
    return _this->local_ahc(connection, std::string(url), std::string(method), upload_data, upload_data_size, ptr);
}

const char HEX2DEC[256] =
{
    /*       0  1  2  3   4  5  6  7   8  9  A  B   C  D  E  F */
    /* 0 */ -1,-1,-1,-1, -1,-1,-1,-1, -1,-1,-1,-1, -1,-1,-1,-1,
    /* 1 */ -1,-1,-1,-1, -1,-1,-1,-1, -1,-1,-1,-1, -1,-1,-1,-1,
    /* 2 */ -1,-1,-1,-1, -1,-1,-1,-1, -1,-1,-1,-1, -1,-1,-1,-1,
    /* 3 */  0, 1, 2, 3,  4, 5, 6, 7,  8, 9,-1,-1, -1,-1,-1,-1,
    
    /* 4 */ -1,10,11,12, 13,14,15,-1, -1,-1,-1,-1, -1,-1,-1,-1,
    /* 5 */ -1,-1,-1,-1, -1,-1,-1,-1, -1,-1,-1,-1, -1,-1,-1,-1,
    /* 6 */ -1,10,11,12, 13,14,15,-1, -1,-1,-1,-1, -1,-1,-1,-1,
    /* 7 */ -1,-1,-1,-1, -1,-1,-1,-1, -1,-1,-1,-1, -1,-1,-1,-1,
    
    /* 8 */ -1,-1,-1,-1, -1,-1,-1,-1, -1,-1,-1,-1, -1,-1,-1,-1,
    /* 9 */ -1,-1,-1,-1, -1,-1,-1,-1, -1,-1,-1,-1, -1,-1,-1,-1,
    /* A */ -1,-1,-1,-1, -1,-1,-1,-1, -1,-1,-1,-1, -1,-1,-1,-1,
    /* B */ -1,-1,-1,-1, -1,-1,-1,-1, -1,-1,-1,-1, -1,-1,-1,-1,
    
    /* C */ -1,-1,-1,-1, -1,-1,-1,-1, -1,-1,-1,-1, -1,-1,-1,-1,
    /* D */ -1,-1,-1,-1, -1,-1,-1,-1, -1,-1,-1,-1, -1,-1,-1,-1,
    /* E */ -1,-1,-1,-1, -1,-1,-1,-1, -1,-1,-1,-1, -1,-1,-1,-1,
    /* F */ -1,-1,-1,-1, -1,-1,-1,-1, -1,-1,-1,-1, -1,-1,-1,-1
};

std::string UriDecode(std::string sSrc)
{
    // Note from RFC1630: "Sequences which start with a percent
    // sign but are not followed by two hexadecimal characters
    // (0-9, A-F) are reserved for future extension"
    
    sSrc = utils::find_replace(sSrc, "+", " ");
    
    const unsigned char * pSrc = (const unsigned char *)sSrc.c_str();
    const size_t SRC_LEN = sSrc.length();
    const unsigned char * const SRC_END = pSrc + SRC_LEN;
    // last decodable '%'
    const unsigned char * const SRC_LAST_DEC = SRC_END - 2;
    
    char * const pStart = new char[SRC_LEN];
    char * pEnd = pStart;
    
    while (pSrc < SRC_LAST_DEC)
    {
        if (*pSrc == '%')
        {
            char dec1, dec2;
            if (-1 != (dec1 = HEX2DEC[*(pSrc + 1)])
                && -1 != (dec2 = HEX2DEC[*(pSrc + 2)]))
            {
                *pEnd++ = (dec1 << 4) + dec2;
                pSrc += 3;
                continue;
            }
        }
        
        *pEnd++ = *pSrc++;
    }
    
    // the last 2- chars
    while (pSrc < SRC_END)
        *pEnd++ = *pSrc++;
    
    std::string sResult(pStart, pEnd);
    delete [] pStart;
    return sResult;
}

std::tuple<std::string, std::string, Httpd::Session*>
Httpd::check_cookie(struct MHD_Connection* connection) {
    const char *value;
    value = MHD_lookup_connection_value (connection,
                                         MHD_COOKIE_KIND,
                                         "Session");
    if(value) {
        std::string key = value;
        
        auto it = _sessions.find(key);
        if(it == _sessions.end()) {
            //Error("Invalid cookie.");
        } else {
            //Debug("Recognizing cookie %X", it->second);
            return {"", it->first, it->second};
        }
    }
    {
        // start a new session
        char value[128];
        char raw_value[65];
        
        for (unsigned int i=0;i<sizeof (raw_value);i++)
            raw_value[i] = 'A' + (rand () % 26); /* bad PRNG! */
        raw_value[64] = '\0';
        snprintf (value, sizeof (value),
                  "%s=%s",
                  "Session",
                  raw_value);
        
        auto ptr = new Session(raw_value);
        _init_session(ptr);
        
        return {value, raw_value, ptr};
    }
}

int Httpd::local_ahc(struct MHD_Connection * connection,
                     std::string url,
                     std::string method,
                     const char * upload_data,
                     size_t * upload_data_size,
                     void ** ptr)
{
    // THIS METHOD GETS CALLED IN AN ARBITRARY THREAD
    // so be careful what you do here.
    
    static int dummy;
    struct MHD_Response * response = nullptr;
    int ret;
    
    if (method != MHD_HTTP_METHOD_GET && method != MHD_HTTP_METHOD_POST) {
        Error("Unknown http method '%S'.", &method);
        return MHD_NO;
    }
    
    if (&dummy != *ptr)
    {
        /* The first time only the headers are valid,
         do not respond in the first round... */
        *ptr = &dummy;
        return MHD_YES;
        
    } else {
        *ptr = NULL; /* clear context pointer */
    }
    
    if (url == "/") {
        url = "/"+_default_page;
    }
    
    auto && [cookie, key, session] = check_cookie(connection);
    
    if(!session->has_access) {
        auto r = _no_access(session);
        if(r.data.empty()) {
            const char *errorstr =
            "<html><body>Access denied.\
            </body></html>";
            response =
            MHD_create_response_from_buffer (strlen (errorstr),
                                             (void *) errorstr,
                                             MHD_RESPMEM_PERSISTENT);
        } else {
            response = MHD_create_response_from_buffer (r.data.size(), (void*) r.data.data(), MHD_RESPMEM_MUST_COPY);
            MHD_add_response_header(response, "Content-Type", r.type.c_str());
            MHD_add_response_header(response, "Cache-Control", "no-cache");
        }
    } else {
        ret = process_request(connection, &response, url, session, upload_data, upload_data_size, method);
        if(ret != -1337)
            return ret;
    }
    
    if(!response)
        return MHD_YES;
    
    if(_sessions.find(key) == _sessions.end()) {
        auto ret = MHD_add_response_header (response,
                                           MHD_HTTP_HEADER_SET_COOKIE,
                                            cookie.c_str());
        if(ret == MHD_NO) {
            Error("Cannot set cookie.");
        }
        else {
            _sessions[key] = session;
            //Debug("Cookie set %S.", &key);
        }
    }
    
    ret = MHD_queue_response(connection,
                             MHD_HTTP_OK,
                             response);
    
    MHD_destroy_response(response);
    return ret;
}

int Httpd::process_request(struct MHD_Connection *connection, struct MHD_Response **response, const std::string& url, Session * session, const char*upload_data, size_t*upload_data_size, std::string method) {
#if !__APPLE__
    const std::string root_dir = "";
#else
    const std::string root_dir = "../Resources/";
#endif
    const std::string sub_dir = "html";
    
    if(utils::endsWith(url, "/favicon.ico") && file::Path(root_dir+"gfx/icon.ico").exists()) {
        auto str = utils::read_file(root_dir+"gfx/icon.ico");
        *response = MHD_create_response_from_buffer (str.size(), (void*) str.data(),
                                                    MHD_RESPMEM_MUST_COPY);
        MHD_add_response_header(*response, "Content-Type", "image/x-icon");
        //MHD_add_response_header(*response, "Cache-Control", "no-cache");
        
    } else if (utils::endsWith(url, ".html") || utils::endsWith(url, ".css") || utils::endsWith(url, ".ttf") || utils::endsWith(url, ".js")) {
        if (!file_exists(sub_dir+url)) {
            const char *errorstr =
            "<html><body>File cannot be found!\
            </body></html>";
            *response =
            MHD_create_response_from_buffer (strlen (errorstr),
                                             (void *) errorstr,
                                             MHD_RESPMEM_PERSISTENT);
            
            auto ret = MHD_queue_response (connection, MHD_HTTP_INTERNAL_SERVER_ERROR,
                                      *response);
            MHD_destroy_response(*response);
            return ret;
            
        } else {
            auto str = utils::read_file(sub_dir+url);
            *response = MHD_create_response_from_buffer (str.size(), (void*) str.data(),
                                                        MHD_RESPMEM_MUST_COPY);
        }
        
    } else if(utils::contains(url, "/get_settings")) {
        std::stringstream ss;
        ss << "{\"keys\":[";
        auto keys = GlobalSettings::map().keys();
        std::map<std::string, std::string> map;
        for (auto &k : keys) {
            map[k] = GlobalSettings::get(k).get().type_name();
        }
        
        auto str = Meta::toStr(map);
        
        *response = MHD_create_response_from_buffer (str.size(),
                                                    (void*) str.data(),
                                                    MHD_RESPMEM_MUST_COPY);
        
    } else if(utils::contains(url, "/setting/")) {
        auto parts = utils::split(url, '/');
        std::string name = parts.back();
        
        std::string str;
        
        try {
            if(!name.empty()) {
                auto prop = GlobalSettings::get(name);
                if(!prop.get().valid())
                    U_EXCEPTION("Setting '%S' not found.", &name);
                
                str = prop.get().valueString();
                if (prop.is_type<std::string>()) {
                    str = prop.value<std::string>();
                }
            }
            
            if(*upload_data_size != 0) {
                std::string data(upload_data, *upload_data_size);
                auto array = utils::split(data, '&');
                Debug("Number of elements: %d", array.size());
                
                for (auto &k : array) {
                    auto split = utils::split(k, '=');
                    auto n = UriDecode(split[0]);
                    auto v = UriDecode(split[1]);
                    
                    Debug("Received: '%S' = '%S'", &n, &v);
                    
                    if(!GlobalSettings::map().has(n)) {
                        return MHD_NO;
                        
                    } else {
                        if(GlobalSettings::access_level(n) > AccessLevelType::PUBLIC) {
                            Error("Cannot write value for '%S' from web interface (access level %s).", &n, GlobalSettings::access_level(n).name());
                        } else {
                            auto prop = GlobalSettings::get(n);
                            try {
                                prop.get().set_value_from_string(v);
                                DebugCallback("%@", &prop.get());
                                
                            } catch(const std::invalid_argument& e) {
                                Error("Value '%S' cannot be converted to type of %@.", &v, &prop.get());
                            } catch(const cmn::illegal_syntax& e) {
                                Error("Value '%S' cannot be converted to type of %@.", &v, &prop.get());
                            } catch(const UtilsException& ex) {
                                
                            }
                        }
                    }
                }
                
                Debug("Downloaded data '%S'", &data);
                
            } else if(method == MHD_HTTP_METHOD_POST)
                Warning("No data uploaded! (%d)", *upload_data_size);
            
        } catch(const UtilsException& ex) {
            
        }
        
        //Debug("Responding with '%S'.", &str);
        *response = MHD_create_response_from_buffer (str.size(),
                                                    (void*) str.data(),
                                                    MHD_RESPMEM_MUST_COPY);
        
    } else {
        auto tmp = _get_url(session, url);
        auto &buffer = tmp.data;
        
        *response = MHD_create_response_from_buffer (buffer.size(),
                                                    (void*) buffer.data(),
                                                    MHD_RESPMEM_MUST_COPY);
        MHD_add_response_header(*response, "Content-Type", tmp.type.c_str());
        MHD_add_response_header(*response, "Cache-Control", "no-cache");
    }
    
    return -1337;
}
#endif

