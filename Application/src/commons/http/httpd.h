#ifndef _HTTPD_H
#define _HTTPD_H

#if WITH_MHD

#include "types.h"
#include <functional>

#include <microhttpd.h>
#include <misc/SpriteMap.h>

namespace cmn {
    class Httpd : public Printable {
    public:
        class Session {
        public:
            std::string key;
            bool has_access;
            sprite::Map map;
            
            Session(const std::string& key) : key(key), has_access(false) {}
        };
        
        struct Response {
            std::vector<uchar> data;
            std::string type;
            
            Response(const std::vector<uchar>& d, const std::string t = "image/jpeg") {
                data = d;
                type = t;
            }
            
            Response(const std::string& str) {
                data = std::vector<uchar>(str.data(), str.data() + str.length());
                type = "text/html";
            }
        };
        
        //typedef std::function<cv::Mat(int)> image_callback;
        typedef std::function<Response(Session*, const std::string&)> url_callback;
        typedef std::function<void(Session*)> init_session;
        typedef std::function<Response(Session*)> no_access;
        const std::string _default_page;
        
    public:
        Httpd(const url_callback &get_image,
              const std::string& default_page = "index.html",
              const init_session & = [](Session* ptr) { ptr->has_access = true; },
              const no_access& = [](auto){ return Response(""); }
        );
        ~Httpd();
        
        UTILS_TOSTRING("HTTPd<>");
        
    private:
        std::map<std::string, Session*> _sessions;
        
        static int ahc_echo(void * cls,
                     struct MHD_Connection * connection,
                     const char * url,
                     const char * method,
                     const char * version,
                     const char * upload_data,
                     size_t * upload_data_size,
                     void ** ptr);
        int local_ahc(struct MHD_Connection * connection,
                      std::string url,
                      std::string method,
                      const char * upload_data,
                      size_t * upload_data_size,
                      void ** ptr);
        
    private:
        struct MHD_Daemon * daemon;
        //const image_callback _get_image;
        const url_callback _get_url;
        const init_session _init_session;
        const no_access _no_access;
        
        int process_request(struct MHD_Connection*, struct MHD_Response**, const std::string& url, Session*, const char*, size_t *, std::string method);
        
        std::tuple<std::string, std::string, Session*> check_cookie(struct MHD_Connection*);
    };
}

#endif
#endif

