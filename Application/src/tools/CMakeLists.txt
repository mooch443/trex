add_executable(term_metrics term_metrics.cpp)
if(COMMONS_SPLIT_SHARED_LIBS AND TREX_ENABLE_SHARED_INTERNAL_LIBS)
    target_link_libraries(term_metrics PUBLIC commons_misc)
else()
    target_link_libraries(term_metrics PUBLIC commons)
endif()

add_executable(settings_docs settings_docs.cpp)
target_link_libraries(settings_docs PUBLIC tracker_misc)
