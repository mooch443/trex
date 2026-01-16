add_executable(term_metrics term_metrics.cpp)
target_link_libraries(term_metrics PUBLIC commons)

add_executable(settings_docs settings_docs.cpp)
target_link_libraries(settings_docs PUBLIC tracker_misc)
