template <typename T, size_t _cache_size>
ThreadedAnalysis<T, _cache_size>::ThreadedAnalysis(const ElementLoading::Type type, const create_type& create_element, const prepare_type& prepare, const loading_type& loading, const processing_type& analysis, const destroy_type& destroy_element) : _loading(loading), _analysis(analysis), _create_element(create_element), _destroy_element(destroy_element), _prepare(prepare), _loading_thread(NULL), _analysis_thread(NULL), _terminate_threads(false), _type(type), _array_index(0), _paused(false)
{
    // create cache
    _cache = new Container*[_cache_size];
    for (size_t i=0; i<cache_size; i++) {
        _cache[i] = new Container(_create_element());
    }
    
    _tmp_object = new T*[MAX_THREADS_CACHE];
    for (size_t i=0; i<MAX_THREADS_CACHE; i++) {
        _tmp_object[i] = _create_element();
    }
    
    _processed_object = create_element();
    
    _threads_paused[0] = _threads_paused[1] = 0;
    
    _loading_thread = new std::thread([this]() {
        this->loading_function();
    });
    
    _analysis_thread = new std::thread([this]() {
        this->analysis_function();
    });
}

template <typename T, size_t _cache_size>
ThreadedAnalysis<T, _cache_size>::~ThreadedAnalysis() {
    terminate();
    
    for(size_t i=0; i<MAX_THREADS_CACHE; i++) {
        _destroy_element(_tmp_object[i]);
    }
    delete [] _tmp_object;
    
    _destroy_element(_processed_object);
    
    for (size_t i=0; i<_cache_size; i++) {
        _destroy_element(_cache[i]->data);
        delete _cache[i];
    }
    delete [] _cache;
}

template <typename T, size_t _cache_size>
void ThreadedAnalysis<T, _cache_size>::loading_function() {
    long_t previous_idx = -1;
    
    while (!_terminate_threads) {
        if(paused()) {
            const std::chrono::milliseconds ms(55);
            std::this_thread::sleep_for(ms);
            
            _threads_paused[PausedIndex::LOADING_THREAD_PAUSED] = true;
            
            continue;
            
        } else {
            _threads_paused[PausedIndex::LOADING_THREAD_PAUSED] = false;
        }
        
        // save locally
        long_t array_idx;
        {
            std::lock_guard<decltype(lock)> l(lock);
            long_t offset = 0;
            _array_index = -1;
            
            // find the first image that hasn't been processed yet
            for (size_t i=0; i<_cache_size; i++) {
                if(_cache[i]->initialized && _cache[i]->processed) {
                    offset = i+1;
                    
                } else {
                    break;
                }
            }
            
            // find the first image that isn't waiting for processing anymore
            for (size_t i=offset; i<=_cache_size; i++) {
                _array_index = i;
                
                if(i == _cache_size || !_cache[i]->initialized || _cache[i]->processed)
                    break;
                
                /*if (_cache[i]->initialized && !_cache[i]->processed) {
                    continue;
                } else {
                    _array_index = i;
                    break;
                }*/
            }
            
            if (_array_index < 1) {
                //_array_index = _cache_size;
                previous_idx = -1;
            }
            
            if (offset > 0) {
                //Debug("Shifting elements to the left by %d (%d).", offset, _array_index);
                
                // shift all elements in the container according to offset.
                // re-use all the items that pop out in the front.
                Container* tmp_array[_cache_size] = { NULL };
                
                for (size_t i=0; i<size_t(offset); i++)
                    tmp_array[i] = _cache[i];
                
                for (size_t i=offset; i<_cache_size; i++)
                    _cache[i-offset] = _cache[i];
                
                for (size_t i=int(_cache_size)-offset; i<_cache_size; i++) {
                    _cache[i] = tmp_array[int(i) - (int(_cache_size)-offset)];
                    _cache[i]->initialized = false;
                }
                
                // some object is currently being processed.
                // be aware of that and render the reference invalid if it
                // doesnt exist anymore.
                if (_currently_processed >= 0) {
                    _currently_processed -= offset;
                    
                    if (_currently_processed < 0) {
                        _currently_processed = -1;
                    }
                }
                
                // the previous id has to be wrapped around as well,
                // so its still valid after this shift
                if(previous_idx != -1) {
                    previous_idx -= offset;
                    if (previous_idx < 0) {
                        previous_idx += _cache_size;
                    }
                }
                
                // next write will be offset, too
                _array_index -= offset;
                
            }
            
            array_idx = _array_index;
        }
        
        if(array_idx != -1 && ((int(_cache_size) != array_idx && _type == ElementLoading::SINGLE_THREADED) || int(_cache_size) - int(array_idx) >= int(cmn::min(_cache_size, MAX_THREADS_CACHE / 2))))
        {
            // fill cache
            const uint32_t max_threads = min(_cache_size, _type == ElementLoading::MULTI_THREADED ? MAX_THREADS_CACHE : size_t(1));
            std::thread* threads[MAX_THREADS_CACHE] = {NULL};
            uint32_t count = min(max_threads, uint32_t(long_t(_cache_size) - long_t(array_idx)));
            
            auto load_img = [this, previous_idx](int offset) {
                T* prev = NULL;
                if(offset == 0)
                    prev = previous_idx != -1 ? _cache[previous_idx]->data : NULL;
                else
                    prev = _tmp_object[offset];
                
                //Debug("Loading %d@%d", _tmp_object[offset]->index, offset);
                _loading(prev, *_tmp_object[offset]);
            };
            
            //Debug("previous_idx %d", previous_idx);
            bool abort = false;
            T* prev_data = previous_idx != -1 ? _cache[previous_idx]->data : NULL;
            for (uint32_t i=0; i<count; i++) {
                if(i)
                    prev_data = _tmp_object[i-1];
                
                bool success = _prepare(prev_data, *_tmp_object[i]);
                if(!success) {
                    for (uint32_t j=0; j<i; j++) {
                        threads[j]->join();
                        delete threads[j];
                    }
                    
                    reset_cache();
                    abort = true;
                    break;
                }
                
                if(count > 1)
                    threads[i] = new std::thread(load_img, i);
                else
                    load_img(i);
            }
            //Debug("Started %d threads for pos %d (max_threads %d)", count, array_idx, max_threads);
            if(abort) {
                continue;
            }
            
            for (uint32_t i=0; count>1 && i<count; i++) {
                threads[i]->join();
                delete threads[i];
            }
            
            {
                std::lock_guard<decltype(lock)> guard(lock);
                
                if (_array_index != array_idx) {
                    continue;
                    //U_EXCEPTION("Array index changed from %d to %d while loading.", array_idx, _array_index);
                }
                
                for (uint32_t i=0; i<count; i++) {
                    //Debug("We have %d@%d", _tmp_object[i]->index, i);
                    std::swap(_cache[_array_index+i]->data, _tmp_object[i]);
                    _cache[_array_index+i]->processed = false;
                    _cache[_array_index+i]->initialized = true;
                    
                    previous_idx = _array_index+i;
                }
                
                
                /*int filled = 0, last_idx = _array_index;
                for (int i=0; i<_cache_size; i++) {
                    if (!_cache[i]->processed && _cache[i]->initialized) {
                        filled++;
                        last_idx = i;
                    }
                }*/
                
                //Debug("Filled cache %d/%d (image %d@%d)", filled, _cache_size, _cache[last_idx]->data->index, last_idx);
            }
            
        } else {
            const std::chrono::milliseconds ms(15);
            std::this_thread::sleep_for(ms);
        }
    }
}

template <typename T, size_t _cache_size>
void ThreadedAnalysis<T, _cache_size>::analysis_function() {
    while (!_terminate_threads) {
        if(paused()) {
            const std::chrono::milliseconds ms(55);
            std::this_thread::sleep_for(ms);
            
            _threads_paused[PausedIndex::ANALYSIS_THREAD_PAUSED] = true;
            
            continue;
            
        } else {
            _threads_paused[PausedIndex::ANALYSIS_THREAD_PAUSED] = false;
        }
        
        lock.lock();
        
        Container *container = NULL;
        T* ptr = NULL;
        
        for (size_t i=0; i<_cache_size; i++) {
            if (_cache[i]->initialized && !_cache[i]->processed) {
                container = _cache[i];
                _currently_processed = i;
                break;
            }
        }
        
        if (container) {
            //*_processed_object = *container->data;
            ptr = container->data;
            //Debug("Trying to process %d...", _currently_processed);
        }
        
        lock.unlock();
        
        if(container == NULL) {
            // Nothing to do. No unprocessed container could be found.
            const std::chrono::nanoseconds ms(1);
            std::this_thread::sleep_for(ms);
            
        } else {
            Queue::Code code = _analysis(*ptr);
            
            if(code == Queue::ITEM_NEXT) {
                std::lock_guard<decltype(lock)> locked(lock);
                
                // array could have been shifted to left/right so that the object doesn't
                // exist anymore. _currently_processed will be adjusted if shifted. if
                // it doesnt exist anymore, it will be -1.
                if (_currently_processed >= 0) {
                    //Debug("Setting %d to processed.", _currently_processed);
                    _cache[_currently_processed]->processed = true;
                    _currently_processed = -1;
                }
                
            } else if(code == Queue::ITEM_WAIT) {
                lock.lock();
                _currently_processed = -1;
                lock.unlock();
                
                // analysis couldnt be done. retry in a few ms.
                const std::chrono::milliseconds ms(10);
                std::this_thread::sleep_for(ms);
                
            } else if(code == Queue::ITEM_REMOVE) {
                std::lock_guard<decltype(lock)> locked(lock);
                
                if (_currently_processed >= 0) {
                    _cache[_currently_processed]->processed = true;
                    _currently_processed = -1;
                }
            }
        }
    }
}

template <typename T, size_t _cache_size>
void ThreadedAnalysis<T, _cache_size>::reset_cache() {
    std::lock_guard<decltype(lock)> locked(lock);
    
    for (size_t i=0; i<_cache_size; i++) {
        _cache[i]->initialized = false;
        _cache[i]->processed = false;
    }
    
    _currently_processed = -1;
    _array_index = -1;
}
