#pragma once

template<class T>
class ReverseAdapter
{
public:
    ReverseAdapter(T& container) : m_container(container) {}
    typename T::reverse_iterator begin() { return m_container.rbegin(); }
    typename T::reverse_iterator end() { return m_container.rend(); }
    
private:
    T& m_container;
};

template<class T>
class ConstReverseAdapter
{
public:
    ConstReverseAdapter(const T& container) : m_container(container) {}
    typename T::const_reverse_iterator begin() { return m_container.rbegin(); }
    typename T::const_reverse_iterator end() { return m_container.rend(); }
    
private:
    const T& m_container;
};


template<class T>
ReverseAdapter<T> MakeReverse(T& container)
{
    return ReverseAdapter<T>(container);
}

template<class T>
ConstReverseAdapter<T> MakeReverse(const T& container)
{
    return ConstReverseAdapter<T>(container);
}
