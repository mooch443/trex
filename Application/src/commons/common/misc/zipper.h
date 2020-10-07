#pragma once

#include <tuple>

namespace
{

   template<typename _Action, size_t N, typename... _TupleArgs>
   struct _ForEachInTupleHelper
   {
      static void Call( std::tuple<_TupleArgs...>& tuple, _Action&& action )
      {
         action.operator()<std::tuple<_TupleArgs...>, N>( tuple );
         _ForEachInTupleHelper<_Action, N - 1, _TupleArgs...>::Call( tuple, std::forward<_Action>( action ) ); // Recursive call for the next tuple element
      }
   };

   template<typename _Action, typename... _TupleArgs>
   struct _ForEachInTupleHelper<_Action, 0, _TupleArgs...>
   {
      static void Call( std::tuple<_TupleArgs...>& tuple, _Action&& action )
      {
         action.operator()<std::tuple<_TupleArgs...>, 0>( tuple );
      }
   };

}

//! \brief Calls a function for each element in the given tuple
//! \param tuple The tuple on which to call the function
//! \tparam _Action Function template that has to define operator() in a way that it is able to take
//!                 a parameter of each of the tuple types
//! \tparam _FuncArgs Additional arguments for the _Action functor
//! \tparam _FirstArg First argument of the tuple
//! \tparam _Rest Optional other tuple arguments
template<typename _Action, typename... _TupleArgs>
void ForEachInTuple( std::tuple<_TupleArgs...>& tuple, _Action&& action )
{
   //TODO Explicitly forbid empty tuples
   static_assert( sizeof...( _TupleArgs ) > 0, "Empty tuple is not allowed!" );
   _ForEachInTupleHelper<_Action, sizeof...( _TupleArgs ) - 1, _TupleArgs...>::Call( tuple, std::forward<_Action>( action ) );
}

namespace
{

   template<size_t N, typename _Tuple>
   struct _AnyEqualHelper
   {
      static bool Compare( const _Tuple& l, const _Tuple& r )
      {
         if ( std::get<N>( l ) == std::get<N>( r ) ) return true;
         return _AnyEqualHelper<N - 1, _Tuple>::Compare( l, r );
      }
   };

   template<typename _Tuple>
   struct _AnyEqualHelper<0, _Tuple>
   {
      static bool Compare( const _Tuple& l, const _Tuple& r )
      {
         return std::get<0>( l ) == std::get<0>( r );
      }
   };

}

//! \brief Returns true if any elements of the two tuples are equal (compares pair-wise, not the whole permutation)
//! \param l First tuple
//! \param r Second tuple
//! \returns True if there is at least one pair of elements that are equal
template<typename _First, typename... _Rest>
bool AnyEqual( const std::tuple<_First, _Rest...>& l, const std::tuple<_First, _Rest...>& r )
{
   return _AnyEqualHelper<sizeof...( _Rest ), std::tuple<_First, _Rest...>>::Compare( l, r );
}

#pragma region Sequence

template<int... S>
struct Sequence{};

template<int N, int... S>
struct SequenceGenerator : SequenceGenerator<N - 1, N - 1, S...> {};

template<int... S>
struct SequenceGenerator<0, S...>
{
   using type = Sequence<S...>;
};

#pragma endregion

namespace Zip
{

#pragma region TemplateHelpers

   template<typename T>
   struct _GetIterator
   {
      using iterator = typename std::conditional< std::is_const<T>::value, 
                                                  typename std::remove_reference<T>::type::const_iterator, 
                                                  typename std::remove_reference<T>::type::iterator >::type;
   };

   //! \brief Pass-through function that can be used to expand a function call on all arguments of a parameter pack
   template<typename... T>
   inline void PassThrough( T&&... ) {}

#pragma endregion

#pragma region Zip

   template<typename... _Iters>
   struct _IterCollection
   {
      template<size_t Index>
      using value_type_t = typename std::tuple_element<Index, std::tuple<_Iters...>>::type::value_type;

      using value_ref_tuple_t = std::tuple<typename std::iterator_traits<_Iters>::reference...>;

      _IterCollection( _Iters&&... iterators ) :
         _iteratorPack( std::forward<_Iters>(iterators)... )
      {
      }

      inline bool MatchAny( const _IterCollection& other ) const
      {
         return AnyEqual( _iteratorPack, other._iteratorPack );
      }

      inline bool operator==( const _IterCollection& other ) const
      {
         return _iteratorPack == other._iteratorPack;
      }

      inline bool operator!=( const _IterCollection& other ) const
      {
         return !operator==( other );
      }

      inline value_ref_tuple_t Deref()
      {
         return DerefInternal( typename SequenceGenerator<sizeof...( _Iters )>::type() );
      }

      inline void Increment()
      {
         IncrementInternal( typename SequenceGenerator<sizeof...( _Iters )>::type() );
      }

   private:
      template<int... S>
      inline value_ref_tuple_t DerefInternal( Sequence<S...> )
      {
         return value_ref_tuple_t( *std::get<S>( _iteratorPack )... );
      }

      template<int... S>
      inline void IncrementInternal( Sequence<S...> )
      {
         PassThrough( std::get<S>( _iteratorPack ).operator++( )... );
      }

      std::tuple<_Iters...> _iteratorPack;
   };

   //! \brief An iterator that iterates over a range of collections simultaneously
   //!
   //! As such, it returns a tuple of elements at the current position when dereferenced. Since
   //! the collections might be of different lengths, this iterator stops when the first collection
   //! is exhausted
   template<typename... _Iters>
   class ZipIterator
   {
      using IterCollection_t = _IterCollection<_Iters...>;
      using value_ref_tuple_t = std::tuple<typename std::iterator_traits<_Iters>::reference...>;
   public:
      ZipIterator( IterCollection_t cur ) :
         _curIters( cur )
      {
      }

      inline value_ref_tuple_t operator*( )
      {
         return _curIters.Deref();
      }

      inline ZipIterator& operator++( )
      {
         _curIters.Increment();
         return *this;
      }

      inline bool operator==( const ZipIterator& other ) const
      {
         return _curIters.MatchAny( other._curIters ); //Again, for the comparison inside a range based for loop, one match is enough!
      }

      inline bool operator!=( const ZipIterator& other ) const
      {
         return !operator==( other );
      }

   private:
      IterCollection_t _curIters;
   };

   //! \brief 'Collection' that zips multiple iterators. This spawns the begin and end iterators
   template<typename... _Iters>
   class ZipCollection
   {
      using IterCollection_t = _IterCollection<_Iters...>;
   public:
      ZipCollection( IterCollection_t&& begins, IterCollection_t&& ends ) :
         _begins( std::forward<IterCollection_t>( begins ) ),
         _ends( std::forward<IterCollection_t>( ends ) )
      {
      }

      inline ZipIterator<_Iters...> begin( )
      {
         return ZipIterator<_Iters...>( _begins );
      }

      inline ZipIterator<_Iters...> end( )
      {
         return ZipIterator<_Iters...>(_ends );
      }
   private:
      IterCollection_t _begins;
      IterCollection_t _ends;
   };

   //! \brief Creates a zip iterator to iterator over a range of collections simultaneously
   //! \param args All the collections to iterator over
   //! \tparam Args Types of collections
   //! \returns A ZipIterator over all the collections
   template<typename... Args>
   ZipCollection<typename _GetIterator<Args>::iterator...> Zip( Args&&... args )
   {
      using IterCollection_t = _IterCollection<typename _GetIterator<Args>::iterator...>;
      return ZipCollection<typename _GetIterator<Args>::iterator...>( IterCollection_t( std::begin( args )... ),
                                                                      IterCollection_t( std::end(args)... ) );
   }

#pragma endregion

}