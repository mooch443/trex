#pragma once

#ifdef __APPLE__
#include <Availability.h>

#ifdef __MAC_OS_X_VERSION_MAX_ALLOWED
#if __MAC_OS_X_VERSION_MAX_ALLOWED < 101200
#define TREX_NO_SHARED_MUTEX
#endif
#endif
#endif

#ifdef TREX_NO_SHARED_MUTEX
using StructMutex_t = std::mutex;
using ReadLock_t = std::unique_lock<StructMutex_t>;
#else
using StructMutex_t = std::shared_mutex;
using ReadLock_t = std::shared_lock<StructMutex_t>;
#endif

template<typename T>
class StructReference {
private:
    mutable ReadLock_t _lock;
    const T* ptr;
    
public:
    StructReference(StructMutex_t& mutex, const T& ptr)
        : _lock(mutex, std::defer_lock), ptr(&ptr)
    { }
    
    StructReference(const StructReference&) = delete;
    ~StructReference() { }

    const T* operator->() const {
        if(!_lock.owns_lock())
            _lock.lock();
        return ptr;
    }
    const T& operator*() const {
        if(!_lock.owns_lock())
            _lock.lock();
        return *ptr;
    }
#if defined(__EMSCRIPTEN__)
    template<typename K = T>
        requires std::is_trivially_copyable< K >::value
    operator K() const {
        if (!_lock.owns_lock())
            _lock.lock();
        return *ptr;
    }
#endif
};

#if defined(__EMSCRIPTEN__)
#define SELECT_TYPE(TYPE) TYPE
#else
#define SELECT_TYPE(TYPE) typename std::conditional< std::is_trivially_copyable< TYPE >::value, std::atomic< TYPE >, TYPE>::type
#endif

#define EVERY_PAIR(TYPE, NAME) SELECT_TYPE( TYPE ) NAME; StructMutex_t mutex_ ## NAME ;
#define STRUCT_STRING_MEMBERS(NAM, a) EVERY_PAIR a

#define EVERY_PAIR_STRING(a, b) #b,
#define STRINGIZE_MEMBERS(NAM, a) EVERY_PAIR_STRING a

#define EVERY_PAIR_GET_A(a, b) #a
#define EVERY_PAIR_GET_B(a, b) #b
#define EVERY_PLAIN_GET_A_NO_COMMA(a, b) a
#define EVERY_PLAIN_GET_B_NO_COMMA(a, b) b
#define EVERY_PLAIN_GET_B_WITH_T(a, b) b ## _t

#define IMPL_ACCESS_ENUM(NAM, TUPLE) \
template<> struct NAM :: AccessEnum<NAM :: Variables:: EVERY_PLAIN_GET_B_NO_COMMA TUPLE > { \
	template<typename T, typename std::enable_if<std::is_convertible<T, EVERY_PLAIN_GET_A_NO_COMMA TUPLE >::value, std::nullptr_t>::type = nullptr> \
	static void set(T v) { set_impl< EVERY_PLAIN_GET_A_NO_COMMA TUPLE >(v, NAM :: impl(). EVERY_PLAIN_GET_B_NO_COMMA TUPLE, NAM :: impl(). STRUCT_CONCATENATE( mutex_ , EXPAND( EVERY_PLAIN_GET_B_NO_COMMA TUPLE )) ); } \
	static auto get() { return get_impl< EVERY_PLAIN_GET_A_NO_COMMA TUPLE >( NAM :: impl(). EVERY_PLAIN_GET_B_NO_COMMA TUPLE , NAM :: impl(). STRUCT_CONCATENATE( mutex_ , EXPAND( EVERY_PLAIN_GET_B_NO_COMMA TUPLE )) ); } \
	static auto copy() { return copy_impl< EVERY_PLAIN_GET_A_NO_COMMA TUPLE >( NAM :: impl(). EVERY_PLAIN_GET_B_NO_COMMA TUPLE , NAM :: impl(). STRUCT_CONCATENATE( mutex_ , EXPAND( EVERY_PLAIN_GET_B_NO_COMMA TUPLE )) ); } \
};

#define EVERY_PAIR_UPDATE_CONDITION(a, b) else if ( key == #b )
#define UPDATE_MEMBERS(NAM, a) EVERY_PAIR_UPDATE_CONDITION a { assert( printf( "Updating %s::%s of type %s\n", #NAM , EVERY_PAIR_GET_B a , EVERY_PAIR_GET_A a ) ); NAM :: update< NAM :: EVERY_PLAIN_GET_B_NO_COMMA a > ( key, value ); NAM :: set < NAM :: EVERY_PLAIN_GET_B_NO_COMMA a > ( value.template value < EVERY_PLAIN_GET_A_NO_COMMA a >() ) ; }

#define EVERY_PLAIN_GET_B(a, b) b,
#define PLAIN_MEMBERS(NAM, a) EVERY_PLAIN_GET_B a

#define EXPRESS_MEMBER_FUNCTIONS(NAM, TUPLE) IMPL_ACCESS_ENUM(NAM, TUPLE)

#define EXPRESS_MEMBER_TYPES(NAM, TUPLE) using EVERY_PLAIN_GET_B_WITH_T TUPLE = EVERY_PLAIN_GET_A_NO_COMMA TUPLE;
#define EXPRESS_MEMBER_GETTERS(NAM, TUPLE) []() -> auto& { if(!cmn::GlobalSettings::get( EVERY_PAIR_GET_B TUPLE ).is_type< EVERY_PLAIN_GET_A_NO_COMMA TUPLE >()) { auto type_name = cmn::GlobalSettings::get( EVERY_PAIR_GET_B TUPLE ).get().type_name(); throw U_EXCEPTION("Settings type ",type_name," is not '", EVERY_PAIR_GET_A TUPLE ,"' for Variable '", #NAM ,"::", EVERY_PAIR_GET_B TUPLE ,"'."); } return cmn::GlobalSettings::get( EVERY_PAIR_GET_B TUPLE ).get(); },


#define STRUCT_CONCATENATE(arg1, arg2)   STRUCT_CONCATENATE1(arg1, arg2)
#define STRUCT_CONCATENATE1(arg1, arg2)  STRUCT_CONCATENATE2(arg1, arg2)
#define STRUCT_CONCATENATE2(arg1, arg2)  arg1 ## arg2
#define STRUCT_STRINGIZE_SINGLE(ARG) #ARG

#define EXPAND(x) x

/*#define STRUCT_FOR_EACH_1(NAM, what, x) what(NAM, x)
#define STRUCT_FOR_EACH_2(NAM, what, x, ...)\
  what(NAM, x)\
  EXPAND(STRUCT_FOR_EACH_1(NAM, what,  __VA_ARGS__))
#define STRUCT_FOR_EACH_3(NAM, what, x, ...)\
  what(NAM, x)\
  EXPAND(STRUCT_FOR_EACH_2(NAM, what, __VA_ARGS__))
#define STRUCT_FOR_EACH_4(NAM, what, x, ...)\
  what(NAM, x)\
  EXPAND(STRUCT_FOR_EACH_3(NAM, what,  __VA_ARGS__))

*/



#define STRUCT_FOR_EACH_1(NAM, what, x) what(NAM, x)
#define STRUCT_FOR_EACH_2(NAM, what, x, ...)\
    what(NAM, x)\
    EXPAND(STRUCT_FOR_EACH_1(NAM, what,  __VA_ARGS__))

#define STRUCT_FOR_EACH_3(NAM, what, x, ...)\
    what(NAM, x)\
    EXPAND(STRUCT_FOR_EACH_2(NAM, what,  __VA_ARGS__))

#define STRUCT_FOR_EACH_4(NAM, what, x, ...)\
    what(NAM, x)\
    EXPAND(STRUCT_FOR_EACH_3(NAM, what,  __VA_ARGS__))

#define STRUCT_FOR_EACH_5(NAM, what, x, ...)\
    what(NAM, x)\
    EXPAND(STRUCT_FOR_EACH_4(NAM, what,  __VA_ARGS__))

#define STRUCT_FOR_EACH_6(NAM, what, x, ...)\
    what(NAM, x)\
    EXPAND(STRUCT_FOR_EACH_5(NAM, what,  __VA_ARGS__))

#define STRUCT_FOR_EACH_7(NAM, what, x, ...)\
    what(NAM, x)\
    EXPAND(STRUCT_FOR_EACH_6(NAM, what,  __VA_ARGS__))

#define STRUCT_FOR_EACH_8(NAM, what, x, ...)\
    what(NAM, x)\
    EXPAND(STRUCT_FOR_EACH_7(NAM, what,  __VA_ARGS__))

#define STRUCT_FOR_EACH_9(NAM, what, x, ...)\
    what(NAM, x)\
    EXPAND(STRUCT_FOR_EACH_8(NAM, what,  __VA_ARGS__))

#define STRUCT_FOR_EACH_10(NAM, what, x, ...)\
    what(NAM, x)\
    EXPAND(STRUCT_FOR_EACH_9(NAM, what,  __VA_ARGS__))

#define STRUCT_FOR_EACH_11(NAM, what, x, ...)\
    what(NAM, x)\
    EXPAND(STRUCT_FOR_EACH_10(NAM, what,  __VA_ARGS__))

#define STRUCT_FOR_EACH_12(NAM, what, x, ...)\
    what(NAM, x)\
    EXPAND(STRUCT_FOR_EACH_11(NAM, what,  __VA_ARGS__))

#define STRUCT_FOR_EACH_13(NAM, what, x, ...)\
    what(NAM, x)\
    EXPAND(STRUCT_FOR_EACH_12(NAM, what,  __VA_ARGS__))

#define STRUCT_FOR_EACH_14(NAM, what, x, ...)\
    what(NAM, x)\
    EXPAND(STRUCT_FOR_EACH_13(NAM, what,  __VA_ARGS__))

#define STRUCT_FOR_EACH_15(NAM, what, x, ...)\
    what(NAM, x)\
    EXPAND(STRUCT_FOR_EACH_14(NAM, what,  __VA_ARGS__))

#define STRUCT_FOR_EACH_16(NAM, what, x, ...)\
    what(NAM, x)\
    EXPAND(STRUCT_FOR_EACH_15(NAM, what,  __VA_ARGS__))

#define STRUCT_FOR_EACH_17(NAM, what, x, ...)\
    what(NAM, x)\
    EXPAND(STRUCT_FOR_EACH_16(NAM, what,  __VA_ARGS__))

#define STRUCT_FOR_EACH_18(NAM, what, x, ...)\
    what(NAM, x)\
    EXPAND(STRUCT_FOR_EACH_17(NAM, what,  __VA_ARGS__))

#define STRUCT_FOR_EACH_19(NAM, what, x, ...)\
    what(NAM, x)\
    EXPAND(STRUCT_FOR_EACH_18(NAM, what,  __VA_ARGS__))

#define STRUCT_FOR_EACH_20(NAM, what, x, ...)\
    what(NAM, x)\
    EXPAND(STRUCT_FOR_EACH_19(NAM, what,  __VA_ARGS__))

#define STRUCT_FOR_EACH_21(NAM, what, x, ...)\
    what(NAM, x)\
    EXPAND(STRUCT_FOR_EACH_20(NAM, what,  __VA_ARGS__))

#define STRUCT_FOR_EACH_22(NAM, what, x, ...)\
    what(NAM, x)\
    EXPAND(STRUCT_FOR_EACH_21(NAM, what,  __VA_ARGS__))

#define STRUCT_FOR_EACH_23(NAM, what, x, ...)\
    what(NAM, x)\
    EXPAND(STRUCT_FOR_EACH_22(NAM, what,  __VA_ARGS__))

#define STRUCT_FOR_EACH_24(NAM, what, x, ...)\
    what(NAM, x)\
    EXPAND(STRUCT_FOR_EACH_23(NAM, what,  __VA_ARGS__))

#define STRUCT_FOR_EACH_25(NAM, what, x, ...)\
    what(NAM, x)\
    EXPAND(STRUCT_FOR_EACH_24(NAM, what,  __VA_ARGS__))

#define STRUCT_FOR_EACH_26(NAM, what, x, ...)\
    what(NAM, x)\
    EXPAND(STRUCT_FOR_EACH_25(NAM, what,  __VA_ARGS__))

#define STRUCT_FOR_EACH_27(NAM, what, x, ...)\
    what(NAM, x)\
    EXPAND(STRUCT_FOR_EACH_26(NAM, what,  __VA_ARGS__))

#define STRUCT_FOR_EACH_28(NAM, what, x, ...)\
    what(NAM, x)\
    EXPAND(STRUCT_FOR_EACH_27(NAM, what,  __VA_ARGS__))

#define STRUCT_FOR_EACH_29(NAM, what, x, ...)\
    what(NAM, x)\
    EXPAND(STRUCT_FOR_EACH_28(NAM, what,  __VA_ARGS__))

#define STRUCT_FOR_EACH_30(NAM, what, x, ...)\
    what(NAM, x)\
    EXPAND(STRUCT_FOR_EACH_29(NAM, what,  __VA_ARGS__))

#define STRUCT_FOR_EACH_31(NAM, what, x, ...)\
    what(NAM, x)\
    EXPAND(STRUCT_FOR_EACH_30(NAM, what,  __VA_ARGS__))

#define STRUCT_FOR_EACH_32(NAM, what, x, ...)\
    what(NAM, x)\
    EXPAND(STRUCT_FOR_EACH_31(NAM, what,  __VA_ARGS__))

#define STRUCT_FOR_EACH_33(NAM, what, x, ...)\
    what(NAM, x)\
    EXPAND(STRUCT_FOR_EACH_32(NAM, what,  __VA_ARGS__))

#define STRUCT_FOR_EACH_34(NAM, what, x, ...)\
    what(NAM, x)\
    EXPAND(STRUCT_FOR_EACH_33(NAM, what,  __VA_ARGS__))

#define STRUCT_FOR_EACH_35(NAM, what, x, ...)\
    what(NAM, x)\
    EXPAND(STRUCT_FOR_EACH_34(NAM, what,  __VA_ARGS__))

#define STRUCT_FOR_EACH_36(NAM, what, x, ...)\
    what(NAM, x)\
    EXPAND(STRUCT_FOR_EACH_35(NAM, what,  __VA_ARGS__))

#define STRUCT_FOR_EACH_37(NAM, what, x, ...)\
    what(NAM, x)\
    EXPAND(STRUCT_FOR_EACH_36(NAM, what,  __VA_ARGS__))

#define STRUCT_FOR_EACH_38(NAM, what, x, ...)\
    what(NAM, x)\
    EXPAND(STRUCT_FOR_EACH_37(NAM, what,  __VA_ARGS__))

#define STRUCT_FOR_EACH_39(NAM, what, x, ...)\
    what(NAM, x)\
    EXPAND(STRUCT_FOR_EACH_38(NAM, what,  __VA_ARGS__))

#define STRUCT_FOR_EACH_40(NAM, what, x, ...)\
    what(NAM, x)\
    EXPAND(STRUCT_FOR_EACH_39(NAM, what,  __VA_ARGS__))

#define STRUCT_FOR_EACH_41(NAM, what, x, ...)\
    what(NAM, x)\
    EXPAND(STRUCT_FOR_EACH_40(NAM, what,  __VA_ARGS__))

#define STRUCT_FOR_EACH_42(NAM, what, x, ...)\
    what(NAM, x)\
    EXPAND(STRUCT_FOR_EACH_41(NAM, what,  __VA_ARGS__))

#define STRUCT_FOR_EACH_43(NAM, what, x, ...)\
    what(NAM, x)\
    EXPAND(STRUCT_FOR_EACH_42(NAM, what,  __VA_ARGS__))

#define STRUCT_FOR_EACH_44(NAM, what, x, ...)\
    what(NAM, x)\
    EXPAND(STRUCT_FOR_EACH_43(NAM, what,  __VA_ARGS__))

#define STRUCT_FOR_EACH_45(NAM, what, x, ...)\
    what(NAM, x)\
    EXPAND(STRUCT_FOR_EACH_44(NAM, what,  __VA_ARGS__))

#define STRUCT_FOR_EACH_46(NAM, what, x, ...)\
    what(NAM, x)\
    EXPAND(STRUCT_FOR_EACH_45(NAM, what,  __VA_ARGS__))

#define STRUCT_FOR_EACH_47(NAM, what, x, ...)\
    what(NAM, x)\
    EXPAND(STRUCT_FOR_EACH_46(NAM, what,  __VA_ARGS__))

#define STRUCT_FOR_EACH_48(NAM, what, x, ...)\
    what(NAM, x)\
    EXPAND(STRUCT_FOR_EACH_47(NAM, what,  __VA_ARGS__))

#define STRUCT_FOR_EACH_49(NAM, what, x, ...)\
    what(NAM, x)\
    EXPAND(STRUCT_FOR_EACH_48(NAM, what,  __VA_ARGS__))

#define STRUCT_FOR_EACH_50(NAM, what, x, ...)\
    what(NAM, x)\
    EXPAND(STRUCT_FOR_EACH_49(NAM, what,  __VA_ARGS__))

#define STRUCT_FOR_EACH_51(NAM, what, x, ...)\
    what(NAM, x)\
    EXPAND(STRUCT_FOR_EACH_50(NAM, what,  __VA_ARGS__))

#define STRUCT_FOR_EACH_52(NAM, what, x, ...)\
    what(NAM, x)\
    EXPAND(STRUCT_FOR_EACH_51(NAM, what,  __VA_ARGS__))

#define STRUCT_FOR_EACH_53(NAM, what, x, ...)\
    what(NAM, x)\
    EXPAND(STRUCT_FOR_EACH_52(NAM, what,  __VA_ARGS__))

#define STRUCT_FOR_EACH_54(NAM, what, x, ...)\
    what(NAM, x)\
    EXPAND(STRUCT_FOR_EACH_53(NAM, what,  __VA_ARGS__))

#define STRUCT_FOR_EACH_55(NAM, what, x, ...)\
    what(NAM, x)\
    EXPAND(STRUCT_FOR_EACH_54(NAM, what,  __VA_ARGS__))

#define STRUCT_FOR_EACH_56(NAM, what, x, ...)\
    what(NAM, x)\
    EXPAND(STRUCT_FOR_EACH_55(NAM, what,  __VA_ARGS__))

#define STRUCT_FOR_EACH_57(NAM, what, x, ...)\
    what(NAM, x)\
    EXPAND(STRUCT_FOR_EACH_56(NAM, what,  __VA_ARGS__))

#define STRUCT_FOR_EACH_58(NAM, what, x, ...)\
    what(NAM, x)\
    EXPAND(STRUCT_FOR_EACH_57(NAM, what,  __VA_ARGS__))

#define STRUCT_FOR_EACH_59(NAM, what, x, ...)\
    what(NAM, x)\
    EXPAND(STRUCT_FOR_EACH_58(NAM, what,  __VA_ARGS__))

#define STRUCT_FOR_EACH_60(NAM, what, x, ...)\
    what(NAM, x)\
    EXPAND(STRUCT_FOR_EACH_59(NAM, what,  __VA_ARGS__))

#define STRUCT_FOR_EACH_61(NAM, what, x, ...)\
    what(NAM, x)\
    EXPAND(STRUCT_FOR_EACH_60(NAM, what,  __VA_ARGS__))

#define STRUCT_FOR_EACH_62(NAM, what, x, ...)\
    what(NAM, x)\
    EXPAND(STRUCT_FOR_EACH_61(NAM, what,  __VA_ARGS__))

#define STRUCT_FOR_EACH_63(NAM, what, x, ...)\
    what(NAM, x)\
    EXPAND(STRUCT_FOR_EACH_62(NAM, what,  __VA_ARGS__))

#define STRUCT_FOR_EACH_64(NAM, what, x, ...)\
    what(NAM, x)\
    EXPAND(STRUCT_FOR_EACH_63(NAM, what,  __VA_ARGS__))

#define STRUCT_FOR_EACH_65(NAM, what, x, ...)\
    what(NAM, x)\
    EXPAND(STRUCT_FOR_EACH_64(NAM, what,  __VA_ARGS__))

#define STRUCT_FOR_EACH_66(NAM, what, x, ...)\
    what(NAM, x)\
    EXPAND(STRUCT_FOR_EACH_65(NAM, what,  __VA_ARGS__))

#define STRUCT_FOR_EACH_67(NAM, what, x, ...)\
    what(NAM, x)\
    EXPAND(STRUCT_FOR_EACH_66(NAM, what,  __VA_ARGS__))

#define STRUCT_FOR_EACH_68(NAM, what, x, ...)\
    what(NAM, x)\
    EXPAND(STRUCT_FOR_EACH_67(NAM, what,  __VA_ARGS__))

#define STRUCT_FOR_EACH_69(NAM, what, x, ...)\
    what(NAM, x)\
    EXPAND(STRUCT_FOR_EACH_68(NAM, what,  __VA_ARGS__))

#define STRUCT_FOR_EACH_70(NAM, what, x, ...)\
    what(NAM, x)\
    EXPAND(STRUCT_FOR_EACH_69(NAM, what,  __VA_ARGS__))

#define STRUCT_FOR_EACH_71(NAM, what, x, ...)\
    what(NAM, x)\
    EXPAND(STRUCT_FOR_EACH_70(NAM, what,  __VA_ARGS__))

#define STRUCT_FOR_EACH_72(NAM, what, x, ...)\
    what(NAM, x)\
    EXPAND(STRUCT_FOR_EACH_71(NAM, what,  __VA_ARGS__))

#define STRUCT_FOR_EACH_73(NAM, what, x, ...)\
    what(NAM, x)\
    EXPAND(STRUCT_FOR_EACH_72(NAM, what,  __VA_ARGS__))

#define STRUCT_FOR_EACH_74(NAM, what, x, ...)\
    what(NAM, x)\
    EXPAND(STRUCT_FOR_EACH_73(NAM, what,  __VA_ARGS__))

#define STRUCT_FOR_EACH_75(NAM, what, x, ...)\
    what(NAM, x)\
    EXPAND(STRUCT_FOR_EACH_74(NAM, what,  __VA_ARGS__))

#define STRUCT_FOR_EACH_76(NAM, what, x, ...)\
    what(NAM, x)\
    EXPAND(STRUCT_FOR_EACH_75(NAM, what,  __VA_ARGS__))

#define STRUCT_FOR_EACH_77(NAM, what, x, ...)\
    what(NAM, x)\
    EXPAND(STRUCT_FOR_EACH_76(NAM, what,  __VA_ARGS__))

#define STRUCT_FOR_EACH_78(NAM, what, x, ...)\
    what(NAM, x)\
    EXPAND(STRUCT_FOR_EACH_77(NAM, what,  __VA_ARGS__))

#define STRUCT_FOR_EACH_79(NAM, what, x, ...)\
    what(NAM, x)\
    EXPAND(STRUCT_FOR_EACH_78(NAM, what,  __VA_ARGS__))

#define STRUCT_FOR_EACH_80(NAM, what, x, ...)\
    what(NAM, x)\
    EXPAND(STRUCT_FOR_EACH_79(NAM, what,  __VA_ARGS__))

#define STRUCT_FOR_EACH_81(NAM, what, x, ...)\
    what(NAM, x)\
    EXPAND(STRUCT_FOR_EACH_80(NAM, what,  __VA_ARGS__))

#define STRUCT_FOR_EACH_82(NAM, what, x, ...)\
    what(NAM, x)\
    EXPAND(STRUCT_FOR_EACH_81(NAM, what,  __VA_ARGS__))

#define STRUCT_FOR_EACH_83(NAM, what, x, ...)\
    what(NAM, x)\
    EXPAND(STRUCT_FOR_EACH_82(NAM, what,  __VA_ARGS__))

#define STRUCT_FOR_EACH_84(NAM, what, x, ...)\
    what(NAM, x)\
    EXPAND(STRUCT_FOR_EACH_83(NAM, what,  __VA_ARGS__))

#define STRUCT_FOR_EACH_85(NAM, what, x, ...)\
    what(NAM, x)\
    EXPAND(STRUCT_FOR_EACH_84(NAM, what,  __VA_ARGS__))

#define STRUCT_FOR_EACH_86(NAM, what, x, ...)\
    what(NAM, x)\
    EXPAND(STRUCT_FOR_EACH_85(NAM, what,  __VA_ARGS__))

#define STRUCT_FOR_EACH_87(NAM, what, x, ...)\
    what(NAM, x)\
    EXPAND(STRUCT_FOR_EACH_86(NAM, what,  __VA_ARGS__))

#define STRUCT_FOR_EACH_88(NAM, what, x, ...)\
    what(NAM, x)\
    EXPAND(STRUCT_FOR_EACH_87(NAM, what,  __VA_ARGS__))

#define STRUCT_FOR_EACH_89(NAM, what, x, ...)\
    what(NAM, x)\
    EXPAND(STRUCT_FOR_EACH_88(NAM, what,  __VA_ARGS__))

#define STRUCT_FOR_EACH_90(NAM, what, x, ...)\
    what(NAM, x)\
    EXPAND(STRUCT_FOR_EACH_89(NAM, what,  __VA_ARGS__))

#define STRUCT_FOR_EACH_91(NAM, what, x, ...)\
    what(NAM, x)\
    EXPAND(STRUCT_FOR_EACH_90(NAM, what,  __VA_ARGS__))

#define STRUCT_FOR_EACH_92(NAM, what, x, ...)\
    what(NAM, x)\
    EXPAND(STRUCT_FOR_EACH_91(NAM, what,  __VA_ARGS__))

#define STRUCT_FOR_EACH_93(NAM, what, x, ...)\
    what(NAM, x)\
    EXPAND(STRUCT_FOR_EACH_92(NAM, what,  __VA_ARGS__))

#define STRUCT_FOR_EACH_94(NAM, what, x, ...)\
    what(NAM, x)\
    EXPAND(STRUCT_FOR_EACH_93(NAM, what,  __VA_ARGS__))

#define STRUCT_FOR_EACH_95(NAM, what, x, ...)\
    what(NAM, x)\
    EXPAND(STRUCT_FOR_EACH_94(NAM, what,  __VA_ARGS__))

#define STRUCT_FOR_EACH_96(NAM, what, x, ...)\
    what(NAM, x)\
    EXPAND(STRUCT_FOR_EACH_95(NAM, what,  __VA_ARGS__))

#define STRUCT_FOR_EACH_97(NAM, what, x, ...)\
    what(NAM, x)\
    EXPAND(STRUCT_FOR_EACH_96(NAM, what,  __VA_ARGS__))

#define STRUCT_FOR_EACH_98(NAM, what, x, ...)\
    what(NAM, x)\
    EXPAND(STRUCT_FOR_EACH_97(NAM, what,  __VA_ARGS__))

#define STRUCT_FOR_EACH_99(NAM, what, x, ...)\
    what(NAM, x)\
    EXPAND(STRUCT_FOR_EACH_98(NAM, what,  __VA_ARGS__))

#define STRUCT_FOR_EACH_100(NAM, what, x, ...)\
    what(NAM, x)\
    EXPAND(STRUCT_FOR_EACH_99(NAM, what,  __VA_ARGS__))

#define STRUCT_FOR_EACH_101(NAM, what, x, ...)\
    what(NAM, x)\
    EXPAND(STRUCT_FOR_EACH_100(NAM, what,  __VA_ARGS__))

#define STRUCT_FOR_EACH_102(NAM, what, x, ...)\
    what(NAM, x)\
    EXPAND(STRUCT_FOR_EACH_101(NAM, what,  __VA_ARGS__))

#define STRUCT_FOR_EACH_103(NAM, what, x, ...)\
    what(NAM, x)\
    EXPAND(STRUCT_FOR_EACH_102(NAM, what,  __VA_ARGS__))

#define STRUCT_FOR_EACH_104(NAM, what, x, ...)\
    what(NAM, x)\
    EXPAND(STRUCT_FOR_EACH_103(NAM, what,  __VA_ARGS__))

#define STRUCT_FOR_EACH_105(NAM, what, x, ...)\
    what(NAM, x)\
    EXPAND(STRUCT_FOR_EACH_104(NAM, what,  __VA_ARGS__))

#define STRUCT_FOR_EACH_106(NAM, what, x, ...)\
    what(NAM, x)\
    EXPAND(STRUCT_FOR_EACH_105(NAM, what,  __VA_ARGS__))

#define STRUCT_FOR_EACH_107(NAM, what, x, ...)\
    what(NAM, x)\
    EXPAND(STRUCT_FOR_EACH_106(NAM, what,  __VA_ARGS__))

#define STRUCT_FOR_EACH_108(NAM, what, x, ...)\
    what(NAM, x)\
    EXPAND(STRUCT_FOR_EACH_107(NAM, what,  __VA_ARGS__))

#define STRUCT_FOR_EACH_109(NAM, what, x, ...)\
    what(NAM, x)\
    EXPAND(STRUCT_FOR_EACH_108(NAM, what,  __VA_ARGS__))

#define STRUCT_FOR_EACH_110(NAM, what, x, ...)\
    what(NAM, x)\
    EXPAND(STRUCT_FOR_EACH_109(NAM, what,  __VA_ARGS__))

#define STRUCT_FOR_EACH_111(NAM, what, x, ...)\
    what(NAM, x)\
    EXPAND(STRUCT_FOR_EACH_110(NAM, what,  __VA_ARGS__))

#define STRUCT_FOR_EACH_112(NAM, what, x, ...)\
    what(NAM, x)\
    EXPAND(STRUCT_FOR_EACH_111(NAM, what,  __VA_ARGS__))

#define STRUCT_FOR_EACH_113(NAM, what, x, ...)\
    what(NAM, x)\
    EXPAND(STRUCT_FOR_EACH_112(NAM, what,  __VA_ARGS__))

#define STRUCT_FOR_EACH_114(NAM, what, x, ...)\
    what(NAM, x)\
    EXPAND(STRUCT_FOR_EACH_113(NAM, what,  __VA_ARGS__))

#define STRUCT_FOR_EACH_115(NAM, what, x, ...)\
    what(NAM, x)\
    EXPAND(STRUCT_FOR_EACH_114(NAM, what,  __VA_ARGS__))

#define STRUCT_FOR_EACH_116(NAM, what, x, ...)\
    what(NAM, x)\
    EXPAND(STRUCT_FOR_EACH_115(NAM, what,  __VA_ARGS__))

#define STRUCT_FOR_EACH_117(NAM, what, x, ...)\
    what(NAM, x)\
    EXPAND(STRUCT_FOR_EACH_116(NAM, what,  __VA_ARGS__))

#define STRUCT_FOR_EACH_118(NAM, what, x, ...)\
    what(NAM, x)\
    EXPAND(STRUCT_FOR_EACH_117(NAM, what,  __VA_ARGS__))

#define STRUCT_FOR_EACH_119(NAM, what, x, ...)\
    what(NAM, x)\
    EXPAND(STRUCT_FOR_EACH_118(NAM, what,  __VA_ARGS__))

#define STRUCT_FOR_EACH_120(NAM, what, x, ...)\
    what(NAM, x)\
    EXPAND(STRUCT_FOR_EACH_119(NAM, what,  __VA_ARGS__))

#define STRUCT_FOR_EACH_121(NAM, what, x, ...)\
    what(NAM, x)\
    EXPAND(STRUCT_FOR_EACH_120(NAM, what,  __VA_ARGS__))

#define STRUCT_FOR_EACH_122(NAM, what, x, ...)\
    what(NAM, x)\
    EXPAND(STRUCT_FOR_EACH_121(NAM, what,  __VA_ARGS__))

#define STRUCT_FOR_EACH_123(NAM, what, x, ...)\
    what(NAM, x)\
    EXPAND(STRUCT_FOR_EACH_122(NAM, what,  __VA_ARGS__))

#define STRUCT_FOR_EACH_ARG_N( _1, _2, _3, _4, _5, _6, _7, _8, _9, _10, _11, _12, _13, _14, _15, _16, _17, _18, _19, _20, _21, _22, _23, _24, _25, _26, _27, _28, _29, _30, _31, _32, _33, _34, _35, _36, _37, _38, _39, _40, _41, _42, _43, _44, _45, _46, _47, _48, _49, _50, _51, _52, _53, _54, _55, _56, _57, _58, _59, _60, _61, _62, _63, _64, _65, _66, _67, _68, _69, _70, _71, _72, _73, _74, _75, _76, _77, _78, _79, _80, _81, _82, _83, _84, _85, _86, _87, _88, _89, _90, _91, _92, _93, _94, _95, _96, _97, _98, _99, _100, _101, _102, _103, _104, _105, _106, _107, _108, _109, _110, _111, _112, _113, _114, _115, _116, _117, _118, _119, _120, _121, _122, _123, _124,  N, ...) N
#define STRUCT_FOR_EACH_RSEQ_N() 124, 123, 122, 121, 120, 119, 118, 117, 116, 115, 114, 113, 112, 111, 110, 109, 108, 107, 106, 105, 104, 103, 102, 101, 100, 99, 98, 97, 96, 95, 94, 93, 92, 91, 90, 89, 88, 87, 86, 85, 84, 83, 82, 81, 80, 79, 78, 77, 76, 75, 74, 73, 72, 71, 70, 69, 68, 67, 66, 65, 64, 63, 62, 61, 60, 59, 58, 57, 56, 55, 54, 53, 52, 51, 50, 49, 48, 47, 46, 45, 44, 43, 42, 41, 40, 39, 38, 37, 36, 35, 34, 33, 32, 31, 30, 29, 28, 27, 26, 25, 24, 23, 22, 21, 20, 19, 18, 17, 16, 15, 14, 13, 12, 11, 10, 9, 8, 7, 6, 5, 4, 3, 2, 1, 0

#define STRUCT_FOR_EACH_NARG(...) STRUCT_FOR_EACH_NARG_(__VA_ARGS__, STRUCT_FOR_EACH_RSEQ_N())
#define STRUCT_FOR_EACH_NARG_(...) EXPAND(STRUCT_FOR_EACH_ARG_N(__VA_ARGS__))

#define STRUCT_FOR_EACH_(N, NAM, what, ...) EXPAND(STRUCT_CONCATENATE(STRUCT_FOR_EACH_, N)(NAM, what, __VA_ARGS__))
#define STRUCT_FOR_EACH(NAM, what, ...) STRUCT_FOR_EACH_( STRUCT_FOR_EACH_NARG( __VA_ARGS__ ) , NAM, what, __VA_ARGS__ )
/*enum Variables {
    test,
    bed
};

constexpr const char* VariableName[] = {
    "test",
    "bed"
};

template<Variables v>
void update() {
    printf("%s\n", VariableName[v]);
}*/

#ifndef NDEBUG
inline void member_destruct(const char*name) {
    printf("Destruction members '%s'\n", name);
}
inline void member_construct(const char*name) {
    printf("Construction members '%s'\n", name);
}
#else
inline void member_destruct(const char*) {}
inline void member_construct(const char*) {}
#endif

template<typename Variables, typename callback_fn_t>
struct CallbackHolder {
    const char *name;
    std::unordered_map<Variables, callback_fn_t> _callbacks;
    CallbackHolder(const char*name) : name(name) {
#ifndef NDEBUG
        printf("CallbackHolder for '%s' created.\n", name);
#endif
    }
    ~CallbackHolder() {
#ifndef NDEBUG
        printf("CallbackHolder for '%s' destructed.\n", name);
#endif
    }
};


#define CREATE_STRUCT(NAM, ...) \
class NAM { \
public: \
    enum Variables { STRUCT_FOR_EACH(NAM, PLAIN_MEMBERS, __VA_ARGS__) }; \
    struct Members { \
        Members() { member_construct(#NAM); } \
        ~Members() { member_destruct(#NAM); } \
        STRUCT_FOR_EACH(NAM, STRUCT_STRING_MEMBERS, __VA_ARGS__) \
    }; \
    STRUCT_FOR_EACH(NAM, EXPRESS_MEMBER_TYPES, __VA_ARGS__) \
private: \
    static constexpr const char * VariableNames[] { STRUCT_FOR_EACH(NAM, STRINGIZE_MEMBERS, __VA_ARGS__) }; \
    template<Variables M> struct AccessEnum { }; \
\
public: \
    static auto& members() { \
        static Members _members; \
        return _members; \
    } \
    typedef std::function<void(const std::string&, const sprite::PropertyType&)> callback_fn_t; \
    static auto& callbacks() { \
        static CallbackHolder<Variables, callback_fn_t> _callbacks(#NAM); \
        return _callbacks._callbacks; \
    } \
    static inline const std::array<std::function<const cmn::sprite::PropertyType&()>, STRUCT_FOR_EACH_NARG(__VA_ARGS__)> _getters { \
        STRUCT_FOR_EACH(NAM, EXPRESS_MEMBER_GETTERS, __VA_ARGS__) \
    }; \
    static auto& get(Variables name) { return _getters[name](); } \
    template<Variables M> static auto get() { return AccessEnum<M>::get(); } \
    template<Variables M> static auto copy() { return AccessEnum<M>::copy(); } \
    template<Variables M, typename T> static void set(T obj) { AccessEnum<M>::set(obj); } \
private: \
 \
    template<typename T, typename K> \
    static inline auto get_impl(const K& obj, StructMutex_t& mutex) { \
        if constexpr (std::is_same<std::atomic<T>, K>::value) { \
            return obj.load(); \
        } \
        else { \
            return StructReference<T>(mutex, obj); \
        } \
    } \
 \
    template<typename T, typename K> \
    static inline T copy_impl(const K& obj, StructMutex_t& mutex) { \
        if constexpr (std::is_same<std::atomic<T>, K>::value) { \
            return obj.load(); \
        } \
        else { \
            ReadLock_t guard(mutex); \
            return obj; \
        } \
    } \
 \
    template<typename T, typename K> \
    static inline void set_impl(T v, K& obj, StructMutex_t& mutex) { \
        if constexpr (std::is_same<std::atomic<T>, K>::value) { \
            obj = v; \
        } \
        else { \
            std::unique_lock guard(mutex); \
            obj = v; \
        } \
    } \
public: \
    inline static NAM :: Members & impl() { return NAM :: members(); } \
    template<Variables M> static void update(const std::string &key, const sprite::PropertyType& value) { auto it = callbacks().find(M); if(it != callbacks().end()) it->second(key, value); } \
    template<Variables M> \
    static const char* name() { \
        return VariableNames[M]; \
    } \
    static void set_callback(Variables v, callback_fn_t f) { callbacks()[v] = f; } \
    static void clear_callbacks() { callbacks().clear(); } \
    static std::vector<std::string> names() { return std::vector<std::string>{ STRUCT_FOR_EACH(NAM, STRINGIZE_MEMBERS, __VA_ARGS__) }; } \
    static void variable_changed (sprite::Map::Signal signal, sprite::Map &, const std::string &key, const sprite::PropertyType& value) { \
        if(signal == sprite::Map::Signal::EXIT) { \
            cmn::GlobalSettings::map().unregister_callback(#NAM); \
            return; \
        } \
        if(false); STRUCT_FOR_EACH(NAM, UPDATE_MEMBERS, __VA_ARGS__) \
    } \
    static inline void init() { \
        static std::once_flag flag; \
        std::call_once(flag, [](){ \
            cmn::GlobalSettings::map().register_callback(#NAM, NAM :: variable_changed ); \
            for(auto &n : NAM :: names()) \
                variable_changed(sprite::Map::Signal::NONE, cmn::GlobalSettings::map(), n, cmn::GlobalSettings::get(n).get()); \
        }); \
    } \
}; \
STRUCT_FOR_EACH(NAM, EXPRESS_MEMBER_FUNCTIONS, __VA_ARGS__)

#define STRUCT_META_EXTENSIONS(NAM) \
 \
template<> inline std::string cmn::Meta::toStr<NAM :: Variables>(const NAM :: Variables& value, const NAM :: Variables* ) { return NAM :: names()[value]; } \
template<> inline std::string cmn::Meta::name<enum NAM :: Variables>(const enum NAM :: Variables*) { return #NAM ; } \
template<> inline enum NAM :: Variables cmn::Meta::fromStr<enum NAM :: Variables>(const std::string& str, const enum NAM :: Variables* ) { \
    size_t index = 0; \
    for(auto &name : NAM :: names()) { \
        if(str == name) { \
            return (NAM :: Variables)index; \
        } \
        ++index; \
    } \
    \
    throw CustomException(cmn::type<std::invalid_argument>, "Cannot find variable '", #NAM ,"::", str.c_str() ,"'."); \
}
