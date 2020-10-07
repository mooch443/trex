#ifndef _HASH_UTILS_H
#define _HASH_UTILS_H

#include <iostream>
#include <map>
#include <assert.h>

namespace utils {
#ifndef _WIN32
    /**
     * Converts a given constant string, given at compile-time, into an
     * unsigned int hash code.
     * @warning does not check for strlen() - if it differs from the array
     * length, please provide the size as second parameter
     * @param str a string in quotes (does not work with variables)
     */
    template<size_t N>
    constexpr unsigned int HASH(const char(&str)[N], size_t I = N) {
        return (I == 1 ? ((2166136261u ^ str[0]) * 16777619u) : ((fun(str, I - 1) ^ str[I - 1]) * 16777619u));
    }
#endif

#ifndef ME_INLINE
#define ME_INLINE inline
#endif

	//! A helper class.
	template <unsigned int N, unsigned int I>
	struct FnvHash
	{
		ME_INLINE static unsigned int Hash(const char(&str)[N])
		{
			return (FnvHash<N, I - 1>::Hash(str) ^ str[I - 1]) * 16777619u;
		}
	};

	//! Another helper class.
	template <unsigned int N>
	struct FnvHash<N, 1>
	{
		ME_INLINE static unsigned int Hash(const char(&str)[N])
		{
			return (2166136261u ^ str[0]) * 16777619u;
		}
	};

	class StringHash;

	class StringHashMaps {
	public:
		StringHashMaps() {
			if (!StringHashMaps::INSTANCE)
				StringHashMaps::INSTANCE = this;
		}
		~StringHashMaps() {
			if (StringHashMaps::INSTANCE == this)
				StringHashMaps::INSTANCE = NULL;
		}

	private:
		//! Saving all known hashes, assigned to the corresponding real-names
		std::map<unsigned int, const char*> hashMap;

		//! The singleton instance of this class.
		static StringHashMaps *INSTANCE;

	public:
		/**
		* Returns the name of a certain hash (if known).
		* @param hash a hash known to the shader class
		* @return NULL if the hash can not be found, or the real-name
		*/
		static const char * nameForHash(unsigned int hash);

		/**
		* Adds a hash for a given name to the global hashMap.
		* @param name real-name of the hash
		* @param hash the hash
		*/
		static void addHashForName(const char *name, unsigned int hash);

		/**
		* Returns the name of a certain StringHash.
		* Basically returns the const char instance that is already saved inside StringHash.
		* @param hash reference to a certain StringHash
		*/
		static const char * nameForHash(const utils::StringHash &hash);
	};

    /**
     * Converts a given constant string, given at compile-time, into an
     * unsigned int hash code.
     * @warning does not check for strlen() - if it differs from the array
     * length, please provide the size as second parameter
     * @param str a string in quotes (does not work with variables)
     */
    class StringHash
    {
    public:
		template <unsigned int N>
		StringHash(const char(&str)[N])
			: m_hash(FnvHash<N, N>::Hash(str)), source_string(str)
		{
			if (!StringHashMaps::nameForHash(m_hash)) {
				StringHashMaps::addHashForName(source_string, m_hash);
			}
		}
        
    public:
        const unsigned int m_hash;
        const char * const source_string;
    };
}

#endif
