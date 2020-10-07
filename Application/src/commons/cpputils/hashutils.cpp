#include "hashutils.h"
#include "utilsexception.h"

namespace utils {
	StringHashMaps *StringHashMaps::INSTANCE = NULL;

	/**
	* Returns the name of a certain hash (if known).
	* @param hash a hash known to the shader class
	* @return NULL if the hash can not be found, or the real-name
	*/
	const char * StringHashMaps::nameForHash(unsigned int hash) {
		if (!StringHashMaps::INSTANCE)
			throw UtilsException("An instance of a utils::StringHashMaps object has to be created before any static method calls.");

		auto pos = StringHashMaps::INSTANCE->hashMap.find(hash);
		if (pos != StringHashMaps::INSTANCE->hashMap.end()) {
			return (*pos).second;
		}

		return NULL;
	}

	/**
	* Adds a hash for a given name to the global hashMap.
	* @param name real-name of the hash
	* @param hash the hash
	*/
	void StringHashMaps::addHashForName(const char *name, unsigned int hash) {
		if (!StringHashMaps::INSTANCE)
			throw UtilsException("An instance of a utils::StringHashMaps object has to be created before any static method calls.");

		assert(nameForHash(hash) == NULL);
		StringHashMaps::INSTANCE->hashMap.insert(std::pair<unsigned int, const char*>(hash, name));
	}

	/**
	* Returns the name of a certain StringHash.
	* Basically returns the const char instance that is already saved inside StringHash.
	* @param hash reference to a certain StringHash
	*/
	const char * StringHashMaps::nameForHash(const utils::StringHash &hash) {
		return hash.source_string;
	}
}
