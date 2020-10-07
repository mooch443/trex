/**
* 𝓞(n³) implementation of the Hungarian algorithm
*
* Copyright (C) 2011, 2014  Mattias Andrée
*
* This program is free software. It comes without any warranty, to
* the extent permitted by applicable law. You can redistribute it
* and/or modify it under the terms of the Do What The Fuck You Want
* To Public License, Version 2, as published by Sam Hocevar. See
* http://sam.zoy.org/wtfpl/COPYING for more details.
*/

#ifndef HUNGARIAN_H
#define HUNGARIAN_H

#include <stdio.h>
#include <stdlib.h>
#include <stdint.h>

#if defined(_MSC_VER)
#include <BaseTsd.h>
namespace track {
typedef SSIZE_T ssize_t;
}
#define __attribute__(X)
#endif

#include <tracking/PairingGraph.h>

namespace track {
typedef int64_t Hungarian_t;
ssize_t** kuhn_match(Hungarian_t** table, size_t n, size_t m);
}

#endif
