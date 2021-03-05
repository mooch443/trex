# - Try to find Pylon
# Once done this will define
#  PYLON_FOUND - System has Pylon
#  PYLON_INCLUDE_DIRS - The Pylon include directories
#  PYLON_LIBRARIES - The libraries needed to use Pylon

IF(${OperatingSystem} MATCHES "Windows")
	if( CMAKE_SIZEOF_VOID_P EQUAL 8 )
		set( PYLON_LIBRARY "$ENV{PYLON_ROOT}/lib/x64" )
	else()
		set( PYLON_LIBRARY "$ENV{PYLON_ROOT}/lib/Win32" )
	endif()
ELSE()
		if( CMAKE_SIZEOF_VOID_P EQUAL 8 )
			set( PYLON_LIBRARY "/opt/pylon5/lib64" )
		else()
			set( PYLON_LIBRARY "/opt/pylon5/lib32" )
		endif()
ENDIF()

message(STATUS "Searching for pylon at " ${PYLON_LIBRARY})

FIND_PATH(	PYLON_INCLUDE_DIR pylon/PylonBase.h
			PATHS
			/opt/pylon5/include
			"$ENV{PYLON_ROOT}/include"
)

FIND_LIBRARY(	PYLONBASE_LIBRARY 
				NAMES 
				pylonbase PylonBase_MD_VC100 PylonBase_v6_0 PylonBase_v5_2
				PATHS
				${PYLON_LIBRARY}
)

FIND_LIBRARY(	PYLON_UTILITY_LIBRARY 
				NAMES 
				pylonutility PylonUtility_MD_VC100 PylonUtility_v6_0 PylonUtility_v5_2
				PATHS
				${PYLON_LIBRARY}
)

FIND_LIBRARY( PYLON_GEN_LIBRARY
	NAMES
	GenApi_gcc_v3_0_Basler_pylon_v5_0
	GenApi_gcc_v3_1_Basler_pylon_v5_1
	GenApi_gcc_v3_1_Basler_pylon 
	GenApi_MD_VC141_v3_1_Basler_pylon 
	PATHS
	${PYLON_LIBRARY}
)

FIND_LIBRARY( PYLON_GEN2_LIBRARY
	NAMES
	GCBase_gcc_v3_0_Basler_pylon_v5_0
	GCBase_gcc_v3_0_Basler_pylon_v5_1
	GCBase_gcc_v3_1_Basler_pylon 
	GCBase_MD_VC141_v3_1_Basler_pylon
	PATHS
	${PYLON_LIBRARY}
)

set( XERCES-C_LIBRARY "" )
FIND_LIBRARY(	XERCES-C_LIBRARY 
				NAMES 
				Xerces-C_gcc40_v2_7 Xerces-C_MD_VC100_v2_7_1
				PATHS
				${PYLON_LIBRARY}
)

if( NOT XERCES-C_LIBRARY)
    set(XERCES-C_LIBRARY "")
endif(NOT XERCES-C_LIBRARY)

set(PYLON_LIBRARIES  ${PYLONBASE_LIBRARY} ${XERCES-C_LIBRARY} ${PYLON_UTILITY_LIBRARY} ${PYLON_GEN_LIBRARY} ${PYLON_GEN2_LIBRARY})
set(PYLON_INCLUDE_DIRS ${PYLON_INCLUDE_DIR})

INCLUDE(FindPackageHandleStandardArgs)

FIND_PACKAGE_HANDLE_STANDARD_ARGS(PYLON DEFAULT_MSG
  PYLON_INCLUDE_DIR
  PYLON_LIBRARY)

mark_as_advanced(PYLON_INCLUDE_DIR PYLON_LIBRARIES)
