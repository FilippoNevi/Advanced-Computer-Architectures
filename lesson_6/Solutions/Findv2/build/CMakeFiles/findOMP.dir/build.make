# CMAKE generated file: DO NOT EDIT!
# Generated by "Unix Makefiles" Generator, CMake Version 3.5

# Delete rule output on recipe failure.
.DELETE_ON_ERROR:


#=============================================================================
# Special targets provided by cmake.

# Disable implicit rules so canonical targets will work.
.SUFFIXES:


# Remove some rules from gmake that .SUFFIXES does not remove.
SUFFIXES =

.SUFFIXES: .hpux_make_needs_suffix_list


# Suppress display of executed commands.
$(VERBOSE).SILENT:


# A target that is always out of date.
cmake_force:

.PHONY : cmake_force

#=============================================================================
# Set environment variables for the build.

# The shell in which to execute make rules.
SHELL = /bin/sh

# The CMake executable.
CMAKE_COMMAND = /usr/bin/cmake

# The command to remove a file.
RM = /usr/bin/cmake -E remove -f

# Escaping for special characters.
EQUALS = =

# The top-level source directory on which CMake was run.
CMAKE_SOURCE_DIR = /mnt/c/Users/Federico/Dropbox/CorsoAA/Esercizi/OpenMP/SoluzioniOMP2v2/Findv2

# The top-level build directory on which CMake was run.
CMAKE_BINARY_DIR = /mnt/c/Users/Federico/Dropbox/CorsoAA/Esercizi/OpenMP/SoluzioniOMP2v2/Findv2/build

# Include any dependencies generated for this target.
include CMakeFiles/findOMP.dir/depend.make

# Include the progress variables for this target.
include CMakeFiles/findOMP.dir/progress.make

# Include the compile flags for this target's objects.
include CMakeFiles/findOMP.dir/flags.make

CMakeFiles/findOMP.dir/Find.cpp.o: CMakeFiles/findOMP.dir/flags.make
CMakeFiles/findOMP.dir/Find.cpp.o: ../Find.cpp
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green --progress-dir=/mnt/c/Users/Federico/Dropbox/CorsoAA/Esercizi/OpenMP/SoluzioniOMP2v2/Findv2/build/CMakeFiles --progress-num=$(CMAKE_PROGRESS_1) "Building CXX object CMakeFiles/findOMP.dir/Find.cpp.o"
	/usr/bin/c++   $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -o CMakeFiles/findOMP.dir/Find.cpp.o -c /mnt/c/Users/Federico/Dropbox/CorsoAA/Esercizi/OpenMP/SoluzioniOMP2v2/Findv2/Find.cpp

CMakeFiles/findOMP.dir/Find.cpp.i: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green "Preprocessing CXX source to CMakeFiles/findOMP.dir/Find.cpp.i"
	/usr/bin/c++  $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -E /mnt/c/Users/Federico/Dropbox/CorsoAA/Esercizi/OpenMP/SoluzioniOMP2v2/Findv2/Find.cpp > CMakeFiles/findOMP.dir/Find.cpp.i

CMakeFiles/findOMP.dir/Find.cpp.s: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green "Compiling CXX source to assembly CMakeFiles/findOMP.dir/Find.cpp.s"
	/usr/bin/c++  $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -S /mnt/c/Users/Federico/Dropbox/CorsoAA/Esercizi/OpenMP/SoluzioniOMP2v2/Findv2/Find.cpp -o CMakeFiles/findOMP.dir/Find.cpp.s

CMakeFiles/findOMP.dir/Find.cpp.o.requires:

.PHONY : CMakeFiles/findOMP.dir/Find.cpp.o.requires

CMakeFiles/findOMP.dir/Find.cpp.o.provides: CMakeFiles/findOMP.dir/Find.cpp.o.requires
	$(MAKE) -f CMakeFiles/findOMP.dir/build.make CMakeFiles/findOMP.dir/Find.cpp.o.provides.build
.PHONY : CMakeFiles/findOMP.dir/Find.cpp.o.provides

CMakeFiles/findOMP.dir/Find.cpp.o.provides.build: CMakeFiles/findOMP.dir/Find.cpp.o


# Object files for target findOMP
findOMP_OBJECTS = \
"CMakeFiles/findOMP.dir/Find.cpp.o"

# External object files for target findOMP
findOMP_EXTERNAL_OBJECTS =

findOMP: CMakeFiles/findOMP.dir/Find.cpp.o
findOMP: CMakeFiles/findOMP.dir/build.make
findOMP: CMakeFiles/findOMP.dir/link.txt
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green --bold --progress-dir=/mnt/c/Users/Federico/Dropbox/CorsoAA/Esercizi/OpenMP/SoluzioniOMP2v2/Findv2/build/CMakeFiles --progress-num=$(CMAKE_PROGRESS_2) "Linking CXX executable findOMP"
	$(CMAKE_COMMAND) -E cmake_link_script CMakeFiles/findOMP.dir/link.txt --verbose=$(VERBOSE)

# Rule to build all files generated by this target.
CMakeFiles/findOMP.dir/build: findOMP

.PHONY : CMakeFiles/findOMP.dir/build

CMakeFiles/findOMP.dir/requires: CMakeFiles/findOMP.dir/Find.cpp.o.requires

.PHONY : CMakeFiles/findOMP.dir/requires

CMakeFiles/findOMP.dir/clean:
	$(CMAKE_COMMAND) -P CMakeFiles/findOMP.dir/cmake_clean.cmake
.PHONY : CMakeFiles/findOMP.dir/clean

CMakeFiles/findOMP.dir/depend:
	cd /mnt/c/Users/Federico/Dropbox/CorsoAA/Esercizi/OpenMP/SoluzioniOMP2v2/Findv2/build && $(CMAKE_COMMAND) -E cmake_depends "Unix Makefiles" /mnt/c/Users/Federico/Dropbox/CorsoAA/Esercizi/OpenMP/SoluzioniOMP2v2/Findv2 /mnt/c/Users/Federico/Dropbox/CorsoAA/Esercizi/OpenMP/SoluzioniOMP2v2/Findv2 /mnt/c/Users/Federico/Dropbox/CorsoAA/Esercizi/OpenMP/SoluzioniOMP2v2/Findv2/build /mnt/c/Users/Federico/Dropbox/CorsoAA/Esercizi/OpenMP/SoluzioniOMP2v2/Findv2/build /mnt/c/Users/Federico/Dropbox/CorsoAA/Esercizi/OpenMP/SoluzioniOMP2v2/Findv2/build/CMakeFiles/findOMP.dir/DependInfo.cmake --color=$(COLOR)
.PHONY : CMakeFiles/findOMP.dir/depend

