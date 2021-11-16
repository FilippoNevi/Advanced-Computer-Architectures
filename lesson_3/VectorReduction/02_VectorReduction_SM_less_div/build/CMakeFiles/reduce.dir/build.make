# CMAKE generated file: DO NOT EDIT!
# Generated by "Unix Makefiles" Generator, CMake Version 3.10

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
CMAKE_SOURCE_DIR = /home/nvidia/ACA/Filippo_Nevi/Advanced-Computer-Architectures/lesson_3/VectorReduction/02_VectorReduction_SM_less_div

# The top-level build directory on which CMake was run.
CMAKE_BINARY_DIR = /home/nvidia/ACA/Filippo_Nevi/Advanced-Computer-Architectures/lesson_3/VectorReduction/02_VectorReduction_SM_less_div/build

# Include any dependencies generated for this target.
include CMakeFiles/reduce.dir/depend.make

# Include the progress variables for this target.
include CMakeFiles/reduce.dir/progress.make

# Include the compile flags for this target's objects.
include CMakeFiles/reduce.dir/flags.make

CMakeFiles/reduce.dir/reduce_generated_reduce.cu.o: CMakeFiles/reduce.dir/reduce_generated_reduce.cu.o.depend
CMakeFiles/reduce.dir/reduce_generated_reduce.cu.o: CMakeFiles/reduce.dir/reduce_generated_reduce.cu.o.cmake
CMakeFiles/reduce.dir/reduce_generated_reduce.cu.o: ../reduce.cu
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --blue --bold --progress-dir=/home/nvidia/ACA/Filippo_Nevi/Advanced-Computer-Architectures/lesson_3/VectorReduction/02_VectorReduction_SM_less_div/build/CMakeFiles --progress-num=$(CMAKE_PROGRESS_1) "Building NVCC (Device) object CMakeFiles/reduce.dir/reduce_generated_reduce.cu.o"
	cd /home/nvidia/ACA/Filippo_Nevi/Advanced-Computer-Architectures/lesson_3/VectorReduction/02_VectorReduction_SM_less_div/build/CMakeFiles/reduce.dir && /usr/bin/cmake -E make_directory /home/nvidia/ACA/Filippo_Nevi/Advanced-Computer-Architectures/lesson_3/VectorReduction/02_VectorReduction_SM_less_div/build/CMakeFiles/reduce.dir//.
	cd /home/nvidia/ACA/Filippo_Nevi/Advanced-Computer-Architectures/lesson_3/VectorReduction/02_VectorReduction_SM_less_div/build/CMakeFiles/reduce.dir && /usr/bin/cmake -D verbose:BOOL=$(VERBOSE) -D build_configuration:STRING= -D generated_file:STRING=/home/nvidia/ACA/Filippo_Nevi/Advanced-Computer-Architectures/lesson_3/VectorReduction/02_VectorReduction_SM_less_div/build/CMakeFiles/reduce.dir//./reduce_generated_reduce.cu.o -D generated_cubin_file:STRING=/home/nvidia/ACA/Filippo_Nevi/Advanced-Computer-Architectures/lesson_3/VectorReduction/02_VectorReduction_SM_less_div/build/CMakeFiles/reduce.dir//./reduce_generated_reduce.cu.o.cubin.txt -P /home/nvidia/ACA/Filippo_Nevi/Advanced-Computer-Architectures/lesson_3/VectorReduction/02_VectorReduction_SM_less_div/build/CMakeFiles/reduce.dir//reduce_generated_reduce.cu.o.cmake

# Object files for target reduce
reduce_OBJECTS =

# External object files for target reduce
reduce_EXTERNAL_OBJECTS = \
"/home/nvidia/ACA/Filippo_Nevi/Advanced-Computer-Architectures/lesson_3/VectorReduction/02_VectorReduction_SM_less_div/build/CMakeFiles/reduce.dir/reduce_generated_reduce.cu.o"

reduce: CMakeFiles/reduce.dir/reduce_generated_reduce.cu.o
reduce: CMakeFiles/reduce.dir/build.make
reduce: /usr/local/cuda-10.2/lib64/libcudart_static.a
reduce: /usr/lib/aarch64-linux-gnu/librt.so
reduce: CMakeFiles/reduce.dir/link.txt
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green --bold --progress-dir=/home/nvidia/ACA/Filippo_Nevi/Advanced-Computer-Architectures/lesson_3/VectorReduction/02_VectorReduction_SM_less_div/build/CMakeFiles --progress-num=$(CMAKE_PROGRESS_2) "Linking CXX executable reduce"
	$(CMAKE_COMMAND) -E cmake_link_script CMakeFiles/reduce.dir/link.txt --verbose=$(VERBOSE)

# Rule to build all files generated by this target.
CMakeFiles/reduce.dir/build: reduce

.PHONY : CMakeFiles/reduce.dir/build

CMakeFiles/reduce.dir/requires:

.PHONY : CMakeFiles/reduce.dir/requires

CMakeFiles/reduce.dir/clean:
	$(CMAKE_COMMAND) -P CMakeFiles/reduce.dir/cmake_clean.cmake
.PHONY : CMakeFiles/reduce.dir/clean

CMakeFiles/reduce.dir/depend: CMakeFiles/reduce.dir/reduce_generated_reduce.cu.o
	cd /home/nvidia/ACA/Filippo_Nevi/Advanced-Computer-Architectures/lesson_3/VectorReduction/02_VectorReduction_SM_less_div/build && $(CMAKE_COMMAND) -E cmake_depends "Unix Makefiles" /home/nvidia/ACA/Filippo_Nevi/Advanced-Computer-Architectures/lesson_3/VectorReduction/02_VectorReduction_SM_less_div /home/nvidia/ACA/Filippo_Nevi/Advanced-Computer-Architectures/lesson_3/VectorReduction/02_VectorReduction_SM_less_div /home/nvidia/ACA/Filippo_Nevi/Advanced-Computer-Architectures/lesson_3/VectorReduction/02_VectorReduction_SM_less_div/build /home/nvidia/ACA/Filippo_Nevi/Advanced-Computer-Architectures/lesson_3/VectorReduction/02_VectorReduction_SM_less_div/build /home/nvidia/ACA/Filippo_Nevi/Advanced-Computer-Architectures/lesson_3/VectorReduction/02_VectorReduction_SM_less_div/build/CMakeFiles/reduce.dir/DependInfo.cmake --color=$(COLOR)
.PHONY : CMakeFiles/reduce.dir/depend

