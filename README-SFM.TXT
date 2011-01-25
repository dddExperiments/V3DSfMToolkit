Readme for the ETH-V3D Structure-and-Motion pipeline

Features and shortcomings
-------------------------

Warning: this software should be considered being in beta stage at best. I'm
continuing to add features and fix bugs and other problems.

Features are:
* Relatively fast by leveraging the GPU and avoiding frequent bundle adjustment
* Has usually no problems with drift and loop closing
* The library contains lots of code useful on its own. Even if the overall SfM
  approach is not suitable for your application, there might be useful
  building blocks for your own code.

Drawbacks are:
* Very little documentation
* Experimental code, especially in later stages in the pipeline
* Currently the utilized approach does not fully exploit correspondences
  between images and often creates independent models residing in their own
  coordinate frame.
* Support for weakly calibrated cameras (i.e. using EXIF tags) is rather new,
  and the performance of the used SfM approach in this setting is not well
  known.


Installation instructions
-------------------------

I assume a recent Linux box. The software depends on a number of 3rd party
libraries, and most of them are easy to install via the software packaging
system of your distribution. Note that in order to compile the software you
have to install the development packages!

Libraries I build on include (ubuntu 10.04 package names in brackets):

boost       [libboost-all-dev]
libjpeg     [libjpeg62-dev] and libpng [libpng12-dev] (for reading and writing images)
sqlite3     [libsqlite3-dev]     (for temporary data files)
arpack      [libarpack2-dev]     (sparse eigenvalue problems)
suitesparse [libsuitesparse-dev] (for COLAMD and sparse Cholesky factorization)
libexif     [libexif-dev]        (reading JPEGs exif tags)
SiftGPU                          (available from http://www.cs.unc.edu/~ccwu/siftgpu/)
Cg                               (for SiftGPU; available from NVidia's webpage)
CUDA                             (for SIFT descriptor quantization and matching; available from NVidia's webpage)

For the loop inference stuff in V3D/Apps/CycleInference you will need:

lp-solve [liblpsolve55-dev] (solving LP/MILP instances)
libdai (inference in graphical models, available from http://people.kyb.tuebingen.mpg.de/jorism/libDAI/)

You may need to edit V3D/Config/local_config.cmake to reflect different
include/library paths (e.g. for SiftGPU).

I'm still using CUDA 2.3, which does not directly support gcc-4.4 in Ubuntu
10.04, hence I installed gcc-4.3 and created a gcc-4.3 folder in $HOME/tmp,
and symlinked gcc to /usr/bin/gcc-4.3. You have to edit the NVCC and NVCC_ARGS
variables in V3D/CMakeLists.txt if you have a different configuration.

If you don't want to play with loop inference, just change
   enable_feature (V3DLIB_ENABLE_LIBDAI)
to 
   #enable_feature (V3DLIB_ENABLE_LIBDAI)
in V3D/Config/local_config.cmake and the respective binaries are not built.

Finally, in V3D/build run
   cmake .. && make
to compile everything. Binaries are available in 
   build/Apps/SfM
and
   build/Apps/CycleInference.


Running the software
--------------------

In the Data/Example folder there is a small image data set and a shell script
to run the SfM pipeline. It should be more or less self-explanatory. Since
SiftGPU and CUDA-based feature matching and descriptor quantization is an
integral part of the software, you need to have an appropriate NVidia GPU.

To see the pipeline working, optionally edit run_sfm.sh and/or conf.txt, and
start everything in the Data/Example directory by typing in
   sh ./run_sfm.sh

Most executables print a usage message if run without arguments.

The final result consists of a VRML files representing the sparse point cloud
(ba-points3d*.wrl), plus the camera poses (models-*-cams.txt) and 3D points
(models-*-points.txt) in easy to parse text format. The camera poses file has
the same format as used for SSBA, see
http://www.inf.ethz.ch/personal/chzach/oss/README-SSBA.txt.


Contributors
------------

I would like to thank the following people, who provided code or data used in
this distribution (in alphabetical order):

Brian Clipp
David Gallup
Arnold Irschara
Manfred Klopschitz
Changchang Wu


Citations
---------

If you find the software or some part of it useful, then the following
publications are relevant:

The particular vocabulary tree implementation is described in A. Irschara,
C. Zach, J.-M. Frahm, and H. Bischof. From SfM Point Clouds to Fast Location
Recognition. CVPR, p. 2599-2606, 2009.

The upgrade from two-view image relations to a common coordinate frame is a
variant of C. Zach and M. Pollefeys. Practical Methods For Convex Multi-View
Reconstruction. ECCV 2010.

Inference over loops to detect bogus two-view relations was presented in
C. Zach, M. Klopschitz, and M. Pollefeys. Disambiguating Visual Relations
Using Loop Constraints. CVPR 2010.


License
-------

ETH-V3D Structure-and-Motion software
Copyright (C) 2010  Christopher Zach, ETH Zurich

This program is free software: you can redistribute it and/or modify
it under the terms of the GNU General Public License as published by
the Free Software Foundation, either version 3 of the License, or
(at your option) any later version.

This program is distributed in the hope that it will be useful,
but WITHOUT ANY WARRANTY; without even the implied warranty of
MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
GNU General Public License for more details.

You should have received a copy of the GNU General Public License
along with this program.  If not, see <http://www.gnu.org/licenses/>.


Contact
-------

Christopher Zach <http://www.inf.ethz.ch/personal/chzach>
