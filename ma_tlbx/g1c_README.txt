G1C (single Gaussian combined with additional information)

Implementation submitted to MIREX'06

All details are described in my thesis.
The approach is a combination of spectral similarity with information 
computed from fluctuation patterns. The spectral similarity is computed
using a single Gaussian (G1) with full covariance and KL distance as 
suggested by Mandel & Ellis at ISMIR'05. From the fluctuation patterns 
the following information is used: the patterns themselves, the energy 
in the bass, and "gravity" (a simple approximation of perceived tempo).

The weighting parameters were optimized using different music collections 
including music from Magnatune. No collection specific optimizations are made.
The normalization parameters for the combination were computed using the DB-L 
collection described in my thesis.

Documentation of the Matlab files
---------------------------------

Both functions will exit if there is an error in the main loops or if 
they finish the computations.

function ma_g1c_FeatureExtraction(in_file,out_dir)
%%
%%         G1C, implementation as described in thesis
%%         for mirex'06, "music audio search" (was audio music similarity)
%%
%% USAGE EXAMPLES
%%       ma_g1c_FeatureExtraction(in_file,out_dir)
%%       ma_g1c_FeatureExtraction('mypath/myInputFile.txt','mypath/myOutputDirectory')
%%       ma_g1c_FeatureExtraction('mypath/myInputFile','mypath/myOutputDirectory/')
%%
%% INPUT ARGUMENTS
%%
%% in_file: path fo file containing list of wav files to extract features from
%%          all files are 22050Hz mono wav (this function checks if this is true)
%% out_dir: directory to which this function can write output. (logfiles,
%%          and extracted features), files written to out_dir must be read
%%          by function "DistanceCumputation"
%%
%% in_file format (text):
%%      path/to/audiofile/0000001.wav
%%      path/to/audiofile/0000002.wav
%%      path/to/audiofile/0000003.wav
%%      ...
%%      path/to/audiofile/9999999.wav
%%

function ma_g1c_ComputeSimilarities(in_dir,out_file)
%%
%%         G1C, implementation as described in thesis
%%         for mirex, "music audio search" (was audio music similarity)
%%
%% USAGE EXAMPLE
%%       ma_g1c_FeatureExtraction(in_file,out_dir)
%%       ma_g1c_ComputeSimilarities(out_dir,'somepath/distance_matrix.txt')
%%
%% INPUT ARGUMENTS
%%
%% in_dir:   directory to which function "FeatureExtraction" was writing to.
%%           a log file will be created in this directory
%% out_file: whole distance matrix in the following format
%%
%%   <start file, exclude this line>
%%     <SUBMISSION_NAME>
%%     1   path\file1.wav
%%     2   path\file2.wav
%%     ...
%%     N  path\fileN.wav
%%     Q\R   1       2      ...      N
%%     1     0.000  10.234  ...   123.32
%%     2    10.234   0.000  ...    23.45
%%     .     ...     ...    0.000   ...
%%     N     4.1234  6.345  ...     0.0
%%   <end file, exclude this line>
%%
%%   delimiter: tabulator space, number format: float

-- elias 27 May 2007