#include <CUDA/v3d_cudamatching.h>
#include <CUDA/v3d_cudamatching.cuh>

#include <cublas.h>
#include <cuda.h>
#include <cuda_runtime.h>

using namespace std;
using namespace V3D;

namespace V3D_CUDA
{

	void
		SIFT_Matching::allocate(int nMaxLeftFeatures, int nMaxRightFeatures, bool useGuidedMatching)
	{
		_useGuidedMatching = useGuidedMatching;

		if ((nMaxLeftFeatures % NTHREADS) != 0)
		{
			if(_warnOnRounding) {
				cerr << "SIFT_Matching::allocate() warning: nMaxLeftFeatures should be a multiple of " << NTHREADS << endl;
				cerr << "Rounding nMaxLeftFeatures down." << endl;
			}
			nMaxLeftFeatures = nMaxLeftFeatures - (nMaxLeftFeatures & (NTHREADS-1));
		}

		if ((nMaxRightFeatures % NTHREADS) != 0)
		{
			if(_warnOnRounding) {
				cerr << "SIFT_Matching::allocate() warning: nMaxRightFeatures should be a multiple of " << NTHREADS << endl;
				cerr << "Rounding nMaxRightFeatures down." << endl;
			}
			nMaxRightFeatures = nMaxRightFeatures - (nMaxRightFeatures & (NTHREADS-1));
		}

		_nMaxLeftFeatures  = nMaxLeftFeatures;
		_nMaxRightFeatures = nMaxRightFeatures;

		cublasAlloc(nMaxLeftFeatures, 128*sizeof(float), (void **)&_d_leftFeatures);
		cublasAlloc(nMaxRightFeatures, 128*sizeof(float), (void **)&_d_rightFeatures);
		cublasAlloc(nMaxLeftFeatures*nMaxRightFeatures, sizeof(float), (void **)&_d_scores);

		if (useGuidedMatching)
		{
			cublasAlloc(2, 2*sizeof(float)*nMaxLeftFeatures, (void **)&_d_leftPositions);
			cublasAlloc(2, 2*sizeof(float)*nMaxRightFeatures, (void **)&_d_rightPositions);
		} // end if (useGuidedMatching)
	}

	void
		SIFT_Matching::deallocate()
	{
		cublasFree(_d_leftFeatures);
		cublasFree(_d_rightFeatures);
		cublasFree(_d_scores);

		if (_useGuidedMatching)
		{
			cublasFree(_d_leftPositions);
			cublasFree(_d_rightPositions);
		}
	}

	void
		SIFT_Matching::setLeftFeatures(int nFeatures, float const * descriptors)
	{
		if ((nFeatures % NTHREADS) != 0)
		{
			if(_warnOnRounding) {
				cerr << "SIFT_Matching::setLeftFeatures() warning: nFeatures should be a multiple of " << NTHREADS << endl;
				cerr << "Rounding nFeatures down." << endl;
			}
			nFeatures = nFeatures - (nFeatures & (NTHREADS-1));
		}

		_nLeftFeatures = std::min(nFeatures, _nMaxLeftFeatures);
		cublasSetMatrix(128, _nLeftFeatures, sizeof(float), descriptors, 128, _d_leftFeatures, 128);
	}

	void
		SIFT_Matching::setRightFeatures(int nFeatures, float const * descriptors)
	{
		if ((nFeatures % NTHREADS) != 0)
		{
			if(_warnOnRounding) {
				cerr << "SIFT_Matching::setRightFeatures() warning: nFeatures should be a multiple of " << NTHREADS << endl;
				cerr << "Rounding nFeatures down." << endl;
			}
			nFeatures = nFeatures - (nFeatures & (NTHREADS-1));
		}

		_nRightFeatures = std::min(nFeatures, _nMaxRightFeatures);;
		cublasSetMatrix(128, _nRightFeatures, sizeof(float), descriptors, 128, _d_rightFeatures, 128);
	}

	void
		SIFT_Matching::findPutativeMatches(std::vector<std::pair<int, int> >& matches,
		float const minScore, float const minRatio)
	{
		matches.clear();
		matches.reserve(std::min(_nLeftFeatures, _nRightFeatures));

		cublasSgemm('T', 'N', _nLeftFeatures, _nRightFeatures, 128, 1.0, _d_leftFeatures, 128, _d_rightFeatures, 128,
			0.0f, _d_scores, _nLeftFeatures);

		vector<int> colMax(_nLeftFeatures, -1);
		vector<int> rowMax(_nRightFeatures, -1);

		if (minRatio < 0)
		{
			computeColumnMaxima(_nRightFeatures, _nLeftFeatures, minScore, _d_scores, &colMax[0]);
			computeRowMaxima(_useReductionPath, _nRightFeatures, _nLeftFeatures, minScore, _d_scores, &rowMax[0]);
		}
		else
		{
			computeColumnMaximaWithRatioTest(_nRightFeatures, _nLeftFeatures, minScore, minRatio, _d_scores, &colMax[0]);
			computeRowMaximaWithRatioTest(_useReductionPath, _nRightFeatures, _nLeftFeatures, minScore, minRatio, _d_scores, &rowMax[0]);
		} // end if

		for (int k = 0; k < _nLeftFeatures; ++k)
		{
			int const l = colMax[k];
			if (l >= 0 && rowMax[l] == k)
				matches.push_back(make_pair(k, l));
		} // end for (k)
	} // end SIFT_Matching::findPutativeMatches()

	void
		SIFT_Matching::runGuidedMatching(V3D::Matrix3x3d const& F, float const distanceThreshold,
		Vector2f const * leftPositions,
		Vector2f const * rightPositions,
		std::vector<std::pair<int, int> >& matches,
		float const minScore, float const minRatio)
	{
		matches.clear();
		matches.reserve(std::min(_nLeftFeatures, _nRightFeatures));

		{
			vector<float> leftPos(2*_nLeftFeatures);
			vector<float> rightPos(2*_nRightFeatures);

			for (int k = 0; k < _nLeftFeatures; ++k)
			{
				leftPos[k]                = leftPositions[k][0];
				leftPos[k+_nLeftFeatures] = leftPositions[k][1];
			}

			for (int k = 0; k < _nRightFeatures; ++k)
			{
				rightPos[k]                 = rightPositions[k][0];
				rightPos[k+_nRightFeatures] = rightPositions[k][1];
			}

			int const leftSize  = 2*sizeof(float)*_nLeftFeatures;
			int const rightSize = 2*sizeof(float)*_nRightFeatures;

			CUDA_SAFE_CALL( cudaMemcpy( _d_leftPositions, &leftPos[0], leftSize, cudaMemcpyHostToDevice) );
			CUDA_SAFE_CALL( cudaMemcpy( _d_rightPositions, &rightPos[0], rightSize, cudaMemcpyHostToDevice) );

			float3 F1, F2, F3;
			F1.x = F[0][0]; F1.y = F[0][1]; F1.z = F[0][2];
			F2.x = F[1][0]; F2.y = F[1][1]; F2.z = F[1][2];
			F3.x = F[2][0]; F3.y = F[2][1]; F3.z = F[2][2];

			eliminateNonEpipolarScores(_d_rightPositions, _d_leftPositions, _d_scores, rightSize, _nRightFeatures, _nLeftFeatures, distanceThreshold, F1, F2, F3); //HA

		} // end scope

		vector<int> colMax(_nLeftFeatures, -1);
		vector<int> rowMax(_nRightFeatures, -1);

		if (minRatio < 0)
		{
			computeColumnMaxima(_nRightFeatures, _nLeftFeatures, minScore, _d_scores, &colMax[0]);
			computeRowMaxima(_useReductionPath, _nRightFeatures, _nLeftFeatures, minScore, _d_scores, &rowMax[0]);
		}
		else
		{
			computeColumnMaximaWithRatioTest(_nRightFeatures, _nLeftFeatures, minScore, minRatio, _d_scores, &colMax[0]);
			computeRowMaximaWithRatioTest(_useReductionPath, _nRightFeatures, _nLeftFeatures, minScore, minRatio, _d_scores, &rowMax[0]);
		} // end if

		for (int k = 0; k < _nLeftFeatures; ++k)
		{
			int const l = colMax[k];
			if (l >= 0 && rowMax[l] == k)
				matches.push_back(make_pair(k, l));
		} // end for (k)

	} // end SIFT_Matching::runGuidedMatching()

	void
		SIFT_Matching::runGuidedHomographyMatching(V3D::Matrix3x3d const& H, float const distanceThreshold,
		Vector2f const * leftPositions,
		Vector2f const * rightPositions,
		std::vector<std::pair<int, int> >& matches,
		float const minScore, float const minRatio)
	{
		matches.clear();
		matches.reserve(std::min(_nLeftFeatures, _nRightFeatures));

		{
			vector<float> leftPos(2*_nLeftFeatures);
			vector<float> rightPos(2*_nRightFeatures);

			for (int k = 0; k < _nLeftFeatures; ++k)
			{
				leftPos[k]                = leftPositions[k][0];
				leftPos[k+_nLeftFeatures] = leftPositions[k][1];
			}

			for (int k = 0; k < _nRightFeatures; ++k)
			{
				rightPos[k]                 = rightPositions[k][0];
				rightPos[k+_nRightFeatures] = rightPositions[k][1];
			}

			int const leftSize  = 2*sizeof(float)*_nLeftFeatures;
			int const rightSize = 2*sizeof(float)*_nRightFeatures;

			CUDA_SAFE_CALL( cudaMemcpy( _d_leftPositions, &leftPos[0], leftSize, cudaMemcpyHostToDevice) );
			CUDA_SAFE_CALL( cudaMemcpy( _d_rightPositions, &rightPos[0], rightSize, cudaMemcpyHostToDevice) );

			float3 H1, H2, H3;
			H1.x = H[0][0]; H1.y = H[0][1]; H1.z = H[0][2];
			H2.x = H[1][0]; H2.y = H[1][1]; H2.z = H[1][2];
			H3.x = H[2][0]; H3.y = H[2][1]; H3.z = H[2][2];

			eliminateNonHomographyScores(_d_rightPositions, _d_leftPositions, _d_scores, rightSize, _nRightFeatures, _nLeftFeatures, distanceThreshold, H1, H2, H3); //HA

		} // end scope

		vector<int> colMax(_nLeftFeatures, -1);
		vector<int> rowMax(_nRightFeatures, -1);

		if (minRatio < 0)
		{
			computeColumnMaxima(_nRightFeatures, _nLeftFeatures, minScore, _d_scores, &colMax[0]);
			computeRowMaxima(_useReductionPath, _nRightFeatures, _nLeftFeatures, minScore, _d_scores, &rowMax[0]);
		}
		else
		{
			computeColumnMaximaWithRatioTest(_nRightFeatures, _nLeftFeatures, minScore, minRatio, _d_scores, &colMax[0]);
			computeRowMaximaWithRatioTest(_useReductionPath, _nRightFeatures, _nLeftFeatures, minScore, minRatio, _d_scores, &rowMax[0]);
		} // end if

		for (int k = 0; k < _nLeftFeatures; ++k)
		{
			int const l = colMax[k];
			if (l >= 0 && rowMax[l] == k)
				matches.push_back(make_pair(k, l));
		} // end for (k)

	} // end SIFT_Matching::runGuidedMatching()

} // end namespace V3D_CUDA

namespace V3D_CUDA
{

	template <int BranchFactor, int Depth>
	VocabularyTreeTraversal<BranchFactor, Depth>::VocabularyTreeTraversal()
		: _nMaxFeatures(0), _d_nodes(0)
	{ }

	template <int BranchFactor, int Depth>
	void
		VocabularyTreeTraversal<BranchFactor, Depth>::allocate(int nMaxFeatures)
	{
		if ((nMaxFeatures % NTHREADS_VOCTREE) != 0)
		{
			cerr << "VocabularyTreeTraversal::allocate() warning: nMaxFeatures should be a multiple of " << NTHREADS_VOCTREE << endl;
			cerr << "Rounding nFeatures down." << endl;
			nMaxFeatures = nMaxFeatures - (nMaxFeatures & (NTHREADS_VOCTREE-1));
		}

		_nMaxFeatures = nMaxFeatures;

		CUDA_SAFE_CALL( cudaMalloc( (void**) &_d_features, nMaxFeatures * 128 * sizeof(float)) );
		CUDA_SAFE_CALL( cudaMalloc( (void**) &_d_leaveIds, nMaxFeatures * sizeof(int)) );
	}

	template <int BranchFactor, int Depth>
	void
		VocabularyTreeTraversal<BranchFactor, Depth>::deallocate()
	{
		CUDA_SAFE_CALL( cudaFree(_d_features) );
		CUDA_SAFE_CALL( cudaFree(_d_leaveIds) );
		if (_d_nodes != 0)
			CUDA_SAFE_CALL( cudaFree(_d_nodes) );
	}

	template <int BranchFactor, int Depth>
	void
		VocabularyTreeTraversal<BranchFactor, Depth>::setNodes(float const * nodes)
	{
		if (_d_nodes != 0)
			CUDA_SAFE_CALL( cudaFree(_d_nodes) );

		int const nNodes = getNodeCount();

		CUDA_SAFE_CALL( cudaMalloc( (void**) &_d_nodes, nNodes * 128 * sizeof(float)) );
		CUDA_SAFE_CALL( cudaMemcpy( _d_nodes, nodes, nNodes * 128 * sizeof(float), cudaMemcpyHostToDevice) );
	}

	template <int BranchFactor, int Depth>
	void
		VocabularyTreeTraversal<BranchFactor, Depth>::traverseTree(int nFeatures, float const * features, int * leaveIds)
	{
		if (nFeatures > _nMaxFeatures)
		{
			cerr << "VocabularyTreeTraversal::traverseTree() warning: nFeatures is larger than nMaxFeatures." << endl;
			cerr << "Reducing nFeatures." << endl;
			nFeatures = _nMaxFeatures;
		}
		
		CUDA_SAFE_CALL( cudaMemcpy( _d_features, features, nFeatures * 128 * sizeof(float), cudaMemcpyHostToDevice) );

		::traverseTree<BranchFactor, Depth>(nFeatures, _d_features, _d_nodes, _d_leaveIds);

		CUDA_SAFE_CALL( cudaMemcpy( leaveIds, _d_leaveIds, nFeatures * sizeof(int), cudaMemcpyDeviceToHost) );
	}

	// Add additional instances you need.
	template struct VocabularyTreeTraversal<10, 5>;
	template struct VocabularyTreeTraversal<50, 3>;
	template struct VocabularyTreeTraversal<300, 2>;

} // end namespace V3D_CUDA