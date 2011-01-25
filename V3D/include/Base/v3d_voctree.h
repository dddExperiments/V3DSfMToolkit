// -*- C++ -*-
#ifndef V3D_VOCTREE_H
#define V3D_VOCTREE_H

#include "Base/v3d_serialization.h"

#include <algorithm>
#include <vector>
#include <list>
#include <set>
#include <map>
#include <fstream>
#include <iostream>
#include <cmath>

namespace V3D
{

   template <int Depth, int BranchFactor, int DescLen = 128>
   struct VocabularyTree
   {
         VocabularyTree()
            : _levelStarts(Depth+1)
         {
            int treeSize = 0;
            int len = 1;
            for (int d = 0; d < Depth; ++d)
            {
               _levelStarts[d] = treeSize;
               len *= BranchFactor;

               treeSize += len;
            }
            _levelStarts[Depth] = treeSize;
            _tree.resize(treeSize * DescLen);
         } // end VocabularyTree()

         bool loadTreeFromFile(char const * filename)
         {
            using namespace std;

            //cout << "_tree.size() = " << _tree.size() << endl;

            ifstream is(filename, ios::binary);
            if (!is)
            {
               cerr << "Could not open " << filename << endl;
               return false;
            }
            is.read((char *)&_tree[0], sizeof(float) * _tree.size());
            return true;
         } // end loadTreeFromFile()

         int visualWordCount() const { return this->leafCount(); }

         int getVisualWord(float const * key) const
         {
            return this->findBestLeafIndex(key);
         }

         int leafCount() const { return _levelStarts[Depth] - _levelStarts[Depth-1]; }

         int findBestLeafIndex(float const * key) const
         {
            using namespace std;

            int leafId = 0;
            int nextLeafId = 0;

            for (int d = 0; d < Depth; ++d)
            {
               float const * nodes = &_tree[nextLeafId * DescLen];
               int const k = this->findBestNodeIndex(key, nodes);

               leafId = nextLeafId + k;
               nextLeafId = (leafId + 1) * BranchFactor;
            }
            return leafId - _levelStarts[Depth-1];
         } // end findBestLeafIndex()

      protected:
         static int findBestNodeIndex(float const * key, float const * nodes)
         {
            int bestNode = -1;
            float maxScore = -1e30f;
            for (int k = 0; k < BranchFactor; ++k)
            {
               float const * curNode = nodes + k*DescLen;

               float score = 0;
               for (int i = 0; i < DescLen; ++i) score += key[i] * curNode[i];
               if (score > maxScore)
               {
                  bestNode = k;
                  maxScore = score;
               }
            } // end for (k)
            return bestNode;
         } // end findBestNodeIndex()

         std::vector<float> _tree;
         std::vector<int>   _levelStarts;
   }; // end struct VocabularyTree

   struct LeafDocuments
   {
         struct entries_type : public std::list<std::pair<int, int> >
         {
               template <typename Archive> void serialize(Archive& ar)
               {
                  unsigned int sz = this->size();
                  ar & sz;

                  SerializationScope<Archive> s(ar);

                  int fst, snd;
				  fst = snd = 0;

                  if (ar.isLoading())
                  {
                     this->clear();
                     for (unsigned i = 0; i < sz; ++i) ar & fst & snd;
                     this->push_back(std::make_pair(fst, snd));
                  }
                  else
                  {
                     for (typename std::list<std::pair<int, int> >::iterator p = this->begin(); p != this->end(); ++p)
                     {
                        fst = p->first;
                        snd = p->second;
                        ar & fst & snd;
                     }
                  } // end if
               }

               V3D_DEFINE_LOAD_SAVE(LeafDocuments::entries_type);
         }; // end struct entries_type

         LeafDocuments(int nLeaves)
            : _nDocuments(0), _invertedFiles(nLeaves)
         { }

         int insertDocument(std::vector<int> const& visualWords)
         {
            using namespace std;

            int const documentId = _nDocuments;
            ++_nDocuments;

            map<int, int> histogram;
            this->getVisualWordHistogram(visualWords, histogram);

            for (map<int, int>::const_iterator p = histogram.begin(); p != histogram.end(); ++p)
            {
               int const leaf  = (*p).first;
               int const count = (*p).second;
               _invertedFiles[leaf].push_back(make_pair(documentId, count));
            }

            _nDocumentFeatures.push_back(visualWords.size());
            _nDocumentLeaves.push_back(histogram.size());
            _rcpNDocumentFeatures.push_back(1.0f / visualWords.size());
            return documentId;
         }

         void getDocumentHistogram(std::vector<int> const& visualWords, std::vector<int>& histogram) const
         {
            using namespace std;

            histogram.resize(_nDocuments, 0);

            for (vector<int>::const_iterator q = visualWords.begin(); q != visualWords.end(); ++q)
            {
               entries_type const& invertedFile = _invertedFiles[*q];
               for (entries_type::const_iterator p = invertedFile.begin(); p != invertedFile.end(); ++p)
                  histogram[(*p).first] += 1;
            } // end for (i)
         } // end getDocumentHistogram()

         void getVisualWordHistogram(std::vector<int> const& visualWords, std::map<int, int>& histogram) const
         {
            using namespace std;

            histogram.clear();

            for (size_t i = 0; i < visualWords.size(); ++i)
            {
               int const leaf = visualWords[i];

               map<int, int>::iterator p = histogram.find(leaf);
               if (p == histogram.end())
                  histogram.insert(make_pair(leaf, 1));
               else
                  ++(*p).second;
            } // end for (i)
         } // end getVisualWordHistogram()

         void compute_L1_Scores(std::vector<int> const& visualWords,
                                std::vector<std::pair<float, int> >& scores, int nBestDocuments = -1)
         {
            using namespace std;

            map<int, int> histogram;
            this->getVisualWordHistogram(visualWords, histogram);

            float const rcpNVisualWords = 1.0f / visualWords.size();

            // For normalized vectors d and q:
            // sum_i |d_i - q_i| = 2 - sum_{i:d_i>0 && q_i>0}(|d_i - q_i| - |d_i| - |q_i|).
            // See also Nister's vocalulary tree paper.
            vector<float> distancesL1(_nDocuments, 2.0f);
            for (map<int, int>::const_iterator p = histogram.begin(); p != histogram.end(); ++p)
            {
               float const qi = float((*p).second) * rcpNVisualWords;

               int const leafId = (*p).first;
               entries_type const& invertedFile = _invertedFiles[leafId];
               for (entries_type::const_iterator q = invertedFile.begin(); q != invertedFile.end(); ++q)
               {
                  int const docId = (*q).first;
                  float const di = float((*q).second) * _rcpNDocumentFeatures[docId];
                  distancesL1[docId] += fabsf(di - qi);
                  distancesL1[docId] -= di;
                  distancesL1[docId] -= qi;
               }
            } // end for (p)

            scores.resize(_nDocuments);

            for (size_t i = 0; i < _nDocuments; ++i)
            {
               scores[i].first = distancesL1[i];
               scores[i].second = i;
            }

            if (nBestDocuments < 0 || scores.size() <= nBestDocuments)
               std::sort(scores.begin(), scores.end());
            else
               std::partial_sort(scores.begin(), scores.begin() + nBestDocuments, scores.end());
         } // end compute_L1_Scores()

         void computeBinomialScores(double const p_match, std::vector<int> const& visualWords,
                                    std::vector<std::pair<float, int> >& scores, int nBestDocuments = -1)
         {
			// scores vector<pair<float,int>>
			// float is the score
		    // int is the document id

            using namespace std;

            vector<int> histogram;
            this->getDocumentHistogram(visualWords, histogram);

            scores.resize(histogram.size());

            double const normalizer = 1.0 / _invertedFiles.size();

            for (size_t i = 0; i < histogram.size(); ++i)
            {
               double const k = histogram[i];
               double const n = std::min(_nDocumentFeatures[i], int(visualWords.size()));

               //cout << "n = " << n << " k = " << k << endl;

               double const p_random = std::max(_nDocumentFeatures[i], int(visualWords.size())) * normalizer;

               double const log_p0   = log(p_random);
               double const log_1_p0 = log(1.0 - p_random);

               double const p_matching = std::min(1.0, p_random + p_match);
               double const log_p1     = log(p_matching);
               double const log_1_p1   = log(1.0 - p_matching);

               //cout << "p_random = " << p_random << " p_matching = " << p_matching << endl;

               double const z0 = k*log_p0 + (n-k)*log_1_p0;
               double const z1 = k*log_p1 + (n-k)*log_1_p1;

               double const score = z0 - z1;

               scores[i].first = score;
               scores[i].second = i;
            } // end for (i)

            if (nBestDocuments < 0 || scores.size() <= nBestDocuments)
               std::sort(scores.begin(), scores.end());
            else
               std::partial_sort(scores.begin(), scores.begin() + nBestDocuments, scores.end());
         } // end computeBinomialScores()

         template <typename Archive> void serialize(Archive& ar)
         {
            ar & _nDocuments;
            ar & _invertedFiles;
            ar & _nDocumentFeatures;
            ar & _nDocumentLeaves;
            ar & _rcpNDocumentFeatures;
         }

         V3D_DEFINE_LOAD_SAVE(LeafDocuments);

      protected:
         int _nDocuments;

         SerializableVector<entries_type> _invertedFiles;
         SerializableVector<int>          _nDocumentFeatures;
         SerializableVector<int>          _nDocumentLeaves;
         SerializableVector<float>        _rcpNDocumentFeatures;
   }; // end struct LeafDocuments

   V3D_DEFINE_IOSTREAM_OPS(LeafDocuments);

} // end namespace V3D

#endif
