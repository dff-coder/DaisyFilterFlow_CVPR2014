#ifndef COMMON_DATASTRUCTURE_HEADER
#define COMMON_DATASTRUCTURE_HEADER

//#include "Tickcount_Header.h"

#include <vector>
using std::vector;
#include <set>
using std::set;
#include <iterator>
using std::iterator;

class DisjointedSets
{
public:
	vector<int> par;
	int labelNum;
	DisjointedSets();
	DisjointedSets(int num);
	~DisjointedSets();
	void ReserveSpace(int num);
	int NewSet();
	int UnionSets(int setA, int setB);
	int FindSetRoot(int set);
};

class GraphStructure
{
public:
	vector<set<int>> adjList;
	int vertexNum;
	GraphStructure();
	GraphStructure(int num);
	~GraphStructure();
	void ReserveSpace(int num);
	void SetVertexNum(int vNum);
	void AddEdge(int s, int e);

	void DeleteEdge(int s, int e);
	void DeleteAllEdge(int s);
};

#endif