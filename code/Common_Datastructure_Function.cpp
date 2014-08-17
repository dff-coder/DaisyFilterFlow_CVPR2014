#include "Common_Datastructure_Header.h"

#pragma region DisjointedSets_Part
DisjointedSets::DisjointedSets()
{
	labelNum = 0;
	par.clear();
}

DisjointedSets::DisjointedSets(int num)
{
	labelNum = 0;
	par.clear();
	ReserveSpace(num);
}

DisjointedSets::~DisjointedSets()
{
	// null
}

void DisjointedSets::ReserveSpace(int num)
{
	par.reserve(num);
}

int DisjointedSets::NewSet()
{
	par.push_back(labelNum);
	return labelNum++;
}

int DisjointedSets::FindSetRoot(int set)
{
	int p = set;

	//while (p != par[p])
	//{
	//	p = par[p];
	//}
	//return p;

	// recursive way using path compression
	if (p != par[p])
	{
		return (par[p] = FindSetRoot(par[p]));
	}
	else return p;
}

int DisjointedSets::UnionSets(int setA, int setB)
{
	// simple way
	int pA = FindSetRoot(setA);
	int pB = FindSetRoot(setB);
	par[pA] = pB;
	return pB;
}

#pragma endregion

#pragma region GraphStructure_Part

GraphStructure::GraphStructure()
{
	vertexNum = 0;
	adjList.clear();
}

GraphStructure::GraphStructure(int num)
{
	vertexNum = 0;
	adjList.clear();
	ReserveSpace(num);
}

GraphStructure::~GraphStructure()
{

}

void GraphStructure::ReserveSpace(int num)
{
	adjList.reserve(num);
}

void GraphStructure::SetVertexNum(int vNum)
{
	vertexNum = vNum;
	adjList.resize(vertexNum);
}

void GraphStructure::AddEdge(int s, int e)
{
	adjList[s].insert(e);
}

void GraphStructure::DeleteEdge(int s, int e)
{
	adjList[s].erase(adjList[s].find(e));
}

void GraphStructure::DeleteAllEdge(int s)
{
	adjList[s].clear();
}

#pragma endregion