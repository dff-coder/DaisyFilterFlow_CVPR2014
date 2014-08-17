#ifndef __SCAN_FILE_HEADER
#define __SCAN_FILE_HEADER

/// ENV:
/// Win32 API
/// REM:
/// GUI_GetFileName is to load a file using GUI dialog(Win32 API)
/// Scan**** is to scan specific format file in the folder and subfolders
#include <Windows.h>
#include <vector>
#include <cstdio>
#include <cstring>
#include <string>
using std::string;
using std::vector;

class ScanFile 
{
public:
	bool scanSubFolder;
	// using formatExt.pushback(".") to scan all files [with an extension name, i.e. has a '.' in its name]
	int ScanSpecificFormatFiles(string &location, vector<string> &formatExt);
	int ScanJPGs(string &location);
	int ScanBMPs(string &location);

	std::string GetFile(int id);
	std::string GetJPG(int id);
	std::string GetBMP(int id);

	ScanFile() : scanSubFolder(false) {};

public:
	// Use GUI interface to open a file
	static void GUI_GetFileName(string &filename);

protected:
private:
	vector<string> localFiles;
	bool FindFileInFolder(const char *chFolderPath, const char *chFilter, bool bFindSubFolder, vector<string> &formatExt);

};

bool ScanFile::FindFileInFolder(const char *chFolderPath, const char *chFilter, bool bFindSubFolder, vector<string> &formatExt)
{
	int nPathLen = strlen(chFolderPath) + MAX_PATH;
	char *pChPath = new char[nPathLen];
	sprintf_s(pChPath, nPathLen, "%s\\%s", chFolderPath, chFilter);
	WIN32_FIND_DATAA fileFindData;

	HANDLE hFind = FindFirstFileA(pChPath, &fileFindData);    
	do
	{
		if (fileFindData.cFileName[0] == '.')
		{
			continue;           
		} 
		sprintf_s(pChPath, nPathLen, "%s\\%s", chFolderPath, fileFindData.cFileName);   
		//printf("in: %s\n", pChPath);
		if (bFindSubFolder && (fileFindData.dwFileAttributes & FILE_ATTRIBUTE_DIRECTORY))
		{
			FindFileInFolder(pChPath, chFilter, bFindSubFolder, formatExt);
		}
		else
		{
			string str = pChPath;
			for (int i=0; i<formatExt.size(); ++i)
			{
				if (str.rfind(formatExt[i]) != string::npos) 
					localFiles.push_back(str);
			}
		}
	}
	while (FindNextFileA(hFind, &fileFindData)); 
	FindClose(hFind);            
	delete pChPath;
	return true;
}

// using formatExt.pushback(".") to scan all files [with an extension name, i.e. has a '.' in its name]
int ScanFile::ScanSpecificFormatFiles(string &location, vector<string> &formatExt)
{
	localFiles.clear();
	FindFileInFolder(location.c_str(), "*.*", scanSubFolder, formatExt);
	//	for (int i=0; i<localJPGFiles.size(); ++i) cout << localJPGFiles[i] << endl;
	return localFiles.size();
}

int ScanFile::ScanJPGs(string &location)
{
	vector<string> fe;
	fe.push_back(".jpg");
	return ScanSpecificFormatFiles(location, fe);
}

int ScanFile::ScanBMPs(string &location)
{
	vector<string> fe;
	fe.push_back(".bmp");
	return ScanSpecificFormatFiles(location, fe);
}

std::string ScanFile::GetFile(int id)
{
	return localFiles[id];
}

std::string ScanFile::GetJPG(int id)
{
	return GetFile(id);
}

std::string ScanFile::GetBMP(int id)
{
	return GetFile(id);
}

void ScanFile::GUI_GetFileName(string &filename)
{
	OPENFILENAME ofn;
#ifdef UNICODE
	wchar_t szFileName[MAX_PATH];
#else
	char szFileName[MAX_PATH];
#endif

	ZeroMemory(&ofn, sizeof(ofn));

	szFileName[0] = '\0';

	ofn.lStructSize = sizeof(ofn);
	ofn.hwndOwner = NULL;
	ofn.lpstrFilter = TEXT("All Files (*.*)\0*.*\0");

	ofn.lpstrFile = szFileName;
#ifdef UNICODE
	ofn.lpstrDefExt = L"";
#else
	ofn.lpstrDefExt = "";
#endif
	ofn.nMaxFile = MAX_PATH;
	ofn.Flags = OFN_EXPLORER | OFN_FILEMUSTEXIST | OFN_HIDEREADONLY;
	

	if (GetOpenFileName(&ofn))
	{
#ifdef UNICODE
		char tempstr[MAX_PATH];
		//wcout << szFileName << endl;

		//wchar_t to char, that's useful
		size_t convertedChars = 0;
		wcstombs_s(&convertedChars, tempstr, wcslen(szFileName) + 1, szFileName, _TRUNCATE);

		//cout << tempstr << endl;

		filename.assign(tempstr);
#else
		filename.assign(szFileName);
#endif
	}
	else 
	{
		puts("Error GetOpenFileName");
		exit(-1);
	}
}

/// end of file
#endif