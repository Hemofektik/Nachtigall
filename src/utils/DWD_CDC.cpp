
#include <vector>
#include <map>
#include <iostream>
#include <iomanip>
#include <chrono>
#include <locale>
#include <cassert>
#include <unzip.h>
#include <sstream>

#include "DWD_CDC.h"

using namespace std;
using namespace date;

namespace n8igall
{
	std::string retrievestringfromarchive(std::string filename)
	{
		// Open the zip file
		unzFile zipfile = unzOpen("../CDC/tageswerte_00044_19710301_20151231_hist.zip");
		if (zipfile == NULL)
		{
			return "";
		}

		unz_file_info info;

		unzLocateFile(zipfile, filename.c_str(), NULL);
		unzOpenCurrentFile(zipfile);
		unzGetCurrentFileInfo(zipfile, &info, NULL, 0, NULL, 0, NULL, 0);
		u8* rwh = (u8*)malloc(info.uncompressed_size);
		unzReadCurrentFile(zipfile, rwh, info.uncompressed_size);

		std::stringstream tempstream;
		tempstream << ((const char*)rwh);
		free(rwh);

		unzClose(zipfile);

		return tempstream.str();
	}

	DWD_CDC::DWD_CDC()
	{
		retrievestringfromarchive("produkt_klima_Tageswerte_19710301_20151231_00044.txt");	
	}

	DWD_CDC::~DWD_CDC()
	{

	}
}