
#include <vector>
#include <map>
#include <iostream>
#include <iomanip>
#include <chrono>
#include <locale>
#include <cassert>
#include <unzip.h>
#include <sstream>
#include <regex>
#include <experimental/filesystem>

#include "DWD_CDC.h"

using namespace std::experimental::filesystem::v1;
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
		const path CDCSourceDir("../CDC/");

		for (directory_iterator di(CDCSourceDir); di != end(di); di++)
		{
			const auto& entity = *di;
			const auto extension = entity.path().extension();
			if (is_regular_file(entity.status()))
			{
				//string filepath = entity.path().string();
				string filename = entity.path().filename().string();

				// example zip file name: tageswerte_00044_19710301_20151231_hist.zip
				std::regex regex("tageswerte_(\\d\\d\\d\\d\\d)_(\\d\\d\\d\\d\\d\\d\\d\\d)_(\\d\\d\\d\\d\\d\\d\\d\\d)_hist\\.zip");
				std::smatch matches;

				if (!std::regex_match(filename, matches, regex)) continue;

				std::string stationId = matches[1].str();
				std::string startDate = matches[2].str();
				std::string endDate = matches[3].str();

				// example csv file name 1: produkt_klima_Tageswerte_19710301_20151231_00044.txt
				// example csv file name 2: Stationsmetadaten_klima_stationen_02303_19710301_20151231.txt

				// TODO: read csv files
			}
		}

		retrievestringfromarchive("produkt_klima_Tageswerte_19710301_20151231_00044.txt");
	}

	DWD_CDC::~DWD_CDC()
	{

	}
}