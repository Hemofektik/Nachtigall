
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
	struct ZipFile
	{
	private:
		unzFile zipfile;

	public:
		ZipFile(const string& filepath)
		{
			zipfile = unzOpen(filepath.c_str());
		}

		~ZipFile()
		{
			unzClose(zipfile);
		}

		bool IsOK() const
		{
			return zipfile != NULL;
		}

		string ReadStringFromFile(const string& filename)
		{
			if (UNZ_OK != unzLocateFile(zipfile, filename.c_str(), NULL))
			{
				return "";
			}

			if (UNZ_OK != unzOpenCurrentFile(zipfile))
			{
				return "";
			}

			unz_file_info info;
			if (UNZ_OK != unzGetCurrentFileInfo(zipfile, &info, NULL, 0, NULL, 0, NULL, 0))
			{
				return "";
			}
			
			string result;
			result.resize(info.uncompressed_size);

			if (info.uncompressed_size != unzReadCurrentFile(zipfile, &result.front(), info.uncompressed_size))
			{
				return "";
			}

			return result;
		}
	};

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
				regex regex("tageswerte_(\\d\\d\\d\\d\\d)_(\\d\\d\\d\\d\\d\\d\\d\\d)_(\\d\\d\\d\\d\\d\\d\\d\\d)_hist\\.zip");
				smatch matches;

				if (!regex_match(filename, matches, regex)) continue;

				string stationId = matches[1].str();
				string startDate = matches[2].str();
				string endDate = matches[3].str();

				ZipFile zipFile(entity.path().string());
				if (!zipFile.IsOK()) continue;

				string dayValuesFilename = "produkt_klima_Tageswerte_" + startDate + "_" + endDate + "_" + stationId + ".txt";
				auto dayValuesCSVString = zipFile.ReadStringFromFile(dayValuesFilename);
				if (dayValuesCSVString.size() == 0) continue;

				string stationMetaDataFilename = "Stationsmetadaten_klima_stationen_" + stationId + "_" + startDate + "_" + endDate + + ".txt";
				auto stationMetaDataCSVString = zipFile.ReadStringFromFile(stationMetaDataFilename);
				if (stationMetaDataCSVString.size() == 0) continue;

				// TODO: parse csv files
			}
		}

		//retrievestringfromarchive("produkt_klima_Tageswerte_19710301_20151231_00044.txt");
	}

	DWD_CDC::~DWD_CDC()
	{

	}
}