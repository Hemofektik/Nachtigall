
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

#pragma warning(push)
#pragma warning(disable:4996)
#pragma warning(disable:4244)
#pragma warning(disable:4267)
#include <csv.h>
#pragma warning(pop)

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

			if (result.back() == 26) // remove substitute character
			{
				result.pop_back();
			}

			// fix broken meta files which have one extra empty column in header but missing in data
			const auto location = result.find("Stationsname;");
			if (location != string::npos)
			{
				result[location + 12] = ' ';
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

				// parse daily values
				{
					stringstream dayValuesCSV;
					dayValuesCSV << dayValuesCSVString;

					io::CSVReader<17, io::trim_chars<' ', '\t'>, io::no_quote_escape<';'>> in("DayValuesCSV", dayValuesCSV);

					in.read_header(io::ignore_extra_column, "STATIONS_ID", "MESS_DATUM", "QUALITAETS_NIVEAU", "LUFTTEMPERATUR", "DAMPFDRUCK", "BEDECKUNGSGRAD", "LUFTDRUCK_STATIONSHOEHE", "REL_FEUCHTE", "WINDGESCHWINDIGKEIT", "LUFTTEMPERATUR_MAXIMUM", "LUFTTEMPERATUR_MINIMUM", "LUFTTEMP_AM_ERDB_MINIMUM", "WINDSPITZE_MAXIMUM", "NIEDERSCHLAGSHOEHE", "NIEDERSCHLAGSHOEHE_IND", "SONNENSCHEINDAUER", "SCHNEEHOEHE");

					int STATIONS_ID;
					u64 MESS_DATUM;
					int QUALITAETS_NIVEAU;
					double LUFTTEMPERATUR, DAMPFDRUCK, BEDECKUNGSGRAD, LUFTDRUCK_STATIONSHOEHE, REL_FEUCHTE, WINDGESCHWINDIGKEIT, LUFTTEMPERATUR_MAXIMUM, LUFTTEMPERATUR_MINIMUM, LUFTTEMP_AM_ERDB_MINIMUM, WINDSPITZE_MAXIMUM, NIEDERSCHLAGSHOEHE, NIEDERSCHLAGSHOEHE_IND, SONNENSCHEINDAUER, SCHNEEHOEHE;
					while (in.read_row(STATIONS_ID, MESS_DATUM, QUALITAETS_NIVEAU, LUFTTEMPERATUR, DAMPFDRUCK, BEDECKUNGSGRAD, LUFTDRUCK_STATIONSHOEHE, REL_FEUCHTE, WINDGESCHWINDIGKEIT, LUFTTEMPERATUR_MAXIMUM, LUFTTEMPERATUR_MINIMUM, LUFTTEMP_AM_ERDB_MINIMUM, WINDSPITZE_MAXIMUM, NIEDERSCHLAGSHOEHE, NIEDERSCHLAGSHOEHE_IND, SONNENSCHEINDAUER, SCHNEEHOEHE))
					{
					}
				}

				// parse meta data
				{
					stringstream stationMetaDataCSV;
					stationMetaDataCSV << stationMetaDataCSVString;

					io::CSVReader<7, io::trim_chars<' ', '\t'>, io::no_quote_escape<';'>> in("stationMetaDataCSV", stationMetaDataCSV);

					in.read_header(io::ignore_extra_column, "Stations_id", "Stationshoehe", "Geogr.Breite", "Geogr.Laenge", "von_datum", "bis_datum", "Stationsname");

					int Stations_id, Stationshoehe;
					double Geogr_Breite, Geogr_Laenge;
					u64 von_datum, bis_datum;
					const char* Stationsname;
					while (in.read_row(Stations_id, Stationshoehe, Geogr_Breite, Geogr_Laenge, von_datum, bis_datum, Stationsname))
					{
					}
				}
			}
		}
	}

	DWD_CDC::~DWD_CDC()
	{

	}
}