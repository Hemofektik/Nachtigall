#pragma once

#include "../n8igallcore.h"

namespace n8igall
{
	// Deutsche Wetter Dienst - Climate Data Center
	// facilitates access to historic data archive of DWD
	class DWD_CDC
	{
	public:

		struct DaySample
		{
			int QUALITAETS_NIVEAU;
			double LUFTTEMPERATUR, DAMPFDRUCK, BEDECKUNGSGRAD, LUFTDRUCK_STATIONSHOEHE, REL_FEUCHTE, WINDGESCHWINDIGKEIT, LUFTTEMPERATUR_MAXIMUM, LUFTTEMPERATUR_MINIMUM, LUFTTEMP_AM_ERDB_MINIMUM, WINDSPITZE_MAXIMUM, NIEDERSCHLAGSHOEHE, NIEDERSCHLAGSHOEHE_IND, SONNENSCHEINDAUER, SCHNEEHOEHE;
		};

		struct Station
		{
			int Stations_id;
			int Stationshoehe;
			double Geogr_Breite;
			double Geogr_Laenge;
			string Stationsname;

			std::map<dayPoint, DaySample> samples;
		};

		DWD_CDC();
		DWD_CDC(const DWD_CDC&) = delete;
		~DWD_CDC();

		size GetNumStations() const;
		const Station& GetStation(size index) const;

	private:
		Station* stations;
		size numStations;
	};
}
