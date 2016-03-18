
#include <vector>
#include <map>
#include <iostream>
#include <iomanip>
#include <chrono>
#include <locale>
#include <cassert>
#include <unzip.h>
#include <sstream>

#include "n8igall.h"
#include "utils/DWD_CDC.h"

using namespace std;
using namespace date;

namespace n8igall
{
	struct Sample : public ISample
	{
	protected:
		u32 numDimensions;
		double* values;

		friend struct GeoClimateCorrelator;

	public:
		Sample(u32 numDimensions)
			: numDimensions(numDimensions)
			, values(new double[numDimensions])
		{
		}

		virtual ~Sample() 
		{
			delete[] values;
		};

		virtual void SetValue(double value, u32 dimension) override
		{
			assert(dimension < numDimensions);
			values[dimension] = value;
		}
	};

	ISample* ISample::Create(u32 numDimensions)
	{
		return new Sample(numDimensions);
	}

	struct GeoTimeSeries : public IGeoTimeSeries
	{
	protected:
		double longitude;
		double latitude;
		u32 numDimensions;

		map<day_point, const Sample*> timeSeries;

		friend struct GeoClimateCorrelator;

	public:

		GeoTimeSeries(double longitude, double latitude, u32 numDimensions)
			: longitude(longitude)
			, latitude(latitude)
			, numDimensions(numDimensions)
		{

		}

		virtual ~GeoTimeSeries() 
		{
			for (auto sample : timeSeries)
			{
				delete sample.second;
			}
		};

		virtual void AddTimeSample(dayPoint dayPoint, const ISample* sample) override
		{
			timeSeries[dayPoint] = dynamic_cast<const Sample*>(sample);
		}
	};


	IGeoTimeSeries* IGeoTimeSeries::Create(double longitude, double latitude, u32 numDimensions)
	{
		return new GeoTimeSeries(longitude, latitude, numDimensions);
	}


	struct GeoClimateCorrelator : public IGeoClimateCorrelator
	{
	private:
		DWD_CDC dwd_CDC;

		const DWD_CDC::Station& FindNearestStation(double lon, double lat) const
		{
			const auto& nearestStation = dwd_CDC.GetStation(0);

			for (size s = 0; s < dwd_CDC.GetNumStations(); s++)
			{
				const auto& station = dwd_CDC.GetStation(s);

				// TODO: check distance
			}

			return nearestStation;
		}

	public:
		GeoClimateCorrelator()
		{
		}

		virtual IGeoTimeSeriesClimateCorrelation* DeriveCorrelation(const IGeoTimeSeries* geoTimeSeries) const override
		{
			const auto* gts = (const GeoTimeSeries*)geoTimeSeries;

			const auto& station = FindNearestStation(gts->longitude, gts->latitude);

			const u32 numDimensions = gts->numDimensions;

			for (const auto& timeSample : gts->timeSeries)
			{
				const auto date = timeSample.first;
				const auto sample = timeSample.second;

				// TODO: collect all needed samples around date and feed them into FANN
				auto stationSample = station.samples.find(date);
				if (stationSample != station.samples.end())
				{
					//sample->values
				}
			}

			return NULL;
		}

		virtual ~GeoClimateCorrelator() {}
	};


	IGeoClimateCorrelator* IGeoClimateCorrelator::Create()
	{
		return new GeoClimateCorrelator();
	}
}
