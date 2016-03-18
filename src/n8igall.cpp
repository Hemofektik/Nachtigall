
#include <vector>
#include <map>
#include <iostream>
#include <iomanip>
#include <chrono>
#include <locale>
#include <limits>
#include <cassert>
#include <unzip.h>
#include <sstream>
#include <doublefann.h>
#include <fann_cpp.h>
#include <proj_api.h>

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
			projCtx ctx = pj_ctx_alloc();

			// create Azimuthal Equidistant projection where requested lon and lat is at its center (0,0) to remove any projection induced distortions
			string aeqdProjStr = "+proj=aeqd +lon_0=" + to_string(lon) + " +lat_0=" + to_string(lat) + "+ellps=WGS84 +datum=WGS84 +unit=m +no_defs";
			projPJ aeqd = pj_init_plus_ctx(ctx, aeqdProjStr.c_str());
			projPJ epsg4326 = pj_init_plus_ctx(ctx, "+proj=longlat +ellps=WGS84 +datum=WGS84 +no_defs");

			double minSqrDistance = numeric_limits<double>::max();
			size nearestStationIndex = string::npos;

			for (size s = 0; s < dwd_CDC.GetNumStations(); s++)
			{
				auto& station = dwd_CDC.GetStation(s);

				double X = station.Geogr_Laenge * DEG_TO_RAD;
				double Y = station.Geogr_Breite * DEG_TO_RAD;
				pj_transform(epsg4326, aeqd, 1, 1, &X, &Y, NULL);
				const double sqrDistance = X * X + Y * Y;

				if (minSqrDistance > sqrDistance)
				{
					minSqrDistance = sqrDistance;
					nearestStationIndex = s;
				}
			}

			pj_free(aeqd);
			pj_free(epsg4326);
			pj_ctx_free(ctx);

			cout << " nearest station " << nearestStationIndex << " " << dwd_CDC.GetStation(nearestStationIndex).Stationsname << endl;
			cout << " distance: " << sqrt(minSqrDistance) << " m (id: " << dwd_CDC.GetStation(nearestStationIndex).Stations_id << ")" << endl;

			return dwd_CDC.GetStation(nearestStationIndex);
		}

	public:
		GeoClimateCorrelator()
			: dwd_CDC("../../test/data/") // TODO: set this by using libconfig
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
