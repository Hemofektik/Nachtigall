
#include <vector>
#include <map>
#include <iostream>
#include <iomanip>
#include <chrono>
#include <locale>
#include <cassert>

#include "n8igall.h"

using namespace std;
using namespace date;

namespace n8igall
{
	struct Sample : public ISample
	{
	private:
		u32 numDimensions;
		double* values;

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

		virtual void SetValue(u32 dimension, double value) override
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
	private:
		double longitude;
		double latitude;
		u32 numDimensions;

		map<day_point, const Sample*> timeSeries;

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

		virtual void AddTimeSample(day_point dayPoint, const ISample* sample) override
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
		virtual ~GeoClimateCorrelator() {};

	};


	IGeoClimateCorrelator* Create()
	{
		return new GeoClimateCorrelator();
	}
}
