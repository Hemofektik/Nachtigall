#pragma once

#include "n8igallcore.h"

namespace n8igall
{
	struct ISample
	{
		virtual ~ISample() {};
		virtual void SetValue(u32 dimension, double value) = 0;

		static ISample* Create(u32 numDimensions = 1);
	};

	struct IGeoTimeSeries
	{
		virtual ~IGeoTimeSeries() {};

		// Adds a sample along with its proper time point. 
		// The sample's ownership is transfered to the GeoTimeSeries.
		virtual void AddTimeSample(date::day_point dayPoint, const ISample* sample) = 0;

		static IGeoTimeSeries* Create(double longitude, double latitude, u32 numDimensions = 1);
	};

	struct IGeoClimateCorrelator
	{
		virtual ~IGeoClimateCorrelator() {};

		static IGeoClimateCorrelator* Create();
	};
}
