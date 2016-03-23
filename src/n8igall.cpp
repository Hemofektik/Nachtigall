
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
#include <algorithm>
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

		map<dayPoint, const Sample*> timeSeries;

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

		void FillFANNTrainingData(const GeoTimeSeries* gts, dayPoint startInclusive, dayPoint endExclusive, const DWD_CDC::Station& station, FANN::training_data& trainingData) const
		{
			const u32 numInputParams = 9;
			const u32 numInputDays = 7;
			const s32 numInputDayOffset = -4;
			const u32 numInputNodes = numInputDays * numInputParams;
			const u32 numOutputNodes = gts->numDimensions;
			const double MissingValue = -999.0;

			vector<double*> input;
			vector<double*> output;
			for (dayPoint day = startInclusive; day < endExclusive; day += days(1))
			{
				double* inputData = NULL;		// weather data of surrounding days for "day"
				const Sample* sample = NULL;	// output sample for "day"

				auto isDayValid = [&]
				{
					const auto sampleIt = gts->timeSeries.find(day);
					if (sampleIt == gts->timeSeries.end()) return false;

					inputData = new double[numInputNodes];
					double* i = inputData;

					for (u32 d = 0; d < numInputDays; d++)
					{
						dayPoint inputDay = day + days(numInputDayOffset + d);

						const auto stationSampleIt = station.samples.find(inputDay);
						if (stationSampleIt == station.samples.end()) return false;

						const auto& stationSample = stationSampleIt->second;

						i[0] = stationSample.BEDECKUNGSGRAD;
						i[1] = stationSample.SONNENSCHEINDAUER;
						i[2] = stationSample.LUFTTEMPERATUR_MAXIMUM;
						i[3] = stationSample.LUFTTEMPERATUR_MINIMUM;
						i[4] = stationSample.NIEDERSCHLAGSHOEHE;
						i[5] = stationSample.SCHNEEHOEHE;
						i[6] = stationSample.REL_FEUCHTE;
						i[7] = stationSample.WINDGESCHWINDIGKEIT;
						i[8] = stationSample.WINDSPITZE_MAXIMUM;
						i += numInputParams;
					}

					// check if any of the inputData values is missing
					for (u32 n = 0; n < numInputNodes; n++)
					{
						if (inputData[n] == MissingValue) return false;
					}

					sample = sampleIt->second;
					return true;
				};

				if (!isDayValid())
				{
					delete[] inputData;
					continue;
				}

				input.push_back(inputData);
				output.push_back(sample->values);
			}

			trainingData.set_train_data((unsigned int)input.size(), numInputNodes, input.data(), numOutputNodes, output.data());

			for (auto i : input)
			{
				delete[] i;
			}
		}

		void TrainAndTestFANN(FANN::training_data& trainingData, FANN::training_data& testData) const
		{
			cout << "num train datasets: " << trainingData.length_train_data() << endl;

			const unsigned int numInputNodes = trainingData.num_input_train_data();
			const unsigned int numOutputNodes = trainingData.num_output_train_data();
			const unsigned int numHiddenNodes = min(trainingData.num_input_train_data() / 2, numOutputNodes * 100);

			FANN::neural_net nn(FANN_NETTYPE_LAYER, 3, numInputNodes, numHiddenNodes, numOutputNodes);

			cout << "initial outputs: " << nn.get_num_output() << endl;

			const float desired_error = 0.001f;
			nn.train_on_data(trainingData, 50, 1, desired_error);

			unsigned int newlayers[3];
			nn.get_layer_array(newlayers);

			cout << "final numlayers: " << nn.get_num_layers() << endl;
			cout << "num neurons in hidden layer: " << newlayers[1] << endl;
			cout << "final outputs: " << nn.get_num_output() << endl;

			nn.reset_MSE();
			float mse = nn.test_data(testData);

			cout << "mse: " << mse << endl;
			cout << "me: " << sqrt(mse) << endl;
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

			dayPoint trainTestDataSplitDay = day_point(year(2012) / month(12) / day(31)); // TODO: this should come from outside (libconfig?)

			FANN::training_data trainingData;
			FillFANNTrainingData(gts, gts->timeSeries.begin()->first, trainTestDataSplitDay, station, trainingData);

			FANN::training_data testData;
			FillFANNTrainingData(gts, trainTestDataSplitDay, gts->timeSeries.rbegin()->first + days(1), station, testData);

			TrainAndTestFANN(trainingData, testData);

			return NULL;
		}

		virtual ~GeoClimateCorrelator() {}
	};


	IGeoClimateCorrelator* IGeoClimateCorrelator::Create()
	{
		return new GeoClimateCorrelator();
	}
}
