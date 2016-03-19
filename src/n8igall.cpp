
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

		void FillFANNTrainingData(const GeoTimeSeries* gts, dayPoint start, dayPoint end, const DWD_CDC::Station& station, FANN::training_data& trainingData) const
		{
			/*
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


			const uint32_t numInputNodes = input[0].width * input[0].height;
			const uint32_t numOutputNodes = output[0].width * output[0].height;

			double** inputTrainData = new double*[input.size()];
			double** outputTrainData = new double*[output.size()];
			for (size_t n = 0; n < input.size(); n++)
			{
				inputTrainData[n] = input[n].data;
				outputTrainData[n] = output[n].data;
			}
			trainingData.set_train_data((int)input.size(), numInputNodes, inputTrainData, numOutputNodes, outputTrainData);
			delete[] inputTrainData;
			delete[] outputTrainData;*/
		}

		void TrainAndTestFANN(FANN::training_data& trainingData, FANN::training_data& testData) const
		{
			cout << "num train datasets: " << trainingData.length_train_data() << endl;

			const unsigned int numInputNodes = trainingData.num_input_train_data();
			const unsigned int numOutputNodes = trainingData.num_output_train_data();
			const unsigned int numHiddenNodes = numOutputNodes * 100;

			FANN::neural_net nn(FANN_NETTYPE_LAYER, 3, numInputNodes, numOutputNodes, numHiddenNodes);

			cout << "initial outputs: " << nn.get_num_output() << endl;

			const float desired_error = 0.001f;
			nn.train_on_data(trainingData, 5000, 1, desired_error);

			unsigned int newlayers[3];
			nn.get_layer_array(newlayers);

			cout << "final numlayers: " << nn.get_num_layers() << endl;
			cout << "num neurons in hidden layer: " << newlayers[1] << endl;
			cout << "final outputs: " << nn.get_num_output() << endl;

			nn.reset_MSE();
			float mse = nn.test_data(testData);

			cout << "mse: " << mse << endl;
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
			FillFANNTrainingData(gts, trainTestDataSplitDay, gts->timeSeries.rbegin()->first, station, testData);

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
