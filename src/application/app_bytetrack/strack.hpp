#ifndef _STRACK_HPP_
#define _STRACK_HPP_

#include <opencv2/opencv.hpp>
#include "kalman_filter.hpp"

namespace ByteTrack
{
	using namespace cv;
	using namespace std;

	enum TrackState
	{
		New = 0,
		Tracked,
		Lost,
		Removed
	};

	class STrack
	{
	public:
		STrack(vector<float> tlwh_, float score, int class_label);
		~STrack();

		vector<float> static tlbr_to_tlwh(vector<float> &tlbr);
		void static multi_predict(vector<STrack *> &stracks, KalmanFilter &kalman_filter);
		void static_tlwh();
		void static_tlbr();
		vector<float> tlwh_to_xyah(vector<float> tlwh_tmp);
		vector<float> to_xyah();
		void mark_lost();
		void mark_removed();
		int next_id();
		int end_frame();

		void activate(KalmanFilter &kalman_filter, int frame_id);
		void re_activate(STrack &new_track, int frame_id, bool new_id = false);
		void update(STrack &new_track, int frame_id);

	public:
		bool is_activated;
		int track_id;
		int state;

		vector<float> _tlwh;
		vector<float> tlwh;
		vector<float> tlbr;
		int frame_id;
		int tracklet_len;
		int start_frame;

		KAL_MEAN mean;
		KAL_COVA covariance;
		float score;
		int class_label;

	private:
		KalmanFilter kalman_filter;
	};
}

#endif // _STRACK_HPP_