#include <dynamic_reconfigure/server.h>
#include <image_transport/camera_publisher.h>
#include <image_transport/image_transport.h>
#include <pcl_ros/point_cloud.h>
#include <sensor_msgs/PointCloud2.h>
#include <velodyne_msgs/VelodynePacket.h>
#include <velodyne_msgs/VelodyneScan.h>
#include <velodyne_puck/VelodynePuckConfig.h>

#include "cloud.h"
#include "constants.h"

namespace velodyne_puck {

using namespace sensor_msgs;
using namespace velodyne_msgs;

class Decoder {
 public:
  explicit Decoder(const ros::NodeHandle& pnh);

  Decoder(const Decoder&) = delete;
  Decoder operator=(const Decoder&) = delete;

  void PacketCb(const VelodynePacketConstPtr& packet_msg);
  void ConfigCb(VelodynePuckConfig& config, int level);

 private:
  /// All of these uses laser index from velodyne which is interleaved

  /// 9.3.1.3 Data Point
  /// A data point is a measurement by one laser channel of a relection of a
  /// laser pulse
  struct DataPoint {
    uint16_t distance;
    uint8_t reflectivity;
  } __attribute__((packed));
  static_assert(sizeof(DataPoint) == 3, "sizeof(DataPoint) != 3");

  /// 9.3.1.1 Firing Sequence
  /// A firing sequence occurs when all the lasers in a sensor are fired. There
  /// are 16 firings per cycle for VLP-16
  struct FiringSequence {
    DataPoint points[kFiringsPerSequence];  // 16
  } __attribute__((packed));
  static_assert(sizeof(FiringSequence) == 48, "sizeof(FiringSequence) != 48");

  /// 9.3.1.4 Azimuth
  /// A two-byte azimuth value (alpha) appears after the flag bytes at the
  /// beginning of each data block
  ///
  /// 9.3.1.5 Data Block
  /// The information from 2 firing sequences of 16 lasers is contained in each
  /// data block. Each packet contains the data from 24 firing sequences in 12
  /// data blocks.
  struct DataBlock {
    uint16_t flag;
    uint16_t azimuth;                              // [0, 35999]
    FiringSequence sequences[kSequencesPerBlock];  // 2
  } __attribute__((packed));
  static_assert(sizeof(DataBlock) == 100, "sizeof(DataBlock) != 100");

  struct Packet {
    DataBlock blocks[kBlocksPerPacket];  // 12
    /// The four-byte time stamp is a 32-bit unsigned integer marking the moment
    /// of the first data point in the first firing sequcne of the first data
    /// block. The time stamp’s value is the number of microseconds elapsed
    /// since the top of the hour.
    uint32_t stamp;
    uint8_t factory[2];
  } __attribute__((packed));
  static_assert(sizeof(Packet) == sizeof(velodyne_msgs::VelodynePacket().data),
                "sizeof(Packet) != 1206");

  void DecodeAndFill(const Packet* const packet_buf, uint64_t time);

 private:
  bool CheckFactoryBytes(const Packet* const packet);
  void Reset();

  // ROS related parameters
  std::string frame_id_;
  ros::NodeHandle pnh_;
  image_transport::ImageTransport it_;
  ros::Subscriber packet_sub_;
  ros::Publisher cloud_pub_;
  image_transport::Publisher intensity_pub_, range_pub_;
  image_transport::CameraPublisher camera_pub_;
  dynamic_reconfigure::Server<VelodynePuckConfig> cfg_server_;
  VelodynePuckConfig config_;

  // cached
  cv::Mat image_;
  std::vector<double> azimuths_;
  std::vector<uint64_t> timestamps_;
  int curr_col_{0};
  std::vector<double> elevations_;
};

Decoder::Decoder(const ros::NodeHandle& pnh)
    : pnh_(pnh), it_(pnh), cfg_server_(pnh) {
  pnh_.param<std::string>("frame_id", frame_id_, "velodyne");
  ROS_INFO("Velodyne frame_id: %s", frame_id_.c_str());
  cfg_server_.setCallback(boost::bind(&Decoder::ConfigCb, this, _1, _2));

  // Pre-compute elevations
  for (int i = 0; i < kFiringsPerSequence; ++i) {
    elevations_.push_back(kMaxElevation - i * kDeltaElevation);
  }
}

bool Decoder::CheckFactoryBytes(const Packet* const packet_buf) {
  // Check return mode and product id, for now just die
  const auto return_mode = packet_buf->factory[0];

  if (!(return_mode == 55 || return_mode == 56)) {
    ROS_ERROR(
        "return mode must be Strongest (55) or Last Return (56), "
        "instead got (%u)",
        return_mode);
    return false;
  }

  const auto product_id = packet_buf->factory[1];
  if (product_id != 34) {
    ROS_ERROR("product id must be VLP-16 or Puck Lite (34), instead got (%u)",
              product_id);
    return false;
  }

  return true;
}

void Decoder::DecodeAndFill(const Packet* const packet_buf, uint64_t time) {
  if (!CheckFactoryBytes(packet_buf)) {
    ros::shutdown();
  }

  // transform
  //            ^ x_l
  //            | -> /
  //            | a /
  //            |  /
  //            | /
  // <----------o
  // y_l

  // For each data block, 12 total
  for (int iblk = 0; iblk < kBlocksPerPacket; ++iblk) {
    const auto& block = packet_buf->blocks[iblk];
    const auto raw_azimuth = block.azimuth;         // nominal azimuth [0,35999]
    const auto azimuth = Raw2Azimuth(raw_azimuth);  // nominal azimuth [0, 2pi)

    ROS_WARN_STREAM_COND(raw_azimuth > kMaxRawAzimuth,
                         "Invalid raw azimuth: " << raw_azimuth);
    ROS_WARN_COND(block.flag != UPPER_BANK, "Invalid block %d", iblk);

    float azimuth_gap{0};
    if (iblk == kBlocksPerPacket - 1) {
      // Last block, 12th
      const auto& prev_block = packet_buf->blocks[iblk - 1];
      const auto prev_azimuth = Raw2Azimuth(prev_block.azimuth);
      azimuth_gap = azimuth - prev_azimuth;
    } else {
      // First 11 blocks
      const auto& next_block = packet_buf->blocks[iblk + 1];
      const auto next_azimuth = Raw2Azimuth(next_block.azimuth);
      azimuth_gap = next_azimuth - azimuth;
    }

    // Adjust for azimuth rollover from 2pi to 0
    if (azimuth_gap < 0) azimuth_gap += kTau;
    const auto half_azimuth_gap = azimuth_gap / 2;

    // for each firing sequence in the data block, 2
    for (int iseq = 0; iseq < kSequencesPerBlock; ++iseq, ++curr_col_) {
      const auto col = iblk * 2 + iseq;
      timestamps_[curr_col_] = time + col * kFiringCycleNs;
      azimuths_[curr_col_] = azimuth + half_azimuth_gap * iseq;

      const auto& seq = block.sequences[iseq];
      // for each laser beam, 16
      for (int lid = 0; lid < kFiringsPerSequence; ++lid) {
        const auto& point = seq.points[lid];
        auto& v = image_.at<cv::Vec3f>(LaserId2Row(lid), curr_col_);
        v[cloud::RANGE] = point.distance * kDistanceResolution;
        v[cloud::INTENSITY] = point.reflectivity;
        const auto offset = lid * kSingleFiringRatio * half_azimuth_gap +
                            iseq * half_azimuth_gap;
        v[cloud::AZIMUTH] = azimuth + offset;
      }
    }
  }
}

void Decoder::PacketCb(const VelodynePacketConstPtr& packet_msg) {
  const auto start = ros::Time::now();

  const auto* packet_buf =
      reinterpret_cast<const Packet*>(&(packet_msg->data[0]));
  DecodeAndFill(packet_buf, packet_msg->stamp.toNSec());

  if (curr_col_ < config_.image_width) {
    return;
  }

  std_msgs::Header header;
  header.frame_id = frame_id_;
  header.stamp.fromNSec(timestamps_.front());

  const ImagePtr image_msg =
      cv_bridge::CvImage(header, "32FC3", image_).toImageMsg();

  // Fill in camera info
  const CameraInfoPtr cinfo_msg(new CameraInfo);
  cinfo_msg->header = header;
  cinfo_msg->height = image_msg->height;
  cinfo_msg->width = image_msg->width;
  cinfo_msg->distortion_model = "VLP16";
  cinfo_msg->K[0] = kFiringCycleNs;  // delta time between two measurements

  // D = [altitude, azimuth]
  cinfo_msg->D = elevations_;
  cinfo_msg->D.insert(cinfo_msg->D.end(), azimuths_.cbegin(), azimuths_.cend());

  // Publish on demand
  if (camera_pub_.getNumSubscribers() > 0) {
    camera_pub_.publish(image_msg, cinfo_msg);
  }

  if (cloud_pub_.getNumSubscribers() > 0) {
    const auto start = ros::Time::now();
    cloud_pub_.publish(cloud::ToCloud(image_msg, *cinfo_msg, false));
    ROS_DEBUG("ToCloud time: %f", (ros::Time::now() - start).toSec());
  }

  if (range_pub_.getNumSubscribers() > 0 ||
      intensity_pub_.getNumSubscribers() > 0) {
    // Publish range and intensity separately
    cv::Mat sep[3];
    cv::split(image_, sep);

    cv::Mat range = sep[cloud::RANGE];
    // should be 2, use 3 for more contrast
    range.convertTo(range, CV_8UC1, 3.0);
    range_pub_.publish(
        cv_bridge::CvImage(header, image_encodings::MONO8, range).toImageMsg());

    // use 300 for more contrast
    cv::Mat intensity = sep[cloud::INTENSITY];
    double a, b;
    cv::minMaxIdx(intensity, &a, &b);
    intensity.convertTo(intensity, CV_8UC1, 255 / (b - a), 255 * a / (a - b));
    intensity_pub_.publish(
        cv_bridge::CvImage(header, image_encodings::MONO8, intensity)
            .toImageMsg());
  }

  // Don't forget to reset
  Reset();
  ROS_DEBUG("PacketCb time: %f", (ros::Time::now() - start).toSec());
}

void Decoder::ConfigCb(VelodynePuckConfig& config, int level) {
  config.image_width /= kSequencesPerPacket;
  config.image_width *= kSequencesPerPacket;

  ROS_INFO("Reconfigure Request: image_width: %d", config.image_width);

  config_ = config;
  Reset();

  if (level < 0) {
    ROS_INFO("Initialize ROS subscriber/publisher...");
    camera_pub_ = it_.advertiseCamera("image", 10);
    cloud_pub_ = pnh_.advertise<PointCloud2>("cloud", 10);
    intensity_pub_ = it_.advertise("intensity", 1);
    range_pub_ = it_.advertise("range", 1);

    packet_sub_ =
        pnh_.subscribe<VelodynePacket>("packet", 256, &Decoder::PacketCb, this);
    ROS_INFO("Decoder initialized");
  }
}

void Decoder::Reset() {
  curr_col_ = 0;
  image_ = cv::Mat(
      kFiringsPerSequence, config_.image_width, CV_32FC3, cv::Scalar(kNaNF));
  azimuths_.clear();
  azimuths_.resize(config_.image_width, kNaND);
  timestamps_.clear();
  timestamps_.resize(config_.image_width, 0);
}

}  // namespace velodyne_puck

int main(int argc, char** argv) {
  ros::init(argc, argv, "velodyne_puck_decoder");
  ros::NodeHandle pnh("~");

  velodyne_puck::Decoder node(pnh);
  ros::spin();
}
