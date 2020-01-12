#pragma once

#include <pcl/point_cloud.h>
#include <pcl/point_types.h>

#include <cv_bridge/cv_bridge.h>
#include <pcl_conversions/pcl_conversions.h>
#include <sensor_msgs/CameraInfo.h>
#include <sensor_msgs/Image.h>

namespace cloud {

using sensor_msgs::CameraInfo;
using sensor_msgs::ImageConstPtr;
using Point = pcl::PointXYZI;
using Cloud = pcl::PointCloud<Point>;
static constexpr auto kNaNF = std::numeric_limits<float>::quiet_NaN();

struct SinCos {
  SinCos() = default;
  SinCos(double rad) : sin{std::sin(rad)}, cos{std::cos(rad)} {}
  double sin, cos;
};

/// Used for indexing into packet and image, (NOISE not used now)
enum Index { RANGE = 0, INTENSITY = 1, AZIMUTH = 2, NOISE = 3 };

template <typename Iter>
std::vector<SinCos> PrecomputeSinCos(Iter first, Iter last) {
  std::vector<SinCos> sc;
  sc.reserve(std::distance(first, last));
  while (first != last)
    sc.emplace_back(*first++);
  return sc;
}

inline void PolarToCart(Point &p, const SinCos &ev, const SinCos &az,
                        double r) {
  p.x = r * ev.cos * az.cos;
  p.y = -r * ev.cos * az.sin;
  p.z = r * ev.sin;
}

/// High precision mode
static Cloud::VectorType ToCloud(const cv::Mat &image,
                                 const std::vector<SinCos> &elevations) {
  Cloud::VectorType points;
  points.reserve(image.total());

  for (int r = 0; r < image.rows; ++r) {
    const auto *const row_ptr = image.ptr<cv::Vec3f>(r);
    for (int c = 0; c < image.cols; ++c) {
      const cv::Vec3f &data = row_ptr[c];
      const auto theta = data[AZIMUTH];

      Point p;
      if (std::isnan(data[RANGE])) {
        p.x = p.y = p.z = p.intensity = kNaNF;
      } else {
        PolarToCart(p, elevations[r], {theta}, data[RANGE]);
        p.intensity = data[INTENSITY];
      }
      points.push_back(p);
    } // c
  }   // r

  return points;
}

/// Low precision mode
static Cloud::VectorType ToCloud(const cv::Mat &image,
                                 const std::vector<SinCos> &elevations,
                                 const std::vector<SinCos> &azimuths) {
  Cloud::VectorType points;
  points.reserve(image.total());

  for (int r = 0; r < image.rows; ++r) {
    const auto *const row_ptr = image.ptr<cv::Vec3f>(r);
    for (int c = 0; c < image.cols; ++c) {
      const cv::Vec3f &data = row_ptr[c];
      Point p;
      if (std::isnan(data[RANGE])) {
        p.x = p.y = p.z = p.intensity = kNaNF;
      } else {
        PolarToCart(p, elevations[r], azimuths[c], data[RANGE]);
        p.intensity = data[INTENSITY];
      }
      points.push_back(p);
    } // c
  }   // r

  return points;
}

Cloud ToCloud(const ImageConstPtr &image_msg, const CameraInfo &cinfo_msg,
              bool high_prec) {
  Cloud cloud;
  const auto image = cv_bridge::toCvShare(image_msg)->image;
  const auto &D = cinfo_msg.D;
  const auto elevations = PrecomputeSinCos(D.cbegin(), D.cbegin() + image.rows);

  if (high_prec) {
    cloud.points = std::move(ToCloud(image, elevations));
  } else {
    const auto azimuths = PrecomputeSinCos(
        D.cbegin() + image.rows, D.cbegin() + image.rows + image.cols);
    cloud.points = std::move(ToCloud(image, elevations, azimuths));
  }

  cloud.header = pcl_conversions::toPCL(image_msg->header);
  cloud.width = image.cols;
  cloud.height = image.rows;
  return cloud;
}

} // namespace cloud
