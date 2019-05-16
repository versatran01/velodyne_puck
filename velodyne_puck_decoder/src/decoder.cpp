/*
 * This file is part of velodyne_puck driver.
 *
 * The driver is free software: you can redistribute it and/or modify
 * it under the terms of the GNU General Public License as published by
 * the Free Software Foundation, either version 3 of the License, or
 * (at your option) any later version.
 *
 * The driver is distributed in the hope that it will be useful,
 * but WITHOUT ANY WARRANTY; without even the implied warranty of
 * MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
 * GNU General Public License for more details.
 *
 * You should have received a copy of the GNU General Public License
 * along with the driver.  If not, see <http://www.gnu.org/licenses/>.
 */

#include "decoder.h"

#include <pcl/point_types.h>
#include <pcl_conversions/pcl_conversions.h>
#include <pcl_ros/point_cloud.h>
#include <sensor_msgs/PointCloud2.h>

namespace velodyne_puck_decoder {

using PointT = pcl::PointXYZI;
using CloudT = pcl::PointCloud<PointT>;
using velodyne_puck_msgs::VelodynePacket;
using velodyne_puck_msgs::VelodyneScan;

/// Compute total points in a sweep
size_t TotalPoints(const VelodyneSweep& sweep) {
  size_t n = 0;
  for (const VelodyneScan& scan : sweep.scans) {
    n += scan.points.size();
  }
  return n;
}

/// p51 8.3.1
double AzimuthResolutionDegree(int rpm) {
  ROS_ASSERT_MSG(rpm % 60 == 0, "rpm must be divisible by 60");
  return rpm / 60.0 * 360.0 * kFiringCycleUs / 1e6;
}

VelodynePuckDecoder::VelodynePuckDecoder(const ros::NodeHandle& n,
                                         const ros::NodeHandle& pn)
    : nh(n), pnh(pn), sweep_data(new VelodyneSweep()) {}

bool VelodynePuckDecoder::loadParameters() {
  pnh.param("min_range", min_range, 0.5);
  pnh.param("max_range", max_range, 100.0);
  ROS_INFO("min_range: %f, max_range: %f", min_range, max_range);

  int rpm;
  pnh.param("rpm", rpm, 300);
  ROS_INFO("azimuth resolution %f degree for rpm %d",
           AzimuthResolutionDegree(rpm), rpm);

  pnh.param("publish_cloud", publish_cloud, true);
  pnh.param<std::string>("frame_id", frame_id, "velodyne");
  return true;
}

bool VelodynePuckDecoder::createRosIO() {
  packet_sub = pnh.subscribe<VelodynePacket>(
      "packet", 100, &VelodynePuckDecoder::packetCallback, this);
  sweep_pub = pnh.advertise<VelodyneSweep>("sweep", 10);
  cloud_pub = pnh.advertise<sensor_msgs::PointCloud2>("cloud", 10);
  return true;
}

bool VelodynePuckDecoder::initialize() {
  if (!loadParameters()) {
    ROS_ERROR("Cannot load all required parameters...");
    return false;
  }

  if (!createRosIO()) {
    ROS_ERROR("Cannot create ROS I/O...");
    return false;
  }

  // Fill in the altitude for each scan.
  for (size_t scan_idx = 0; scan_idx < 16; ++scan_idx) {
    size_t remapped_scan_idx =
        scan_idx % 2 == 0 ? scan_idx / 2 : scan_idx / 2 + 8;
    sweep_data->scans[remapped_scan_idx].elevation = scan_elevation[scan_idx];
  }

  // Create the sin and cos table for different azimuth values.
  for (size_t i = 0; i < 6300; ++i) {
    double angle = static_cast<double>(i) / 1000.0;
    cos_azimuth_table[i] = cos(angle);
    sin_azimuth_table[i] = sin(angle);
  }

  return true;
}

bool VelodynePuckDecoder::checkPacketValidity(const RawPacket* packet) {
  for (size_t blk_idx = 0; blk_idx < BLOCKS_PER_PACKET; ++blk_idx) {
    if (packet->blocks[blk_idx].header != UPPER_BANK) {
      ROS_WARN("Skip invalid VLP-16 packet: block %lu header is %x", blk_idx,
               packet->blocks[blk_idx].header);
      return false;
    }
  }
  return true;
}

void VelodynePuckDecoder::decodePacket(const RawPacket* packet) {
  // Compute the azimuth angle for each firing.
  for (size_t fir_idx = 0; fir_idx < FIRINGS_PER_PACKET; fir_idx += 2) {
    size_t blk_idx = fir_idx / 2;
    firings[fir_idx].firing_azimuth =
        rawAzimuthToDouble(packet->blocks[blk_idx].rotation);
  }

  // Interpolate the azimuth values
  for (size_t fir_idx = 1; fir_idx < FIRINGS_PER_PACKET; fir_idx += 2) {
    size_t lfir_idx = fir_idx - 1;
    size_t rfir_idx = fir_idx + 1;

    if (fir_idx == FIRINGS_PER_PACKET - 1) {
      lfir_idx = fir_idx - 3;
      rfir_idx = fir_idx - 1;
    }

    double azimuth_diff =
        firings[rfir_idx].firing_azimuth - firings[lfir_idx].firing_azimuth;
    azimuth_diff = azimuth_diff < 0 ? azimuth_diff + 2 * M_PI : azimuth_diff;

    firings[fir_idx].firing_azimuth =
        firings[fir_idx - 1].firing_azimuth + azimuth_diff / 2.0;
    firings[fir_idx].firing_azimuth =
        firings[fir_idx].firing_azimuth > 2 * M_PI
            ? firings[fir_idx].firing_azimuth - 2 * M_PI
            : firings[fir_idx].firing_azimuth;
  }

  // Fill in the distance and intensity for each firing.
  for (size_t blk_idx = 0; blk_idx < BLOCKS_PER_PACKET; ++blk_idx) {
    const RawBlock& raw_block = packet->blocks[blk_idx];

    for (size_t blk_fir_idx = 0; blk_fir_idx < FIRINGS_PER_BLOCK;
         ++blk_fir_idx) {
      size_t fir_idx = blk_idx * FIRINGS_PER_BLOCK + blk_fir_idx;

      double azimuth_diff = 0.0;
      if (fir_idx < FIRINGS_PER_PACKET - 1)
        azimuth_diff = firings[fir_idx + 1].firing_azimuth -
                       firings[fir_idx].firing_azimuth;
      else
        azimuth_diff = firings[fir_idx].firing_azimuth -
                       firings[fir_idx - 1].firing_azimuth;

      for (size_t scan_fir_idx = 0; scan_fir_idx < kFiringsPerCycle;
           ++scan_fir_idx) {
        size_t byte_idx =
            RAW_SCAN_SIZE * (kFiringsPerCycle * blk_fir_idx + scan_fir_idx);

        // Azimuth
        firings[fir_idx].azimuth[scan_fir_idx] =
            firings[fir_idx].firing_azimuth +
            (scan_fir_idx * DSR_TOFFSET / kFiringCycleUs) * azimuth_diff;

        // Distance
        TwoBytes raw_distance;
        raw_distance.bytes[0] = raw_block.data[byte_idx];
        raw_distance.bytes[1] = raw_block.data[byte_idx + 1];
        firings[fir_idx].distance[scan_fir_idx] =
            static_cast<double>(raw_distance.distance) * DISTANCE_RESOLUTION;

        // Intensity
        firings[fir_idx].intensity[scan_fir_idx] =
            static_cast<double>(raw_block.data[byte_idx + 2]);
      }
    }
  }
}

void VelodynePuckDecoder::packetCallback(const VelodynePacketConstPtr& msg) {
  // Convert the msg to the raw packet type.
  const RawPacket* raw_packet = (const RawPacket*)(&(msg->data[0]));

  // Check if the packet is valid
  if (!checkPacketValidity(raw_packet)) return;

  // Decode the packet
  decodePacket(raw_packet);

  // Find the start of a new revolution
  //    If there is one, new_sweep_start will be the index of the start firing,
  //    otherwise, new_sweep_start will be FIRINGS_PER_PACKET.
  size_t new_sweep_start = 0;
  do {
    if (firings[new_sweep_start].firing_azimuth < last_azimuth)
      break;
    else {
      last_azimuth = firings[new_sweep_start].firing_azimuth;
      ++new_sweep_start;
    }
  } while (new_sweep_start < FIRINGS_PER_PACKET);

  // The first sweep may not be complete. So, the firings with
  // the first sweep will be discarded. We will wait for the
  // second sweep in order to find the 0 azimuth angle.
  size_t start_fir_idx = 0;
  size_t end_fir_idx = new_sweep_start;
  if (is_first_sweep && new_sweep_start == FIRINGS_PER_PACKET) {
    // The first sweep has not ended yet.
    return;
  } else {
    if (is_first_sweep) {
      is_first_sweep = false;
      start_fir_idx = new_sweep_start;
      end_fir_idx = FIRINGS_PER_PACKET;
      sweep_start_time = msg->stamp.toSec() +
                         kFiringCycleUs * (end_fir_idx - start_fir_idx) * 1e-6;
    }
  }

  for (size_t fir_idx = start_fir_idx; fir_idx < end_fir_idx; ++fir_idx) {
    for (size_t scan_idx = 0; scan_idx < kFiringsPerCycle; ++scan_idx) {
      // Check if the point is valid.
      if (!isPointInRange(firings[fir_idx].distance[scan_idx])) continue;

      // Convert the point to xyz coordinate
      size_t table_idx =
          floor(firings[fir_idx].azimuth[scan_idx] * 1000.0 + 0.5);
      // cout << table_idx << endl;
      double cos_azimuth = cos_azimuth_table[table_idx];
      double sin_azimuth = sin_azimuth_table[table_idx];

      // double x = firings[fir_idx].distance[scan_idx] *
      //  cos_scan_altitude[scan_idx] * sin(firings[fir_idx].azimuth[scan_idx]);
      // double y = firings[fir_idx].distance[scan_idx] *
      //  cos_scan_altitude[scan_idx] * cos(firings[fir_idx].azimuth[scan_idx]);
      // double z = firings[fir_idx].distance[scan_idx] *
      //  sin_scan_altitude[scan_idx];

      double x = firings[fir_idx].distance[scan_idx] *
                 cos_scan_elevation[scan_idx] * sin_azimuth;
      double y = firings[fir_idx].distance[scan_idx] *
                 cos_scan_elevation[scan_idx] * cos_azimuth;
      double z =
          firings[fir_idx].distance[scan_idx] * sin_scan_elevation[scan_idx];

      double x_coord = y;
      double y_coord = -x;
      double z_coord = z;

      // Compute the time of the point
      double time =
          packet_start_time + kFiringCycleUs * fir_idx + DSR_TOFFSET * scan_idx;

      // Remap the index of the scan
      int remapped_scan_idx =
          scan_idx % 2 == 0 ? scan_idx / 2 : scan_idx / 2 + 8;
      sweep_data->scans[remapped_scan_idx].points.push_back(
          velodyne_puck_msgs::VelodynePoint());
      velodyne_puck_msgs::VelodynePoint& new_point =
          sweep_data->scans[remapped_scan_idx]
              .points[sweep_data->scans[remapped_scan_idx].points.size() - 1];

      // Pack the data into point msg
      new_point.time = time;
      new_point.x = x_coord;
      new_point.y = y_coord;
      new_point.z = z_coord;
      new_point.azimuth = firings[fir_idx].azimuth[scan_idx];
      new_point.distance = firings[fir_idx].distance[scan_idx];
      new_point.intensity = firings[fir_idx].intensity[scan_idx];
    }
  }

  packet_start_time += kFiringCycleUs * (end_fir_idx - start_fir_idx);

  // A new sweep begins
  if (end_fir_idx != FIRINGS_PER_PACKET) {
    // Publish the last revolution
    sweep_data->header.stamp = ros::Time(sweep_start_time);
    sweep_pub.publish(sweep_data);

    if (publish_cloud) {
      publishCloud(*sweep_data);
    }

    sweep_data = velodyne_puck_msgs::VelodyneSweepPtr(
        new velodyne_puck_msgs::VelodyneSweep());

    // Prepare the next revolution
    sweep_start_time = msg->stamp.toSec() +
                       kFiringCycleUs * (end_fir_idx - start_fir_idx) * 1e-6;
    packet_start_time = 0.0;
    last_azimuth = firings[FIRINGS_PER_PACKET - 1].firing_azimuth;

    start_fir_idx = end_fir_idx;
    end_fir_idx = FIRINGS_PER_PACKET;

    for (size_t fir_idx = start_fir_idx; fir_idx < end_fir_idx; ++fir_idx) {
      for (size_t scan_idx = 0; scan_idx < kFiringsPerCycle; ++scan_idx) {
        // Check if the point is valid.
        if (!isPointInRange(firings[fir_idx].distance[scan_idx])) continue;

        // Convert the point to xyz coordinate
        size_t table_idx =
            floor(firings[fir_idx].azimuth[scan_idx] * 1000.0 + 0.5);
        // cout << table_idx << endl;
        double cos_azimuth = cos_azimuth_table[table_idx];
        double sin_azimuth = sin_azimuth_table[table_idx];

        // double x = firings[fir_idx].distance[scan_idx] *
        //  cos_scan_altitude[scan_idx] *
        //  sin(firings[fir_idx].azimuth[scan_idx]);
        // double y = firings[fir_idx].distance[scan_idx] *
        //  cos_scan_altitude[scan_idx] *
        //  cos(firings[fir_idx].azimuth[scan_idx]);
        // double z = firings[fir_idx].distance[scan_idx] *
        //  sin_scan_altitude[scan_idx];

        double x = firings[fir_idx].distance[scan_idx] *
                   cos_scan_elevation[scan_idx] * sin_azimuth;
        double y = firings[fir_idx].distance[scan_idx] *
                   cos_scan_elevation[scan_idx] * cos_azimuth;
        double z =
            firings[fir_idx].distance[scan_idx] * sin_scan_elevation[scan_idx];

        double x_coord = y;
        double y_coord = -x;
        double z_coord = z;

        // Compute the time of the point
        double time = packet_start_time +
                      kFiringCycleUs * (fir_idx - start_fir_idx) +
                      DSR_TOFFSET * scan_idx;

        // Remap the index of the scan
        int remapped_scan_idx =
            scan_idx % 2 == 0 ? scan_idx / 2 : scan_idx / 2 + 8;
        sweep_data->scans[remapped_scan_idx].points.push_back(
            velodyne_puck_msgs::VelodynePoint());
        velodyne_puck_msgs::VelodynePoint& new_point =
            sweep_data->scans[remapped_scan_idx]
                .points[sweep_data->scans[remapped_scan_idx].points.size() - 1];

        // Pack the data into point msg
        new_point.time = time;
        new_point.x = x_coord;
        new_point.y = y_coord;
        new_point.z = z_coord;
        new_point.azimuth = firings[fir_idx].azimuth[scan_idx];
        new_point.distance = firings[fir_idx].distance[scan_idx];
        new_point.intensity = firings[fir_idx].intensity[scan_idx];
      }
    }

    packet_start_time += kFiringCycleUs * (end_fir_idx - start_fir_idx);
  }
}

void VelodynePuckDecoder::publishCloud(const VelodyneSweep& sweep_msg) {
  CloudT::Ptr cloud = boost::make_shared<CloudT>();

  cloud->header = pcl_conversions::toPCL(sweep_msg.header);
  cloud->header.frame_id = frame_id;
  cloud->height = 1;
  cloud->reserve(TotalPoints(sweep_msg));

  for (const VelodyneScan& scan : sweep_msg.scans) {
    // The first and last point in each scan is ignored, which
    // seems to be corrupted based on the received data.
    // TODO: The two end points should be removed directly in the scans.
    if (scan.points.size() <= 2) continue;
    for (size_t j = 1; j < scan.points.size() - 1; ++j) {
      // TODO: compute here instead of saving them in scan, waste space
      const auto& vlp_point = scan.points[j];
      pcl::PointXYZI point;
      point.x = vlp_point.x;
      point.y = vlp_point.y;
      point.z = vlp_point.z;
      point.intensity = vlp_point.intensity;
      // cloud->push_back does extra work, so we don't use it
      cloud->points.push_back(point);
    }
  }

  cloud->width = cloud->size();
  cloud_pub.publish(cloud);
  ROS_DEBUG("Total cloud %zu", cloud->size());
}

}  //  namespace velodyne_puck_decoder