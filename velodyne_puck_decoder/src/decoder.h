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
#pragma once

#include <ros/ros.h>

#include <velodyne_puck_msgs/VelodynePacket.h>
#include <velodyne_puck_msgs/VelodyneSweep.h>

#include "constants.h"

namespace velodyne_puck_decoder {

using velodyne_puck_msgs::VelodynePacketConstPtr;
using velodyne_puck_msgs::VelodyneSweep;

/**
 * @brief The VelodynePuckDecoder class
 */
class VelodynePuckDecoder {
 public:
  VelodynePuckDecoder(const ros::NodeHandle& n, const ros::NodeHandle& pn);
  VelodynePuckDecoder(const VelodynePuckDecoder&) = delete;
  VelodynePuckDecoder operator=(const VelodynePuckDecoder&) = delete;

  using Ptr = boost::shared_ptr<VelodynePuckDecoder>;
  using ConstPtr = boost::shared_ptr<const VelodynePuckDecoder>;

  bool initialize();

 private:
  union TwoBytes {
    uint16_t distance;
    uint8_t bytes[2];
  };

  struct RawBlock {
    uint16_t header;    ///< UPPER_BANK or LOWER_BANK
    uint16_t rotation;  ///< 0-35999, divide by 100 to get degrees
    uint8_t data[BLOCK_DATA_SIZE];
  };

  struct RawPacket {
    RawBlock blocks[BLOCKS_PER_PACKET];
    uint32_t time_stamp;
    uint8_t factory[2];
    // uint16_t revolution;
    // uint8_t status[PACKET_STATUS_SIZE];
  };

  struct Firing {
    // Azimuth associated with the first shot within this firing.
    double firing_azimuth;
    double azimuth[kFiringsPerCycle];
    double distance[kFiringsPerCycle];
    double intensity[kFiringsPerCycle];
  };

  // Intialization sequence
  bool loadParameters();
  bool createRosIO();

  // Callback function for a single velodyne packet.
  bool checkPacketValidity(const RawPacket* packet);
  void decodePacket(const RawPacket* packet);
  void packetCallback(const VelodynePacketConstPtr& packet_msg);

  // Publish data
  void publishCloud(const VelodyneSweep& sweep_msg);

  // Check if a point is in the required range.
  bool isPointInRange(double distance) const {
    return distance >= min_range && distance <= max_range;
  }

  // Configuration parameters
  double min_range;
  double max_range;
  bool publish_cloud{true};

  double cos_azimuth_table[6300];
  double sin_azimuth_table[6300];

  bool is_first_sweep{true};
  double last_azimuth{0.0};
  double sweep_start_time{0.0};
  double packet_start_time{0.0};

  Firing firings[FIRINGS_PER_PACKET];

  // ROS related parameters
  ros::NodeHandle nh;
  ros::NodeHandle pnh;
  std::string frame_id;

  ros::Subscriber packet_sub;
  ros::Publisher sweep_pub;
  ros::Publisher cloud_pub;

  velodyne_puck_msgs::VelodyneSweepPtr sweep_data;
};

}  // end namespace velodyne_puck_decoder