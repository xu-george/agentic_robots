#include <array>
#include <chrono>
#include <cmath>
#include <cstdlib>
#include <iomanip>
#include <iostream>
#include <sstream>
#include <stdexcept>
#include <string>
#include <thread>

#include <franka/exception.h>
#include <franka/gripper.h>
#include <franka/gripper_state.h>
#include <franka/robot.h>
#include <franka/robot_state.h>

namespace {

struct Quaternion {
  double w;
  double x;
  double y;
  double z;
};

struct WorkspaceBounds {
  double x_min;
  double x_max;
  double y_min;
  double y_max;
  double z_min;
  double z_max;
};

struct Vector3 {
  double x;
  double y;
  double z;
};

double parse_double(const char* value, const char* name) {
  char* end = nullptr;
  const double result = std::strtod(value, &end);
  if (end == value || *end != '\0') {
    throw std::runtime_error(std::string("Invalid ") + name + ": " + value);
  }
  return result;
}

WorkspaceBounds parse_workspace_bounds(int argc, char** argv) {
  if (argc < 13) {
    throw std::runtime_error(
        "move-relative usage: <host> move-relative <dx> <dy> <dz> <duration_s> "
        "<x_min> <x_max> <y_min> <y_max> <z_min> <z_max>");
  }

  WorkspaceBounds bounds{
      parse_double(argv[7], "x_min"),
      parse_double(argv[8], "x_max"),
      parse_double(argv[9], "y_min"),
      parse_double(argv[10], "y_max"),
      parse_double(argv[11], "z_min"),
      parse_double(argv[12], "z_max"),
  };
  if (bounds.x_min >= bounds.x_max || bounds.y_min >= bounds.y_max ||
      bounds.z_min >= bounds.z_max) {
    throw std::runtime_error("workspace min values must be smaller than max values");
  }
  return bounds;
}

Vector3 parse_tcp_offset(int first_index, int argc, char** argv) {
  if (argc != first_index + 3) {
    throw std::runtime_error(
        "expected TCP offset arguments: <tcp_offset_x> <tcp_offset_y> <tcp_offset_z>");
  }
  return Vector3{
      parse_double(argv[first_index], "tcp_offset_x"),
      parse_double(argv[first_index + 1], "tcp_offset_y"),
      parse_double(argv[first_index + 2], "tcp_offset_z"),
  };
}

std::array<double, 7> parse_joint_positions(int first_index, char** argv) {
  std::array<double, 7> joints{};
  for (size_t i = 0; i < joints.size(); ++i) {
    joints[i] = parse_double(argv[first_index + static_cast<int>(i)], "joint_position");
  }
  return joints;
}

Vector3 transform_point(const std::array<double, 16>& transform, const Vector3& point) {
  return Vector3{
      transform[12] + transform[0] * point.x + transform[4] * point.y + transform[8] * point.z,
      transform[13] + transform[1] * point.x + transform[5] * point.y + transform[9] * point.z,
      transform[14] + transform[2] * point.x + transform[6] * point.y + transform[10] * point.z,
  };
}

std::array<double, 16> tcp_transform_from_ee(
    const std::array<double, 16>& ee_transform,
    const Vector3& tcp_offset_ee) {
  std::array<double, 16> tcp_transform = ee_transform;
  const Vector3 tcp_position = transform_point(ee_transform, tcp_offset_ee);
  tcp_transform[12] = tcp_position.x;
  tcp_transform[13] = tcp_position.y;
  tcp_transform[14] = tcp_position.z;
  return tcp_transform;
}

Quaternion rotation_to_quaternion(const std::array<double, 16>& transform) {
  const double r00 = transform[0];
  const double r01 = transform[4];
  const double r02 = transform[8];
  const double r10 = transform[1];
  const double r11 = transform[5];
  const double r12 = transform[9];
  const double r20 = transform[2];
  const double r21 = transform[6];
  const double r22 = transform[10];

  const double trace = r00 + r11 + r22;
  Quaternion q{};
  if (trace > 0.0) {
    const double scale = std::sqrt(trace + 1.0) * 2.0;
    q.w = 0.25 * scale;
    q.x = (r21 - r12) / scale;
    q.y = (r02 - r20) / scale;
    q.z = (r10 - r01) / scale;
  } else if (r00 > r11 && r00 > r22) {
    const double scale = std::sqrt(1.0 + r00 - r11 - r22) * 2.0;
    q.w = (r21 - r12) / scale;
    q.x = 0.25 * scale;
    q.y = (r01 + r10) / scale;
    q.z = (r02 + r20) / scale;
  } else if (r11 > r22) {
    const double scale = std::sqrt(1.0 + r11 - r00 - r22) * 2.0;
    q.w = (r02 - r20) / scale;
    q.x = (r01 + r10) / scale;
    q.y = 0.25 * scale;
    q.z = (r12 + r21) / scale;
  } else {
    const double scale = std::sqrt(1.0 + r22 - r00 - r11) * 2.0;
    q.w = (r10 - r01) / scale;
    q.x = (r02 + r20) / scale;
    q.y = (r12 + r21) / scale;
    q.z = 0.25 * scale;
  }
  return q;
}

void print_transform_array(const std::array<double, 16>& transform) {
  for (size_t i = 0; i < transform.size(); ++i) {
    std::cout << transform[i];
    if (i + 1 != transform.size()) {
      std::cout << ",";
    }
  }
}

void print_pose_json(const std::array<double, 16>& ee_transform, const Vector3& tcp_offset_ee) {
  const std::array<double, 16> tcp_transform = tcp_transform_from_ee(ee_transform, tcp_offset_ee);
  const Quaternion q = rotation_to_quaternion(tcp_transform);
  std::cout << std::fixed << std::setprecision(6);
  std::cout << "{\"position\":[" << tcp_transform[12] << "," << tcp_transform[13] << ","
            << tcp_transform[14] << "],\"quaternion_wxyz\":[" << q.w << "," << q.x << ","
            << q.y << "," << q.z << "],\"tcp_offset_from_ee\":[" << tcp_offset_ee.x << ","
            << tcp_offset_ee.y << "," << tcp_offset_ee.z << "],\"O_T_TCP\":[";
  print_transform_array(tcp_transform);
  std::cout << "],\"O_T_EE\":[";
  print_transform_array(ee_transform);
  std::cout << "]}\n";
}

void print_gripper_json(const franka::GripperState& state) {
  std::cout << std::fixed << std::setprecision(6);
  std::cout << "{\"width\":" << state.width << ",\"max_width\":" << state.max_width
            << ",\"is_grasped\":" << (state.is_grasped ? "true" : "false")
            << ",\"temperature\":" << state.temperature << "}\n";
}

void command_pose(const char* host, int argc, char** argv) {
  if (argc != 6) {
    throw std::runtime_error("pose usage: <host> pose <tcp_offset_x> <tcp_offset_y> <tcp_offset_z>");
  }
  const Vector3 tcp_offset_ee = parse_tcp_offset(3, argc, argv);
  franka::Robot robot(host, franka::RealtimeConfig::kIgnore);
  bool printed = false;
  robot.read([&printed, &tcp_offset_ee](const franka::RobotState& state) {
    if (!printed) {
      print_pose_json(state.O_T_EE, tcp_offset_ee);
      printed = true;
    }
    return !printed;
  });
}

void command_move_relative(const char* host, int argc, char** argv) {
  if (argc != 16) {
    throw std::runtime_error(
        "move-relative usage: <host> move-relative <dx> <dy> <dz> <duration_s> "
        "<x_min> <x_max> <y_min> <y_max> <z_min> <z_max> "
        "<tcp_offset_x> <tcp_offset_y> <tcp_offset_z>");
  }

  const double dx = parse_double(argv[3], "dx");
  const double dy = parse_double(argv[4], "dy");
  const double dz = parse_double(argv[5], "dz");
  const double duration = parse_double(argv[6], "duration_s");
  const WorkspaceBounds bounds = parse_workspace_bounds(argc, argv);
  const Vector3 tcp_offset_ee = parse_tcp_offset(13, argc, argv);

  if (duration < 2.0 || duration > 20.0) {
    throw std::runtime_error("duration_s must be in [2.0, 20.0]");
  }
  if (std::abs(dx) > 0.05 || std::abs(dy) > 0.05 || std::abs(dz) > 0.05) {
    throw std::runtime_error("Each Cartesian delta must be <= 0.05 m");
  }

  franka::Robot robot(host, franka::RealtimeConfig::kIgnore);
  std::array<double, 16> initial_pose{};
  std::array<double, 16> final_pose{};
  double time = 0.0;
  bool initialized = false;

  robot.control([&](const franka::RobotState& state, franka::Duration period) {
    time += period.toSec();
    if (!initialized) {
      initial_pose = state.O_T_EE_c;
      const Vector3 initial_tcp_position = transform_point(initial_pose, tcp_offset_ee);
      const double target_x = initial_tcp_position.x + dx;
      const double target_y = initial_tcp_position.y + dy;
      const double target_z = initial_tcp_position.z + dz;
      if (target_x < bounds.x_min || target_x > bounds.x_max || target_y < bounds.y_min ||
          target_y > bounds.y_max || target_z < bounds.z_min || target_z > bounds.z_max) {
        std::ostringstream message;
        message << "Target is outside configured workspace x=[" << bounds.x_min << ","
                << bounds.x_max << "], y=[" << bounds.y_min << "," << bounds.y_max
                << "], z=[" << bounds.z_min << "," << bounds.z_max << "]";
        throw std::runtime_error(message.str());
      }
      initialized = true;
    }

    const double clamped_time = std::min(time, duration);
    const double alpha = 0.5 * (1.0 - std::cos(M_PI * clamped_time / duration));
    std::array<double, 16> pose = initial_pose;
    pose[12] += alpha * dx;
    pose[13] += alpha * dy;
    pose[14] += alpha * dz;
    final_pose = pose;

    if (time >= duration) {
      return franka::MotionFinished(franka::CartesianPose(pose));
    }
    return franka::CartesianPose(pose);
  });

  print_pose_json(final_pose, tcp_offset_ee);
}

void command_joint_move(const char* host, int argc, char** argv) {
  if (argc != 14) {
    throw std::runtime_error(
        "joint-move usage: <host> joint-move <q0> <q1> <q2> <q3> <q4> <q5> <q6> "
        "<duration_s> <tcp_offset_x> <tcp_offset_y> <tcp_offset_z>");
  }

  const std::array<double, 7> q_goal = parse_joint_positions(3, argv);
  const double duration = parse_double(argv[10], "duration_s");
  const Vector3 tcp_offset_ee = parse_tcp_offset(11, argc, argv);
  if (duration < 2.0 || duration > 20.0) {
    throw std::runtime_error("duration_s must be in [2.0, 20.0]");
  }

  franka::Robot robot(host, franka::RealtimeConfig::kIgnore);
  std::array<double, 7> q_start{};
  std::array<double, 16> final_pose{};
  double time = 0.0;
  bool initialized = false;

  robot.control([&](const franka::RobotState& state, franka::Duration period) {
    time += period.toSec();
    if (!initialized) {
      q_start = state.q;
      initialized = true;
    }

    const double clamped_time = std::min(time, duration);
    const double alpha = 0.5 * (1.0 - std::cos(M_PI * clamped_time / duration));
    std::array<double, 7> q_command{};
    for (size_t i = 0; i < q_command.size(); ++i) {
      q_command[i] = q_start[i] + alpha * (q_goal[i] - q_start[i]);
    }

    if (time >= duration) {
      return franka::MotionFinished(franka::JointPositions(q_goal));
    }
    return franka::JointPositions(q_command);
  });

  final_pose = robot.readOnce().O_T_EE;
  print_pose_json(final_pose, tcp_offset_ee);
}

void command_gripper(const char* host, const std::string& command, int argc, char** argv) {
  franka::Gripper gripper(host);
  if (command == "gripper-read") {
    print_gripper_json(gripper.readOnce());
    return;
  }

  if (command == "gripper-open") {
    const double width = argc >= 4 ? parse_double(argv[3], "width") : 0.04;
    const double speed = argc >= 5 ? parse_double(argv[4], "speed") : 0.03;
    if (width < 0.0 || width > 0.08) {
      throw std::runtime_error("gripper open width must be in [0.0, 0.08]");
    }
    if (!gripper.move(width, speed)) {
      throw std::runtime_error("gripper open move returned false");
    }
    std::this_thread::sleep_for(std::chrono::milliseconds(300));
    print_gripper_json(gripper.readOnce());
    return;
  }

  if (command == "gripper-close") {
    const double width = argc >= 4 ? parse_double(argv[3], "width") : 0.01;
    const double speed = argc >= 5 ? parse_double(argv[4], "speed") : 0.03;
    if (width < 0.0 || width > 0.04) {
      throw std::runtime_error("gripper close width must be in [0.0, 0.04]");
    }
    if (!gripper.move(width, speed)) {
      throw std::runtime_error("gripper close move returned false");
    }
    std::this_thread::sleep_for(std::chrono::milliseconds(300));
    print_gripper_json(gripper.readOnce());
    return;
  }

  throw std::runtime_error("unknown gripper command");
}

void print_usage(const char* program) {
  std::cerr << "Usage:\n"
            << "  " << program << " <host> pose <tcp_offset_x> <tcp_offset_y> <tcp_offset_z>\n"
            << "  " << program
            << " <host> move-relative <dx> <dy> <dz> <duration_s> "
               "<x_min> <x_max> <y_min> <y_max> <z_min> <z_max> "
               "<tcp_offset_x> <tcp_offset_y> <tcp_offset_z>\n"
            << "  " << program
            << " <host> joint-move <q0> <q1> <q2> <q3> <q4> <q5> <q6> "
               "<duration_s> <tcp_offset_x> <tcp_offset_y> <tcp_offset_z>\n"
            << "  " << program << " <host> gripper-read\n"
            << "  " << program << " <host> gripper-open [width_m] [speed_m_s]\n"
            << "  " << program << " <host> gripper-close [width_m] [speed_m_s]\n";
}

}  // namespace

int main(int argc, char** argv) {
  if (argc < 3) {
    print_usage(argv[0]);
    return 2;
  }

  const char* host = argv[1];
  const std::string command = argv[2];

  try {
    if (command == "pose") {
      command_pose(host, argc, argv);
    } else if (command == "move-relative") {
      command_move_relative(host, argc, argv);
    } else if (command == "joint-move") {
      command_joint_move(host, argc, argv);
    } else if (
        command == "gripper-read" || command == "gripper-open" || command == "gripper-close") {
      command_gripper(host, command, argc, argv);
    } else {
      print_usage(argv[0]);
      return 2;
    }
  } catch (const franka::Exception& e) {
    std::cerr << "libfranka error: " << e.what() << "\n";
    return 1;
  } catch (const std::exception& e) {
    std::cerr << "error: " << e.what() << "\n";
    return 1;
  }

  return 0;
}
