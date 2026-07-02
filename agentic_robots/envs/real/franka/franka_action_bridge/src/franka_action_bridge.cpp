#include <algorithm>
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

Quaternion normalize_quaternion(const Quaternion& q) {
  const double norm = std::sqrt(q.w * q.w + q.x * q.x + q.y * q.y + q.z * q.z);
  if (norm <= 0.0) {
    throw std::runtime_error("quaternion norm must be positive");
  }
  return Quaternion{q.w / norm, q.x / norm, q.y / norm, q.z / norm};
}

Quaternion parse_quaternion(int first_index, char** argv) {
  return normalize_quaternion(Quaternion{
      parse_double(argv[first_index], "qw"),
      parse_double(argv[first_index + 1], "qx"),
      parse_double(argv[first_index + 2], "qy"),
      parse_double(argv[first_index + 3], "qz"),
  });
}

double quaternion_dot(const Quaternion& a, const Quaternion& b) {
  return a.w * b.w + a.x * b.x + a.y * b.y + a.z * b.z;
}

Quaternion slerp(Quaternion start, Quaternion goal, double alpha) {
  double dot = quaternion_dot(start, goal);
  if (dot < 0.0) {
    goal = Quaternion{-goal.w, -goal.x, -goal.y, -goal.z};
    dot = -dot;
  }

  if (dot > 0.9995) {
    return normalize_quaternion(Quaternion{
        start.w + alpha * (goal.w - start.w),
        start.x + alpha * (goal.x - start.x),
        start.y + alpha * (goal.y - start.y),
        start.z + alpha * (goal.z - start.z),
    });
  }

  const double theta_0 = std::acos(std::max(-1.0, std::min(1.0, dot)));
  const double theta = theta_0 * alpha;
  const double sin_theta = std::sin(theta);
  const double sin_theta_0 = std::sin(theta_0);
  const double s0 = std::cos(theta) - dot * sin_theta / sin_theta_0;
  const double s1 = sin_theta / sin_theta_0;
  return Quaternion{
      s0 * start.w + s1 * goal.w,
      s0 * start.x + s1 * goal.x,
      s0 * start.y + s1 * goal.y,
      s0 * start.z + s1 * goal.z,
  };
}

void set_rotation_from_quaternion(std::array<double, 16>& transform, const Quaternion& q_in) {
  const Quaternion q = normalize_quaternion(q_in);
  const double ww = q.w * q.w;
  const double xx = q.x * q.x;
  const double yy = q.y * q.y;
  const double zz = q.z * q.z;
  const double wx = q.w * q.x;
  const double wy = q.w * q.y;
  const double wz = q.w * q.z;
  const double xy = q.x * q.y;
  const double xz = q.x * q.z;
  const double yz = q.y * q.z;

  // libfranka uses column-major homogeneous transforms.
  transform[0] = ww + xx - yy - zz;
  transform[4] = 2.0 * (xy - wz);
  transform[8] = 2.0 * (xz + wy);
  transform[1] = 2.0 * (xy + wz);
  transform[5] = ww - xx + yy - zz;
  transform[9] = 2.0 * (yz - wx);
  transform[2] = 2.0 * (xz - wy);
  transform[6] = 2.0 * (yz + wx);
  transform[10] = ww - xx - yy + zz;
}

Vector3 rotate_vector(const std::array<double, 16>& transform, const Vector3& point) {
  return Vector3{
      transform[0] * point.x + transform[4] * point.y + transform[8] * point.z,
      transform[1] * point.x + transform[5] * point.y + transform[9] * point.z,
      transform[2] * point.x + transform[6] * point.y + transform[10] * point.z,
  };
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

void command_move_pose(const char* host, int argc, char** argv) {
  if (argc != 20) {
    throw std::runtime_error(
        "move-pose usage: <host> move-pose <dx> <dy> <dz> <qw> <qx> <qy> <qz> "
        "<duration_s> <x_min> <x_max> <y_min> <y_max> <z_min> <z_max> "
        "<tcp_offset_x> <tcp_offset_y> <tcp_offset_z>");
  }

  const double dx = parse_double(argv[3], "dx");
  const double dy = parse_double(argv[4], "dy");
  const double dz = parse_double(argv[5], "dz");
  const Quaternion target_quaternion = parse_quaternion(6, argv);
  const double duration = parse_double(argv[10], "duration_s");
  const WorkspaceBounds bounds{
      parse_double(argv[11], "x_min"),
      parse_double(argv[12], "x_max"),
      parse_double(argv[13], "y_min"),
      parse_double(argv[14], "y_max"),
      parse_double(argv[15], "z_min"),
      parse_double(argv[16], "z_max"),
  };
  const Vector3 tcp_offset_ee = parse_tcp_offset(17, argc, argv);

  if (duration < 2.0 || duration > 20.0) {
    throw std::runtime_error("duration_s must be in [2.0, 20.0]");
  }
  if (bounds.x_min >= bounds.x_max || bounds.y_min >= bounds.y_max ||
      bounds.z_min >= bounds.z_max) {
    throw std::runtime_error("workspace min values must be smaller than max values");
  }

  franka::Robot robot(host, franka::RealtimeConfig::kIgnore);
  std::array<double, 16> initial_pose{};
  std::array<double, 16> target_pose{};
  std::array<double, 16> final_pose{};
  Quaternion initial_quaternion{};
  double time = 0.0;
  bool initialized = false;

  robot.control([&](const franka::RobotState& state, franka::Duration period) {
    time += period.toSec();
    if (!initialized) {
      initial_pose = state.O_T_EE_c;
      initial_quaternion = rotation_to_quaternion(initial_pose);
      const Vector3 initial_tcp_position = transform_point(initial_pose, tcp_offset_ee);
      const Vector3 target_tcp_position{
          initial_tcp_position.x + dx,
          initial_tcp_position.y + dy,
          initial_tcp_position.z + dz,
      };
      if (target_tcp_position.x < bounds.x_min || target_tcp_position.x > bounds.x_max ||
          target_tcp_position.y < bounds.y_min || target_tcp_position.y > bounds.y_max ||
          target_tcp_position.z < bounds.z_min || target_tcp_position.z > bounds.z_max) {
        std::ostringstream message;
        message << "Target is outside configured workspace x=[" << bounds.x_min << ","
                << bounds.x_max << "], y=[" << bounds.y_min << "," << bounds.y_max
                << "], z=[" << bounds.z_min << "," << bounds.z_max << "]";
        throw std::runtime_error(message.str());
      }

      target_pose = initial_pose;
      set_rotation_from_quaternion(target_pose, target_quaternion);
      const Vector3 target_offset_base = rotate_vector(target_pose, tcp_offset_ee);
      target_pose[12] = target_tcp_position.x - target_offset_base.x;
      target_pose[13] = target_tcp_position.y - target_offset_base.y;
      target_pose[14] = target_tcp_position.z - target_offset_base.z;
      initialized = true;
    }

    const double clamped_time = std::min(time, duration);
    const double alpha = 0.5 * (1.0 - std::cos(M_PI * clamped_time / duration));
    std::array<double, 16> pose = initial_pose;
    const Quaternion q = slerp(initial_quaternion, target_quaternion, alpha);
    set_rotation_from_quaternion(pose, q);
    pose[12] = initial_pose[12] + alpha * (target_pose[12] - initial_pose[12]);
    pose[13] = initial_pose[13] + alpha * (target_pose[13] - initial_pose[13]);
    pose[14] = initial_pose[14] + alpha * (target_pose[14] - initial_pose[14]);
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
    const double force = argc >= 6 ? parse_double(argv[5], "force") : 20.0;
    const double epsilon_inner = argc >= 7 ? parse_double(argv[6], "epsilon_inner") : 0.005;
    const double epsilon_outer = argc >= 8 ? parse_double(argv[7], "epsilon_outer") : 0.04;
    if (width < 0.0 || width > 0.04) {
      throw std::runtime_error("gripper close width must be in [0.0, 0.04]");
    }
    if (force <= 0.0 || force > 70.0) {
      throw std::runtime_error("gripper close force must be in (0.0, 70.0]");
    }
    if (epsilon_inner < 0.0 || epsilon_outer < 0.0) {
      throw std::runtime_error("gripper close epsilons must be non-negative");
    }
    if (!gripper.grasp(width, speed, force, epsilon_inner, epsilon_outer)) {
      throw std::runtime_error("gripper grasp returned false");
    }
    std::this_thread::sleep_for(std::chrono::milliseconds(300));
    print_gripper_json(gripper.readOnce());
    return;
  }

  throw std::runtime_error("unknown gripper command");
}

void command_recover_errors(const char* host, int argc) {
  if (argc != 3) {
    throw std::runtime_error("recover-errors usage: <host> recover-errors");
  }
  franka::Robot robot(host, franka::RealtimeConfig::kIgnore);
  robot.automaticErrorRecovery();
  std::cout << "{\"ok\":true,\"command\":\"recover-errors\"}\n";
}

void print_usage(const char* program) {
  std::cerr << "Usage:\n"
            << "  " << program << " <host> pose <tcp_offset_x> <tcp_offset_y> <tcp_offset_z>\n"
            << "  " << program
            << " <host> move-relative <dx> <dy> <dz> <duration_s> "
               "<x_min> <x_max> <y_min> <y_max> <z_min> <z_max> "
               "<tcp_offset_x> <tcp_offset_y> <tcp_offset_z>\n"
            << "  " << program
            << " <host> move-pose <dx> <dy> <dz> <qw> <qx> <qy> <qz> "
               "<duration_s> <x_min> <x_max> <y_min> <y_max> <z_min> <z_max> "
               "<tcp_offset_x> <tcp_offset_y> <tcp_offset_z>\n"
            << "  " << program
            << " <host> joint-move <q0> <q1> <q2> <q3> <q4> <q5> <q6> "
               "<duration_s> <tcp_offset_x> <tcp_offset_y> <tcp_offset_z>\n"
            << "  " << program << " <host> gripper-read\n"
            << "  " << program << " <host> gripper-open [width_m] [speed_m_s]\n"
            << "  " << program << " <host> gripper-close [width_m] [speed_m_s]\n"
            << "  " << program << " <host> recover-errors\n";
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
    } else if (command == "move-pose") {
      command_move_pose(host, argc, argv);
    } else if (command == "joint-move") {
      command_joint_move(host, argc, argv);
    } else if (
        command == "gripper-read" || command == "gripper-open" || command == "gripper-close") {
      command_gripper(host, command, argc, argv);
    } else if (command == "recover-errors") {
      command_recover_errors(host, argc);
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
