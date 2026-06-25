cmake_minimum_required(VERSION 3.16)
project(franka_action_bridge LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

find_package(Franka REQUIRED)

add_executable(franka_action_bridge src/franka_action_bridge.cpp)
target_link_libraries(franka_action_bridge Franka::Franka)
