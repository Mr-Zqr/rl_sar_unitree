/*
 * Copyright (c) 2024-2025 Ziqi Fan
 * SPDX-License-Identifier: Apache-2.0
 */

#include "rl_real_g1.hpp"
#include "onnxruntime_cxx_api.h"


// 全局指针用于信号处理
RL_Real* g_rl_real_instance = nullptr;
RL_Real::RL_Real()
#if defined(USE_ROS2) && defined(USE_ROS)
    : rclcpp::Node("rl_real_node")
#endif
{
#if defined(USE_ROS1) && defined(USE_ROS)
    ros::NodeHandle nh;
    this->cmd_vel_subscriber = nh.subscribe<geometry_msgs::Twist>("/cmd_vel", 10, &RL_Real::CmdvelCallback, this);
#elif defined(USE_ROS2) && defined(USE_ROS)
    this->cmd_vel_subscriber = this->create_subscription<geometry_msgs::msg::Twist>(
        "/cmd_vel", rclcpp::SystemDefaultsQoS(),
        [this] (const geometry_msgs::msg::Twist::SharedPtr msg) {this->CmdvelCallback(msg);}
    );
#endif

    // read params from yaml
    this->ang_vel_type = "ang_vel_body";
    this->robot_name = "g1";
    this->ReadYamlBase(this->robot_name);

    // auto load FSM by robot_name
    if (FSMManager::GetInstance().IsTypeSupported(this->robot_name))
    {
        auto fsm_ptr = FSMManager::GetInstance().CreateFSM(this->robot_name, this);
        if (fsm_ptr)
        {
            this->fsm = *fsm_ptr;
        }
    }
    else
    {
        std::cout << LOGGER::ERROR << "No FSM registered for robot: " << this->robot_name << std::endl;
    }

    // init torch
    torch::autograd::GradMode::set_enabled(false);
    torch::set_num_threads(4);

    // init robot
    this->mode_pr = Mode::PR;
    this->mode_machine = 0;
    this->InitLowCmd();
    this->InitOutputs();
    this->InitControl();
    // init MotionSwitcherClient
    this->msc.SetTimeout(5.0f);
    this->msc.Init();
    // Shut down motion control-related service
    std::string form, name;
    while (this->msc.CheckMode(form, name), !name.empty())
    {
        if (this->msc.ReleaseMode())
        {
            std::cout << "Failed to switch to Release Mode" << std::endl;
        }
        sleep(5);
    }
    // create lowcmd publisher
    this->lowcmd_publisher.reset(new ChannelPublisher<LowCmd_>(HG_CMD_TOPIC));
    this->lowcmd_publisher->InitChannel();
    // create lowstate subscriber
    this->lowstate_subscriber.reset(new ChannelSubscriber<LowState_>(HG_STATE_TOPIC));
    this->lowstate_subscriber->InitChannel(std::bind(&RL_Real::LowStateHandler, this, std::placeholders::_1), 1);
    // create imutorso subscriber
    this->imutorso_subscriber.reset(new ChannelSubscriber<IMUState_>(HG_IMU_TORSO));
    this->imutorso_subscriber->InitChannel(std::bind(&RL_Real::ImuTorsoHandler, this, std::placeholders::_1), 1);

    // loop
    this->loop_keyboard = std::make_shared<LoopFunc>("loop_keyboard", 0.05, std::bind(&RL_Real::KeyboardInterface, this));
    this->loop_control = std::make_shared<LoopFunc>("loop_control", this->params.dt, std::bind(&RL_Real::RobotControl, this));
    this->loop_rl = std::make_shared<LoopFunc>("loop_rl", this->params.dt * this->params.decimation, std::bind(&RL_Real::RunModel, this));
    this->loop_keyboard->start();
    this->loop_control->start();
    this->loop_rl->start();

#ifdef PLOT
    this->plot_t = std::vector<int>(this->plot_size, 0);
    this->plot_real_joint_pos.resize(this->params.num_of_dofs);
    this->plot_target_joint_pos.resize(this->params.num_of_dofs);
    for (auto &vector : this->plot_real_joint_pos) { vector = std::vector<double>(this->plot_size, 0); }
    for (auto &vector : this->plot_target_joint_pos) { vector = std::vector<double>(this->plot_size, 0); }
    this->loop_plot = std::make_shared<LoopFunc>("loop_plot", 0.002, std::bind(&RL_Real::Plot, this));
    this->loop_plot->start();
#endif
#ifdef CSV_LOGGER
    this->CSVInit(this->robot_name);
#endif
    // 初始化日志器
    this->logger_ = std::make_unique<RLLogger>();
    this->logging_active_ = false;
    this->previous_rl_init_done_ = false;
    this->start_time_ = std::chrono::high_resolution_clock::now();
    this->last_log_time_ = this->start_time_;
    this->last_inference_time_ = 0.0;
}

RL_Real::~RL_Real()
{
    // 在退出时保存日志
    if (this->logging_active_ && this->logger_->HasData()) {
        std::cout << LOGGER::INFO << "Saving log data before exit..." << std::endl;
        this->SaveCurrentLog();
    }
    
    this->loop_keyboard->shutdown();
    this->loop_control->shutdown();
    this->loop_rl->shutdown();
#ifdef PLOT
    this->loop_plot->shutdown();
#endif
    std::cout << LOGGER::INFO << "RL_Real exit" << std::endl;
}

void RL_Real::GetState(RobotState<double> *state)
{
    if (this->mode_machine != this->unitree_low_state.mode_machine())
    {
        if (this->mode_machine == 0)
        {
            std::cout << "G1 type: " << unsigned(this->unitree_low_state.mode_machine()) << std::endl;
        }
        this->mode_machine = this->unitree_low_state.mode_machine();
    }

    memcpy(this->remote_data_rx.buff, &unitree_low_state.wireless_remote()[0], 40);
    this->gamepad.update(this->remote_data_rx.RF_RX);

    if (this->gamepad.A.pressed) this->control.SetGamepad(Input::Gamepad::A);
    if (this->gamepad.B.pressed) this->control.SetGamepad(Input::Gamepad::B);
    if (this->gamepad.X.pressed) this->control.SetGamepad(Input::Gamepad::X);
    if (this->gamepad.Y.pressed) this->control.SetGamepad(Input::Gamepad::Y);
    if (this->gamepad.R1.pressed) this->control.SetGamepad(Input::Gamepad::RB);
    if (this->gamepad.L1.pressed) this->control.SetGamepad(Input::Gamepad::LB);
    if (this->gamepad.F1.pressed) this->control.SetGamepad(Input::Gamepad::LStick);
    if (this->gamepad.F2.pressed) this->control.SetGamepad(Input::Gamepad::RStick);
    if (this->gamepad.up.pressed) this->control.SetGamepad(Input::Gamepad::DPadUp);
    if (this->gamepad.down.pressed) this->control.SetGamepad(Input::Gamepad::DPadDown);
    if (this->gamepad.left.pressed) this->control.SetGamepad(Input::Gamepad::DPadLeft);
    if (this->gamepad.right.pressed) this->control.SetGamepad(Input::Gamepad::DPadRight);
    if (this->gamepad.L1.pressed && this->gamepad.A.pressed) this->control.SetGamepad(Input::Gamepad::LB_A);
    if (this->gamepad.L1.pressed && this->gamepad.B.pressed) this->control.SetGamepad(Input::Gamepad::LB_B);
    if (this->gamepad.L1.pressed && this->gamepad.X.pressed) this->control.SetGamepad(Input::Gamepad::LB_X);
    if (this->gamepad.L1.pressed && this->gamepad.Y.pressed) this->control.SetGamepad(Input::Gamepad::LB_Y);
    if (this->gamepad.L1.pressed && this->gamepad.F1.pressed) this->control.SetGamepad(Input::Gamepad::LB_LStick);
    if (this->gamepad.L1.pressed && this->gamepad.F2.pressed) this->control.SetGamepad(Input::Gamepad::LB_RStick);
    if (this->gamepad.L1.pressed && this->gamepad.up.pressed) this->control.SetGamepad(Input::Gamepad::LB_DPadUp);
    if (this->gamepad.L1.pressed && this->gamepad.down.pressed) this->control.SetGamepad(Input::Gamepad::LB_DPadDown);
    if (this->gamepad.L1.pressed && this->gamepad.left.pressed) this->control.SetGamepad(Input::Gamepad::LB_DPadLeft);
    if (this->gamepad.L1.pressed && this->gamepad.right.pressed) this->control.SetGamepad(Input::Gamepad::LB_DPadRight);
    if (this->gamepad.R1.pressed && this->gamepad.A.pressed) this->control.SetGamepad(Input::Gamepad::RB_A);
    if (this->gamepad.R1.pressed && this->gamepad.B.pressed) this->control.SetGamepad(Input::Gamepad::RB_B);
    if (this->gamepad.R1.pressed && this->gamepad.X.pressed) this->control.SetGamepad(Input::Gamepad::RB_X);
    if (this->gamepad.R1.pressed && this->gamepad.Y.pressed) this->control.SetGamepad(Input::Gamepad::RB_Y);
    if (this->gamepad.R1.pressed && this->gamepad.F1.pressed) this->control.SetGamepad(Input::Gamepad::RB_LStick);
    if (this->gamepad.R1.pressed && this->gamepad.F2.pressed) this->control.SetGamepad(Input::Gamepad::RB_RStick);
    if (this->gamepad.R1.pressed && this->gamepad.up.pressed) this->control.SetGamepad(Input::Gamepad::RB_DPadUp);
    if (this->gamepad.R1.pressed && this->gamepad.down.pressed) this->control.SetGamepad(Input::Gamepad::RB_DPadDown);
    if (this->gamepad.R1.pressed && this->gamepad.left.pressed) this->control.SetGamepad(Input::Gamepad::RB_DPadLeft);
    if (this->gamepad.R1.pressed && this->gamepad.right.pressed) this->control.SetGamepad(Input::Gamepad::RB_DPadRight);
    if (this->gamepad.L1.pressed && this->gamepad.R1.pressed) this->control.SetGamepad(Input::Gamepad::LB_RB);

    this->control.x = this->gamepad.ly;
    this->control.y = -this->gamepad.lx;
    this->control.yaw = -this->gamepad.rx;

    state->imu.quaternion[0] = this->unitree_low_state.imu_state().quaternion()[0]; // w
    state->imu.quaternion[1] = this->unitree_low_state.imu_state().quaternion()[1]; // x
    state->imu.quaternion[2] = this->unitree_low_state.imu_state().quaternion()[2]; // y
    state->imu.quaternion[3] = this->unitree_low_state.imu_state().quaternion()[3]; // z

    for (int i = 0; i < 3; ++i)
    {
        state->imu.gyroscope[i] = this->unitree_low_state.imu_state().gyroscope()[i];
    }
    for (int i = 0; i < this->params.num_of_dofs; ++i)
    {
        state->motor_state.q[i] = this->unitree_low_state.motor_state()[this->params.joint_mapping[i]].q();
        state->motor_state.dq[i] = this->unitree_low_state.motor_state()[this->params.joint_mapping[i]].dq();
        state->motor_state.tau_est[i] = this->unitree_low_state.motor_state()[this->params.joint_mapping[i]].tau_est();
    }
}

void RL_Real::SetCommand(const RobotCommand<double> *command)
{
    this->unitree_low_command.mode_pr() = static_cast<uint8_t>(this->mode_pr);
    this->unitree_low_command.mode_machine() = this->mode_machine;

    for (int i = 0; i < this->params.num_of_dofs; ++i)
    {
        this->unitree_low_command.motor_cmd()[this->params.joint_mapping[i]].mode() = 1; // 1:Enable, 0:Disable
        this->unitree_low_command.motor_cmd()[this->params.joint_mapping[i]].q() = command->motor_command.q[i];
        this->unitree_low_command.motor_cmd()[this->params.joint_mapping[i]].dq() = command->motor_command.dq[i];
        this->unitree_low_command.motor_cmd()[this->params.joint_mapping[i]].kp() = command->motor_command.kp[i];
        this->unitree_low_command.motor_cmd()[this->params.joint_mapping[i]].kd() = command->motor_command.kd[i];
        this->unitree_low_command.motor_cmd()[this->params.joint_mapping[i]].tau() = command->motor_command.tau[i];
    }

    this->unitree_low_command.crc() = Crc32Core((uint32_t *)&unitree_low_command, (sizeof(LowCmd_) >> 2) - 1);
    lowcmd_publisher->Write(unitree_low_command);
}

void RL_Real::RobotControl()
{
    this->motiontime++;

    if (this->control.current_keyboard == Input::Keyboard::W)
    {
        this->control.x += 0.1;
        this->control.current_keyboard = this->control.last_keyboard;
    }
    if (this->control.current_keyboard == Input::Keyboard::S)
    {
        this->control.x -= 0.1;
        this->control.current_keyboard = this->control.last_keyboard;
    }
    if (this->control.current_keyboard == Input::Keyboard::A)
    {
        this->control.y += 0.1;
        this->control.current_keyboard = this->control.last_keyboard;
    }
    if (this->control.current_keyboard == Input::Keyboard::D)
    {
        this->control.y -= 0.1;
        this->control.current_keyboard = this->control.last_keyboard;
    }
    if (this->control.current_keyboard == Input::Keyboard::Q)
    {
        this->control.yaw += 0.1;
        this->control.current_keyboard = this->control.last_keyboard;
    }
    if (this->control.current_keyboard == Input::Keyboard::E)
    {
        this->control.yaw -= 0.1;
        this->control.current_keyboard = this->control.last_keyboard;
    }
    if (this->control.current_keyboard == Input::Keyboard::Space)
    {
        this->control.x = 0;
        this->control.y = 0;
        this->control.yaw = 0;
        this->control.current_keyboard = this->control.last_keyboard;
    }
    if (this->control.current_keyboard == Input::Keyboard::N || this->control.current_gamepad == Input::Gamepad::X)
    {
        this->control.navigation_mode = !this->control.navigation_mode;
        std::cout << std::endl << LOGGER::INFO << "Navigation mode: " << (this->control.navigation_mode ? "ON" : "OFF") << std::endl;
        this->control.current_keyboard = this->control.last_keyboard;
    }

    this->GetState(&this->robot_state);
    this->StateController(&this->robot_state, &this->robot_command);
    this->SetCommand(&this->robot_command);
    // 处理日志记录
    this->HandleLogging();
}

void RL_Real::RunModel()
{
    if (this->rl_init_done)
    {
        auto inference_start = std::chrono::high_resolution_clock::now();
        
        this->episode_length_buf += 1;
        this->obs.ang_vel = torch::tensor(this->robot_state.imu.gyroscope).unsqueeze(0);
        if (this->control.navigation_mode)
        {
#if !defined(USE_CMAKE) && defined(USE_ROS)
            this->obs.commands = torch::tensor({{this->cmd_vel.linear.x, this->cmd_vel.linear.y, this->cmd_vel.angular.z}});
#endif
        }
        else
        {
            this->obs.commands = torch::tensor({{this->control.x, this->control.y, this->control.yaw}});
        }
        this->obs.base_quat = torch::tensor(this->robot_state.imu.quaternion).unsqueeze(0);
        this->obs.dof_pos = torch::tensor(this->robot_state.motor_state.q).narrow(0, 0, this->params.num_of_dofs).unsqueeze(0);
        this->obs.dof_vel = torch::tensor(this->robot_state.motor_state.dq).narrow(0, 0, this->params.num_of_dofs).unsqueeze(0);

        this->obs.actions = this->Forward();
        this->ComputeOutput(this->obs.actions, this->output_dof_pos, this->output_dof_vel, this->output_dof_tau);

        auto inference_end = std::chrono::high_resolution_clock::now();
        // 计算推理时间并转换为毫秒
        auto inference_duration = std::chrono::duration_cast<std::chrono::microseconds>(inference_end - inference_start);
        this->last_inference_time_ = static_cast<double>(inference_duration.count()) * 1e-6; // 转换为秒

        if (this->output_dof_pos.defined() && this->output_dof_pos.numel() > 0)
        {
            output_dof_pos_queue.push(this->output_dof_pos);
        }
        if (this->output_dof_vel.defined() && this->output_dof_vel.numel() > 0)
        {
            output_dof_vel_queue.push(this->output_dof_vel);
        }
        if (this->output_dof_tau.defined() && this->output_dof_tau.numel() > 0)
        {
            output_dof_tau_queue.push(this->output_dof_tau);
        }

        // this->TorqueProtect(this->output_dof_tau);
        // this->AttitudeProtect(this->robot_state.imu.quaternion, 75.0f, 75.0f);

#ifdef CSV_LOGGER
        torch::Tensor tau_est = torch::tensor(this->robot_state.motor_state.tau_est).unsqueeze(0);
        this->CSVLogger(this->output_dof_tau, tau_est, this->obs.dof_pos, this->output_dof_pos, this->obs.dof_vel);
#endif
    }
    else
    {
        this->last_inference_time_ = 0.0; // RL未初始化时推理时间为0
    }
}

torch::Tensor RL_Real::Forward()
{
    torch::autograd::GradMode::set_enabled(false);

    // Try ONNX inference first if model is loaded
    if (this->onnx_engine.IsModelLoaded()) {
        // try {
            std::vector<float> clamped_obs_float = this->ComputeObservationFloat();
            float motion_step = static_cast<float>(this->episode_length_buf);

            std::vector<Ort::Value> policy_output;
            // if (!this->params.observations_history.empty()) {
            //     torch::Tensor obs_tensor = this->ComputeObservation();
            //     this->history_obs_buf.insert(obs_tensor);
            //     this->history_obs = this->history_obs_buf.get_obs_vec(this->params.observations_history);
            //     std::vector<float> history_obs_vec = this->TensorToVector(this->history_obs);
            //     std::vector<int64_t> input_shape = {1, static_cast<int64_t>(history_obs_vec.size())};
            //     policy_output = this->onnx_engine.Forward(history_obs_vec, motion_step);
            // } else {
                policy_output = this->onnx_engine.Forward(clamped_obs_float, motion_step);
            // }
            
            auto actions = this->onnx_engine.ExtractTensorData(policy_output[0]);
            auto body_quat_w = this->onnx_engine.ExtractTensorData(policy_output[4]);

            std::vector<float> motion_anchor_quat_w = {body_quat_w[28], 
                                                        body_quat_w[29],
                                                        body_quat_w[30],
                                                        body_quat_w[31]};

            this->ref_joint_pos = this->VectorToTensor(this->onnx_engine.ExtractTensorData(policy_output[1]), {1, 29});
            this->ref_joint_vel = this->VectorToTensor(this->onnx_engine.ExtractTensorData(policy_output[2]), {1, 29});
            this->ref_body_quat_w = this->VectorToTensor(motion_anchor_quat_w, {1, 4});

            // Convert back to tensor
            torch::Tensor actions_tensor = this->VectorToTensor(actions, {1, 29});

            // Apply clipping
            if (this->params.clip_actions_upper.numel() != 0 && this->params.clip_actions_lower.numel() != 0) {
                return torch::clamp(actions_tensor, this->params.clip_actions_lower, this->params.clip_actions_upper);
            } else {
                return actions_tensor;
            }
        // } catch (const std::exception& e) {
        //     std::cerr << "[Forward] ONNX inference failed: " << e.what() << ", falling back to PyTorch" << std::endl;
        // }
    }

    // Fallback to PyTorch inference only if PyTorch model is loaded
    if (!this->pytorch_model_loaded) {
        throw std::runtime_error("No valid inference model available (neither ONNX nor PyTorch model loaded)");
    }

    torch::Tensor clamped_obs = this->ComputeObservation();

    torch::Tensor actions;
    if (!this->params.observations_history.empty())
    {
        this->history_obs_buf.insert(clamped_obs);
        this->history_obs = this->history_obs_buf.get_obs_vec(this->params.observations_history);
        actions = this->model.forward({this->history_obs}).toTensor();
    }
    else
    {
        actions = this->model.forward({clamped_obs}).toTensor();
    }

    if (this->params.clip_actions_upper.numel() != 0 && this->params.clip_actions_lower.numel() != 0)
    {
        return torch::clamp(actions, this->params.clip_actions_lower, this->params.clip_actions_upper);
    }
    else
    {
        return actions;
    }
}

void RL_Real::Plot()
{
    this->plot_t.erase(this->plot_t.begin());
    this->plot_t.push_back(this->motiontime);
    plt::cla();
    plt::clf();
    for (int i = 0; i < this->params.num_of_dofs; ++i)
    {
        this->plot_real_joint_pos[i].erase(this->plot_real_joint_pos[i].begin());
        this->plot_target_joint_pos[i].erase(this->plot_target_joint_pos[i].begin());
        this->plot_real_joint_pos[i].push_back(this->unitree_low_state.motor_state()[i].q());
        this->plot_target_joint_pos[i].push_back(this->unitree_low_command.motor_cmd()[i].q());
        plt::subplot(this->params.num_of_dofs, 1, i + 1);
        plt::named_plot("_real_joint_pos", this->plot_t, this->plot_real_joint_pos[i], "r");
        plt::named_plot("_target_joint_pos", this->plot_t, this->plot_target_joint_pos[i], "b");
        plt::xlim(this->plot_t.front(), this->plot_t.back());
    }
    // plt::legend();
    plt::pause(0.0001);
}

uint32_t RL_Real::Crc32Core(uint32_t *ptr, uint32_t len)
{
    unsigned int xbit = 0;
    unsigned int data = 0;
    unsigned int CRC32 = 0xFFFFFFFF;
    const unsigned int dwPolynomial = 0x04c11db7;

    for (unsigned int i = 0; i < len; ++i)
    {
        xbit = 1 << 31;
        data = ptr[i];
        for (unsigned int bits = 0; bits < 32; bits++)
        {
            if (CRC32 & 0x80000000)
            {
                CRC32 <<= 1;
                CRC32 ^= dwPolynomial;
            }
            else
            {
                CRC32 <<= 1;
            }

            if (data & xbit)
            {
                CRC32 ^= dwPolynomial;
            }
            xbit >>= 1;
        }
    }

    return CRC32;
}

void RL_Real::InitLowCmd()
{
    for (int i = 0; i < 32; ++i)
    {
        this->unitree_low_command.motor_cmd()[i].mode() = (1); // 1:Enable, 0:Disable
        this->unitree_low_command.motor_cmd()[i].q() = (0);
        this->unitree_low_command.motor_cmd()[i].kp() = (0);
        this->unitree_low_command.motor_cmd()[i].dq() = (0);
        this->unitree_low_command.motor_cmd()[i].kd() = (0);
        this->unitree_low_command.motor_cmd()[i].tau() = (0);
    }
}

void RL_Real::LowStateHandler(const void *message)
{
    this->unitree_low_state = *(const LowState_ *)message;
}

void RL_Real::ImuTorsoHandler(const void *message)
{
    this->unitree_imu_torso = *(const IMUState_ *)message;
}

void signalHandler(int signum)
{
    if (g_rl_real_instance && g_rl_real_instance->logging_active_ && g_rl_real_instance->logger_->HasData()) {
        std::cout << std::endl << LOGGER::INFO << "💾 Saving log data before exit..." << std::endl;
        g_rl_real_instance->SaveCurrentLog();
    }
    exit(0);
}

void RL_Real::HandleLogging()
{
    // 检查是否应该开始记录日志（进入活动状态）
    if (!this->logging_active_ && this->rl_init_done && !this->previous_rl_init_done_) {
        std::cout << LOGGER::INFO << "🔴 Starting data logging - RL system initialized" << std::endl;
        this->logging_active_ = true;
        this->logger_->Clear(); // 清空之前的数据
        this->start_time_ = std::chrono::high_resolution_clock::now();
        this->last_log_time_ = this->start_time_;
    }
    
    // 检查是否应该停止记录并保存（回到passive模式）
    else if (this->logging_active_ && !this->rl_init_done && this->previous_rl_init_done_) {
        std::cout << LOGGER::INFO << "🟢 Stopping data logging - RL system deactivated" << std::endl;
        this->SaveCurrentLog();
        this->logging_active_ = false;
    }
    
    // 如果正在记录日志，则记录当前数据
    if (this->logging_active_) {
        this->RecordControlData();
    }
    
    this->previous_rl_init_done_ = this->rl_init_done;
}

void RL_Real::RecordControlData()
{
    auto current_time = std::chrono::high_resolution_clock::now();
    auto timestamp_duration = std::chrono::duration_cast<std::chrono::microseconds>(current_time - this->start_time_);
    auto loop_time_duration = std::chrono::duration_cast<std::chrono::microseconds>(current_time - this->last_log_time_);
    
    double timestamp = static_cast<double>(timestamp_duration.count()) / 1000000.0; // 转换为秒
    double loop_time = static_cast<double>(loop_time_duration.count()) / 1000000.0; // 转换为秒
    
    // 记录时间戳
    this->logger_->Record("timestamp", timestamp);
    this->logger_->Record("loop_time", loop_time);
    this->logger_->Record("motion_time", this->motiontime);
    
    // 记录RL推理相关数据
    this->logger_->Record("rl_inference_time", this->last_inference_time_);
    this->logger_->Record("episode_length_buf", this->episode_length_buf);
    this->logger_->Record("rl_init_done", this->rl_init_done ? 1.0 : 0.0);
    
    // 记录关节数据
    for (int i = 0; i < this->params.num_of_dofs; ++i) {
        
        // 从robot_command中获取目标值和增益
        // target_q = this->robot_command.motor_command.q[i] * 180.0 / M_PI; // 转换为度
        // kp = this->robot_command.motor_command.kp[i];
        // kd = this->robot_command.motor_command.kd[i];
        
        this->logger_->RecordJointData(
            i,
            this->robot_command.motor_command.q[i] * 180.0 / M_PI,
            this->robot_state.motor_state.q[i] * 180.0 / M_PI, // 转换为度
            this->robot_state.motor_state.dq[i] * 180.0 / M_PI, // 转换为度
            this->robot_command.motor_command.kp[i],
            this->robot_command.motor_command.kd[i],
            this->robot_state.motor_state.tau_est[i]
        );
    }
    
    // 记录控制命令
    this->logger_->Record("control_x", this->control.x);
    this->logger_->Record("control_y", this->control.y);
    this->logger_->Record("control_yaw", this->control.yaw);
    this->logger_->Record("navigation_mode", this->control.navigation_mode ? 1.0 : 0.0);
    
    // 记录IMU数据
    this->logger_->Record("imu_quat_w", this->robot_state.imu.quaternion[0]);
    this->logger_->Record("imu_quat_x", this->robot_state.imu.quaternion[1]);
    this->logger_->Record("imu_quat_y", this->robot_state.imu.quaternion[2]);
    this->logger_->Record("imu_quat_z", this->robot_state.imu.quaternion[3]);

    this->logger_->Record("imu_acc_x", this->robot_state.imu.accelerometer[0]);
    this->logger_->Record("imu_acc_y", this->robot_state.imu.accelerometer[1]);
    this->logger_->Record("imu_acc_z", this->robot_state.imu.accelerometer[2]);
    
    for (int i = 0; i < 3; ++i) {
        this->logger_->Record("imu_gyro_" + std::to_string(i), this->robot_state.imu.gyroscope[i]);
    }
    
    // 记录RL状态
    this->logger_->Record("rl_init_done", this->rl_init_done ? 1.0 : 0.0);
    this->logger_->Record("episode_length", this->episode_length_buf);
    
    this->last_log_time_ = current_time;
}

void RL_Real::SaveCurrentLog()
{
    if (this->logger_->HasData()) {
        try {
            this->logger_->SaveToCSV();
            std::cout << LOGGER::INFO << "📊 Log data saved successfully!" << std::endl;
        } catch (const std::exception& e) {
            std::cout << LOGGER::ERROR << "❌ Error saving log: " << e.what() << std::endl;
        }
    } else {
        std::cout << LOGGER::WARNING << "⚠️  No data to save" << std::endl;
    }
}



#if !defined(USE_CMAKE) && defined(USE_ROS)
void RL_Real::CmdvelCallback(
#if defined(USE_ROS1) && defined(USE_ROS)
    const geometry_msgs::Twist::ConstPtr &msg
#elif defined(USE_ROS2) && defined(USE_ROS)
    const geometry_msgs::msg::Twist::SharedPtr msg
#endif
)
{
    this->cmd_vel = *msg;
}
#endif

#if defined(USE_ROS1) && defined(USE_ROS)
void signalHandler(int signum)
{
    ros::shutdown();
    exit(0);
}
#endif

int main(int argc, char **argv)
{
    if (argc < 2)
    {
        std::cout << "Usage: " << argv[0] << " networkInterface [wheel]" << std::endl;
        exit(-1);
    }
    ChannelFactory::Instance()->Init(0, argv[1]);
#if defined(USE_ROS1) && defined(USE_ROS)
    signal(SIGINT, signalHandler);
    ros::init(argc, argv, "rl_sar");
    RL_Real rl_sar;
    ros::spin();
#elif defined(USE_ROS2) && defined(USE_ROS)
    rclcpp::init(argc, argv);
    rclcpp::spin(std::make_shared<RL_Real>());
    rclcpp::shutdown();
#elif defined(USE_CMAKE) || !defined(USE_ROS)
    signal(SIGINT, signalHandler);
    RL_Real rl_sar;
    g_rl_real_instance = &rl_sar;
    while (1) { sleep(10); }
#endif
    return 0;
}
