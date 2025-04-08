import torch
import torch.nn as nn
import streamlit as st
from PIL import Image
import torchvision.transforms as transforms
import numpy as np
from torchvision.models import resnet18
import os
import time
import glob
from pathlib import Path
import pandas as pd
from io import BytesIO
import cv2
import tempfile
import hashlib
import json
import base64
from datetime import datetime


# ===== 用户认证系统 =====

class AuthenticationSystem:
    """用户认证系统，处理登录、注册和用户信息管理"""

    def __init__(self, user_db_path="users.json"):
        """初始化认证系统

        Args:
            user_db_path: 用户数据库文件路径
        """
        self.user_db_path = user_db_path
        self.users = self._load_users()

    def _load_users(self):
        """加载用户数据"""
        if os.path.exists(self.user_db_path):
            try:
                with open(self.user_db_path, "r") as f:
                    return json.load(f)
            except Exception as e:
                st.error(f"加载用户数据失败: {str(e)}")
                return {}
        return {}

    def _save_users(self):
        """保存用户数据"""
        try:
            with open(self.user_db_path, "w") as f:
                json.dump(self.users, f, indent=4)
            return True
        except Exception as e:
            st.error(f"保存用户数据失败: {str(e)}")
            return False

    def _hash_password(self, password):
        """密码哈希处理"""
        return hashlib.sha256(password.encode()).hexdigest()

    def register_user(self, username, password, fullname, avatar=None):
        """注册新用户

        Args:
            username: 用户名
            password: 密码
            fullname: 用户全名
            avatar: 用户头像图像

        Returns:
            bool: 注册是否成功
        """
        # 检查用户名是否已存在
        if username in self.users:
            return False, "用户名已存在"

        # 处理头像
        avatar_data = None
        if avatar is not None:
            try:
                # 调整头像大小并转换为base64
                avatar_img = Image.open(avatar)
                avatar_img = avatar_img.resize((128, 128))
                buffered = BytesIO()
                avatar_img.save(buffered, format="PNG")
                avatar_data = base64.b64encode(buffered.getvalue()).decode()
            except Exception as e:
                st.error(f"处理头像失败: {str(e)}")

        # 创建用户记录
        self.users[username] = {
            "password_hash": self._hash_password(password),
            "fullname": fullname,
            "avatar": avatar_data,
            "created_at": datetime.now().isoformat(),
            "last_login": None
        }

        # 保存用户数据
        if self._save_users():
            return True, "注册成功"
        return False, "注册失败，无法保存用户数据"

    def authenticate(self, username, password):
        """验证用户登录

        Args:
            username: 用户名
            password: 密码

        Returns:
            bool: 登录是否成功
        """
        if username not in self.users:
            return False, "用户不存在"

        stored_hash = self.users[username]["password_hash"]
        input_hash = self._hash_password(password)

        if stored_hash == input_hash:
            # 更新最后登录时间
            self.users[username]["last_login"] = datetime.now().isoformat()
            self._save_users()
            return True, "登录成功"

        return False, "密码错误"

    def get_user_avatar(self, username):
        """获取用户头像数据

        Args:
            username: 用户名

        Returns:
            str: base64编码的头像数据
        """
        if username in self.users and self.users[username].get("avatar"):
            return self.users[username]["avatar"]
        return None

    def get_user_info(self, username):
        """获取用户信息

        Args:
            username: 用户名

        Returns:
            dict: 用户信息
        """
        if username in self.users:
            info = self.users[username].copy()
            # 删除密码哈希，不返回给前端
            if "password_hash" in info:
                del info["password_hash"]
            return info
        return None


# 显示用户登录界面
def show_login_page(auth_system):
    """显示登录界面"""
    st.title("工业热误差预测系统 - 用户登录")

    login_username = st.text_input("用户名", key="login_username")
    login_password = st.text_input("密码", type="password", key="login_password")

    login_button = st.button("登录")

    if login_button:
        if not login_username or not login_password:
            st.error("请输入用户名和密码")
        else:
            success, message = auth_system.authenticate(login_username, login_password)
            if success:
                st.session_state["authenticated"] = True
                st.session_state["username"] = login_username
                st.success(message)
                # 强制页面重新加载，进入主系统
                st.rerun()
            else:
                st.error(message)

    st.markdown("没有账号？[点击注册](?page=register)")


# 显示用户注册界面
def show_register_page(auth_system):
    """显示注册界面"""
    st.title("工业热误差预测系统 - 用户注册")

    reg_username = st.text_input("用户名 (必填)", key="reg_username")
    reg_password = st.text_input("密码 (必填)", type="password", key="reg_password")
    reg_confirm_password = st.text_input("确认密码 (必填)", type="password", key="reg_confirm_password")
    reg_fullname = st.text_input("姓名 (必填)", key="reg_fullname")
    reg_avatar = st.file_uploader("上传头像图片 (可选)", type=["jpg", "jpeg", "png"])

    if reg_avatar:
        try:
            avatar_preview = Image.open(reg_avatar)
            st.image(avatar_preview, caption="头像预览", width=128)
        except Exception as e:
            st.error(f"头像预览失败: {str(e)}")

    register_button = st.button("注册")

    if register_button:
        if not reg_username or not reg_password or not reg_fullname:
            st.error("请填写所有必填字段")
        elif reg_password != reg_confirm_password:
            st.error("两次输入的密码不一致")
        else:
            success, message = auth_system.register_user(
                reg_username, reg_password, reg_fullname, reg_avatar
            )
            if success:
                st.success(message + " 请使用新账号登录。")
            else:
                st.error(message)

    st.markdown("已有账号？[点击登录](?page=login)")


# 显示用户信息
def display_user_info(auth_system, username):
    """在侧边栏显示用户信息"""
    user_info = auth_system.get_user_info(username)

    if user_info:
        st.sidebar.subheader(f"👋 欢迎, {user_info['fullname']}")

        # 显示用户头像
        if user_info.get("avatar"):
            try:
                avatar_bytes = base64.b64decode(user_info["avatar"])
                avatar_image = Image.open(BytesIO(avatar_bytes))
                st.sidebar.image(avatar_image, width=100)
            except Exception as e:
                st.sidebar.warning("无法加载头像")

        # 显示上次登录时间
        if user_info.get("last_login"):
            try:
                last_login = datetime.fromisoformat(user_info["last_login"])
                st.sidebar.text(f"上次登录: {last_login.strftime('%Y-%m-%d %H:%M')}")
            except:
                pass

        # 退出按钮
        if st.sidebar.button("退出登录"):
            st.session_state["authenticated"] = False
            st.session_state["username"] = None
            st.rerun()
    else:
        st.sidebar.error("无法加载用户信息")


# ==== 热误差预测模型 ====

class ThermalErrorModel(nn.Module):
    """热误差预测模型 - 与训练代码保持完全一致"""

    def __init__(self):
        super().__init__()
        base_model = resnet18(weights=None)
        self.feature_extractor = nn.Sequential(*list(base_model.children())[:-2])

        # 增强回归头 - 与训练代码保持一致
        self.regressor = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Flatten(),
            nn.Linear(512, 512),
            nn.BatchNorm1d(512),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(512, 256),
            nn.ReLU(),
            nn.Linear(256, 1)
        )

    def forward(self, x):
        features = self.feature_extractor(x)
        return self.regressor(features).squeeze()


def load_model(model_file):
    """加载训练好的模型"""
    model = ThermalErrorModel()

    try:
        # 读取上传的模型文件
        model_bytes = model_file.read()
        # 加载模型参数
        state_dict = torch.load(BytesIO(model_bytes), map_location='cpu')

        # 处理可能的'module.'前缀（用于兼容分布式训练）
        state_dict = {k.replace('module.', ''): v for k, v in state_dict.items()}

        model.load_state_dict(state_dict)
        model.eval()
        return model

    except Exception as e:
        st.error(f"""
        ## 模型加载失败
        **错误原因**: {str(e)}
        """)
        return None


def preprocess_image(image):
    """图像预处理 - 与训练代码中的预处理保持一致"""
    try:
        # 确保图像是RGB格式
        if isinstance(image, np.ndarray):
            # 如果是OpenCV图像(numpy array)，转换为PIL
            if image.shape[2] == 3:  # 确保有3个通道
                image = Image.fromarray(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
            else:
                image = Image.fromarray(image)
        else:
            # 如果已经是PIL图像，确保是RGB
            image = image.convert('RGB')

        # 使用与训练相同的预处理流程
        transform = transforms.Compose([
            transforms.Resize(224),  # ResNet标准输入尺寸
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])  # ImageNet标准
        ])

        tensor = transform(image).unsqueeze(0)  # 增加批次维度
        return tensor

    except Exception as e:
        st.error(f"预处理失败: {str(e)}")
        return None


def predict_single_image(model, image):
    """预测单张图像的热误差"""
    input_tensor = preprocess_image(image)
    if input_tensor is None:
        return None

    try:
        with torch.no_grad():
            prediction = model(input_tensor).item()
        return prediction
    except Exception as e:
        st.error(f"预测失败: {str(e)}")
        return None


def process_folder(model, folder_path, interval=0.01):
    """处理文件夹中的所有图像，每隔interval秒处理一张"""
    # 获取所有图像文件
    image_extensions = ['*.jpg', '*.jpeg', '*.png']
    image_files = []

    for ext in image_extensions:
        image_files.extend(glob.glob(os.path.join(folder_path, ext)))
        image_files.extend(glob.glob(os.path.join(folder_path, ext.upper())))

    if not image_files:
        st.warning(f"文件夹 '{folder_path}' 中未找到图像文件")
        return

    # 排序确保处理顺序一致
    image_files.sort()

    # 创建结果容器
    results = []

    # 创建进度条
    progress_bar = st.progress(0)
    status_text = st.empty()
    current_image = st.empty()
    result_text = st.empty()

    # 处理每张图像
    for i, img_path in enumerate(image_files):
        # 更新进度
        progress = int(100 * (i / len(image_files)))
        progress_bar.progress(progress)
        status_text.text(f"处理中: {i + 1}/{len(image_files)} - {os.path.basename(img_path)}")

        try:
            # 读取图像
            image = Image.open(img_path)

            # 显示当前图像
            current_image.image(image, caption=f"正在处理: {os.path.basename(img_path)}", width=300)

            # 预测热误差
            prediction = predict_single_image(model, image)

            if prediction is not None:
                # 更新结果显示
                result_text.metric(
                    "当前热误差预测",
                    f"{abs(prediction):.2f} nm",
                )

                # 添加到结果列表
                results.append({
                    "文件名": os.path.basename(img_path),
                    "路径": img_path,
                    "预测热误差(nm)": prediction,
                    "预测绝对热误差(nm)": abs(prediction)
                })

            # 等待指定的间隔时间
            time.sleep(interval)

        except Exception as e:
            st.error(f"处理图像 {img_path} 时出错: {str(e)}")

    # 完成进度条
    progress_bar.progress(100)
    status_text.text("处理完成!")

    # 将结果转换为DataFrame并返回
    if results:
        return pd.DataFrame(results)
    else:
        return None


def process_video(model, video_file, frame_interval=5, display_interval=0.01):
    """处理视频文件并进行热误差预测

    Args:
        model: 预测模型
        video_file: 上传的视频文件
        frame_interval: 每隔多少帧处理一次
        display_interval: 显示结果的时间间隔(秒)
    """
    # 创建临时文件保存上传的视频
    tfile = tempfile.NamedTemporaryFile(delete=False)
    tfile.write(video_file.read())
    tfile_path = tfile.name
    tfile.close()

    # 打开视频文件
    cap = cv2.VideoCapture(tfile_path)
    if not cap.isOpened():
        st.error("无法打开视频文件")
        os.unlink(tfile_path)  # 删除临时文件
        return None

    # 获取视频信息
    frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    fps = cap.get(cv2.CAP_PROP_FPS)
    duration = frame_count / fps if fps > 0 else 0

    st.info(f"视频信息: {frame_count} 帧, {fps:.2f} FPS, 时长约 {duration:.2f} 秒")

    # 创建结果容器
    results = []

    # 创建UI元素
    progress_bar = st.progress(0)
    status_text = st.empty()
    col1, col2 = st.columns(2)
    frame_display = col1.empty()
    result_display = col2.empty()
    chart_placeholder = st.empty()

    # 预测记录
    frame_times = []  # 记录帧时间点
    error_values = []  # 记录误差值
    timestamps = []  # 记录时间戳

    # 初始化当前帧计数
    current_frame = 0
    total_frames_processed = 0

    try:
        # 处理视频帧
        while True:
            ret, frame = cap.read()
            if not ret:
                break

            # 只处理指定间隔的帧
            if current_frame % frame_interval == 0:
                # 更新进度
                progress = min(100, int(100 * (current_frame / frame_count)))
                progress_bar.progress(progress)

                # 计算时间戳
                timestamp = current_frame / fps if fps > 0 else 0
                status_text.text(f"处理中: 帧 {current_frame}/{frame_count} - 时间: {timestamp:.2f}秒")

                # 显示当前帧
                frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                frame_display.image(frame_rgb, caption=f"当前帧: {current_frame}", width=400)

                # 预测热误差
                prediction = predict_single_image(model, frame)

                if prediction is not None:
                    # 更新结果显示
                    result_display.metric(
                        "当前热误差预测",
                        f"{abs(prediction):.2f} nm",
                    )

                    # 添加到结果列表
                    results.append({
                        "帧号": current_frame,
                        "时间戳(秒)": timestamp,
                        "预测热误差(nm)": prediction,
                        "预测绝对热误差(nm)": abs(prediction)
                    })

                    # 更新图表数据
                    frame_times.append(current_frame)
                    error_values.append(prediction)
                    timestamps.append(timestamp)

                    # 每10帧更新一次图表
                    if len(frame_times) % 10 == 0 or len(frame_times) == 1:
                        # 创建图表数据
                        chart_data = pd.DataFrame({
                            "时间(秒)": timestamps,
                            "热误差(nm)": error_values
                        })
                        # 更新图表
                        chart_placeholder.line_chart(chart_data, x="时间(秒)", y="热误差(nm)")

                total_frames_processed += 1

                # 等待显示间隔
                time.sleep(display_interval)

            current_frame += 1

    except Exception as e:
        st.error(f"处理视频时出错: {str(e)}")

    finally:
        # 释放资源
        cap.release()
        os.unlink(tfile_path)  # 删除临时文件

    # 完成进度条
    progress_bar.progress(100)
    status_text.text(f"处理完成! 共处理 {total_frames_processed} 帧")

    # 将结果转换为DataFrame并返回
    if results:
        results_df = pd.DataFrame(results)

        # 显示统计信息
        st.subheader("视频热误差统计")
        col1, col2, col3 = st.columns(3)
        with col1:
            st.metric("平均热误差", f"{results_df['预测热误差(nm)'].mean():.2f} nm")
        with col2:
            st.metric("平均绝对热误差", f"{results_df['预测绝对热误差(nm)'].mean():.2f} nm")
        with col3:
            st.metric("最大绝对热误差", f"{results_df['预测绝对热误差(nm)'].max():.2f} nm")

        # 显示最终热误差图表
        st.subheader("热误差变化趋势")
        st.line_chart(results_df, x="时间戳(秒)", y="预测热误差(nm)")

        return results_df
    else:
        st.warning("未能从视频中获取有效预测结果")
        return None


# ===== 主程序入口 =====

def main():
    # 设置页面配置
    st.set_page_config(
        page_title="静压主轴热伸长误差预测系统",
        page_icon="🌡️",
        layout="wide"
    )

    # 初始化会话状态
    if "authenticated" not in st.session_state:
        st.session_state["authenticated"] = False
    if "username" not in st.session_state:
        st.session_state["username"] = None

    # 初始化认证系统
    auth_system = AuthenticationSystem()

    # 获取当前页面
    page = st.query_params.get("page", ["login"])[0]

    # 检查用户是否已经登录
    if not st.session_state["authenticated"]:
        if page == "login":
            # 显示登录页面
            show_login_page(auth_system)
        elif page == "register":
            # 显示注册页面
            show_register_page(auth_system)
    else:
        # 显示主应用程序
        username = st.session_state["username"]

        # 在侧边栏中显示用户信息
        display_user_info(auth_system, username)

        # 应用标题
        st.title("静压主轴热伸长误差预测系统")
        st.sidebar.title("配置面板")

        # 上传模型文件
        model_file = st.sidebar.file_uploader(
            "上传模型文件 (.pth)",
            type=["pth"],
            help="选择训练好的热误差预测模型文件"
        )

        # 添加功能选择
        if model_file:
            # 加载模型
            model = load_model(model_file)
            if model is None:
                st.stop()

            # 模式选择
            mode = st.sidebar.radio(
                "选择分析模式",
                ["单张图片分析", "文件夹批量分析", "视频动态检测"],
                index=0
            )

            if mode == "单张图片分析":
                # 单张图片分析模式
                uploaded_file = st.file_uploader(
                    "上传热像图",
                    type=["jpg", "jpeg", "png"],
                    help="上传需要进行热误差预测的图像"
                )

                if uploaded_file:
                    cols = st.columns([3, 2])

                    with cols[0]:
                        try:
                            image = Image.open(uploaded_file)
                            # 显示原始图像
                            st.image(image, caption="原始热像图", use_container_width=True)
                        except Exception as e:
                            st.error(f"图像加载失败: {str(e)}")
                            st.stop()

                    with cols[1]:
                        st.subheader("分析结果")
                        with st.spinner("正在处理..."):
                            prediction = predict_single_image(model, image)

                            if prediction is not None:
                                # 显示结果
                                st.metric(
                                    "预测热误差",
                                    f"{abs(prediction):.2f} nm",
                                )


            elif mode == "文件夹批量分析":
                # 文件夹批量分析模式
                st.subheader("文件夹批量分析")

                # 输入文件夹路径
                folder_path = st.text_input("输入图片文件夹路径", "")

                # 设置处理间隔（默认0.1秒）
                interval = st.slider("处理间隔(秒)", 0.01, 2.0, 0.01, 0.01)

                if folder_path and os.path.isdir(folder_path):
                    if st.button("开始批量分析"):
                        st.subheader("批量分析进度")

                        # 处理文件夹
                        results_df = process_folder(model, folder_path, interval)

                        if results_df is not None and not results_df.empty:
                            # 显示结果表格
                            st.subheader("分析结果")
                            st.dataframe(results_df)

                            # 计算统计数据
                            st.subheader("统计信息")
                            col1, col2, col3 = st.columns(3)
                            with col1:
                                st.metric("平均热误差", f"{results_df['预测热误差(nm)'].mean():.2f} nm")
                            with col2:
                                st.metric("平均绝对热误差", f"{results_df['预测绝对热误差(nm)'].mean():.2f} nm")
                            with col3:
                                st.metric("最大绝对热误差", f"{results_df['预测绝对热误差(nm)'].max():.2f} nm")

                            # 提供下载结果的选项
                            csv = results_df.to_csv(index=False).encode('utf-8')
                            st.download_button(
                                "下载分析结果CSV",
                                csv,
                                "热误差分析结果.csv",
                                "text/csv",
                                key='download-csv'
                            )
                elif folder_path:
                    st.error(f"文件夹路径 '{folder_path}' 不存在或无效")

            else:
                # 视频动态检测模式
                st.subheader("视频动态检测")

                # 上传视频文件
                video_file = st.file_uploader(
                    "上传视频文件",
                    type=["mp4", "avi", "mov", "mkv"],
                    help="上传包含热像图的视频文件进行动态分析"
                )

                if video_file:
                    # 配置选项
                    st.sidebar.subheader("视频处理设置")

                    # 视频预览
                    video_preview = st.video(video_file)

                    # 处理间隔设置
                    col1, col2 = st.columns(2)
                    with col1:
                        frame_interval = st.slider(
                            "处理帧间隔",
                            1, 30, 5,
                            help="每隔多少帧处理一次，数值越大处理速度越快但精度降低"
                        )

                    with col2:
                        display_interval = st.slider(
                            "显示更新间隔(秒)",
                            0.0, 1.0, 0.01, 0.01,
                            help="结果显示的更新频率，值越小更新越快但可能影响性能"
                        )

                    # 开始处理
                    if st.button("开始视频分析"):
                        st.subheader("视频分析进度")

                        # 处理视频
                        results_df = process_video(
                            model,
                            video_file,
                            frame_interval=frame_interval,
                            display_interval=display_interval
                        )

                        if results_df is not None and not results_df.empty:
                            # 显示结果表格
                            st.subheader("详细分析结果")
                            st.dataframe(results_df)

                            # 提供下载结果的选项
                            csv = results_df.to_csv(index=False).encode('utf-8')
                            st.download_button(
                                "下载分析结果CSV",
                                csv,
                                "视频热误差分析结果.csv",
                                "text/csv",
                                key='download-video-csv'
                            )
        else:
            # 显示欢迎信息和使用说明
            st.markdown(f"""
            ## 欢迎使用静压主轴热伸长误差预测系统

            该系统可以帮助您分析热像图并预测热误差值。

            ### 使用流程:
            1. 在左侧配置面板上传预训练的模型文件 (.pth格式)
            2. 选择您需要的分析模式:
               - **单张图片分析**: 适合单张热像图的快速分析
               - **文件夹批量分析**: 适合批量处理一个文件夹中的所有图像
               - **视频动态检测**: 适合对视频流进行实时热误差监测

            ### 关于用户账户:
            - 您当前已登录账户: **{st.session_state["username"]}**
            - 您可以在左侧边栏查看您的账户信息和头像
            - 使用左侧的"退出登录"按钮可以安全退出系统

            ### 系统日志记录:
            - 所有分析操作都会被记录在您的用户账户中
            - 您的所有分析结果可以导出保存
            """)


if __name__ == "__main__":
    main()
