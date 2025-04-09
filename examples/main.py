# -*- coding: utf-8 -*-
"""
@File      : main.py
@Time      : 2025-04-08 17:39
@Author    : Xiaodong Zheng
@Email     : zxd_xjtu@stu.xjtu.edu.cn
@Description: 此 Python 脚本是 PowerZoo 的主脚本，用于解析命令行参数并打印解析结果。
- 关键组件及职责：
  - `argparse` 库：用于创建命令行参数解析器。
  - `main` 函数：
    - 创建解析器对象，设置脚本描述为 'PowerZoo Main Script'。
    - 添加多个命令行参数，包括配置文件路径、日志文件路径等。
    - 解析命令行参数并打印解析结果。
- 工作流程：
  1. 导入 `argparse` 库。
  2. 定义 `main` 函数。
  3. 在 `main` 函数中创建解析器，添加参数，解析参数并打印。
  4. 若脚本作为主程序运行，则调用 `main` 函数。
"""
import argparse

def main():
    # 创建一个解析器对象
    parser = argparse.ArgumentParser(description='PowerZoo Main Script')

    # 添加参数
    parser.add_argument('--config', type=str, help='Path to the configuration file')
    parser.add_argument('--log', type=str, help='Path to the log file')
    parser.add_argument('--verbose', action='store_true', help='Increase output verbosity')
    parser.add_argument('--mode', type=str, choices=['train', 'test', 'eval'], help='Mode to run the script in')
    parser.add_argument('--epochs', type=int, default=10, help='Number of epochs to run')
    
    # 解析参数
    args = parser.parse_args()

    # 打印解析到的参数
    print(args)

if __name__ == "__main__":
    main()
