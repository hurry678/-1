"""查看服务器运行状态。"""

from __future__ import annotations

import paramiko


HOST = "192.168.182.155"
USER = "hy"
PASSWORD = "admin@4567"
REMOTE_ROOT = "/data2/hy/cts/shumo1"


def main() -> None:
    client = paramiko.SSHClient()
    client.set_missing_host_key_policy(paramiko.AutoAddPolicy())
    client.connect(HOST, username=USER, password=PASSWORD, timeout=30)

    def run(command: str) -> tuple[str, str]:
        _stdin, stdout, stderr = client.exec_command(command, timeout=60)
        return (
            stdout.read().decode("utf-8", errors="replace"),
            stderr.read().decode("utf-8", errors="replace"),
        )

    commands = [
        f"cat {REMOTE_ROOT}/run.pid 2>/dev/null || echo '无 pid 文件'",
        f"ps -p $(cat {REMOTE_ROOT}/run.pid 2>/dev/null) -o pid,etime,pcpu,pmem,cmd 2>/dev/null || echo '进程未运行'",
        f"tail -n 8 {REMOTE_ROOT}/run.log 2>/dev/null || echo '无日志'",
        f"ls -la {REMOTE_ROOT}/output/problem2/v2_standard_grid/200x100/M1/run_1 2>/dev/null || echo '无输出目录'",
    ]
    for command in commands:
        out, err = run(command)
        print(f"$ {command}")
        print(out.strip())
        if err.strip():
            print("[stderr]", err.strip()[:200])
        print("-" * 50)
    client.close()


if __name__ == "__main__":
    main()
