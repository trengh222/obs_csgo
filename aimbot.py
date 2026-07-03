#!/usr/bin/env python3
"""
CSGO 自动瞄准 - 平滑追踪头部
使用 YOLOv8 检测目标头部，指数衰减平滑移动鼠标
支持热键开关 和 视频测试模式
"""

import os
import cv2
import numpy as np
import time
import math
import json
import threading
from ultralytics import YOLO

# ============== 配置 ==============
PROJECT_DIR = os.path.dirname(os.path.abspath(__file__))
MODEL_PATH = os.path.join(PROJECT_DIR, "runs/detect/runs/detect/csgo_detect-4/weights/best.pt")
OUTPUT_DIR = os.path.join(PROJECT_DIR, "output")
SCREEN_W, SCREEN_H = 1920, 1080
MONITOR = {"top": 0, "left": 0, "width": SCREEN_W, "height": SCREEN_H}
HEAD_CLASSES = [1, 3]       # CT_head=1, T_head=3
CONF_THRESHOLD = 0.5
# ====================================


# ============== 热键监听 ==============
class HotkeyListener:
    """热键控制开关，支持 macOS (Quartz) 和通用方案 (pynput)"""

    def __init__(self, toggle_key="x"):
        self.enabled = False
        self.toggle_key = toggle_key.lower()
        self._running = True
        self._thread = threading.Thread(target=self._listen, daemon=True)
        self._method = None

        # macOS 原生方案 (无需额外依赖)
        try:
            import Quartz
            self._method = "quartz"
            print(f"✅ 热键监听: macOS Quartz (按 [{toggle_key}] 切换)")
        except ImportError:
            try:
                from pynput import keyboard
                self._method = "pynput"
                print(f"✅ 热键监听: pynput (按 [{toggle_key}] 切换)")
            except ImportError:
                print("⚠️  未找到热键库，使用 pyautogui 鼠标控制")
                self._method = None

        self._thread.start()

    def _listen(self):
        if self._method == "quartz":
            self._listen_quartz()
        elif self._method == "pynput":
            self._listen_pynput()

    def _listen_quartz(self):
        import Quartz

        target_keycode = self._char_to_keycode(self.toggle_key)

        def callback(proxy, event_type, event, refcon):
            if event_type == Quartz.kCGEventKeyDown:
                keycode = Quartz.CGEventGetIntegerValueField(event, Quartz.kCGKeyboardEventKeycode)
                if keycode == target_keycode:
                    self.enabled = not self.enabled
                    state = "开启" if self.enabled else "关闭"
                    print(f"\n🎯 自动瞄准已{state}")
            return event

        tap = Quartz.CGEventTapCreate(
            Quartz.kCGSessionEventTap,
            Quartz.kCGHeadInsertEventTap,
            Quartz.kCGEventTapOptionListenOnly,
            Quartz.CGEventMaskBit(Quartz.kCGEventKeyDown),
            callback,
            None,
        )
        run_loop_source = Quartz.CFMachPortCreateRunLoopSource(None, tap, 0)
        Quartz.CFRunLoopAddSource(Quartz.CFRunLoopGetCurrent(), run_loop_source, Quartz.kCFRunLoopCommonModes)
        Quartz.CGEventTapEnable(tap, True)
        Quartz.CFRunLoopRun()

    def _listen_pynput(self):
        from pynput import keyboard

        hotkey = keyboard.Key.from_char(self.toggle_key)

        def on_press(key):
            try:
                if hasattr(key, 'char') and key.char and key.char.lower() == self.toggle_key:
                    self.enabled = not self.enabled
                    state = "开启" if self.enabled else "关闭"
                    print(f"\n🎯 自动瞄准已{state}")
            except AttributeError:
                pass

        with keyboard.Listener(on_press=on_press) as listener:
            listener.join()

    @staticmethod
    def _char_to_keycode(char):
        """macOS 字符到 keycode 映射"""
        keymap = {
            'a': 0, 's': 1, 'd': 2, 'f': 3, 'h': 4, 'g': 5, 'z': 6, 'x': 7,
            'c': 8, 'v': 9, 'b': 11, 'q': 12, 'w': 13, 'e': 14, 'r': 15,
            'y': 16, 't': 17, '1': 18, '2': 19, '3': 20, '4': 21, '6': 22,
            '5': 23, '=': 24, '9': 25, '7': 26, '-': 27, '8': 28, '0': 29,
            ']': 30, 'o': 31, 'u': 32, '[': 33, 'i': 34, 'p': 35,
            'n': 45, 'm': 46, '.': 47, 'l': 37, 'j': 38, 'k': 40,
            'space': 49, 'tab': 48, 'return': 36, 'escape': 53,
        }
        return keymap.get(char, 0)


# ============== 平滑鼠标控制 ==============
class SmoothAim:
    """平滑鼠标控制器"""

    def __init__(self, smoothing=0.15, dead_zone=3.0):
        self.smoothing = smoothing
        self.dead_zone = dead_zone
        self.move_history = []  # 记录移动轨迹用于调试

        try:
            import Quartz
            self._move = self._move_quartz
            self._get_pos = self._get_pos_quartz
            print("✅ 鼠标控制: macOS Quartz")
        except ImportError:
            try:
                import pyautogui
                pyautogui.FAILSAFE = False
                self._move = self._move_pyautogui
                self._get_pos = self._get_pos_pyautogui
                print("✅ 鼠标控制: pyautogui")
            except ImportError:
                raise RuntimeError("请安装 pyautogui: pip install pyautogui")

    # ---- macOS 原生鼠标控制 ----
    def _move_quartz(self, x, y):
        import Quartz
        point = Quartz.CGPointMake(int(x), int(y))
        event = Quartz.CGEventCreateMouseEvent(
            None, Quartz.kCGEventMouseMoved, point, 0
        )
        Quartz.CGEventPost(Quartz.kCGHIDEventTap, event)

    def _get_pos_quartz(self):
        import Quartz
        pos = Quartz.CGEventGetLocation(Quartz.CGEventCreate(None))
        return pos.x, pos.y

    # ---- pyautogui 备选 ----
    def _move_pyautogui(self, x, y):
        import pyautogui
        pyautogui.moveTo(int(x), int(y), duration=0)

    def _get_pos_pyautogui(self):
        import pyautogui
        return pyautogui.position()

    def update(self, target_x, target_y):
        """指数衰减平滑 + 速度限制"""
        cur_x, cur_y = self._get_pos()

        dx = target_x - cur_x
        dy = target_y - cur_y
        dist = math.sqrt(dx * dx + dy * dy)

        if dist < self.dead_zone:
            return cur_x, cur_y

        # 动态平滑 - 加快响应速度
        dynamic_smooth = self.smoothing
        if dist > 100:
            dynamic_smooth = min(0.5, self.smoothing + dist * 0.0015)
        elif dist < 30:
            dynamic_smooth = max(0.08, self.smoothing * 0.6)

        new_x = cur_x + dx * dynamic_smooth
        new_y = cur_y + dy * dynamic_smooth

        # 速度限制 - 提高最大速度
        max_speed = max(120, dist * 0.7)
        move_dx = new_x - cur_x
        move_dy = new_y - cur_y
        move_dist = math.sqrt(move_dx**2 + move_dy**2)

        if move_dist > max_speed:
            scale = max_speed / move_dist
            new_x = cur_x + move_dx * scale
            new_y = cur_y + move_dy * scale

        self.move_history.append((new_x, new_y))
        if len(self.move_history) > 120:
            self.move_history.pop(0)

        return new_x, new_y

    def move_to(self, x, y):
        self._move(x, y)


# ============== 检测逻辑 ==============
def find_nearest_head(results, ref_x, ref_y, head_classes=None, conf_threshold=0.5):
    """从检测结果中找离参考点最近的头部"""
    if head_classes is None:
        head_classes = HEAD_CLASSES
    best = None
    min_dist = float('inf')

    for box in results[0].boxes:
        cls = int(box.cls[0])
        if cls not in head_classes:
            continue
        if float(box.conf[0]) < conf_threshold:
            continue

        x1, y1, x2, y2 = box.xyxy[0].tolist()
        head_cx = (x1 + x2) / 2
        head_cy = y1 + (y2 - y1) * 0.3

        dist = math.sqrt((head_cx - ref_x)**2 + (head_cy - ref_y)**2)
        if dist < min_dist:
            min_dist = dist
            best = (head_cx, head_cy, dist)

    return best


def get_all_detections(results, conf_threshold=0.5):
    """获取所有检测结果"""
    detections = []
    for box in results[0].boxes:
        cls = int(box.cls[0])
        conf = float(box.conf[0])
        if conf < conf_threshold:
            continue
        x1, y1, x2, y2 = box.xyxy[0].tolist()
        cx = (x1 + x2) / 2
        cy = (y1 + y2) / 2
        head_cy = y1 + (y2 - y1) * 0.3
        detections.append({
            "cls": cls, "conf": conf,
            "box": (x1, y1, x2, y2),
            "center": (cx, cy),
            "head": (cx, head_cy),
        })
    return detections


# ============== 模式1: 实时截屏瞄准 ==============
def run_aimbot(args):
    """实时截屏 + 检测 + 鼠标控制"""
    import mss

    print("加载模型...")
    model = YOLO(args.model)
    aim = SmoothAim(smoothing=args.smoothing, dead_zone=args.dead_zone)
    hotkey = HotkeyListener(toggle_key=args.toggle_key)

    print(f"目标类别: {HEAD_CLASSES} (CT_head, T_head)")
    print(f"置信度阈值: {args.conf}")
    print(f"平滑系数: {args.smoothing} | 死区: {args.dead_zone}px")
    print("=" * 50)
    print(f"按 [{args.toggle_key.upper()}] 键切换开关")
    print(f"{args.delay}秒后开始...")
    time.sleep(args.delay)
    print("就绪！等待按键开启...")

    fps_history = []

    with mss.mss() as sct:
        while True:
            t0 = time.perf_counter()

            # 截屏
            img = np.array(sct.grab(MONITOR))
            img = img[:, :, :3]

            # 推理
            results = model(img, conf=args.conf, verbose=False)

            # 获取鼠标位置
            cur_x, cur_y = aim._get_pos()

            if hotkey.enabled:
                target = find_nearest_head(results, cur_x, cur_y,
                                           conf_threshold=args.conf)
                if target:
                    tx, ty, dist = target
                    new_x, new_y = aim.update(tx, ty)
                    aim.move_to(new_x, new_y)
                    status = f"🎯 锁定 ({tx:.0f}, {ty:.0f}) dist={dist:.0f}px"
                else:
                    status = "🔍 搜索中..."
            else:
                status = "⏸  已暂停 (按 [X] 开启)"

            # FPS
            elapsed = time.perf_counter() - t0
            fps_history.append(elapsed)
            if len(fps_history) > 60:
                fps_history.pop(0)
            avg_fps = 1.0 / (sum(fps_history) / len(fps_history))

            print(f"\r[FPS: {avg_fps:.1f}] {status}  ", end="", flush=True)


# ============== 模式2: 视频测试 ==============
def run_video_test(args):
    """视频测试：检测视频中的目标，可视化 + 模拟鼠标移动"""
    import cv2

    cap = cv2.VideoCapture(args.video)
    if not cap.isOpened():
        print(f"❌ 无法打开视频: {args.video}")
        return

    print("加载模型...")
    model = YOLO(args.model)

    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    video_fps = cap.get(cv2.CAP_PROP_FPS)
    video_w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    video_h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    print(f"视频: {args.video}")
    print(f"分辨率: {video_w}x{video_h} | FPS: {video_fps:.1f} | 总帧数: {total_frames}")
    print(f"目标类别: 全部 (CT, CT_head, T, T_head)")
    print("=" * 50)

    # 颜色映射
    COLORS = {
        0: (255, 100, 100),   # CT - 蓝
        1: (100, 100, 255),   # CT_head - 红
        2: (100, 255, 100),   # T - 绿
        3: (100, 255, 255),   # T_head - 黄
    }
    NAMES = {0: "CT", 1: "CT_head", 2: "T", 3: "T_head"}

    # 模拟鼠标位置（视频中心）
    mouse_x, mouse_y = video_w / 2, video_h / 2
    fps_history = []

    # 写入输出视频（默认保存到 output/ 目录）
    writer = None
    output_path = args.output
    if not output_path:
        video_name = os.path.splitext(os.path.basename(args.video))[0]
        output_path = os.path.join(OUTPUT_DIR, f"{video_name}_output.mp4")

    os.makedirs(OUTPUT_DIR, exist_ok=True)

    if output_path:
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        writer = cv2.VideoWriter(output_path, fourcc, video_fps, (video_w, video_h))

    cv2.namedWindow("YOLO Detection Test", cv2.WINDOW_NORMAL)
    cv2.resizeWindow("YOLO Detection Test", min(video_w, 1280), min(video_h, 720))

    frame_idx = 0
    detection_summary = {
        "total_frames": total_frames,
        "video": args.video,
        "fps": video_fps,
        "frames_with_detections": 0,
        "total_detections": 0,
        "detections_by_class": {},
        "frames": []
    }

    while True:
        ret, frame = cap.read()
        if not ret:
            break
        frame_idx += 1
        t0 = time.perf_counter()

        # 推理 - 使用 imgsz=416 加速
        results = model(frame, conf=args.conf, imgsz=416, verbose=False)
        detections = get_all_detections(results, conf_threshold=args.conf)

        # 收集检测结果数据
        frame_detection_data = []
        if detections:
            detection_summary["frames_with_detections"] += 1
            for det in detections:
                cls_id = det["cls"]
                cls_name = NAMES.get(cls_id, str(cls_id))
                detection_summary["total_detections"] += 1
                detection_summary["detections_by_class"][cls_name] = detection_summary["detections_by_class"].get(cls_name, 0) + 1
                frame_detection_data.append({
                    "class": cls_name,
                    "class_id": cls_id,
                    "confidence": round(det["conf"], 3),
                    "box": [round(v, 1) for v in det["box"]],
                    "center": [round(v, 1) for v in det["center"]],
                    "head": [round(v, 1) for v in det["head"]]
                })

            if args.summary:
                detection_summary["frames"].append({
                    "frame": frame_idx,
                    "detections": frame_detection_data
                })

        # 屏幕中心作为准星
        cross_x, cross_y = video_w / 2, video_h / 2

        # 找最近头部用于模拟鼠标，没有头部则回退到身体
        ref_x, ref_y = mouse_x, mouse_y
        best = None
        min_dist = float('inf')
        for det in detections:
            hx, hy = det["head"] if det["cls"] in HEAD_CLASSES else det["center"]
            dist = math.sqrt((hx - ref_x)**2 + (hy - ref_y)**2)
            if dist < min_dist:
                min_dist = dist
                best = det

        # 模拟平滑移动 - 加快响应
        if best and args.mouse_sim:
            tx, ty = best["head"]
            dx = tx - mouse_x
            dy = ty - mouse_y
            dist = math.sqrt(dx*dx + dy*dy)
            if dist > 3:
                smooth = args.smoothing
                if dist > 100:
                    smooth = min(0.5, smooth + dist * 0.0015)
                elif dist < 30:
                    smooth = max(0.08, smooth * 0.6)
                mouse_x += dx * smooth
                mouse_y += dy * smooth

        # ---- 绘制 ----
        # 绘制所有检测框
        for det in detections:
            x1, y1, x2, y2 = [int(v) for v in det["box"]]
            cls = det["cls"]
            conf = det["conf"]
            color = COLORS.get(cls, (200, 200, 200))
            name = NAMES.get(cls, str(cls))

            cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
            label = f"{name} {conf:.2f}"
            (tw, th), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1)
            cv2.rectangle(frame, (x1, y1 - th - 8), (x1 + tw, y1), color, -1)
            cv2.putText(frame, label, (x1, y1 - 4),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)

            # 头部瞄准点
            hx, hy = det["head"]
            cv2.circle(frame, (int(hx), int(hy)), 4, (0, 0, 255), -1)

        # 绘制准星
        cx, cy = int(cross_x), int(cross_y)
        cv2.line(frame, (cx - 15, cy), (cx + 15, cy), (0, 255, 0), 2)
        cv2.line(frame, (cx, cy - 15), (cx, cy + 15), (0, 255, 0), 2)
        cv2.circle(frame, (cx, cy), 20, (0, 255, 0), 1)

        # 绘制模拟鼠标
        if args.mouse_sim:
            mx, my = int(mouse_x), int(mouse_y)
            cv2.circle(frame, (mx, my), 6, (0, 165, 255), -1)
            cv2.circle(frame, (mx, my), 12, (0, 165, 255), 2)
            # 准星到模拟鼠标的连线
            if best:
                tx, ty = best["head"]
                cv2.line(frame, (mx, my), (int(tx), int(ty)),
                        (0, 165, 255), 1, cv2.LINE_AA)

        # HUD 信息
        elapsed = time.perf_counter() - t0
        fps_history.append(elapsed)
        if len(fps_history) > 60:
            fps_history.pop(0)
        avg_fps = 1.0 / (sum(fps_history) / len(fps_history))

        status_text = f"FPS: {avg_fps:.1f} | Frame: {frame_idx}/{total_frames}"
        if best:
            hx, hy = best["head"]
            status_text += f" | Target: ({hx:.0f},{hy:.0f}) conf={best['conf']:.2f}"
        else:
            status_text += " | No target"

        cv2.rectangle(frame, (0, 0), (video_w, 30), (0, 0, 0), -1)
        cv2.putText(frame, status_text, (10, 22),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)

        if writer:
            writer.write(frame)

        # 保存有检测结果的帧图片（用于测试展示）
        if detections and args.save_frames:
            save_dir = os.path.join(OUTPUT_DIR, "frames")
            os.makedirs(save_dir, exist_ok=True)
            frame_path = os.path.join(save_dir, f"frame_{frame_idx:05d}.jpg")
            cv2.imwrite(frame_path, frame)

        cv2.imshow("YOLO Detection Test", frame)

        key = cv2.waitKey(1) & 0xFF
        if key == ord('q') or key == 27:  # q 或 ESC 退出
            break
        elif key == ord(' '):  # 空格暂停
            cv2.waitKey(0)

    cap.release()
    if writer:
        writer.release()
    cv2.destroyAllWindows()

    # 保存检测汇总
    detection_summary["processed_frames"] = frame_idx
    if args.summary:
        summary_path = os.path.join(OUTPUT_DIR, "detection_summary.json")
        with open(summary_path, "w") as f:
            json.dump(detection_summary, f, indent=2, ensure_ascii=False)
        print(f"检测汇总: {summary_path}")

    print(f"\n处理完成: {frame_idx} 帧")
    print(f"有检测结果的帧: {detection_summary['frames_with_detections']}")
    print(f"总检测目标数: {detection_summary['total_detections']}")
    if args.output:
        print(f"输出视频: {args.output}")


# ============== 主入口 ==============
def main():
    import argparse

    parser = argparse.ArgumentParser(description="CSGO 自动瞄准系统")
    subparsers = parser.add_subparsers(dest="mode", help="运行模式")

    # 模式1: 实时瞄准
    aim_parser = subparsers.add_parser("aim", help="实时截屏 + 瞄准（热键控制）")
    aim_parser.add_argument("--model", type=str, default=MODEL_PATH, help="模型路径")
    aim_parser.add_argument("--conf", type=float, default=CONF_THRESHOLD, help="置信度阈值")
    aim_parser.add_argument("--smoothing", type=float, default=0.15, help="平滑系数")
    aim_parser.add_argument("--dead-zone", type=float, default=3.0, help="死区像素")
    aim_parser.add_argument("--delay", type=int, default=3, help="启动前等待秒数")
    aim_parser.add_argument("--toggle-key", type=str, default="x", help="切换开关按键")

    # 模式2: 视频测试
    test_parser = subparsers.add_parser("test", help="视频检测 + 模拟鼠标追踪")
    test_parser.add_argument("video", type=str, help="视频文件路径")
    test_parser.add_argument("--model", type=str, default=MODEL_PATH, help="模型路径")
    test_parser.add_argument("--conf", type=float, default=CONF_THRESHOLD, help="置信度阈值")
    test_parser.add_argument("--smoothing", type=float, default=0.25, help="鼠标平滑系数")
    test_parser.add_argument("--mouse-sim", action="store_true", default=True, help="模拟鼠标移动")
    test_parser.add_argument("--no-mouse-sim", dest="mouse_sim", action="store_false", help="不模拟鼠标")
    test_parser.add_argument("--output", type=str, default="", help="输出视频路径 (默认: <项目目录>/output/<视频名>_output.mp4)")
    test_parser.add_argument("--save-frames", action="store_true", default=False, help="保存有检测结果的帧图片到 <项目目录>/output/frames/")
    test_parser.add_argument("--summary", action="store_true", default=False, help="生成检测结果汇总JSON到 <项目目录>/output/detection_summary.json")

    args = parser.parse_args()

    if args.mode == "aim":
        run_aimbot(args)
    elif args.mode == "test":
        run_video_test(args)
    else:
        parser.print_help()
        print("\n示例:")
        print("  python aimbot.py aim                        # 实时瞄准")
        print("  python aimbot.py aim --toggle-key x          # 按 X 切换")
        print("  python aimbot.py test gameplay.mp4           # 视频测试")
        print("  python aimbot.py test gameplay.mp4 --output result.mp4  # 保存结果")


if __name__ == "__main__":
    main()
