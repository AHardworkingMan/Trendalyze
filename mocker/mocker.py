import json
import random
from datetime import datetime, timedelta
import uuid
import numpy as np  # 建议使用 numpy 生成正态分布的时间，更像人

# ===========================
# 1. 基础配置与辅助函数
# ===========================

device_types = {
    "Switch": ["switch"],
    "TV": ["volume", "switch", "channel", "mute"],
    "AirConditioner": ["temperature", "humidity", "switch", "mode"],
    "Curtain": ["shadeLevel", "switch"],
    "SoundBar": ["volume", "switch", "mute"],
    "Light": ["switch", "switchLevel", "temperature"],
    "DoorLock": ["lock", "unlock", "status"],
    "SmartFridge": ["temperature", "humidity", "doorStatus", "mode"],
    "Washer": ["washMode", "spinSpeed", "status", "temperature"],
    "SmartSpeaker": ["volume", "switch", "playback", "mute", "equalizer"]
}


def generate_uuid():
    return str(uuid.uuid4())


# 辅助：根据给定的基准时间和偏差生成随机时间
# 例如：基准是8:00，sigma是15分钟，生成的时间会在 7:45-8:15 之间呈正态分布
def get_random_time(base_datetime, sigma_minutes=15):
    offset_seconds = int(random.gauss(0, sigma_minutes * 60))
    new_time = base_datetime + timedelta(seconds=offset_seconds)
    return int(new_time.timestamp())


# ===========================
# 2. 设备生成 (逻辑保持不变，稍作优化)
# ===========================

def generate_device_data(num_devices=15):
    devices = []
    # 强制包含关键设备以便形成剧本
    required_types = ["DoorLock", "Curtain", "AirConditioner", "TV", "SmartFridge", "Washer"]

    # 1. 先生成必选设备
    for dtype in required_types:
        devices.append({
            "device_id": generate_uuid(),
            "device_type": dtype,
            "capabilities": device_types[dtype]
        })

    # 2. 生成多个灯和开关 (模拟客厅灯、卧室灯、厨房开关等)
    for _ in range(4):  # 4个灯
        devices.append({"device_id": generate_uuid(), "device_type": "Light", "capabilities": device_types["Light"]})
    for _ in range(3):  # 3个开关
        devices.append({"device_id": generate_uuid(), "device_type": "Switch", "capabilities": device_types["Switch"]})

    # 3. 补足剩余数量
    while len(devices) < num_devices:
        dtype = random.choice(list(device_types.keys()))
        devices.append({
            "device_id": generate_uuid(),
            "device_type": dtype,
            "capabilities": device_types[dtype]
        })

    return devices


# 辅助：按类型查找设备ID列表
def get_devices_by_type(devices_list, dtype):
    return [d for d in devices_list if d['device_type'] == dtype]


# ===========================
# 3. 事件生成 (核心：基于剧本的生活流)
# ===========================

def generate_daily_routine(date, devices_map, events_list):
    """
    生成某一天的所有事件。
    date: datetime 对象 (当天的 00:00)
    devices_map: 按类型分类的设备字典
    """

    # 判断是工作日还是周末
    is_weekend = date.weekday() >= 5

    # --- 设定当天的基准时间点 ---
    if is_weekend:
        wake_hour = 9  # 周末睡懒觉
        leave_hour = 11  # 周末出门玩或不出门
        home_hour = 16
        sleep_hour = 23
    else:
        wake_hour = 7  # 工作日
        leave_hour = 8.5
        home_hour = 18.5
        sleep_hour = 22.5

    # 获取特定设备 (随机取一个作为主设备，例如主卧窗帘)
    curtains = devices_map.get("Curtain", [])
    lights = devices_map.get("Light", [])
    locks = devices_map.get("DoorLock", [])
    acs = devices_map.get("AirConditioner", [])
    tvs = devices_map.get("TV", [])
    fridges = devices_map.get("SmartFridge", [])

    main_curtain = curtains[0] if curtains else None
    main_lock = locks[0] if locks else None
    living_room_light = lights[0] if lights else None
    bedroom_light = lights[1] if len(lights) > 1 else (lights[0] if lights else None)
    main_ac = acs[0] if acs else None
    main_tv = tvs[0] if tvs else None

    # ==========================
    # 场景 1: 起床模式 (Morning Routine)
    # 逻辑：闹钟 -> 窗帘开 -> 卧室灯关(因为天亮了) -> 去厨房(冰箱)
    # ==========================
    base_time = date.replace(hour=int(wake_hour), minute=30)

    # 1. 窗帘开启
    if main_curtain:
        t = get_random_time(base_time, 5)  # 偏差5分钟
        events_list.append(create_event(t, main_curtain, "Curtain.switch", "on"))  # on代表开
        events_list.append(create_event(t + 2, main_curtain, "Curtain.shadeLevel", "100"))

    # 2. 如果之前开了卧室灯，现在关闭
    if bedroom_light:
        t = get_random_time(base_time + timedelta(minutes=5), 2)
        events_list.append(create_event(t, bedroom_light, "Light.switch", "off"))

    # 3. 打开冰箱拿早餐
    if fridges:
        t = get_random_time(base_time + timedelta(minutes=15), 10)
        events_list.append(create_event(t, fridges[0], "SmartFridge.doorStatus", "open"))
        events_list.append(create_event(t + 15, fridges[0], "SmartFridge.doorStatus", "close"))  # 15秒后关门

    # ==========================
    # 场景 2: 离家模式 (Leaving Home) - 仅工作日或周末出门
    # 逻辑：关空调 -> 关所有灯 -> 门锁上锁
    # ==========================
    if not is_weekend or (is_weekend and random.random() > 0.3):  # 周末有30%概率宅在家
        base_time = date.replace(hour=int(leave_hour), minute=0)

        # 1. 关空调
        if main_ac:
            t = get_random_time(base_time - timedelta(minutes=2), 2)
            events_list.append(create_event(t, main_ac, "AirConditioner.switch", "off"))

        # 2. 关掉所有亮着的灯
        for light in lights:
            t = get_random_time(base_time - timedelta(minutes=1), 1)
            events_list.append(create_event(t, light, "Light.switch", "off"))

        # 3. 锁门 (最后一步)
        if main_lock:
            t = get_random_time(base_time, 2)
            events_list.append(create_event(t, main_lock, "DoorLock.lock", "locked"))

    # ==========================
    # 场景 3: 回家模式 (Coming Home)
    # 逻辑：指纹解锁 -> 门锁开 -> 玄关灯开 -> 客厅灯开 -> 空调开 -> 电视开
    # ==========================
    # 如果没出门就不触发回家，这里简化处理，假设都会触发回家（或者从外面回来）
    base_time = date.replace(hour=int(home_hour), minute=0)

    # 1. 解锁
    if main_lock:
        t = get_random_time(base_time, 5)
        events_list.append(create_event(t, main_lock, "DoorLock.unlock", "unlocked"))
        entry_time = t  # 记录进门时间

        # 2. 开客厅灯 (进门后10秒)
        if living_room_light:
            events_list.append(create_event(entry_time + 10, living_room_light, "Light.switch", "on"))

        # 3. 开空调 (进门后30秒)
        if main_ac:
            events_list.append(create_event(entry_time + 30, main_ac, "AirConditioner.switch", "on"))
            events_list.append(create_event(entry_time + 32, main_ac, "AirConditioner.temperature", "24"))

        # 4. 晚上看电视 (进门后30分钟)
        if main_tv:
            t_tv = get_random_time(datetime.fromtimestamp(entry_time) + timedelta(minutes=30), 10)
            events_list.append(create_event(t_tv, main_tv, "TV.switch", "on"))
            # 换几个台
            events_list.append(create_event(t_tv + 60, main_tv, "TV.channel", "set"))

    # ==========================
    # 场景 4: 睡觉模式 (Bedtime)
    # 逻辑：关电视 -> 关窗帘 -> 关客厅灯 -> 关空调(或者调睡眠模式)
    # ==========================
    base_time = date.replace(hour=int(sleep_hour), minute=30)

    # 1. 关电视
    if main_tv:
        t = get_random_time(base_time, 10)
        events_list.append(create_event(t, main_tv, "TV.switch", "off"))

    # 2. 关窗帘
    if main_curtain:
        t = get_random_time(base_time + timedelta(minutes=5), 2)
        events_list.append(create_event(t, main_curtain, "Curtain.switch", "off"))  # 关窗帘

    # 3. 关灯
    for light in lights:
        t = get_random_time(base_time + timedelta(minutes=10), 2)
        events_list.append(create_event(t, light, "Light.switch", "off"))


def create_event(timestamp, device, event_type, value=None):
    """封装事件对象"""
    e = {
        "event_id": generate_uuid(),
        "timestamp": int(timestamp),
        "device_id": device["device_id"],
        "event_type": event_type
    }
    # 可以在这里扩展，把具体的 value (如温度24度) 放入 extra_data 字段，
    # 但为了配合你之前的 LSTM 代码，主要看 event_type
    return e


def generate_full_simulation(days=60):
    # 1. 生成设备
    devices = generate_device_data(15)

    # 将设备按类型归类，方便剧本调用
    devices_map = {}
    for d in devices:
        dtype = d['device_type']
        if dtype not in devices_map:
            devices_map[dtype] = []
        devices_map[dtype].append(d)

    # 2. 生成事件
    events = []
    end_date = datetime.now()
    start_date = end_date - timedelta(days=days)

    current_date = start_date
    while current_date < end_date:
        # 每天生成一次剧本
        generate_daily_routine(current_date, devices_map, events)
        current_date += timedelta(days=1)

    # 3. 按时间戳排序 (非常重要，否则时序模型无法训练)
    events.sort(key=lambda x: x['timestamp'])

    return devices, events


# ===========================
# 4. 执行与保存
# ===========================

if __name__ == "__main__":
    print("正在生成模拟数据...")
    devices, events = generate_full_simulation(days=360)

    # 保存设备
    with open("devices.json", "w") as f:
        json.dump(devices, f, indent=4)

    # 保存事件
    with open("device_events.json", "w") as f:
        json.dump(events, f, indent=4)

    print(f"生成完成！\n设备数量: {len(devices)}\n事件数量: {len(events)}")
    print("数据已保存至 devices.json 和 device_events.json")