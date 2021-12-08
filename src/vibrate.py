import busio
import board
import time
import numpy as np
from adafruit_bus_device.i2c_device import I2CDevice

DEVICE_ADDRESS = 0x4A  # device address of the haptic


# The follow is for I2C communications
i2c = busio.I2C(board.SCL, board.SDA)
device = I2CDevice(i2c, DEVICE_ADDRESS)


def write_register(dev, register, mask, value, n_bytes=1, _startPosition=0):
    # Write a wregister number and value
    init = np.ubyte(0)
    init = read_register(dev, register, n_bytes)
    init &= mask
    init |= value << _startPosition
    buf = bytearray(1 + n_bytes)
    buf[0] = register
    buf[1:] = init.tobytes()
    with dev:
        dev.write(buf)


def read_register(dev, register, n_bytes=1) -> np.ubyte:
    # write a register number then read back the value
    reg = register.to_bytes(1, "little")
    buf = bytearray(n_bytes)
    with dev:
        dev.write_then_readinto(reg, buf)
    return np.frombuffer(buf, dtype=np.ubyte)


def begin():
    chipRev = np.ubyte(0)
    btn_status = read_register(device, 0x00)
    chipRev |= btn_status << 8
    chipRev |= btn_status
    return 0xBA == chipRev[0]


def vibrate(val):
    accelState = read_register(device, 0x13)[0]
    accelState &= 0x04
    accelState = accelState >> 2
    if accelState == 0x01 and val > 0x7F:
        val = 0x7F
    elif val > 0xFF:
        val = 0xFF

    write_register(device, 0x23, 0x00, val, 7)


def get_vibrate():
    return read_register(device, 0x23)


if not begin():
    print("Unable to find motor")


def set_actuator_type():
    ## LRA_TYPE == 0x00
    write_register(device, 0x13, 0xDF, 0x00, 5, 5)


def set_actuator_abs_volt(voltage):
    voltage = voltage / (23.4 * pow(10, -3))
    write_register(device, 0x0D, 0x00, np.ubyte(voltage), 7)


def set_actuator_nom_volt(voltage):
    voltage = voltage / (23.4 * pow(10, -3))
    write_register(device, 0x0C, 0x00, np.ubyte(voltage), 7)


def set_actuator_imax(max_c):
    max_c = (max_c - 28.6) / 7.2
    write_register(device, 0x0E, 0xE0, np.ubyte(max_c), 4)


def set_actuator_impedence(resistance):
    max_c = read_register(device, 0x0E) | 0x1F
    v2iFactor = np.ubyte((resistance * (max_c + 4)) / 1.6104)
    msbImpedance = (v2iFactor - (v2iFactor & 0x00FF)) / 256
    lsbImpedance = v2iFactor - (256 * (v2iFactor & 0x00FF))
    write_register(device, 0x10, 0x00, np.ubyte(lsbImpedance), 7)
    write_register(device, 0x0F, 0x00, np.ubyte(msbImpedance), 7)


def enable_freq_track(enable):
    write_register(device, 0x13, 0xF7, enable, 1, 3)


def set_actuator_lra_freq(frequency):
    lraPeriod = np.ubyte(1 / (frequency * (1333.32 * pow(10, -9))))
    msbFrequency = (lraPeriod - (lraPeriod & 0x007F)) / 128
    lsbFrequency = lraPeriod - 128 * (lraPeriod & 0xFF00)

    write_register(device, 0x46, 0x00, np.ubyte(msbFrequency), 7)
    write_register(device, 0x47, 0x80, np.ubyte(lsbFrequency), 7)


def set_operation_in_dro_mode():
    ## DRO_MODE == 0x01
    write_register(device, 0x22, 0xF8, 0x01, 1)


def set_all_defaults():
    set_actuator_type()
    set_actuator_abs_volt(2.5)
    set_actuator_nom_volt(2.5)
    set_actuator_imax(165.4)
    set_actuator_impedence(13.8)
    set_actuator_lra_freq(170)
    enable_freq_track(False)


set_all_defaults()
time.sleep(1)

set_operation_in_dro_mode()

while True:
    try:
        vibrate(0x64)
        print(get_vibrate())
        # don't slam the i2c bus
        time.sleep(1)
        vibrate(np.ubyte(0))
    except KeyboardInterrupt:
        break
